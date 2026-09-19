"""Streaming prequential log-loss estimates with exact file coverage.

Equal file segments are decoded in lockstep with persistent KV caches. Each
segment's first token uses a fixed-width code; every subsequent segment token is
scored before training on it, including short final chunks and updates. Fewer
than `batch` tokens left outside the equal segments use the same fixed-width
code at the end of a complete run.

Cached states survive optimizer updates and retain historical context; this is a
stateful predictor, not a fresh forward of the current window. Coding order and
chunk/eviction policy can affect predictions. The windowed train.py loop is also
a valid predictor with different context resets.

Reported sizes include ideal model log loss and a verified raw BPE dictionary,
not finite-precision arithmetic coding, metadata or decoder-program overhead.
An actual codec must reproduce the model and cache updates deterministically.
"""
import os
import gc
import math
import time
import torch
from torch.nn import functional as F
import numpy as np
from model import GPT, GPTConfig, HashedNgram
import wandb
from muon import SingleDeviceMuon
from sparse_adam import SparseRowAdam
from coding_data import load_tokens, validate_dictionary
from loss_forecast import LossForecast

# ---------------------------------------------------------------------------
# run mode
# ---------------------------------------------------------------------------
SWEEP = False
SWEEP_COUNT = 10000
SWEEP_ID = None

DEFAULTS = {
	"architecture": "transformer",
	"forecast_interval": 256, # updates per forecast block; first estimate after 9 blocks
	"forecast_ema": 0.8,    # smoothing on the reported forecast; 0 -> report each raw fit
	"token_path": "10k_enwik9",
	"source_path": "enwik9",
	"dictionary_first": "first",
	"dictionary_second": "second",
	"ensemble_k": 1,       # members in the uniform mixture
	"seed": 0,             # member k initialises from seed+k; vary to measure seed spread
	# Replay (extra gradient passes over already-coded data) was removed. It was a large win
	# before the hashed n-gram table existed and is harmful with it. A forward-only probe showed
	# the model generalises fine on its own -- already-coded loss runs 0.84-0.92 of online -- and
	# that switching on replay gradients drives that ratio to ~0.34 while the pre-clip gradient
	# norm climbs from ~0.23 to 0.56+. Tried and failed: blended into one step, separate steps,
	# LR x0.5 and x0.7, and freezing the n-gram table during replay. Freezing the table changed
	# nothing, so neither Muon's renormalisation nor the table's capacity is the cause -- 7.6M
	# dense params overfit replayed data unaided. The remaining suspect is what replay *samples*:
	# ~2.2e8 replayed tokens over a full run, drawn uniformly from a GROWING prefix, so early data
	# is resampled far more than late. Fitting already-coded data earns nothing here; it is never
	# scored again. Any revival needs a different sampling rule, not another weight or optimizer.
	"context_length": GPTConfig.context_length,
	"vocab_size": GPTConfig.vocab_size,
	"n_layer": GPTConfig.n_layer,
	"n_head": GPTConfig.n_head,
	"n_embd": GPTConfig.n_embd,
	# streaming shape. batch*chunk*accum tokens between optimizer steps; keep that product equal
	# to train.py's total_batch_size*1024 to stay on the same LR scale.
	#
	# chunk and accum are interchangeable for update semantics -- both just set how many tokens are
	# coded with stale weights -- but not for cost: each forward re-reads the whole KV window, so
	# per-token cache traffic goes as (window+chunk)/chunk. Prefer a large chunk and accum=1 over a
	# small chunk and accum>1. What does trade bits against compute is their product.
	"batch": 32,            # parallel streams over equal segments of the file. 1 == one stream
	"chunk": 128,          # tokens coded per forward pass, per stream. 1 == one token at a time
	"accum": 1,            # forwards per optimizer step
	"window": 0,           # cached positions; 0 -> context_length
	"evict_tokens": 128,   # how much of the window to drop when it fills
	"max_tokens": 0,       # stop early; 0 -> code the whole file
	"compile": True,
	# Hashed n-gram embedding table (modded-nanogpt PR #360). Rows are reached only when their
	# n-gram occurs, so lookup and optimizer cost scale with distinct n-grams per chunk, not with
	# table size, and the table costs nothing in bits -- the decoder rebuilds it from what it has
	# already decoded. Worth -15.2 MB in total (144.20 -> 129.00).
	#
	# Measured on 10k_enwik9 at full coverage, MB of payload:
	#   rows        off 144.20 | 2M 131.16 | 8M 129.00        (more rows always won, at fixed width)
	#   ngram_lr    0.05 140.35 | 0.25 134.72 | 1.0 133.01    (at 2M rows); 1.0 129.00 | 2.0 129.75
	#   orders      (2,3) 129.00 | (2,3,4) 129.01 | (2,3,4,6) 130.22 | (2,3,16) 130.45
	#   row split   shared 129.00 | (1,1) 128.88 | (7,3) 129.28 | bigram-only 131.21
	#   table_dim   8Mx256 129.00 | 8Mx64 132.74 | 32Mx64 131.56
	#
	# The three structural variants (split, longer orders, narrow rows + projection) were built,
	# measured and removed. A row holds a learned representation rather than a count, so width is
	# the scarce resource: narrowing 256 -> 64 costs 3.74 MB and 4x the rows wins back only 1.18.
	# That also explains why collisions are benign and why order 3 works at ~10x load (79.3M
	# distinct contexts in 8M rows), and why the trigram channel is still worth 2.21 MB.
	#
	# ngram_lr tracks just under wte_lr (1.0 vs 1.31):
	# a table row wants roughly the step size of the dense embedding it is summed onto. That is
	# the bracket to use for any future sparse component -- an initial guess of 0.01, two orders
	# of magnitude low, made the whole approach look marginal.
	# 4M, not 8M. At 1540 B/row an 8M table peaks at ~13.6 GB of a 16 GB card, which leaves ~2 GB
	# for a desktop that is also driving the display -- that starves the compositor and hangs the
	# session. 4M peaks near 7.5 GB and leaves real headroom. Rows are sublinear anyway
	# (2M 131.16 | 8M 129.00 at n_layer=3), so this costs roughly 1 MB. Check free VRAM before
	# raising it, and do not raise it on a machine whose GPU is also driving a display.
	"ngram_rows": 4_000_000,   # 0 -> disabled
	"ngram_orders": (2, 3),    # hash widths; (2,3) == the PR's bigram + trigram channels
	"ngram_lr": 1.0,
	# Width of a stored row; 0 -> n_embd. Pin it to hold table memory fixed while n_embd grows:
	# rows are n_rows*row_dim, so widening the model would otherwise multiply 8M rows past VRAM.
	# Depth is the wrong axis here -- 256/6 is width:depth 43 against GPT-2's 64 and GPT-3's 128,
	# and doubling depth bought only 1.82 MB, uniformly across the stream rather than late.
	"ngram_row_dim": 0,
	"ngram_state_bf16": True,  # bf16 second moment: 516 B/row instead of 1028
	# Newton-Schulz iterations in Muon. Cost is per optimizer step, not per token: at this shape
	# Muon is ~1-3 ms of a ~12 ms step, so 3 -> 5 is a few percent of wall clock. It was dropped
	# to 3 purely for speed back when Muon ran twice per iteration and was 42% of the step; that
	# reason no longer applies. Fewer iterations leave the update less orthogonal -- closer to
	# raw momentum. Set to 5 on an observed loss win from a partial run; a completed A/B against
	# ns_steps=3 was started four times and killed by confounds each time, so the size effect is
	# still unmeasured.
	"ns_steps": 5,
	# Reverted to the pre-sweep values. Sweep xdf89ktn (run fvnatuns) found muon 0.04205 /
	# wte 1.8561 / lm_head 0.004112 / b 0.05168 / ngram 1.2427, worth -0.18 MB at n_layer=3
	# (128.82 vs 129.00). But a single-objective Bayesian search pushes learning rates as high as
	# they go without diverging, so the winner sits on the stability cliff by construction: those
	# values raised lm_head_lr 5.5x and b_lr 2.5x, and at n_layer=6 the pre-clip gradient norm
	# spiked to 3-4x baseline across roughly a quarter of the run. 0.18 MB is a poor trade for a
	# configuration that will not survive a change of depth, replay, or step count.
	#
	# Note clip_grad_norm_ is hardcoded at 1.0 while typical norms are ~0.23 and the worst spikes
	# reach ~0.95, so clipping never engages and offers no protection here.
	"muon_lr": 0.03530142167198156,
	"wte_lr": 1.3135687240595386,
	"lm_head_lr": 0.000745750997735,
	"b_lr": 0.02088101953753996,
	"muon_wd": 0.0,
	# Swept at fixed incumbent LRs on 10k_enwik9, full coverage (MB): 0.0 144.92, 0.2 144.77,
	# 0.4 144.45, 0.6 144.26, 0.8 144.20, 1.0 144.41. Unimodal with the optimum at 0.8; 0.6-0.8
	# are within noise of each other. 1.0 regresses because it leaves no constant phase, so the
	# LR is already decayed across the early stream where loss -- and so bits -- is highest.
	"cooldown_frac": 0.8,
	"warmup_frac": 0.0,
}

LOG2E = float(np.log2(np.e))

class Stream:
	"""Walks `batch` equal segments of the file in lockstep, handing out (input, target) chunks.

	Coding token t means feeding token t-1 and reading the logits above it, so inputs sit one
	position left of targets. Token 0 of each segment has no context and is not predicted by the
	model: those cost ceil(log2(vocab_size)) bits each.
	"""
	def __init__(self, tokens, chunk, batch):
		if chunk <= 0 or batch <= 0 or len(tokens) < batch:
			raise ValueError("Need positive chunk/batch and at least one token per segment")
		self.tokens = tokens
		self.chunk = chunk
		self.seg_len = len(tokens) // batch
		self.starts = np.arange(batch) * self.seg_len
		self.pos = 1                     # next token to code, relative to each segment start

	def next_chunk(self, n=None):
		n = self.chunk if n is None else n
		assert 0 < n <= self.chunk
		assert self.pos + n <= self.seg_len, "ran off the end of a segment"
		idx = self.starts[:, None] + (self.pos - 1) + np.arange(n)[None, :]
		x = torch.from_numpy(self.tokens[idx].astype(np.int64))
		y = torch.from_numpy(self.tokens[idx + 1].astype(np.int64))
		self.pos += n
		return x, y

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if device.startswith("cuda") else "cpu"
print(f"using device: {device}")

torch.set_float32_matmul_precision('high')


def train(config=None):
	run = wandb.init(project="hutter", config={**DEFAULTS, **(config or {})})
	config = wandb.config

	K = config.ensemble_k
	batch = config.batch
	chunk = config.chunk
	accum = config.accum
	assert K > 0 and batch > 0 and chunk > 0 and accum > 0
	assert config.max_tokens >= 0
	tokens_per_update = batch * chunk * accum
	print(f"coding {batch} streams x {chunk} tokens/forward, "
	      f"{tokens_per_update} tokens between optimizer steps")

	print("Validating BPE dictionary against the complete source file...")
	accounting = validate_dictionary(config.token_path, config.source_path,
		config.dictionary_first, config.dictionary_second, config.vocab_size)
	print(f"Verified {accounting['source_bytes']} source bytes; dictionary: {accounting['dictionary_bytes']} bytes")
	tokens = load_tokens(config.token_path)
	stream = Stream(tokens, chunk, batch)
	run.config.update({"num_tokens": len(tokens), "segment_len": stream.seg_len}, allow_val_change=True)

	# -----------------------------------------------------------------------
	# build K models, each with its own optimizer set and its own KV cache
	# -----------------------------------------------------------------------
	model_config = GPTConfig(
		context_length=config.context_length,
		vocab_size=config.vocab_size,
		n_layer=config.n_layer,
		n_head=config.n_head,
		n_embd=config.n_embd,
	)
	window = config.window or model_config.context_length

	# ONE table, shared by every ensemble member. It memorises stream statistics, which are a
	# property of the data rather than of any member, so copying it per member would multiply the
	# largest object in the model (6.2 GB at 4M rows) for no diversity. Members differ in their
	# dense weights only. The lookup is also evaluated once per chunk and the same tensor is fed
	# to every member, so autograd sums their gradients into the shared rows.
	shared_ngram = (HashedNgram(config.ngram_rows, model_config.n_embd, tuple(config.ngram_orders),
		max_batch_size=batch, row_dim=config.ngram_row_dim).to(device)
		if config.ngram_rows else None)

	def build_member(seed: int):
		torch.manual_seed(seed)
		raw_model = GPT(model_config)
		raw_model.setup_caches(max_batch_size=batch, chunk=chunk, window=window)
		raw_model.ngram = None          # the table is shared; see shared_ngram above
		raw_model = raw_model.to(device)
		# dynamic=False: letting dynamo mark the sequence dim dynamic breaks the rope table slice.
		model = torch.compile(raw_model, mode='max-autotune-no-cudagraphs', dynamic=False) if config.compile else raw_model

		optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=config.wte_lr, betas=(0.0, 0.999), fused=(device_type == "cuda"))
		optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=config.lm_head_lr, betas=(0.0, 0.999), fused=(device_type == "cuda"))
		w_params = [m.weight for m in raw_model.transformer.h.modules() if isinstance(m, torch.nn.Linear)]
		b_params = [m.bias for m in raw_model.transformer.h.modules() if isinstance(m, torch.nn.Linear)]
		b_params.append(raw_model.lm_head.bias)
		optimizer3 = SingleDeviceMuon(w_params, lr=config.muon_lr, momentum=0.95, weight_decay=config.muon_wd,
			ns_steps=config.ns_steps)
		optimizer4 = torch.optim.Adam(b_params, lr=config.b_lr, betas=(0.0, 0.999), fused=(device_type == "cuda"))
		optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]

		for opt in optimizers:
			for group in opt.param_groups:
				group["initial_lr"] = group["lr"]

		return raw_model, model, optimizers

	raw_models, models, optimizer_sets = [], [], []
	for k in range(K):
		raw_model, model, optimizers = build_member(seed=config.seed + k)
		raw_models.append(raw_model)
		models.append(model)
		optimizer_sets.append(optimizers)

	# Optimizers over the shared table and its projection: stepped once per update, not per
	# member, so the table's effective step size does not grow with K.
	shared_optimizers = []
	if shared_ngram is not None:
		# Only rows this step's gradient reached are touched, so step cost tracks distinct
		# n-grams rather than table size. See sparse_adam.py.
		shared_optimizers.append(SparseRowAdam([shared_ngram.table.weight], lr=config.ngram_lr,
			state_dtype=torch.bfloat16 if config.ngram_state_bf16 else torch.float32))
		if shared_ngram.proj is not None:
			shared_optimizers.append(SingleDeviceMuon([shared_ngram.proj.weight], lr=config.muon_lr,
				momentum=0.95, weight_decay=config.muon_wd, ns_steps=config.ns_steps))
	for opt in shared_optimizers:
		for group in opt.param_groups:
			group["initial_lr"] = group["lr"]

	ngram_count = shared_ngram.table.weight.numel() if shared_ngram is not None else 0
	weight_count = sum(p.numel() for p in raw_models[0].parameters())
	print(f"weights per model: {weight_count} | total across ensemble: {weight_count * K}"
	      + (f" | shared hashed n-gram table: {ngram_count/1e6:.1f}M params over "
	         f"{config.ngram_rows} rows (one copy for all {K} members)" if ngram_count else ""))

	# max_tokens caps model-scored tokens, rounded down to a whole lockstep round.
	positions_to_code = stream.seg_len - 1
	if config.max_tokens:
		positions_to_code = min(positions_to_code, config.max_tokens // batch)
	max_steps = math.ceil(positions_to_code / (chunk * accum))
	complete_file = positions_to_code == stream.seg_len - 1
	print(f"{max_steps} updates; {batch * positions_to_code} model-scored tokens")

	cooldown_frac = config.cooldown_frac
	warmup_frac = config.warmup_frac
	def get_lr(step: int):
		x = step / max_steps
		assert 0 <= x < 1
		if x < warmup_frac:
			return x / warmup_frac
		if x > 1 - cooldown_frac:
			w = (1 - x) / cooldown_frac
			return w * 1.0 + (1 - w) * 0.1
		return 1.0

	log_dir = "log"
	os.makedirs(log_dir, exist_ok=True)

	log_K = math.log(K)

	def token_logprobs(logits, y):
		"""log p(y_t | x_<t) for every position, shape [B, T]."""
		V = logits.size(-1)
		nll = F.cross_entropy(logits.reshape(-1, V).float(), y.flatten(), reduction='none')
		return -nll.view(y.shape)

	def code_chunk(x, y, scale):
		"""Code one chunk with the current weights, then backward into the gradient buffer.

		Each member extends its own KV cache by the chunk, so the context every member scores
		against is the whole stream so far (truncated to `window`), not just this chunk. Returns
		(nats under the uniform mixture, mean nats per member, per-segment mixture nats) --
		summed over the chunk. The per-segment vector is the spread the forecaster uses to
		separate content difficulty from learning trend; segments are far apart in the file, so
		they are near-independent samples at the same within-segment position.
		"""
		# One lookup for the whole ensemble, outside autocast (the table is fp32 and the lookup is
		# not a matmul). Members consume a detached leaf rather than the lookup's own graph: each
		# member's backward would otherwise free the shared subgraph and break the next member,
		# and retain_graph would keep every member's full graph alive instead of just this one
		# small one. Their gradients accumulate on the leaf; it is pushed back into the table
		# once, averaged over members so the table's step size does not grow with K.
		ng = shared_ngram(x, use_state=True) if shared_ngram is not None else None
		ng_leaf = ng.detach().requires_grad_(True) if ng is not None else None
		member_logprobs = []
		for raw_model, model in zip(raw_models, models):
			input_pos = raw_model.next_positions(x.shape[1], config.evict_tokens)
			with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
				logits = model(x, input_pos, ng_leaf)
			lp = token_logprobs(logits, y)
			(-lp.mean() * scale).backward()
			member_logprobs.append(lp.detach())
		if ng is not None and ng_leaf.grad is not None:
			ng.backward(ng_leaf.grad / K)

		with torch.no_grad():
			stacked = torch.stack(member_logprobs, dim=0)            # [K, batch, chunk]
			mix = torch.logsumexp(stacked, dim=0) - log_K            # [batch, chunk]
			return -mix.sum(), -stacked.mean(dim=0).sum(), -mix.sum(dim=1)

	def optimizer_step(step):
		"""Clip, set LR, step every member, then zero grads. Returns mean grad norm."""
		norms = []
		for raw_model, optimizers in zip(raw_models, optimizer_sets):
			# The n-gram table is excluded: its gradient is sparse, which clip_grad_norm_ cannot
			# consume, and a global norm over a table millions of rows tall would be meaningless
			# anyway when only a few thousand rows carry gradient.
			clipped = [p for p in raw_model.parameters() if p.grad is None or not p.grad.is_sparse]
			norms.append(torch.nn.utils.clip_grad_norm_(clipped, 1).detach())
			for opt in optimizers:
				for group in opt.param_groups:
					group["lr"] = group["initial_lr"] * get_lr(step)
				opt.step()
		for opt in shared_optimizers:                 # the shared table: stepped once, not per member
			for group in opt.param_groups:
				group["lr"] = group["initial_lr"] * get_lr(step)
			opt.step()
		for model in models:
			model.zero_grad(set_to_none=True)
		if shared_ngram is not None:
			shared_ngram.zero_grad(set_to_none=True)
		return torch.stack(norms).mean()

	ema_loss = None
	ema_beta = 0.98

	# Segment starts are transmitted before the first model-scored round.
	literal_bits = math.ceil(math.log2(config.vocab_size))
	first_token_bytes = batch * literal_bits / 8
	auc_loss = first_token_bytes
	predicted_tokens = 0
	forecast = LossForecast(batch * (stream.seg_len - 1),
		first_token_bytes + (len(tokens) - batch * stream.seg_len) * literal_bits / 8,
		accounting["dictionary_bytes"], config.forecast_interval, config.forecast_ema)

	for model in models:
		model.zero_grad(set_to_none=True)
	if shared_ngram is not None:
		shared_ngram.zero_grad(set_to_none=True)

	for step in range(0, max_steps):
		t0 = time.time()

		# ---- online pass: code tokens_per_update tokens with the CURRENT weights, one chunk
		# per forward per stream, then update once. Nothing in here is trained on before it is
		# coded; nats measure ideal payload cost before coder overhead.
		prefix_len = stream.pos          # tokens transmitted per segment, before this step
		nats = torch.zeros((), device=device)
		member_nats = torch.zeros((), device=device)
		seg_nats = torch.zeros(batch, device=device)

		step_positions = min(chunk * accum, positions_to_code - (stream.pos - 1))
		step_tokens = batch * step_positions
		remaining = step_positions
		while remaining:
			n = min(chunk, remaining)
			x, y = stream.next_chunk(n)
			remaining -= n
			x, y = x.to(device), y.to(device)
			mix_nats, mem_nats, chunk_seg_nats = code_chunk(x, y, n / step_positions)
			nats += mix_nats
			member_nats += mem_nats
			seg_nats += chunk_seg_nats

		weight_norm = optimizer_step(step)

		if device_type == "cuda":
			torch.cuda.synchronize()
		t1 = time.time()
		dt = t1 - t0

		step_loss = nats.item() / step_tokens        # nats/token under the mixture
		member_loss = member_nats.item() / step_tokens
		tok_per_sec = step_tokens / dt

		print(f"step {step:6d} | loss: {step_loss:.6f} | mean_member: {member_loss:.6f} | "
		      f"lr {get_lr(step):.4e} | dt: {dt*1000:.2f}ms | "
		      f"tok/sec: {tok_per_sec:.1f}")

		ema_loss = step_loss if ema_loss is None else ema_beta * ema_loss + (1 - ema_beta) * step_loss
		auc_loss += nats.item() * LOG2E / 8
		predicted_tokens += step_tokens

		segment_nats = seg_nats.detach().cpu().numpy()
		forecast_metrics = forecast.update(step_tokens, nats.item(), segment_nats,
			force=step == max_steps - 1)
		if forecast_metrics:
			print(f"projected final payload: {forecast_metrics['forecast/final_auc_loss']/1e6:.2f} MB | "
			      f"model spread: {forecast_metrics['forecast/model_spread_low_bytes']/1e6:.2f}–"
			      f"{forecast_metrics['forecast/model_spread_high_bytes']/1e6:.2f} MB")
		wandb.log({
			**forecast_metrics,
			"loss": step_loss,
			"ema_loss": ema_loss,
			"mean_member_loss": member_loss,
			"ensemble_gain": member_loss - step_loss,
			"prefix_len": prefix_len,
			"context_len": raw_models[0].cache_len,
			"lr": get_lr(step),
			"weight_norm": weight_norm,
			"bits_per_token": step_loss * LOG2E,
			"segment_loss_spread": float(segment_nats.std(ddof=1) / step_positions),
			'step': step,
			'dt': dt * 1000,
			"tokens": batch + predicted_tokens,
			"predicted_tokens": predicted_tokens,
			"estimated_total_bytes": auc_loss + accounting["dictionary_bytes"],
			"tok_per_sec": tok_per_sec,
			"auc_loss": auc_loss,
		})

	# Equal segments cover batch * seg_len tokens. Charge any remaining suffix
	# literally only after all segment prefixes have been transmitted.
	tail_tokens = len(tokens) - batch * stream.seg_len if complete_file else 0
	tail_bytes = tail_tokens * literal_bits / 8
	auc_loss += tail_bytes
	coded_tokens = batch + predicted_tokens + tail_tokens
	assert not complete_file or coded_tokens == len(tokens)
	run.summary.update(accounting)
	run.summary["first_token_bytes"] = first_token_bytes
	run.summary["literal_tail_tokens"] = tail_tokens
	run.summary["literal_tail_bytes"] = tail_bytes
	run.summary["predicted_tokens"] = predicted_tokens
	run.summary["complete_file"] = complete_file
	run.summary["estimated_total_bytes"] = auc_loss + accounting["dictionary_bytes"]
	if complete_file:
		run.summary["estimated_bits_per_byte"] = 8 * (auc_loss + accounting["dictionary_bytes"]) / accounting["source_bytes"]
	run.summary["size_kind"] = "ideal payload plus raw BPE dictionary; excludes coder, metadata, and decoder overhead"
	run.summary["final_auc_loss"] = auc_loss
	run.summary["final_ema_loss"] = ema_loss
	run.summary["compute_units"] = K
	run.summary["coded_tokens"] = coded_tokens

	run.finish()

	del raw_models, models, optimizer_sets
	gc.collect()
	torch._dynamo.reset()
	if device_type == "cuda":
		torch.cuda.empty_cache()


# Full-coverage runs, not prefixes. The objective is the area under the learning curve, so a
# truncated run is not a noisy estimate of the target -- it is a different target, one that
# systematically rewards configurations that adapt fast early. At ~10 min per pass a full run is
# affordable, so prefixes are used only to prune the diverging tail (see early_terminate).
#
# Learning rates are sampled log-uniformly: the previous uniform ranges put 92% of lm_head_lr
# draws above the incumbent, and capped wte_lr at 1.0 when the incumbent is 1.31 -- the sweep
# could not reach its own best known value. Ranges now bracket each incumbent by ~1.5 decades.
#
# Do not add warmup_frac/cooldown_frac here while early_terminate is on: get_lr keys off
# step/max_steps, so a pruned run compresses the whole schedule into the prefix.
sweep_config = {
	'method': 'bayes',
	'metric': {
		'name': 'estimated_total_bytes',   # compare runs with identical coverage
		'goal': 'minimize'
	},
	'early_terminate': {'type': 'hyperband', 'min_iter': 5000, 'eta': 3},
	# Roughly a decade around each incumbent, not 2-3. Five dimensions at 1.5 decades each is
	# ~10 decades of joint volume, which 60 trials cannot cover -- the first attempt put only 45%
	# of muon_lr draws inside its plausible 0.01-0.08 band and wasted the rest. Muon governs only
	# the transformer block weights, whose shape has not changed since these were tuned, so the
	# incumbents are good priors and deserve tight brackets. Widen one dimension if the winner
	# lands against an edge.
	'parameters': {
		'muon_lr':    {'distribution': 'log_uniform_values', 'min': 8e-3, 'max': 1.2e-1},  # 0.0353
		# the floor is left well below the incumbent on purpose: the n-gram table now supplies a
		# lot of the embedding-like capacity, so wte may want a smaller step than it did alone
		'wte_lr':     {'distribution': 'log_uniform_values', 'min': 3e-1, 'max': 2e+0},    # 1.3136
		'lm_head_lr': {'distribution': 'log_uniform_values', 'min': 1.5e-4, 'max': 8e-3},  # 0.00075
		'b_lr':       {'distribution': 'log_uniform_values', 'min': 5e-3, 'max': 1e-1},    # 0.0209
		# already bracketed at ~1.0, but it is summed into the same residual as wte, so it is
		# swept rather than pinned: a shift in wte_lr should be free to move it.
		'ngram_lr':   {'distribution': 'log_uniform_values', 'min': 4e-1, 'max': 2.5e+0},  # 1.0
	}
}

if __name__ == "__main__":
	if SWEEP:
		sweep_id = SWEEP_ID or wandb.sweep(sweep_config, project="hutter")
		print(f"sweep id: {sweep_id}")
		wandb.agent(sweep_id, train, count=SWEEP_COUNT, project="hutter")
	else:
		train()

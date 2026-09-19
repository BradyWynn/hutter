import os
import gc
import math
import time
import torch
from torch.nn import functional as F
import numpy as np
from model import GPT, GPTConfig
import wandb
from muon import SingleDeviceMuon
from coding_data import load_tokens, DataLoaderLite, validate_dictionary
# import lzma

# ---------------------------------------------------------------------------
# run mode
# ---------------------------------------------------------------------------
SWEEP = False          # False: one run using DEFAULTS. True: launch/join a wandb sweep.
SWEEP_COUNT = 10000       # runs for this agent to execute
SWEEP_ID = None        # set to an existing sweep id to join it instead of creating one

# Defaults for a single run. Under a sweep, wandb overrides the swept keys.
DEFAULTS = {
	"architecture": "transformer",
	"token_path": "5k_enwik9",
	"source_path": "enwik9",
	"dictionary_first": "first",
	"dictionary_second": "second",
	"compile": True,
	"ensemble_k": 1,       # members in the uniform mixture
	"replay_steps": 1,     # extra optimizer steps per online step, on already-coded data
	"replay_seed": 1234,   # replay sampling must be reproducible: the decoder replays it too
	"context_length": GPTConfig.context_length,
	"vocab_size": GPTConfig.vocab_size,
	"n_layer": GPTConfig.n_layer,
	"n_head": GPTConfig.n_head,
	"n_embd": GPTConfig.n_embd,
	"total_batch_size": 4,   # in sequences, not tokens
	"B": 4,                   # micro batch size
	"muon_lr": 0.03530142167198156,
	"wte_lr": 1.3135687240595386,
	"lm_head_lr": 0.000186437749434,
	"b_lr": 0.02088101953753996,
	"muon_wd": 0.0,
	"cooldown_frac": 0.0,
	"warmup_frac": 0.0,
}

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class ReplaySampler:
	"""Samples random windows from tokens[:limit] -- data the decoder has already seen.

	Only ever called with limit == the number of tokens transmitted so far, so nothing
	here leaks future data into the model. Uses its own RNG so the sampling sequence is
	reproducible independently of any other use of numpy's global random state.
	"""
	def __init__(self, tokens, T, seed):
		self.tokens = tokens
		self.T = T
		self.rng = np.random.default_rng(seed)

	def available(self, limit):
		return limit - self.T - 1

	def batch(self, n_seq, limit):
		T = self.T
		max_start = self.available(limit)          # need x[s:s+T] and y[s+1:s+T+1]
		starts = self.rng.integers(0, max_start + 1, size=n_seq)
		x_idx = starts[:, None] + np.arange(T)[None, :]

		x = self.tokens[x_idx]
		y = self.tokens[x_idx + 1]

		x = torch.from_numpy(x.astype(np.int64))
		y = torch.from_numpy(y.astype(np.int64))
		return x, y

device = "cpu"
if torch.cuda.is_available():
	device = "cuda"
print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.set_float32_matmul_precision('high')


def train(config=None):
	# Under an agent, wandb.init() picks up the sweep's parameters; values swept
	# override the matching keys in DEFAULTS.
	run = wandb.init(project="hutter", config={**DEFAULTS, **(config or {})})
	config = wandb.config

	K = config.ensemble_k
	replay_steps = config.replay_steps
	total_batch_size = config.total_batch_size
	B = config.B
	assert K > 0 and B > 0 and total_batch_size > 0 and config.context_length > 0
	assert replay_steps >= 0
	assert total_batch_size % B == 0, "make sure total_batch_size is divisible by B"
	grad_accum_steps = total_batch_size // B
	print(f"total desired batch size: {total_batch_size}")
	print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

	print("Validating BPE dictionary against the complete source file...")
	accounting = validate_dictionary(config.token_path, config.source_path,
		config.dictionary_first, config.dictionary_second, config.vocab_size)
	print(f"Verified {accounting['source_bytes']} source bytes; dictionary: {accounting['dictionary_bytes']} bytes")
	train_loader = DataLoaderLite(B=B, T=config.context_length, tokens=load_tokens(config.token_path))
	replay_sampler = ReplaySampler(train_loader.tokens, config.context_length, config.replay_seed)
	run.config.update({"num_tokens": len(train_loader.tokens)}, allow_val_change=True)

	# -----------------------------------------------------------------------
	# build K models, each with its own optimizer set
	# -----------------------------------------------------------------------
	model_config = GPTConfig(
		context_length=config.context_length,
		vocab_size=config.vocab_size,
		n_layer=config.n_layer,
		n_head=config.n_head,
		n_embd=config.n_embd,
	)

	def build_member(seed: int):
		torch.manual_seed(seed)
		raw_model = GPT(model_config).to(device)
		model = torch.compile(raw_model, mode='max-autotune-no-cudagraphs') if config.compile else raw_model

		optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=config.wte_lr, betas=(0.0, 0.999), fused=(device_type == "cuda"))
		optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=config.lm_head_lr, betas=(0.0, 0.999), fused=(device_type == "cuda"))
		w_params = [m.weight for m in raw_model.transformer.h.modules() if isinstance(m, torch.nn.Linear)]
		b_params = [m.bias for m in raw_model.transformer.h.modules() if isinstance(m, torch.nn.Linear)]
		b_params.append(raw_model.lm_head.bias)
		optimizer3 = SingleDeviceMuon(w_params, lr=config.muon_lr, momentum=0.95, weight_decay=config.muon_wd)
		optimizer4 = torch.optim.Adam(b_params, lr=config.b_lr, betas=(0.0, 0.999), fused=(device_type == "cuda"))
		optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]

		for opt in optimizers:
			for group in opt.param_groups:
				group["initial_lr"] = group["lr"]

		return raw_model, model, optimizers

	raw_models, models, optimizer_sets = [], [], []
	for k in range(K):
		raw_model, model, optimizers = build_member(seed=k)   # different seed => diverse members
		raw_models.append(raw_model)
		models.append(model)
		optimizer_sets.append(optimizers)

	weight_count = sum(p.numel() for p in raw_models[0].parameters())
	total_weight_count = weight_count * K
	print(f"weights per model: {weight_count} | total across ensemble: {total_weight_count}")

	tokens_per_update = total_batch_size * config.context_length
	max_steps = math.ceil((train_loader.n - 1) / tokens_per_update)

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

	def ensemble_nll(logprob_list):
		"""NLL of the uniform mixture: -log( (1/K) * sum_k p_k(y) ), averaged over tokens."""
		stacked = torch.stack(logprob_list, dim=0)             # [K, B, T]
		mix = torch.logsumexp(stacked, dim=0) - log_K          # [B, T]
		return -mix.mean(), -mix[:, -1].mean()                 # (all positions, last position)

	def accumulate_micro(xy_per_model, scale):
		"""Forward each of the K members on its own (x, y) micro-batch, backward each
		member's own CE. `xy_per_model` is a list of K (x, y) pairs -- pass the same
		pair K times to give every member an identical batch, or K different pairs to
		diversify what each member sees (as replay does).

		Returns detached (ensemble nll, ensemble nll at last position, mean member nll).
		`scale` is this micro-batch's fraction of the update's scored tokens.

		Note: the "ensemble nll" mixture is only meaningful when every member is scored
		against the same targets -- with diversified batches, use mean member nll instead.
		"""
		member_logprobs = []
		for model, (x, y) in zip(models, xy_per_model):
			with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
				logits = model(x)
			lp = token_logprobs(logits, y)
			(-lp.mean() * scale).backward()        # only touches this member's graph
			member_logprobs.append(lp.detach())    # values survive the freed graph

		with torch.no_grad():
			ens_loss, ens_loss_last = ensemble_nll(member_logprobs)
			member_loss = sum(-lp.mean() for lp in member_logprobs) / K
		return ens_loss, ens_loss_last, member_loss

	def optimizer_step(step):
		"""Clip, set LR, step every member, then zero grads. Returns mean grad norm."""
		norms = []
		for raw_model, optimizers in zip(raw_models, optimizer_sets):
			norms.append(torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1).detach())
			for opt in optimizers:
				for group in opt.param_groups:
					group["lr"] = group["initial_lr"] * get_lr(step)
				opt.step()
		for model in models:
			model.zero_grad(set_to_none=True)
		return torch.stack(norms).mean()

	ema_loss = None
	ema_beta = 0.98  # averages over roughly the last 1/(1-beta) ≈ 50 steps

	# First token uses a fixed-width vocabulary code. Remaining tokens use ideal NLL.
	first_token_bytes = math.ceil(math.log2(config.vocab_size)) / 8
	auc_loss = first_token_bytes

	for model in models:
		model.zero_grad(set_to_none=True)

	for step in range(0, max_steps):
		t0 = time.time()

		# ---- online pass: code the next chunk with the CURRENT weights, then learn from
		# it. The loss below is measured before any update that uses this chunk, so it is
		# the ideal log-loss cost, before finite-precision coder overhead.
		prefix_len = train_loader.pos      # tokens the decoder already has, before this chunk

		accum = {
			"ens_loss": 0.0,
			"ens_loss_last": 0.0,
			"mean_member_loss": 0.0,
			"bits": 0.0,
		}

		step_tokens = min(tokens_per_update, train_loader.n - train_loader.pos)
		processed = 0
		online_sequences = 0
		while processed < step_tokens:
			x, y = train_loader.next_batch(step_tokens - processed)
			count = y.numel()
			online_sequences += y.shape[0]
			scale = count / step_tokens
			processed += count
			x, y = x.to(device), y.to(device)
			ens_loss, ens_loss_last, member_loss = accumulate_micro([(x, y)] * K, scale)
			accum["ens_loss"] += ens_loss * scale
			accum["ens_loss_last"] += ens_loss_last * scale
			accum["mean_member_loss"] += member_loss * scale

		weight_norm = optimizer_step(step)

		# ---- replay: extra passes over data already transmitted (prefix + the chunk we
		# just coded). Free in bits -- the decoder can reproduce these updates -- so the
		# only cost is compute. Losses here are diagnostics, NOT added to auc_loss.
		replay_len = train_loader.pos
		replay_loss = float('nan')
		replay_done = 0

		if replay_steps > 0 and replay_sampler.available(replay_len) >= 0:
			for _ in range(replay_steps):
				r_accum = 0.0
				for micro_step in range(grad_accum_steps):
					# each member gets its own randomly sampled replay window, rather
					# than all members seeing the same window -- diversifies what the
					# ensemble sees during replay. Still fully reproducible: the
					# per-model batches are drawn in a fixed order from the seeded RNG.
					xy_per_model = []
					for _ in range(K):
						xr, yr = replay_sampler.batch(B, replay_len)
						xy_per_model.append((xr.to(device), yr.to(device)))
					# ensemble nll isn't meaningful across mismatched targets, so use
					# mean member nll as the replay diagnostic instead.
					_, _, member_loss = accumulate_micro(xy_per_model, 1.0 / grad_accum_steps)
					r_accum += member_loss / grad_accum_steps
				optimizer_step(step)
				replay_loss = r_accum.item()
				replay_done += 1

		if device_type == "cuda":
			torch.cuda.synchronize()
		t1 = time.time()
		dt = t1 - t0 # time difference in seconds
		seq_processed = online_sequences + train_loader.B * grad_accum_steps * replay_done
		seq_per_sec = seq_processed / dt

		step_loss = accum["ens_loss"].item()      # <- the number that matters: ensembled loss
		member_loss = accum["mean_member_loss"].item()

		print(f"step {step:5d} | ens_loss: {step_loss:.6f} | mean_member: {member_loss:.6f} | "
		      f"replay: {replay_loss:.6f} | lr {get_lr(step):.4e} | dt: {dt*1000:.2f}ms | "
		      f"seq/sec: {seq_per_sec:.2f}")

		ema_loss = step_loss if ema_loss is None else ema_beta * ema_loss + (1 - ema_beta) * step_loss
		auc_loss += (step_loss * np.log2(np.e) * step_tokens) / 8

		wandb.log({
			"loss": step_loss,
			"ema_loss": ema_loss,
			"mean_member_loss": member_loss,
			"ensemble_gain": member_loss - step_loss,
			"replay_loss": replay_loss,
			"replay_generalization_gap": step_loss - replay_loss,
			"prefix_len": prefix_len,
			# "bits_per_weight": accum["bits"]/total_weight_count,
			"lr": get_lr(step),
			"weight_norm": weight_norm,
			"last_loss": accum["ens_loss_last"].item(),
			'step': step,
			'dt': dt*1000,
			"tokens": train_loader.pos,
			"predicted_tokens": train_loader.pos - 1,
			"estimated_total_bytes": auc_loss + accounting["dictionary_bytes"],
			"auc_loss": auc_loss
		})

	# Summary metrics are what the sweep optimises against -- one scalar per run,
	# rather than whatever value happened to be logged last.
	run.summary["final_auc_loss"] = auc_loss
	run.summary["first_token_bytes"] = first_token_bytes
	run.summary["coded_tokens"] = train_loader.pos
	run.summary["predicted_tokens"] = train_loader.pos - 1
	run.summary.update(accounting)
	run.summary["estimated_total_bytes"] = auc_loss + accounting["dictionary_bytes"]
	run.summary["estimated_bits_per_byte"] = 8 * (auc_loss + accounting["dictionary_bytes"]) / accounting["source_bytes"]
	run.summary["size_kind"] = "ideal payload plus raw BPE dictionary; excludes coder, metadata, and decoder overhead"
	run.summary["final_ema_loss"] = ema_loss
	run.summary["compute_units"] = K * (1 + replay_steps)

	# optional checkpointing of the whole ensemble
	# checkpoint = {
	# 	'models': [m.state_dict() for m in raw_models],
	# 	'optimizers': [[o.state_dict() for o in opts] for opts in optimizer_sets],
	# 	'config': GPTConfig,
	# 	'step': step,
	# }
	# torch.save(checkpoint, os.path.join(log_dir, f"ensemble_{step:05d}.pt"))

	run.finish()

	# Sweeps run many models back to back in one process: drop everything and reset
	# dynamo, or compiled graphs and optimizer state pile up until you OOM.
	del raw_models, models, optimizer_sets
	gc.collect()
	torch._dynamo.reset()
	if device_type == "cuda":
		torch.cuda.empty_cache()


sweep_config = {
	'method': 'bayes',
	'metric': {
		'name': 'estimated_total_bytes',   # ideal payload bytes plus dictionary bytes
		'goal': 'minimize'
	},
	'parameters': {
		'muon_lr': {'distribution': 'uniform', 'min': 0.0, 'max': 0.08},
		'wte_lr': {'distribution': 'uniform', 'min': 0.0, 'max': 1.0},
		'lm_head_lr': {'distribution': 'uniform', 'min': 0.0, 'max': 0.01},
		'b_lr': {'distribution': 'uniform', 'min': 0.0, 'max': 0.03},
		# 'muon_wd': {'distribution': 'uniform', 'min': 0.0, 'max': 0.029088727454488157*5},
		# 'cooldown_frac': {'distribution': 'uniform', 'min': 0.0, 'max': 0.4},
		# Sweeping these changes the compute budget, so runs stop being comparable on
		# loss alone. Fix them per sweep and compare sweeps at equal K*(1+R) instead.
		# 'ensemble_k': {'values': [1, 2, 4]},
		# 'replay_steps': {'values': [0, 1, 3]},
	}
}

if __name__ == "__main__":
	if SWEEP:
		sweep_id = SWEEP_ID or wandb.sweep(sweep_config, project="hutter")
		print(f"sweep id: {sweep_id}")
		wandb.agent(sweep_id, train, count=SWEEP_COUNT, project="hutter")
	else:
		train()
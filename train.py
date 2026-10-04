import os
import gc
import dataclasses
import math
import time
import torch
from torch.nn import functional as F
import numpy as np
from model import GPT, GPTConfig
import wandb
from muon import SingleDeviceMuon
from coding_data import load_tokens, doc_start_tokens
import hints as hint_lib
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
	"compile": True,
	"log_interval": 100,   # print/log every this many steps (plus the last step)
	"data_seed": 1234,     # seed for random window sampling
	"hints": True,        # feed retrieval hints (hints.py; run it first to build the arrays)
	"hint_lr": 0.05,
	"doc_mask": True,      # attention never crosses a <page> boundary (coding must reset context there too)
	"context_length": GPTConfig.context_length,
	"vocab_size": GPTConfig.vocab_size,
	"n_layer": GPTConfig.n_layer,
	"n_head": GPTConfig.n_head,
	"n_embd": GPTConfig.n_embd,
	"total_batch_size": 32,   # in sequences, not tokens
	"B": 32,                   # micro batch size
	"muon_lr": 0.020466009566743288,
	"wte_lr": 0.18293164620959812,
	"lm_head_lr": 0.00181864153812,
	"b_lr": 0.02088101953753996,
	"muon_wd": 0.0,
	"cooldown_frac": 0.5,
	"warmup_frac": 0.0,
	"max_steps": 30_000,
	"lr_scale": 1.0,       # multiplies the whole schedule (every optimizer's peak LR)
	"resume": "",          # checkpoint path: continue its weights, optimizer state and data stream
}

# A resumed run must rebuild exactly the checkpoint's model and data path, so these come from
# the checkpoint's run config, whatever DEFAULTS say.
RESUME_KEYS = ("token_path", "context_length", "vocab_size", "n_layer", "n_head", "n_embd", "hints", "doc_mask")

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class RandomWindowLoader:
	"""Every batch is B independent, uniformly random windows from the token stream.
	No epochs, no ordering: windows can overlap and repeat.

	The whole token stream lives on `device` and windows are sampled there, so a batch
	never touches the CPU or waits on a host->device copy."""
	def __init__(self, tokens, B, T, seed, device, hint_tok=None, hint_bkt=None, doc_starts=None):
		self.B, self.T = B, T
		self.n = len(tokens)
		assert self.n > T, "token stream shorter than one window"
		# int32: torch has little uint16 support, and int32 halves the footprint of int64
		self.tokens = torch.from_numpy(np.asarray(tokens, dtype=np.int32)).to(device)
		self.offsets = torch.arange(T + 1, device=device)
		# hints are indexed by target position, so a window's hints are gathered at its y
		self.hint_tok = None if hint_tok is None else torch.from_numpy(np.asarray(hint_tok)).to(device)
		self.hint_bkt = None if hint_bkt is None else torch.from_numpy(np.asarray(hint_bkt)).to(device)
		# token id -> starts a new document; a window's segment ids are a running count of those
		# in its inputs, so they need nothing outside the window (the decoder derives the same)
		self.is_start = None
		if doc_starts is not None:
			size = max(int(self.tokens.max()), max(doc_starts)) + 1   # some start ids may never occur
			self.is_start = torch.zeros(size, dtype=torch.int32, device=device)
			self.is_start[doc_starts] = 1
		self.rng = torch.Generator(device=device)
		self.rng.manual_seed(seed)

	def next_batch(self):
		# need x[s:s+T] and y[s+1:s+T+1], so starts run over [0, n-T)
		starts = torch.randint(0, self.n - self.T, (self.B,), generator=self.rng, device=self.tokens.device)
		idx = starts[:, None] + self.offsets
		w = self.tokens[idx].long()
		x, y = w[:, :-1], w[:, 1:]
		seg = None if self.is_start is None else torch.cumsum(self.is_start[x], dim=1)
		if self.hint_tok is None:
			return x, y, None, None, seg
		tgt = idx[:, 1:]
		return x, y, self.hint_tok[tgt].long(), self.hint_bkt[tgt].long(), seg

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
	resume_ck = None
	if config.resume:
		resume_ck = torch.load(config.resume, map_location="cpu", weights_only=False)
		config.update({k: resume_ck["run_config"][k] for k in RESUME_KEYS}, allow_val_change=True)
		print(f"resuming {config.resume} (step {resume_ck['step']})")

	total_batch_size = config.total_batch_size
	B = config.B
	assert B > 0 and total_batch_size > 0 and config.context_length > 0
	assert total_batch_size % B == 0, "make sure total_batch_size is divisible by B"
	grad_accum_steps = total_batch_size // B
	print(f"total desired batch size: {total_batch_size}")
	print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

	hint_arrays = [np.load(p, mmap_mode="r") for p in hint_lib.hint_paths(config.token_path)] if config.hints else [None, None]
	doc_starts = doc_start_tokens(vocab_size=config.vocab_size) if config.doc_mask else None
	train_loader = RandomWindowLoader(load_tokens(config.token_path), B, config.context_length, config.data_seed, device,
	                                  *hint_arrays, doc_starts=doc_starts)
	num_tokens = train_loader.n
	run.config.update({"num_tokens": num_tokens}, allow_val_change=True)

	model_config = GPTConfig(
		context_length=config.context_length,
		vocab_size=config.vocab_size,
		n_layer=config.n_layer,
		n_head=config.n_head,
		n_embd=config.n_embd,
		n_hint=len(hint_lib.MATCH_MIN) if config.hints else 0,
		n_hint_bkt=hint_lib.N_BUCKETS,
	)

	torch.manual_seed(0)
	raw_model = GPT(model_config).to(device)

	def token_nll(x, y, hint_tok, hint_bkt, doc_seg):
		"""Per-position NLL, shape [B, T]. Model and loss are compiled as one graph so the
		fp32 cast and softmax fuse instead of materialising full-vocab logits several times."""
		with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
			logits = raw_model(x, hint_tok=hint_tok, hint_bkt=hint_bkt, doc_seg=doc_seg)
		V = logits.size(-1)
		return F.cross_entropy(logits.reshape(-1, V).float(), y.flatten(), reduction='none').view(y.shape)

	# Every batch is exactly [B, T], so CUDA graphs can replay the whole forward/backward
	# and skip per-kernel launch overhead.
	loss_fn = torch.compile(token_nll, mode='max-autotune') if config.compile else token_nll

	optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=config.wte_lr, betas=(0.0, 0.999), fused=(device_type == "cuda"))
	optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=config.lm_head_lr, betas=(0.0, 0.999), fused=(device_type == "cuda"))
	w_params = [m.weight for m in raw_model.transformer.h.modules() if isinstance(m, torch.nn.Linear)]
	b_params = [m.bias for m in raw_model.transformer.h.modules() if isinstance(m, torch.nn.Linear)]
	b_params.append(raw_model.lm_head.bias)
	optimizer3 = SingleDeviceMuon(w_params, lr=config.muon_lr, momentum=0.95, weight_decay=config.muon_wd)
	optimizer4 = torch.optim.Adam(b_params, lr=config.b_lr, betas=(0.0, 0.999), fused=(device_type == "cuda"))
	optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]
	if config.hints:
		optimizers.append(torch.optim.Adam([raw_model.hint_gate, raw_model.hint_logit], lr=config.hint_lr,
		                                   betas=(0.0, 0.999), fused=(device_type == "cuda")))

	for opt in optimizers:
		for group in opt.param_groups:
			group["initial_lr"] = group["lr"]

	start_step = 0
	if resume_ck is not None:
		raw_model.load_state_dict(resume_ck["model"])
		assert len(resume_ck["optimizers"]) == len(optimizers), "optimizer set differs from the checkpoint's"
		for opt, state in zip(optimizers, resume_ck["optimizers"]):
			peak = [g["initial_lr"] for g in opt.param_groups]
			opt.load_state_dict(state)       # moments and momentum buffers; this also restores old lrs,
			for g, lr in zip(opt.param_groups, peak):
				g["initial_lr"] = lr         # so put this run's peak LRs back (the schedule sets lr per step)
		train_loader.rng.set_state(resume_ck["rng_state"])   # keep drawing new windows, not a replay
		start_step = resume_ck.get("total_step", resume_ck["step"] + 1)

	weight_count = sum(p.numel() for p in raw_model.parameters())
	print(f"weights: {weight_count}")

	tokens_per_update = total_batch_size * config.context_length
	max_steps = config.max_steps

	cooldown_frac = config.cooldown_frac
	warmup_frac = config.warmup_frac
	lr_scale = config.lr_scale
	def get_lr(step: int):
		x = step / max_steps
		assert 0 <= x < 1
		if x < warmup_frac:
			return lr_scale * x / warmup_frac
		if x > 1 - cooldown_frac:
			w = (1 - x) / cooldown_frac
			return lr_scale * (w * 1.0 + (1 - w) * 0.1)
		return lr_scale

	log_dir = "log"
	os.makedirs(log_dir, exist_ok=True)

	def optimizer_step(step):
		"""Clip, set LR, step, then zero grads. Returns grad norm."""
		norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1).detach()
		for opt in optimizers:
			for group in opt.param_groups:
				group["lr"] = group["initial_lr"] * get_lr(step)
			opt.step()
		raw_model.zero_grad(set_to_none=True)
		return norm

	ema_loss = None
	ema_last_loss = None
	ema_beta = 0.98  # averages over roughly the last 1/(1-beta) ≈ 50 steps

	raw_model.zero_grad(set_to_none=True)

	log_interval = config.log_interval
	t0 = time.time()
	last_log_step = -1

	for step in range(0, max_steps):

		accum = {
			"loss": 0.0,
			"loss_last": 0.0,
		}

		scale = 1.0 / grad_accum_steps
		for micro_step in range(grad_accum_steps):
			x, y, hint_tok, hint_bkt, doc_seg = train_loader.next_batch()
			# Mark once per optimizer step, not per micro-step. A mark lets the next replay
			# overwrite every earlier CUDA graph output -- including micro-step 0's gradients,
			# which param.grad aliases until the next backward accumulates into it. Marking
			# per micro-step therefore crashes under grad accumulation ("accessing tensor
			# output of CUDAGraphs that has been overwritten"). nll is reduced before reuse.
			if micro_step == 0:
				torch.compiler.cudagraph_mark_step_begin()
			nll = loss_fn(x, y, hint_tok, hint_bkt, doc_seg)
			(nll.mean() * scale).backward()
			nll = nll.detach()
			accum["loss"] += nll.mean() * scale
			accum["loss_last"] += nll[:, -1].mean() * scale

		weight_norm = optimizer_step(step)

		# EMAs stay on the device so steps that don't log never sync with the GPU
		step_loss, last_loss = accum["loss"], accum["loss_last"]
		ema_loss = step_loss if ema_loss is None else ema_beta * ema_loss + (1 - ema_beta) * step_loss
		ema_last_loss = last_loss if ema_last_loss is None else ema_beta * ema_last_loss + (1 - ema_beta) * last_loss

		if (step + 1) % log_interval == 0 or step == max_steps - 1:
			if device_type == "cuda":
				torch.cuda.synchronize()
			t1 = time.time()
			steps_since = step - last_log_step
			dt = (t1 - t0) / steps_since   # mean seconds per step since the last log
			tok_per_sec = tokens_per_update / dt
			t0, last_log_step = t1, step

			step_loss, last_loss = step_loss.item(), last_loss.item()
			# size of the whole dataset if coded with the current model's loss
			bytes_estimate = (step_loss * np.log2(np.e) * num_tokens) / 8

			print(f"step {step:5d} | loss: {step_loss:.6f} | ema: {ema_loss.item():.6f} | "
			      f"lr {get_lr(step):.4e} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:.2f}")

			wandb.log({
				"loss": step_loss,
				"ema_loss": ema_loss.item(),
				"lr": get_lr(step),
				"weight_norm": weight_norm.item(),
				"last_loss": last_loss,
				"ema_last_loss": ema_last_loss.item(),
				'step': step,
				'total_step': start_step + step + 1,
				'dt': dt*1000,
				"tokens": (step + 1) * tokens_per_update,
				"bytes": bytes_estimate,
			}, step=step)

	run.summary["final_ema_loss"] = ema_loss.item()
	run.summary["final_ema_last_loss"] = ema_last_loss.item()

	checkpoint = {
		'model': raw_model.state_dict(),
		'optimizers': [o.state_dict() for o in optimizers],
		'model_config': dataclasses.asdict(model_config),
		'run_config': dict(config),
		'step': step,
		'total_step': start_step + step + 1,   # optimizer steps across this run and any it resumed
		'resumed_from': config.resume,
		'rng_state': train_loader.rng.get_state(),
	}
	checkpoint_path = os.path.join(log_dir, f"model_{run.id}_{step:05d}.pt")
	torch.save(checkpoint, checkpoint_path)
	print(f"saved checkpoint to {checkpoint_path}")

	run.finish()

	# Sweeps run many models back to back in one process: drop everything and reset
	# dynamo, or compiled graphs and optimizer state pile up until you OOM.
	del raw_model, loss_fn, optimizers, train_loader
	gc.collect()
	torch._dynamo.reset()
	if device_type == "cuda":
		torch.cuda.empty_cache()


sweep_config = {
	'method': 'bayes',
	'metric': {
		'name': 'bytes',
		'goal': 'minimize'
	},
	'parameters': {
		'muon_lr': {'distribution': 'uniform', 'min': 0.0, 'max': 0.08},
		'wte_lr': {'distribution': 'uniform', 'min': 0.0, 'max': 1.0},
		'lm_head_lr': {'distribution': 'uniform', 'min': 0.0, 'max': 0.01},
		'b_lr': {'distribution': 'uniform', 'min': 0.0, 'max': 0.03},
		# 'muon_wd': {'distribution': 'uniform', 'min': 0.0, 'max': 0.029088727454488157*5},
		# 'cooldown_frac': {'distribution': 'uniform', 'min': 0.0, 'max': 0.4},
	}
}

if __name__ == "__main__":
	if SWEEP:
		sweep_id = SWEEP_ID or wandb.sweep(sweep_config, project="hutter")
		print(f"sweep id: {sweep_id}")
		wandb.agent(sweep_id, train, count=SWEEP_COUNT, project="hutter")
	else:
		train()
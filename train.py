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
from coding_data import load_tokens
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
	"data_seed": 1234,     # seed for random window sampling
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
	"cooldown_frac": 0.0,
	"warmup_frac": 0.0,
}

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class RandomWindowLoader:
	"""Every batch is B independent, uniformly random windows from the token stream.
	No epochs, no ordering: windows can overlap and repeat."""
	def __init__(self, tokens, B, T, seed):
		self.tokens = tokens
		self.B, self.T = B, T
		self.n = len(tokens)
		assert self.n > T, "token stream shorter than one window"
		self.rng = np.random.default_rng(seed)

	def next_batch(self):
		starts = self.rng.integers(0, self.n - self.T, size=self.B)   # need x[s:s+T] and y[s+1:s+T+1]
		idx = starts[:, None] + np.arange(self.T)[None, :]
		x = torch.from_numpy(self.tokens[idx].astype(np.int64))
		y = torch.from_numpy(self.tokens[idx + 1].astype(np.int64))
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

	total_batch_size = config.total_batch_size
	B = config.B
	assert B > 0 and total_batch_size > 0 and config.context_length > 0
	assert total_batch_size % B == 0, "make sure total_batch_size is divisible by B"
	grad_accum_steps = total_batch_size // B
	print(f"total desired batch size: {total_batch_size}")
	print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

	train_loader = RandomWindowLoader(load_tokens(config.token_path), B, config.context_length, config.data_seed)
	num_tokens = train_loader.n
	run.config.update({"num_tokens": num_tokens}, allow_val_change=True)

	model_config = GPTConfig(
		context_length=config.context_length,
		vocab_size=config.vocab_size,
		n_layer=config.n_layer,
		n_head=config.n_head,
		n_embd=config.n_embd,
	)

	torch.manual_seed(0)
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

	weight_count = sum(p.numel() for p in raw_model.parameters())
	print(f"weights: {weight_count}")

	tokens_per_update = total_batch_size * config.context_length
	max_steps = math.ceil((num_tokens - 1) / tokens_per_update)   # same token budget as one pass

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

	def optimizer_step(step):
		"""Clip, set LR, step, then zero grads. Returns grad norm."""
		norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1).detach()
		for opt in optimizers:
			for group in opt.param_groups:
				group["lr"] = group["initial_lr"] * get_lr(step)
			opt.step()
		model.zero_grad(set_to_none=True)
		return norm

	ema_loss = None
	ema_beta = 0.98  # averages over roughly the last 1/(1-beta) ≈ 50 steps

	model.zero_grad(set_to_none=True)

	for step in range(0, max_steps):
		t0 = time.time()

		accum = {
			"loss": 0.0,
			"loss_last": 0.0,
		}

		scale = 1.0 / grad_accum_steps
		for micro_step in range(grad_accum_steps):
			x, y = train_loader.next_batch()
			x, y = x.to(device), y.to(device)
			with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
				logits = model(x)
			V = logits.size(-1)
			nll = F.cross_entropy(logits.reshape(-1, V).float(), y.flatten(), reduction='none').view(y.shape)
			(nll.mean() * scale).backward()
			nll = nll.detach()
			accum["loss"] += nll.mean() * scale
			accum["loss_last"] += nll[:, -1].mean() * scale

		weight_norm = optimizer_step(step)

		if device_type == "cuda":
			torch.cuda.synchronize()
		t1 = time.time()
		dt = t1 - t0 # time difference in seconds
		tok_per_sec = tokens_per_update / dt

		step_loss = accum["loss"].item()

		print(f"step {step:5d} | loss: {step_loss:.6f} | lr {get_lr(step):.4e} | "
		      f"dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:.2f}")

		ema_loss = step_loss if ema_loss is None else ema_beta * ema_loss + (1 - ema_beta) * step_loss
		# size of the whole dataset if coded with the current model's loss
		bytes_estimate = (step_loss * np.log2(np.e) * num_tokens) / 8

		wandb.log({
			"loss": step_loss,
			"ema_loss": ema_loss,
			"lr": get_lr(step),
			"weight_norm": weight_norm,
			"last_loss": accum["loss_last"].item(),
			'step': step,
			'dt': dt*1000,
			"tokens": (step + 1) * tokens_per_update,
			"bytes": bytes_estimate,
		})

	run.summary["final_ema_loss"] = ema_loss

	checkpoint = {
		'model': raw_model.state_dict(),
		'optimizers': [o.state_dict() for o in optimizers],
		'model_config': dataclasses.asdict(model_config),
		'run_config': dict(config),
		'step': step,
		'rng_state': train_loader.rng.bit_generator.state,
	}
	checkpoint_path = os.path.join(log_dir, f"model_{run.id}_{step:05d}.pt")
	torch.save(checkpoint, checkpoint_path)
	print(f"saved checkpoint to {checkpoint_path}")

	run.finish()

	# Sweeps run many models back to back in one process: drop everything and reset
	# dynamo, or compiled graphs and optimizer state pile up until you OOM.
	del raw_model, model, optimizers
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
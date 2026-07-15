import os
import time
import torch
from torch.nn import functional as F
import numpy as np
from model import GPT, GPTConfig, Quantized_Linear, Quantized_Embedding
import wandb
from muon import SingleDeviceMuon
# import lzma

def load_tokens():
	with open('5k_enwik9', 'rb') as f:
		data = f.read()
	tokens = np.frombuffer(data, dtype=np.uint16)
	return tokens  # keep as numpy array

class DataLoaderLite:
	def __init__(self, B, T):
		self.B, self.T = B, T
		self.tokens = load_tokens()
		self.n = len(self.tokens)
		self.pos = 0

	def next_batch(self):
		B, T = self.B, self.T

		starts = self.pos + np.arange(B) * T
		offsets = np.arange(T)
		x_idx = (starts[:, None] + offsets[None, :]) % self.n
		y_idx = (x_idx + 1) % self.n

		x = self.tokens[x_idx]  # numpy fancy indexing, works fine with uint16
		y = self.tokens[y_idx]

		self.pos = (self.pos + B * T) % self.n

		x = torch.from_numpy(x.astype(np.int64))
		y = torch.from_numpy(y.astype(np.int64))

		return x, y

device = "cpu"
if torch.cuda.is_available():
	device = "cuda"
print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

total_batch_size = 1 # 2**19, ~0.5M, in number of tokens
B = 1 # micro batch size
assert total_batch_size % B == 0, "make sure total_batch_size is divisible by B"
grad_accum_steps = total_batch_size // B
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=GPTConfig.context_length)

torch.set_float32_matmul_precision('high')

wandb.init(
	project="hutter",

	config={
	"architecture": "transformer",
	"context_length": GPTConfig.context_length,
	"vocab_size": GPTConfig.vocab_size,
	"n_layer": GPTConfig.n_layer,
	"n_head": GPTConfig.n_head,
	"n_embd": GPTConfig.n_embd,
	"total_batch_size": total_batch_size,
	"B": B,
	"muon_lr": 0.02405482011843835,
	"wte_lr": 0.06453128852460155,
	"lm_head_lr": 0.011223604923484058,
	# "b_lr": 0.06198621925528761,
	"num_tokens": len(train_loader.tokens)
	}
)

# def train():
torch.manual_seed(0)
# wandb.init(project='hutter')
config = wandb.config
raw_model = GPT()

# Load the checkpoint once so we can pull both model weights and (if present)
# optimizer state out of the same file.
# checkpoint = torch.load('log/laced-microwave-5555.pt', weights_only=False)
# raw_model.load_state_dict(checkpoint['model'])
raw_model = raw_model.to(device)
model = torch.compile(raw_model, mode='max-autotune-no-cudagraphs')

weight_count = sum(p.numel() for p in model.parameters())
print(weight_count)

# max_steps = len(train_loader.tokens) // (total_batch_size * 1024)
max_steps = 5000

cooldown_frac = 0.0
warmup_frac = 0.0
def get_lr(step: int):
	x = step / max_steps
	assert 0 <= x < 1
	if x < warmup_frac:
		return x / warmup_frac
	if x > 1 - cooldown_frac:
		w = (1 - x) / cooldown_frac
		return w * 1.0 + (1 - w) * 0.1
	return 1.0

optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=config.wte_lr, betas=(0.9, 0.95), fused=True)
optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=config.lm_head_lr, betas=(0.9, 0.95), fused=True)
w_params = [m.weight for m in raw_model.transformer.h.modules() if isinstance(m, torch.nn.Linear)]
# b_params = [m.bias for m in raw_model.transformer.h.modules() if isinstance(m, torch.nn.Linear)]
optimizer3 = SingleDeviceMuon(w_params, lr=config.muon_lr, momentum=0.95, weight_decay=0.1)
# optimizer4 = torch.optim.Adam(b_params, lr=config.b_lr, betas=(0.9, 0.95), fused=True)
optimizers = [optimizer1, optimizer2, optimizer3]

# If the checkpoint we resumed from has optimizer state (it won't the first
# time you run this against an old checkpoint that predates this change),
# load it into each optimizer in the same order they were created/saved.
# for opt, state in zip(optimizers, checkpoint['optimizers']):
	# opt.load_state_dict(state)

# for group in optimizer1.param_groups:
# 	group["lr"] = config.wte_lr
# for group in optimizer2.param_groups:
# 	group["lr"] = config.lm_head_lr
# for group in optimizer3.param_groups:
# 	group["lr"] = config.muon_lr
# for group in optimizer4.param_groups:
# 	group["lr"] = config.b_lr

for opt in optimizers:
	for group in opt.param_groups:
		group["initial_lr"] = group["lr"]

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

ema_loss = None
ema_beta = 0.98  # averages over roughly the last 1/(1-beta) ≈ 50 steps

for step in range(0, max_steps):
	t0 = time.time()

	# once in a while evaluate our validation loss
	# if step == (max_steps - 1) or step % 500 == 0:
	# 	# optionally write model checkpoints
	# 	checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
	# 	checkpoint = {
	# 		'model': raw_model.state_dict(),
	# 		'config': model.config,
	# 		'step': step,
	# 		'optimizers': [opt.state_dict() for opt in optimizers],
	# 	}
		# you might also want to add rng seeds etc., if you wanted to more
		# exactly resume training
		# torch.save(checkpoint, checkpoint_path)
		# total_bits = 0
		# for m in raw_model.modules():
		# 	if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
		# 		bits = lzma.compress(m.weight.cpu().detach().numpy())
		# 		total_bits += len(bits)
		# wandb.log({'lzma_bits': total_bits})

	model.zero_grad(set_to_none=True)
	accum = {
		"loss": 0.0,
		"loss_last": 0.0,
		"ce_loss": 0.0,
		"bits": 0.0,
	}
	for micro_step in range(grad_accum_steps):
		x, y = train_loader.next_batch()
		x, y = x.to(device), y.to(device)
		# added after video, this field is also used by the forward pass.
		with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
			logits = model(x)
		loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.flatten(), ignore_index=-1)
		ce_loss = loss.detach()
		# bits = sum([m.get_bits() for m in model.modules() if isinstance(m, (Quantized_Linear, Quantized_Embedding))])
		loss_last = F.cross_entropy(logits[:, -1, :].reshape(-1, logits.size(-1)), y[:, -1].view(-1), ignore_index=-1)
		# loss = (len(train_loader.tokens) * loss * np.log2(np.e) + bits)/(8*10**8)
		loss = loss / grad_accum_steps
		ce_loss = ce_loss / grad_accum_steps
		loss_last = loss_last / grad_accum_steps

		accum["loss"] += loss.detach()
		accum["ce_loss"] += ce_loss
		accum["loss_last"] += loss_last.detach()
		# accum["bits"] += bits / grad_accum_steps

		loss.backward()
	weight_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 100)
	for opt in optimizers:
		for group in opt.param_groups:
			group["lr"] = group["initial_lr"] * get_lr(step)

	for opt in optimizers:
		opt.step()
	torch.cuda.synchronize() # wait for the GPU to finish work
	t1 = time.time()
	dt = t1 - t0 # time difference in seconds
	seq_processed = train_loader.B * grad_accum_steps
	seq_per_sec = seq_processed / dt
	print(f"step {step:5d} |loss: {accum['loss'].item():.6f} | lr {get_lr(step):.4e} | dt: {dt*1000:.2f}ms | seq/sec: {seq_per_sec:.2f}")

	step_loss = accum["ce_loss"].item()
	ema_loss = step_loss if ema_loss is None else ema_beta * ema_loss + (1 - ema_beta) * step_loss

	wandb.log({
		"loss": step_loss,
		"ema_loss": ema_loss,
		"bytes": (len(train_loader.tokens) * step_loss * np.log2(np.e) + accum["bits"])/8,
		"bits_per_weight": accum["bits"]/weight_count,
		"lr": get_lr(step),
		"weight_norm": weight_norm,
		"last_loss": accum["loss_last"].item(),
		'step': step,
		'dt': dt*1000,
	})

# sweep_config = {
#     'method': 'bayes',
#     'metric': {
#         'name': 'ema_loss',
#         'goal': 'minimize'
#     },
#     'parameters': {
#         'muon_lr': {'distribution': 'uniform', 'min': 0.0, 'max': 0.03082128076593258*5},
#         'wte_lr': {'distribution': 'uniform', 'min': 0.0, 'max': 0.19020436245909905*5},
#         'lm_head_lr': {'distribution': 'uniform', 'min': 0.0, 'max': 0.00631756779266*5},
#     }
#     }

# sweep_id = wandb.sweep(sweep_config, project="hutter")
# wandb.agent(sweep_id, train)
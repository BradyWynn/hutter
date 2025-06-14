import os
import time
import torch
from torch.nn import functional as F
import numpy as np
from model import GPT, GPTConfig
import wandb
from muon import SingleDeviceMuon

def load_tokens():
	tokens = np.load("tokenized_enwik9.npy")
	tokens = torch.tensor(tokens, dtype=torch.long)
	return tokens

class DataLoaderLite:
	def __init__(self, B, T):
		self.B = B
		self.T = T
		self.tokens = load_tokens()
		self.reset()

	def reset(self):
		total_len = len(self.tokens)
		self.n_sequences = (total_len - 1) // self.T
		all_starts = torch.arange(self.n_sequences) * self.T
		perm = torch.randperm(self.n_sequences)
		self.sequence_indices = all_starts[perm]
		self.current_batch_idx = 0

	def next_batch(self):
		if self.current_batch_idx + self.B >= len(self.sequence_indices):
			self.reset()

		idx = self.sequence_indices[self.current_batch_idx : self.current_batch_idx + self.B]
		self.current_batch_idx += self.B

		offsets = torch.arange(self.T)
		idx = idx.unsqueeze(1) + offsets
		x = self.tokens[idx]
		y = self.tokens[idx + 1]

		return x, y

# attempt to autodetect device
device = "cpu"
if torch.cuda.is_available():
	device = "cuda"
print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

total_batch_size = 2**18 # 2**19, ~0.5M, in number of tokens
B = 32 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T)

torch.set_float32_matmul_precision('high')

wandb.init(
	project="calebgpt",

	config={
	"architecture": "transformer",
	"context_length": GPTConfig.context_length,
	"vocab_size": GPTConfig.vocab_size,
	"n_layer": GPTConfig.n_layer,
	"n_head": GPTConfig.n_head,
	"n_embd": GPTConfig.n_embd,
	"total_batch_size": total_batch_size,
	"B": B
	}
)

model = GPT()
model = torch.compile(model)
model.to(device)
model.train()

max_steps = len(train_loader.tokens) // total_batch_size

cooldown_frac = 0.4
def get_lr(step: int):
	x = step / max_steps # progress in training
	assert 0 <= x < 1
	if x < 1 - cooldown_frac:
		return 1.0
	else:
		w = (1 - x) / cooldown_frac
		return w * 1.0 + (1 - w) * 0.1

# init the optimizer(s)
optimizer1 = torch.optim.AdamW(
	model.lm_head.parameters(),
	lr=0.008847293888235502,
	betas=(0.9, 0.95),
	weight_decay=0.0,
)
optimizer2 = SingleDeviceMuon(
	model.transformer.h.parameters(),
	lr=0.03268883417940545,
	momentum=0.95,
)

optimizers = [optimizer1, optimizer2]

for opt in optimizers:
	for group in opt.param_groups:
		group["initial_lr"] = group["lr"]

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

for step in range(max_steps):
	t0 = time.time()

	# once in a while evaluate our validation loss
	if step == (max_steps - 1):
		# optionally write model checkpoints
		checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
		checkpoint = {
			'model': model.state_dict(),
			'config': model.config,
			'step': step,
		}
		# you might also want to add optimizer.state_dict() and
		# rng seeds etc., if you wanted to more exactly resume training
		torch.save(checkpoint, checkpoint_path)

	# do one step of the optimization
	model.zero_grad(set_to_none=True)
	loss_accum = 0.0
	for micro_step in range(grad_accum_steps):
		x, y = train_loader.next_batch()
		x, y = x.to(device), y.to(device)
		# added after video, this field is also used by the forward pass.
		with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
			logits = model(x)
		loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
		# we have to scale the loss to account for gradient accumulation,
		# because the gradients just add on each successive backward().
		# addition of gradients corresponds to a SUM in the objective, but
		# instead of a SUM we want MEAN. Scale the loss here so it comes out right
		loss = loss / grad_accum_steps
		loss_accum += loss.detach()
		loss.backward()
	# norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
	# determine and set the learning rate for this iteration
	for opt in optimizers:
		for group in opt.param_groups:
			group["lr"] = group["initial_lr"] * get_lr(step)

	for opt in optimizers:
		opt.step()
	if device_type == "cuda":
		torch.cuda.synchronize() # wait for the GPU to finish work
	t1 = time.time()
	dt = t1 - t0 # time difference in seconds
	tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
	tokens_per_sec = tokens_processed / dt
	print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {get_lr(step):.4e} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
	wandb.log({"loss": loss_accum.item(), "lr": get_lr(step)})

# sweep_config = {
#     'method': 'bayes',
# 	'metric': {
# 		'name': 'loss',
# 		'goal': 'minimize'
# 	},
# 	'parameters': {
# 		'muon_lr': {'distribution': 'uniform', 'min': 0.0001, 'max': 0.5},
# 		'adamw_lr': {'distribution': 'uniform', 'min': 0.00001, 'max': 0.01},
# 		'weight_decay': {'distribution': 'uniform', 'min': 0.0001, 'max': 0.1},
# 	}
#     }

# sweep_id = wandb.sweep(sweep_config, project="calebgpt")

# print(sweep_id)

# wandb.agent("tpit7q9f", train)
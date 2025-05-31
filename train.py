import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from model import GPT, GPTConfig
import wandb

def load_tokens():
	tokens = np.load("tokenized_enwik9.npy")
	tokens = torch.tensor(tokens, dtype=torch.long)
	return tokens.flatten()

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B  # batch size
        self.T = T  # sequence length
        self.reset()
    
    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = None
        self.tokens = load_tokens()
        
        # Calculate number of complete sequences we can make
        self.n_sequences = len(self.tokens) // self.T
        
        # Create sequence starting indices and shuffle them
        self.sequence_indices = np.arange(0, self.n_sequences * self.T, self.T)
        np.random.shuffle(self.sequence_indices)
        
        # Track which sequences we've used
        self.current_batch_idx = 0
        
    def next_batch(self):
        B, T = self.B, self.T
        
        # If we've used all sequences, reset and reshuffle
        if self.current_batch_idx + B >= len(self.sequence_indices):
            self.reset()
        
        # Get the starting indices for this batch
        batch_start_indices = self.sequence_indices[self.current_batch_idx:self.current_batch_idx + B]
        
        # Initialize tensors for the batch
        x = torch.zeros((B, T), dtype=self.tokens.dtype)
        y = torch.zeros((B, T), dtype=self.tokens.dtype)
        
        # Fill the batch
        for i, start_idx in enumerate(batch_start_indices):
            x[i] = self.tokens[start_idx:start_idx + T]
            y[i] = self.tokens[start_idx + 1:start_idx + T + 1]
        
        # Advance batch counter
        self.current_batch_idx += B
        
        return x, y

# attempt to autodetect device
device = "cpu"
if torch.cuda.is_available():
	device = "cuda"
print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

total_batch_size = 2**20 # 2**19, ~0.5M, in number of tokens
B = 8 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, split="train")

torch.set_float32_matmul_precision('high')

wandb.init(
	project="calebgpt",

	config={
	"architecture": "transformer",
	"context_length": GPTConfig.context_length,
	"vocab_size": GPTConfig.vocab_size,
	"n_layer": GPTConfig.n_layer,
	"n_head": GPTConfig.n_head,
	"dim": GPTConfig.dim,
	"total_batch_size": total_batch_size,
	"B": B
	}
)

model = GPT()
# model = torch.compile(model)
model.to(device)
model.train()

max_steps = 312043861 // total_batch_size

# warmup_steps = 0
# def get_lr(it):
#     # 1) linear warmup for warmup_iters steps
#     if it < warmup_steps:
#         return 1.0 * (it+1) / warmup_steps
#     # 2) if it > lr_decay_iters, return min learning rate
#     if it > max_steps:
#         return 0.1
#     # 3) in between, use cosine decay down to min learning rate
#     decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
#     return 0.1 + coeff * (1.0 - 0.1)

cooldown_frac = 0.4
def get_lr(step: int):
    x = step / max_steps # progress in training
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

# optimize!
optimizers = model.configure_optimizers(weight_decay=0.01, muon_lr=0.02, adamw_lr=3e-4, device_type=device_type)

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
	print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {get_lr(step):.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
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
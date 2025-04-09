import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from model import GPT
import wandb

def load_tokens():
	tokens = np.load("enwik9.npy")
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

wandb.init(
	# set the wandb project where this run will be logged
	project="calebgpt",

	# track hyperparameters and run metadata
	config={
	"architecture": "transfomer",
	}
)

# attempt to autodetect device
device = "cpu"
if torch.cuda.is_available():
	device = "cuda"
print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
	torch.cuda.manual_seed(1337)

total_batch_size = 2**19 # 2**19, ~0.5M, in number of tokens
B = 64 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, split="train")
# val_loader = DataLoaderLite(B=B, T=T, split="val")

torch.set_float32_matmul_precision('high')

# create model
model = GPT()
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
raw_model = model # always contains the "raw" unwrapped model

max_steps = 10000 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(step: int):
	x = step / max_steps # progress in training
	assert 0 <= x < 1
	if x < 1 - 0.4:
		return 1.0
	else:
		w = (1 - x) / 0.4
		return w * 1.0 + (1 - w) * 0.1

# optimize!
optimizers = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

for opt in optimizers:
	for group in opt.param_groups:
		group["initial_lr"] = group["lr"]

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
	pass

for step in range(max_steps):
	t0 = time.time()
	last_step = (step == max_steps - 1)

	# once in a while evaluate our validation loss
	if step % 250 == 0 or last_step:
	# 	model.eval()
	# 	val_loader.reset()
	# 	with torch.no_grad():
	# 		val_loss_accum = 0.0
	# 		val_loss_steps = 20
	# 		for _ in range(val_loss_steps):
	# 			x, y = val_loader.next_batch()
	# 			x, y = x.to(device), y.to(device)
	# 			with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
	# 				logits = model(x)
	# 			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
	# 			loss = loss / val_loss_steps
	# 			val_loss_accum += loss.detach()
		# print(f"validation loss: {val_loss_accum.item():.4f}")
		# with open(log_file, "a") as f:
		# 	f.write(f"{step} val {val_loss_accum.item():.4f}\n")
		if step > 0 and (step % 5000 == 0 or last_step):
			# optionally write model checkpoints
			checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
			checkpoint = {
				'model': raw_model.state_dict(),
				'config': raw_model.config,
				'step': step,
			}
			# you might also want to add optimizer.state_dict() and
			# rng seeds etc., if you wanted to more exactly resume training
			torch.save(checkpoint, checkpoint_path)

	# do one step of the optimization
	model.train()
	model.zero_grad(set_to_none=True)
	# optimizer.zero_grad()
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
	norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
	with open(log_file, "a") as f:
		f.write(f"{step} train {loss_accum.item():.6f}\n")
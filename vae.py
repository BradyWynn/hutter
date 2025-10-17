import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
	def __init__(self, fan_in, fan_out):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(fan_in, fan_out),
			nn.ReLU(),
			nn.LayerNorm(fan_out),
        )
	def forward(self, x):
		return x + self.layers(x)

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
			Block(512, 512),
            nn.Linear(512, 1024),
			Block(1024, 1024),
            Block(1024, 1024),
            Block(1024, 1024),
            Block(1024, 1024),
            Block(1024, 1024),
            Block(1024, 1024),
            Block(1024, 1024),
            Block(1024, 1024),
            nn.Linear(1024, 512),
			nn.Linear(512, 256),
            nn.ReLU(),
			nn.Linear(256, 512),
            Block(512, 512),
			Block(512, 512),
            nn.Linear(512, 512)
        )
    def forward(self, x):
        return self.layers(x)
    
import os
import time
import torch
from torch.nn import functional as F
import wandb

torch.manual_seed(0)

def load_tokens():
	tokens = torch.load('model/lm_head.weight.pt')
	return tokens

class DataLoaderLite:
	def __init__(self, B):
		self.B = B
		self.tokens = load_tokens()

	def next_batch(self):
		ix = torch.randint(len(self.tokens), (self.B,))
		return self.tokens[ix]

# attempt to autodetect device
device = "cpu"
if torch.cuda.is_available():
	device = "cuda"
print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

total_batch_size = 50256 # 2**19, ~0.5M, in number of tokens
B = 50256 # micro batch size
grad_accum_steps = total_batch_size // B
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B)

torch.set_float32_matmul_precision('high')

# def train():
wandb.init(project="vae")
# config = wandb.config

raw_model = AE()
raw_model = raw_model.to(device).bfloat16()
model = torch.compile(raw_model)

max_steps = 100000 * (len(train_loader.tokens) // total_batch_size)

# init the optimizer(s)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, betas=(0.9, 0.95), fused=True)

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
            'model': raw_model.state_dict(),
            'step': step,
        }
        # you might also want to add optimizer.state_dict() and
        # rng seeds etc., if you wanted to more exactly resume training
        torch.save(checkpoint, checkpoint_path)

    # do one step of the optimization
    model.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x = train_loader.next_batch()
        x = x.to(device)
        # added after video, this field is also used by the forward pass.
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits = model(x)
        loss = torch.mean(1 - F.cosine_similarity(logits, x))
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration


    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {1.0:.4e} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    wandb.log({"loss": loss_accum.item(), "lr": optimizer.param_groups[0]['lr'], "norm": norm})
	
# sweep_config = {
#     'method': 'bayes',
# 	'metric': {
# 		'name': 'loss',
# 		'goal': 'minimize'
# 	},
# 	'parameters': {
# 		'lr': {'distribution': 'uniform', 'min': 0.00001, 'max': 0.02},
# 	}
#     }

# sweep_id = wandb.sweep(sweep_config, project="vae")
# wandb.agent(sweep_id, train)

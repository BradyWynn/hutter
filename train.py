import os
import time
import torch
from torch.nn import functional as F
import numpy as np
from model import GPT, GPTConfig, Quantized_Linear, Rotary
import wandb
from muon import SingleDeviceMuon

def load_tokens():
	with open('tokenized_enwik9', 'rb') as f:
		data = f.read()
	tokens = np.frombuffer(data, dtype=np.uint16)
	tokens = torch.tensor(tokens, dtype=torch.uint16)
	return tokens

class DataLoaderLite:
	def __init__(self, B, T):
		self.B, self.T = B, T
		self.tokens = load_tokens()

	def next_batch(self):
		ix = torch.randint(len(self.tokens) - self.T, (self.B,))
		x = torch.stack([self.tokens[i:i+self.T] for i in ix])
		y = torch.stack([self.tokens[i+1:i+self.T+1] for i in ix])
		return x.long(), y.long()

# attempt to autodetect device
device = "cpu"
if torch.cuda.is_available():
	device = "cuda"
print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

total_batch_size = 2**7 # 2**19, ~0.5M, in number of tokens
B = 16 # micro batch size
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
	"muon_lr": 0.08461969484908016,
	"wte_lr": 0.5956666725475398,
	"lm_head_lr": 0.00380042592753,
	"b_lr": 0.034601962195880175,
	"e_lr": 0.034601962195880175,
	"num_tokens": len(train_loader.tokens)
	}
)

# def train():
torch.manual_seed(0)
# wandb.init(project='hutter')
config = wandb.config
raw_model = GPT()
# raw_model.load_state_dict(torch.load('log/driven-waterfall-2818.pt', weights_only=False)['model'])
raw_model = raw_model.to(device)
model = torch.compile(raw_model)

weight_count = 12*GPTConfig.n_layer*GPTConfig.n_embd**2 + GPTConfig.vocab_size*GPTConfig.n_embd

max_steps = 100000#(len(train_loader.tokens) // total_batch_size)

cooldown_frac = 0.0
def get_lr(step: int):
	x = step / max_steps # progress in training
	assert 0 <= x < 1
	if x < 1 - cooldown_frac:
		return 1.0
	else:
		w = (1 - x) / cooldown_frac
		return w * 1.0 + (1 - w) * 0.1

# init the optimizer(s)
optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=config.wte_lr, betas=(0.9, 0.95), fused=True)
optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=config.lm_head_lr, betas=(0.9, 0.95), fused=True)
w_params = [m.weight for m in raw_model.transformer.h.modules() if isinstance(m, Quantized_Linear)]
e_params = [m.e for m in raw_model.modules() if isinstance(m, Quantized_Linear)]
b_params = [m.b for m in raw_model.modules() if isinstance(m, Quantized_Linear)]
optimizer3 = SingleDeviceMuon(w_params, lr=config.muon_lr, momentum=0.95)
optimizer4 = torch.optim.Adam(e_params, lr=config.e_lr, betas=(0.9, 0.95), fused=True)
optimizer5 = torch.optim.Adam(b_params, lr=config.b_lr, betas=(0.9, 0.95), fused=True)
optimizers = [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5]

for opt in optimizers:
	for group in opt.param_groups:
		group["initial_lr"] = group["lr"]

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

for step in range(max_steps):
	t0 = time.time()

	# once in a while evaluate our validation loss
	if step == (max_steps - 1) or step % 500 == 0:
		# optionally write model checkpoints
		checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
		checkpoint = {
			'model': raw_model.state_dict(),
			'config': model.config,
			'step': step,
		}
		# you might also want to add optimizer.state_dict() and
		# rng seeds etc., if you wanted to more exactly resume training
		torch.save(checkpoint, checkpoint_path)

	# do one step of the optimization
	model.zero_grad(set_to_none=True)
	loss_accum = 0.0
	loss_last_accum = 0.0
	ce_loss_accum = 0.0
	q_loss_accum = 0.0
	for micro_step in range(grad_accum_steps):
		x, y = train_loader.next_batch()
		x, y = x.to(device), y.to(device)
		# added after video, this field is also used by the forward pass.
		# with te.autocast(enabled=True, recipe=fp8_recipe):
		with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
			logits = model(x)
		loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
		ce_loss = loss.detach()
		Q = torch.stack([m.qbits() for m in model.modules() if isinstance(m, Quantized_Linear)]).sum()
		Q = Q / weight_count
		loss = (len(train_loader.tokens) * loss * np.log2(np.e) + weight_count * Q)/(8*10**8)
		loss_last = F.cross_entropy(logits[:, -1, :].reshape(-1, logits.size(-1)), y[:, -1].view(-1), ignore_index=-1)
		loss = loss / grad_accum_steps
		ce_loss = ce_loss / grad_accum_steps
		Q = Q / grad_accum_steps
		ce_loss_accum += ce_loss
		q_loss_accum += Q.detach()
		loss_last = loss_last / grad_accum_steps
		loss_accum += loss.detach()
		loss_last_accum += loss_last.detach()
		loss.backward()
	norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10000)
	# determine and set the learning rate for this iteration
	for opt in optimizers:
		for group in opt.param_groups:
			group["lr"] = group["initial_lr"] * get_lr(step)
	# prevent nan
	with torch.no_grad():
		for m in raw_model.modules():
			if isinstance(m, Quantized_Linear):
				m.e.clamp_(-16.0, 16.0)
				m.b.clamp_(0, 16.0)

	for opt in optimizers:
		opt.step()
	if device_type == "cuda":
		torch.cuda.synchronize() # wait for the GPU to finish work
	t1 = time.time()
	dt = t1 - t0 # time difference in seconds
	seq_processed = train_loader.B * grad_accum_steps
	seq_per_sec = seq_processed / dt
	print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {get_lr(step):.4e} | dt: {dt*1000:.2f}ms | seq/sec: {seq_per_sec:.2f}")

	es = torch.cat([m.e.detach().float().view(-1) for m in raw_model.modules() if isinstance(m, Quantized_Linear)])
	bs = torch.cat([m.b.detach().float().view(-1) for m in raw_model.modules() if isinstance(m, Quantized_Linear)])

	wandb.log({
		"loss": ce_loss_accum.item(),
		"bytes": (len(train_loader.tokens) * ce_loss_accum.item() * np.log2(np.e) + weight_count * q_loss_accum)/8, 
		"bits_per_weight": q_loss_accum,
		"lr": get_lr(step),
		"norm": norm,
		"last_loss": loss_last_accum.item(),
		'e_min': es.min().item(), 'e_max': es.max().item(), 'e_mean': es.mean().item(),
		'b_min': bs.min().item(), 'b_max': bs.max().item(), 'b_mean': bs.mean().item(),
		})

# sweep_config = {
#     'method': 'bayes',
# 	'metric': {
# 		'name': 'bytes',
# 		'goal': 'minimize'
# 	},
# 	'parameters': {
# 		'muon_lr': {'distribution': 'uniform', 'min': 0.01, 'max': 0.1},
# 		'wte_lr': {'distribution': 'uniform', 'min': 0.01, 'max': 0.8},
# 		'lm_head_lr': {'distribution': 'uniform', 'min': 0.001, 'max': 0.1},
# 		# 'b_lr': {'distribution': 'uniform', 'min': 0.001, 'max': 0.2},
# 		# 'e_lr': {'distribution': 'uniform', 'min': 0.001, 'max': 0.1},
# 	}
#     }

# sweep_id = wandb.sweep(sweep_config, project="hutter")
# wandb.agent(sweep_id, train)

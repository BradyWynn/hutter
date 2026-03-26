import os
import time
import torch
from torch.nn import functional as F
import numpy as np
from model import GPT, GPTConfig, Quantized_Linear, Quantized_Embedding
import wandb
from muon import SingleDeviceMuon
import lzma

def load_tokens():
	with open('30k_enwik9', 'rb') as f:
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
B = 8 # micro batch size
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
	"muon_lr": 0.00320264250353693,
	"wte_lr": 0.057634333366200406,
	"lm_head_lr": 0.0003347659883507895,
	# "b_lr": 0.034601962195880175/10,
	"num_tokens": len(train_loader.tokens)
	}
)

# def train():
torch.manual_seed(0)
# wandb.init(project='hutter')
config = wandb.config
raw_model = GPT()
# raw_model.load_masks('log/prime-microwave-4201.pt')
# raw_model.load_weights('log/valiant-river-4625.pt')
raw_model.load_state_dict(torch.load('log/tiny-bullet-bill-4780.pt', weights_only=False)['model'])
raw_model = raw_model.to(device)
model = torch.compile(raw_model)

weight_count = 12*GPTConfig.n_layer*GPTConfig.n_embd**2 + 2*GPTConfig.vocab_size*GPTConfig.n_embd
print(weight_count)

max_steps = 100000 #(len(train_loader.tokens) // total_batch_size)
# p_nonzero = (10**6/weight_count)

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

# def get_frac0(step: int):
# 	x = step / max_steps
# 	start = 0.5
# 	end = 0.05
# 	return start * (end / start) ** x


# init the optimizer(s)
optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight], lr=config.wte_lr, betas=(0.9, 0.95), fused=True)
optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=config.lm_head_lr, betas=(0.9, 0.95), fused=True)
w_params = [m.weight for m in raw_model.transformer.h.modules() if isinstance(m, torch.nn.Linear)]
# b_params = [m.b for m in raw_model.modules() if isinstance(m, (Quantized_Linear, Quantized_Embedding))]
optimizer3 = SingleDeviceMuon(w_params, lr=config.muon_lr, momentum=0.0)
# optimizer4 = torch.optim.Adam(b_params, lr=config.b_lr, betas=(0.9, 0.95), fused=True)
optimizers = [optimizer1, optimizer2, optimizer3]

for opt in optimizers:
	for group in opt.param_groups:
		group["initial_lr"] = group["lr"]

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

for step in range(0, max_steps):
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
		# total_bits = 0
		# for m in raw_model.modules():
		# 	if isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
		# 		bits = lzma.compress(m.weight.cpu().detach().numpy())
		# 		total_bits += len(bits)
		# wandb.log({'lzma_bits': total_bits})

	# do one step of the optimization
	model.zero_grad(set_to_none=True)
	loss_accum = 0.0
	loss_last_accum = 0.0
	ce_loss_accum = 0.0
	# q_loss_accum = 0.0
	act_norm_accum = 0.0
	# bits_accum = 0.0
	# weight_count_rounded = sum([m.z.round().sum().item() for m in raw_model.modules() if isinstance(m, (Quantized_Linear, Quantized_Embedding))])
	for micro_step in range(grad_accum_steps):
		x, y = train_loader.next_batch()
		x, y = x.to(device), y.to(device)
		# added after video, this field is also used by the forward pass.
		with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
			logits, act_norm = model(x)
		loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.flatten(), ignore_index=-1)
		ce_loss = loss.detach()
		# Q = torch.stack([m.qbits() for m in model.modules() if isinstance(m, (Quantized_Linear, Quantized_Embedding))]).sum()
		# bits = sum([m.get_bits() for m in model.modules() if isinstance(m, (Quantized_Linear, Quantized_Embedding))])
		loss_last = F.cross_entropy(logits[:, -1, :].reshape(-1, logits.size(-1)), y[:, -1].view(-1), ignore_index=-1)
		# loss = (len(train_loader.tokens) * loss * np.log2(np.e) + bits)/(8*10**8)
		loss = loss / grad_accum_steps
		ce_loss = ce_loss / grad_accum_steps
		# Q = Q / grad_accum_steps
		ce_loss_accum += ce_loss
		# q_loss_accum += Q.detach()
		act_norm_accum += act_norm.item() / grad_accum_steps
		# bits_accum += bits / grad_accum_steps
		loss_last = loss_last / grad_accum_steps
		loss_accum += loss.detach()
		loss_last_accum += loss_last.detach()
		loss.backward()
	torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1)
	weight_norm = torch.nn.utils.clip_grad_norm_(w_params + [raw_model.transformer.wte.weight, raw_model.lm_head.weight], 1)
	# determine and set the learning rate for this iteration
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
	print(f"step {step:5d} |loss: {loss_accum.item():.6f} | lr {get_lr(step):.4e} | dt: {dt*1000:.2f}ms | seq/sec: {seq_per_sec:.2f}")

	wandb.log({
		"loss": ce_loss_accum.item(),
		# "bytes": (len(train_loader.tokens) * ce_loss_accum.item() * np.log2(np.e) + bits_accum)/8, 
		# "bits_per_weight": bits_accum/weight_count,
		"lr": get_lr(step),
		"weight_norm": weight_norm,
		"last_loss": loss_last_accum.item(),
		# "params": weight_count_rounded,
		# "params_unrounded": q_loss_accum,
		'step': step,
		'act_norm': act_norm_accum,
		# 'model_bytes': bits_accum/8,
		# "real_params": bits_accum,
		'dt': dt*1000,
		})

# sweep_config = {
#     'method': 'bayes',
# 	'metric': {
# 		'name': 'loss',
# 		'goal': 'minimize'
# 	},
# 	'parameters': {
# 		'muon_lr': {'distribution': 'uniform', 'min': 0.0, 'max': 0.010031281386408198},
# 		'wte_lr': {'distribution': 'uniform', 'min': 0.0, 'max': 0.07826406405724602},
# 		'lm_head_lr': {'distribution': 'uniform', 'min': 0.0, 'max': 0.00107116745635},
# 	}
#     }

# sweep_id = wandb.sweep(sweep_config, project="hutter")
# wandb.agent(sweep_id, train)

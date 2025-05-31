"""
adapted from https://github.com/pytorch-labs/gpt-fast
"""
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import inspect
from muon import Muon

@dataclass
class GPTConfig:
	context_length: int = 1024
	vocab_size: int = 50256
	n_layer: int = 4
	n_head: int = 4
	dim: int = 256

	@property
	def head_dim(self):
		return self.dim // self.n_head

class TransformerBlock(nn.Module):
	def __init__(self, config: GPTConfig) -> None:
		super().__init__()
		self.attn = Attention(config)
		self.mlp = FeedForward(config)

	def forward(self, x: Tensor) -> Tensor:
		h = x + self.attn(F.layer_norm(x, (x.size(-1),)))
		out = h + self.mlp(F.layer_norm(h, (h.size(-1),)))
		return out

class Attention(nn.Module):
	def __init__(self, config: GPTConfig):
		super().__init__()
		assert config.dim % config.n_head == 0
		self.config = config
		# key, query, value projections for all heads, but in a batch
		self.c_attn = nn.Linear(config.dim, 3*config.dim, bias=True)
		self.c_proj = nn.Linear(config.dim, config.dim, bias=True)
		# self.c_proj.weight.data.zero_()
		self.c_proj.NANOGPT_SCALE_INIT = 1

	def forward(self, x: Tensor) -> Tensor:
		bsz, seqlen, _ = x.shape

		q, k, v = self.c_attn(x).split([self.config.dim, self.config.dim, self.config.dim], dim=-1)

		q = q.view(bsz, seqlen, self.config.n_head, self.config.head_dim)
		k = k.view(bsz, seqlen, self.config.n_head, self.config.head_dim)
		v = v.view(bsz, seqlen, self.config.n_head, self.config.head_dim)

		q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

		y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
		y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.config.dim)
		return self.c_proj(y)

class FeedForward(nn.Module):
	def __init__(self, config: GPTConfig) -> None:
		super().__init__()
		self.c_fc = nn.Linear(config.dim, 4*config.dim, bias=True)
		self.c_proj = nn.Linear(4*config.dim, config.dim, bias=True)
		# self.c_proj.weight.data.zero_()
		self.c_proj.NANOGPT_SCALE_INIT = 1

	def forward(self, x: Tensor) -> Tensor:
		return self.c_proj(F.relu(self.c_fc(x)).square())

class GPT(nn.Module):
	def __init__(self, config: GPTConfig=GPTConfig()) -> None:
		super().__init__()
		self.config = config

		transformer = {
		'wte' : nn.Embedding(config.vocab_size, config.dim),
		'wpe' : nn.Embedding(config.context_length, config.dim),
		'h'   : nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer)),
		}

		self.transformer = nn.ModuleDict(transformer)
		self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
		# self.lm_head.weight.data.zero_()

		self.apply(self._init_weights)

	def _init_weights(self, module):
		std = (self.config.dim)**-0.5
		if isinstance(module, nn.Linear):
			if hasattr(module, 'NANOGPT_SCALE_INIT'):
				std *= (2 * self.config.n_layer) ** -0.5
			torch.nn.init.normal_(module.weight, mean=0.0, std=std)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=std)

	def forward(self, idx: Tensor) -> Tensor:
		input_pos = torch.arange(idx.shape[1], device=idx.device)

		x = self.transformer.wte(idx) + self.transformer.wpe(input_pos)

		for _, layer in enumerate(self.transformer.h):
			x = layer(x)

		x = F.layer_norm(x, (x.size(-1),))
		logits = self.lm_head(x)
		return logits
	
	def configure_optimizers(self, weight_decay, muon_lr, adamw_lr, device_type):
		# start with all of the candidate parameters
		param_dict = {pn: p for pn, p in self.named_parameters()}
		# filter out those that do not require grad
		param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
		# create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
		# i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
		decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
		nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
		optim_groups = [
			{'params': decay_params, 'weight_decay': weight_decay},
			{'params': nodecay_params, 'weight_decay': 0.0}
		]
		num_decay_params = sum(p.numel() for p in decay_params)
		num_nodecay_params = sum(p.numel() for p in nodecay_params)
		print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
		print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
		# Create AdamW optimizer and use the fused version if it is available
		fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
		use_fused = fused_available and device_type == 'cuda'
		# extra_args = dict(fused=True) if use_fused else dict()
		# optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, **extra_args)
		print(f"using fused AdamW: {use_fused}")

		muon_params = decay_params
		adamw_params = nodecay_params

		optimizers = [Muon(muon_params, lr=muon_lr, momentum=0.95), torch.optim.AdamW(adamw_params, lr=adamw_lr, betas=(0.90, 0.95), weight_decay=weight_decay)]

		return optimizers
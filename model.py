"""
adapted from https://github.com/pytorch-labs/gpt-fast
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

@dataclass
class GPTConfig:
	context_length: int = 1024
	vocab_size: int = 50256
	n_layer: int = 4
	n_head: int = 4
	n_embd: int = 512

	@property
	def head_n_embd(self):
		return self.n_embd // self.n_head

def norm(x):
	return F.rms_norm(x, (x.size(-1),))

class Rotary(torch.nn.Module):
	def __init__(self, dim, base=10000):
		super().__init__()
		inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
		self.register_buffer('inv_freq', inv_freq)
		self.seq_len_cached = None
		self.cos_cached = None
		self.sin_cached = None

	def forward(self, x):
		seq_len = x.shape[1]
		if seq_len != self.seq_len_cached:
			self.seq_len_cached = seq_len
			t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
			freqs = torch.outer(t, self.inv_freq).to(x.device)
			self.cos_cached = freqs.cos()
			self.sin_cached = freqs.sin()
		return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
	assert x.ndim == 4  # multihead attention
	d = x.shape[3] // 2
	x1 = x[..., :d]
	x2 = x[..., d:]
	y1 = x1 * cos + x2 * sin
	y2 = x1 * (-sin) + x2 * cos
	return torch.cat([y1, y2], 3).type_as(x)

class TransformerBlock(nn.Module):
	def __init__(self, config: GPTConfig) -> None:
		super().__init__()
		self.attn = Attention(config)
		self.mlp = FeedForward(config)
		self.attn_scale = (1 / (2 * config.n_layer)**0.5)

	def forward(self, x: Tensor) -> Tensor:
		x = x + self.attn_scale * self.attn(norm(x))
		x = x + self.mlp(norm(x))
		return x

class Attention(nn.Module):
	def __init__(self, config: GPTConfig):
		super().__init__()
		assert config.n_embd % config.n_head == 0
		self.config = config
		# key, query, value projections for all heads, but in a batch
		self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=False)
		self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
		self.rotary = Rotary(config.head_n_embd)
		self.c_proj.weight.data.zero_()

	def forward(self, x: Tensor) -> Tensor:
		bsz, seqlen, _ = x.shape

		q, k, v = self.c_attn(x).split([self.config.n_embd, self.config.n_embd, self.config.n_embd], dim=-1)

		q = q.view(bsz, seqlen, self.config.n_head, self.config.head_n_embd)
		k = k.view(bsz, seqlen, self.config.n_head, self.config.head_n_embd)
		v = v.view(bsz, seqlen, self.config.n_head, self.config.head_n_embd)

		cos, sin = self.rotary(q)
		q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

		q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

		y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
		y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.config.n_embd)
		return self.c_proj(y)

class FeedForward(nn.Module):
	def __init__(self, config: GPTConfig) -> None:
		super().__init__()
		self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=False)
		self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=False)
		self.c_proj.weight.data.zero_()

	def forward(self, x: Tensor) -> Tensor:
		return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))

class GPT(nn.Module):
	def __init__(self, config: GPTConfig=GPTConfig()) -> None:
		super().__init__()
		self.config = config

		transformer = {
		'wte' : nn.Embedding(config.vocab_size, config.n_embd),
		'h'   : nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer)),
		}

		self.transformer = nn.ModuleDict(transformer)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
		# self.transformer.wte.weight = self.lm_head.weight
		self.lm_head.weight.data.zero_()

		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=(self.config.n_embd)**-0.5)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=(self.config.n_embd)**-0.5)

	def forward(self, idx: Tensor) -> Tensor:
		x = self.transformer.wte(idx)

		for layer in self.transformer.h:
			x = layer(x)

		x = norm(x)
		logits = self.lm_head(x)
		return logits
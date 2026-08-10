"""
adapted from https://github.com/pytorch-labs/gpt-fast
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
import matplotlib.pyplot as plt

@dataclass
class GPTConfig:
	context_length: int = 1024
	vocab_size: int = 5256
	n_layer: int = 24
	n_head: int = 16
	n_embd: int = 1024

	@property
	def head_n_embd(self):
		return self.n_embd // self.n_head

def norm(x):
	return F.rms_norm(x + 1e-2, (x.size(-1),))

def find_multiple(n: int, k: int) -> int:
	if n % k == 0:
		return n
	return n + k - (n % k)

def topk_sparsify(x, k):
	B = x.shape[0]
	flat = x.reshape(B, -1)
	_, idx = flat.abs().topk(k, dim=-1)
	mask = torch.zeros_like(flat).scatter_(-1, idx, 1.0)
	return (flat * mask).view_as(x)

class KVCache(nn.Module):
	def __init__(self, max_batch_size, max_seq_length, n_heads, head_n_embd, dtype=torch.bfloat16):
		super().__init__()
		cache_shape = (max_batch_size, n_heads, max_seq_length, head_n_embd)
		self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
		self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

	def update(self, input_pos, k_val, v_val):
		assert input_pos.shape[0] == k_val.shape[2]
		k_out = self.k_cache
		v_out = self.v_cache
		k_out[:, :, input_pos] = k_val
		v_out[:, :, input_pos] = v_val
		return k_out, v_out

class Rotary(torch.nn.Module):
	def __init__(self, dim, base=10000, max_seq_len=GPTConfig.context_length):
		super().__init__()
		inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
		t = torch.arange(max_seq_len).float()
		freqs = torch.outer(t, inv_freq)
		self.register_buffer('cos_cached', freqs.cos())
		self.register_buffer('sin_cached', freqs.sin())

	def forward(self, x, input_pos=None):
		if input_pos is not None:
			cos = self.cos_cached[input_pos]
			sin = self.sin_cached[input_pos]
		else:
			cos = self.cos_cached[:x.shape[1]]
			sin = self.sin_cached[:x.shape[1]]
		return cos[None, :, None, :], sin[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
	assert x.ndim == 4  # multihead attention
	d = x.shape[3] // 2
	x1 = x[..., :d]
	x2 = x[..., d:]
	y1 = x1 * cos + x2 * sin
	y2 = x1 * (-sin) + x2 * cos
	return torch.cat([y1, y2], 3).type_as(x)

class FeedForward(nn.Module):
	def __init__(self, config: GPTConfig) -> None:
		super().__init__()
		self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=True)
		self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=True)
		self.c_proj.weight.data.zero_()

	def forward(self, x: Tensor) -> Tensor:
		return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))

class TransformerBlock(nn.Module):
	def __init__(self, config: GPTConfig) -> None:
		super().__init__()
		self.attn = Attention(config)
		self.mlp = FeedForward(config)
		self.attn_scale = (1 / (2 * config.n_layer)**0.5)

	def forward(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
		x = x + self.attn_scale * self.attn(norm(x), input_pos, mask)
		x = x + self.mlp(norm(x))
		return x

class Attention(nn.Module):
	def __init__(self, config: GPTConfig):
		super().__init__()
		assert config.n_embd % config.n_head == 0
		self.config = config
		# key, query, value projections for all heads, but in a batch
		self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=True)
		self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
		self.rotary = Rotary(config.head_n_embd, max_seq_len=config.context_length)
		self.c_proj.weight.data.zero_()
		self.kv_cache = None

	def forward(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
		bsz, seqlen, _ = x.shape

		q, k, v = self.c_attn(x).split([self.config.n_embd, self.config.n_embd, self.config.n_embd], dim=-1)

		q = q.view(bsz, seqlen, self.config.n_head, self.config.head_n_embd)
		k = k.view(bsz, seqlen, self.config.n_head, self.config.head_n_embd)
		v = v.view(bsz, seqlen, self.config.n_head, self.config.head_n_embd)

		q, k = norm(q), norm(k)

		cos, sin = self.rotary(q, input_pos)
		q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

		q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

		if self.kv_cache is not None:
			k, v = self.kv_cache.update(input_pos, k, v)

		y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
		y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.config.n_embd)
		return self.c_proj(y)

class GPT(nn.Module):
	def __init__(self, config: GPTConfig=GPTConfig()) -> None:
		super().__init__()
		self.config = config

		transformer = {
		'wte' : nn.Embedding(config.vocab_size, config.n_embd),
		'h'   : nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer)),
		}

		self.transformer = nn.ModuleDict(transformer)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
		# self.transformer.wte.weight = self.lm_head.weight
		self.lm_head.weight.data.zero_()

		causal_mask = torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)).view(1, 1, config.context_length, config.context_length)
		self.register_buffer('causal_mask', causal_mask)

	def setup_caches(self, max_batch_size, max_seq_length):
		max_seq_length = find_multiple(max_seq_length, 8)
		self.max_seq_length = max_seq_length
		self.max_batch_size = max_batch_size
		for b in self.transformer.h:
			b.attn.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, self.config.head_n_embd)

		causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).view(1, 1, self.max_seq_length, self.max_seq_length)
		self.register_buffer('causal_mask', causal_mask)  # now moves with .to(device)

	def forward(self, idx: Tensor, input_pos: Tensor=None) -> Tensor:
		x = self.transformer.wte(idx)

		if input_pos is None:
			input_pos = torch.arange(GPTConfig.context_length, device=idx.device)

		mask = self.causal_mask[:, :, input_pos]

		for layer in self.transformer.h:
			x = layer(x, input_pos, mask)

		x = norm(x)

		logits = self.lm_head(x)
		return logits

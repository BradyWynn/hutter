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
    context_length: int = 2048
    vocab_size: int = 241
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256

    @property
    def head_n_embd(self):
        return self.n_embd // self.n_head

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

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
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.rotary = Rotary(config.head_n_embd)
        self.c_proj.weight.data.zero_()
        self.kv_cache = None

    def forward(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape

        q, k, v = self.c_attn(x).split([self.config.n_embd, self.config.n_embd, self.config.n_embd], dim=-1)

        q = q.view(bsz, seqlen, self.config.n_head, self.config.head_n_embd)
        k = k.view(bsz, seqlen, self.config.n_head, self.config.head_n_embd)
        v = v.view(bsz, seqlen, self.config.n_head, self.config.head_n_embd)

        q, k = norm(q), norm(k)

        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
        self.transformer.wte.weight = self.lm_head.weight
        # self.lm_head.weight.data.zero_()

    def setup_caches(self, max_batch_size, max_seq_length):
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.transformer.h:
            b.attn.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, self.config.head_n_embd)

        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).view(1, 1, self.max_seq_length, self.max_seq_length)

    def forward(self, idx: Tensor, input_pos: Tensor=None) -> Tensor:
        x = self.transformer.wte(idx)

        # mask = self.causal_mask[:, :, input_pos]
        mask = None

        for layer in self.transformer.h:
            x = layer(x, input_pos, mask)

        x = norm(x)
        logits = self.lm_head(x)
        return logits

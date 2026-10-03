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
	# Sized by measurement to code all of enwik9 inside ~10 minutes on a 5070 Ti. Throughput
	# tracks n_layer far more than parameter count: the step is ~90% GPU-busy but reaches only
	# 9-16 of the card's 88 TFLOP/s, because most kernels are bandwidth-bound elementwise work
	# (norms, rotary, residuals, loss) whose cost scales with tokens rather than weights.
	#
	# n_layer=3 is a budget choice, not a quality one. At the pre-sweep LRs, depth 6 reaches
	# 127.18 MB / 1.0174 bpb against depth 3's 129.00 / 1.0320 -- but takes 16.8 min instead of
	# ~10.8. An earlier sweep concluded depth was expensive and barely helped; that was measured
	# with LRs which destabilise at depth 6 (pre-clip gradient norm spiking to ~0.95 against a
	# 0.26 baseline), so it understated depth. Depth 8+ is untested at stable LRs.
	context_length: int = 1024
	vocab_size: int = 10256
	n_layer: int = 12
	n_head: int = 3
	n_embd: int = 192
	# Apply attn_scale to the mlp branch too. The scale used to land on attention alone, an
	# asymmetry no one chose: it arrived with the rmsnorm commit and was never revisited.
	# Making it symmetric is worth -0.44 MB at n_layer=6 (126.46 -> 126.02). See
	# TransformerBlock, and prequential.py for what this does NOT yet distinguish.
	scale_mlp: bool = True
	# Boundary operator, arXiv 2609.19107 eq. 3: BO(h, e) = Norm(h) + alpha*e, applied once
	# between the core and the coda. Renormalising the stream stops each block's update from
	# being a shrinking fraction of a residual that grows with depth (the curse of depth), and
	# re-injecting the prelude output e keeps late blocks conditioned on the input. The paper
	# splits n_layer into prelude/core/coda roughly evenly. alpha=0 disables the whole thing.
	bo_alpha: float = 0.0
	bo_prelude: int = 0      # blocks before e is taken
	bo_coda: int = 0         # blocks after the operator fires

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
	"""Sliding-window KV cache holding the `window` most recent positions.

	Entry i sits in slot i and carries the rotary phase of position i, so the cache always
	looks like a fresh prefill of positions 0..length-1. Length/eviction bookkeeping lives
	in GPT so every layer stays in lockstep. Buffers are non-persistent: checkpoints keep
	the same keys they had before caches existed.
	"""
	def __init__(self, max_batch_size, window, n_heads, head_n_embd, dtype=torch.bfloat16):
		super().__init__()
		cache_shape = (max_batch_size, n_heads, window, head_n_embd)
		self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype), persistent=False)
		self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype), persistent=False)

	def write(self, input_pos, k_val, v_val):
		# detached on purpose: the cache is data, not part of anyone's graph. Gradients reach
		# the k/v of the tokens currently in flight through the cat in Attention.forward and
		# stop at the window boundary (truncated BPTT).
		with torch.no_grad():
			self.k_cache[:, :, input_pos] = k_val.detach().to(self.k_cache.dtype)
			self.v_cache[:, :, input_pos] = v_val.detach().to(self.v_cache.dtype)

	@torch.no_grad()
	def evict(self, n, length, cos, sin):
		"""Drop the oldest n entries; survivors slide down n slots.

		Sliding changes which position each entry claims to be at, so the keys are rotated
		by -n positions to match. Rope phase is linear in position, so this is exact --
		the shifted entry is bit-for-bit what a prefill would have written there (up to
		bf16 rounding), and the caller passes cos/sin for the -n rotation.
		"""
		keep = length - n
		k = self.k_cache[:, :, n:length].clone()   # clone: src and dst overlap in memory
		v = self.v_cache[:, :, n:length].clone()
		self.k_cache[:, :, :keep] = apply_rotary_emb(k, cos, sin)
		self.v_cache[:, :, :keep] = v
		# slots [keep, length) keep stale values; they are masked out, so no need to clear.

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
		# GPT-2 scales residual projections by 1/sqrt(N) over N=2*n_layer residual branches, to
		# stop init-time variance accumulating down the stream. Here BOTH c_proj weights are
		# zero-initialised, so nothing accumulates at init and that rationale does not apply --
		# what the constant actually does is damp attention's function-space step relative to
		# the mlp's, permanently and by 1/sqrt(2L). It reached the mlp branch in neither form.
		# Note it moves with depth: n_layer 3 -> 6 took it from 0.408 to 0.289, so a depth
		# change silently re-weights attention against the mlp.
		self.attn_scale = (1 / (2 * config.n_layer)**0.5)
		self.mlp_scale = self.attn_scale if config.scale_mlp else 1.0

	def forward(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
		x = x + self.attn_scale * self.attn(norm(x), input_pos, mask)
		x = x + self.mlp_scale * self.mlp(norm(x))
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

	def evict_cache(self, n: int, length: int) -> None:
		# rope at position -n: cos is even, sin is odd. Rotating the cached keys by this
		# undoes n positions worth of phase.
		cos = self.rotary.cos_cached[n].view(1, 1, 1, -1)
		sin = -self.rotary.sin_cached[n].view(1, 1, 1, -1)
		self.kv_cache.evict(n, length, cos, sin)

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

		if self.kv_cache is not None and input_pos is not None:
			# cat before write, so the past half is the cache as it was on entry. The window
			# is read whole (static shape, no recompiles); slots past `length` hold stale
			# entries that `mask` drops.
			k_all = torch.cat([self.kv_cache.k_cache.to(k.dtype), k], dim=2)
			v_all = torch.cat([self.kv_cache.v_cache.to(v.dtype), v], dim=2)
			self.kv_cache.write(input_pos, k, v)
			k, v = k_all, v_all

		y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
		y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.config.n_embd)
		return self.c_proj(y)

_NG_MASK  = 0x7FFFFFFFFFFFFFFF                 # clamp into non-negative int64
_NG_SALT  = 0x9E3779B97F4A7C15 - (1 << 64)     # golden ratio, as a signed int64
_NG_PRIME = 0x100000001B3                      # FNV-1a
_NG_MIX   = 0xBF58476D1CE4E5B9 - (1 << 64)     # splitmix64, as a signed int64

class HashedNgram(nn.Module):
	"""Learned embeddings indexed by hashes of the last n input tokens, added to wte.

	A row is reached only when its n-gram occurs, so both the lookup and the optimizer cost scale
	with distinct n-grams in the chunk rather than with table size -- the table can hold orders of
	magnitude more parameters than the transformer without adding FLOPs or per-step optimizer work.
	Costs nothing in bits: the decoder rebuilds it from tokens it has already decoded.

	Zero-initialised, so an untrained table is exactly a no-op rather than noise added to wte.

	Predicting target t conditions on input tokens ..., idx[t-1], idx[t], so an order-n hash needs
	n-1 tokens of left context. Across a streaming chunk boundary those live in `prev`, which is
	kept in lockstep with the KV caches. Replay windows pass use_state=False and start from zeros:
	their first n-1 positions get a context-free hash, which is wrong for those few positions but
	must not consume stream state that the coding path owns.

	`row_dim` decouples the stored row width from the model width. Row memory is n_rows*row_dim,
	so with row_dim pinned the model can be widened without multiplying millions of rows -- which
	is otherwise what blocks widening, since n_embd would be the row width. A row_dim BELOW dim
	was measured and lost badly (8Mx64 lifted to 256 cost 3.74 MB against 8Mx256 at matched rows):
	a row holds a learned representation, not a count, so width is what it is short of. This is
	the opposite direction -- rows stay wide, the model gets wider than them.

	Three variants were built, measured on enwik9 and removed; see prequential.py for the numbers.
	Per-channel row partitions and extra/longer order channels both lost to the plain shared
	table, for the same reason: collisions are benign because colliding contexts share an
	expressive vector, so more rows or better-allocated rows buy less than keeping them wide.

	`heads` (K) splits each order into K independently-salted lookups of width row_dim/K, whose
	results are CONCATENATED back to row_dim -- DeepSeek's Engram (arXiv 2601.07372 eq. 1-2).
	Budget is invariant in K by construction: K partitions of n_rows rows at row_dim/K each.
	The point is collision redundancy. At K=1 a collision is unrecoverable, since the two
	contexts get literally the same vector; under concat they must collide in all K heads to
	be indistinguishable. Measured: 20.66% of distinct trigrams collide at K=1 and 0.000% at
	K=2 -- two independent hashes into 4M rows each expect ~0.1 colliding pairs in the whole
	table, so this ends collisions rather than reducing them. What it costs is the width of
	each retrieved block, the scarce resource here (see table_dim above).

	Zero collisions at half width won, and the curve then turns back up once there is no
	aliasing left to buy: K=1 129.37 MB, K=2 128.80, K=4 128.97. Set K to the smallest value
	whose offline collision count is zero -- that is a unique() over hashed n-grams on a token
	sample, so it never needs a training sweep. One run per arm; see prequential.py.

	Heads partition the rows rather than sharing them, unlike orders, which share. The
	asymmetry is real: order channels are SUMMED, so a shared row serves two contexts in one
	coordinate space, while head k lands in coordinates [k*head_dim, (k+1)*head_dim) of the
	concat. A row reached by two different heads would have to encode unrelated things in
	unrelated slots, which is not the benign kind of collision.
	"""
	def __init__(self, n_rows: int, dim: int, orders=(2, 3), max_batch_size: int = 1,
	             row_dim: int = 0, heads: int = 1, table_dtype=torch.float32):
		super().__init__()
		assert n_rows > 0 and all(o >= 1 for o in orders)
		assert heads >= 1
		rd = row_dim or dim
		assert rd % heads == 0, f"row width {rd} must split evenly into {heads} heads"
		self.n_rows = n_rows
		self.orders = tuple(orders)
		self.heads = heads
		self.head_dim = rd // heads
		self.carry = max(self.orders) - 1
		self.table = nn.Embedding(n_rows * heads, self.head_dim, sparse=True, dtype=table_dtype)
		self.table.weight.data.zero_()
		# zero table + ordinary projection, never the reverse: this keeps the module a no-op at
		# init while gradient still reaches the rows on step one. With a zeroed projection,
		# d(loss)/d(row) would be zero and the table could never start learning.
		self.proj = nn.Linear(rd, dim, bias=False) if rd != dim else None
		self.register_buffer('prev', torch.zeros(max_batch_size, max(self.carry, 1), dtype=torch.long),
		                     persistent=False)
		# P^j mod 2^64 for a polynomial window hash, folded into signed int64. A per-token chain
		# would cost `order` kernel launches per channel; this is one multiply and one sum whatever
		# the order, which is what makes long channels affordable.
		pw, x = [], 1
		for _ in range(max(self.orders)):
			pw.append(x - (1 << 64) if x >= (1 << 63) else x)
			x = (x * _NG_PRIME) % (1 << 64)
		self.register_buffer('powers', torch.tensor(pw, dtype=torch.long), persistent=False)
		self.salts = [(s * _NG_SALT) & _NG_MASK
		              for s in range(1, len(self.orders) * self.heads + 1)]

	def reset(self):
		self.prev.zero_()

	def row_ids(self, idx, use_state: bool = True):
		"""Row index per channel for each position, as a list of [B, T] int64 tensors.

		Order-major and `heads` long per order: [o0h0, o0h1, ..., o1h0, ...]. The polynomial
		is computed once per order and only the finalizer reruns per head, so K heads cost K
		cheap elementwise chains rather than K unfolds.

		torch has no uint64, so every intermediate is masked back into the non-negative int64
		range: that keeps `>>` a logical shift (on a negative int64 it sign-extends and destroys
		the mixing) and makes the result safe to index with. Multiplication overflow wraps, as
		intended for a hash.
		"""
		B, T = idx.shape
		k = self.carry
		left = self.prev[:B, :k] if use_state else idx.new_zeros(B, k)
		ctx = torch.cat([left, idx], dim=1) if k else idx
		out = []
		for i, order in enumerate(self.orders):
			# window ending at position t is ctx[k+t-order+1 : k+t+1]
			w = ctx.unfold(1, order, 1)[:, k - order + 1: k - order + 1 + T]   # [B, T, order]
			base = (w * self.powers[:order]).sum(-1) & _NG_MASK
			for j in range(self.heads):
				h = base ^ self.salts[i * self.heads + j]
				h = h ^ (h >> 29)
				h = (h * _NG_MIX) & _NG_MASK
				h = h ^ (h >> 32)
				# + j*n_rows keeps head j inside its own row partition; see the class docstring
				out.append(torch.remainder(h, self.n_rows) + j * self.n_rows)
		return out

	def forward(self, idx, use_state: bool = True):
		out, ids, K = None, self.row_ids(idx, use_state), self.heads
		for i in range(0, len(ids), K):
			# concat within an order, sum across orders
			e = self.table(ids[i]) if K == 1 else \
				torch.cat([self.table(r) for r in ids[i:i + K]], dim=-1)
			out = e if out is None else out + e
		# a low-precision table stays low-precision only in storage; everything downstream of
		# the gather is fp32. Row values peak near 264 against fp16's 65504, so the sum across
		# orders cannot overflow on the way here.
		out = out.float()
		if self.proj is not None:
			out = self.proj(out)      # channels sum in row space, one projection -- linearity
		if use_state and self.carry:
			B, T, k = idx.shape[0], idx.shape[1], self.carry
			with torch.no_grad():
				# a short final chunk can be narrower than the carry, in which case the new carry
				# is the tail of the old one followed by the whole chunk
				self.prev[:B, :k] = idx[:, -k:].detach() if T >= k else \
					torch.cat([self.prev[:B, T:k], idx.detach()], dim=1)
		return out

class GPT(nn.Module):
	def __init__(self, config: GPTConfig=GPTConfig()) -> None:
		super().__init__()
		self.config = config

		transformer = {
		'wte' : nn.Embedding(config.vocab_size, config.n_embd),
		'h'   : nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer)),
		}

		self.transformer = nn.ModuleDict(transformer)
		if config.bo_alpha:
			assert config.bo_prelude > 0 and config.bo_coda > 0, "boundary operator needs both stages"
			assert config.bo_prelude < config.n_layer - config.bo_coda, \
				"prelude and coda leave no core between them"
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
		# self.transformer.wte.weight = self.lm_head.weight
		self.lm_head.weight.data.zero_()

		# unused (attention takes the is_causal fast path), kept so old checkpoints still load
		causal_mask = torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)).view(1, 1, config.context_length, config.context_length)
		self.register_buffer('causal_mask', causal_mask)

		self.cache_window = 0
		self.cache_chunk = 0
		self.cache_len = 0

	def setup_caches(self, max_batch_size, chunk, window=None, dtype=torch.bfloat16):
		"""Enable streaming attention: a `window`-slot cache behind `chunk` tokens in flight.

		A coded token sees at most `window` tokens of context (its own slot included -- the
		in-flight tokens are written into the window too). Attention runs over window+chunk keys,
		the extra `chunk` being the not-yet-written copies of the tokens in flight.
		"""
		if window is None:
			window = self.config.context_length
		assert chunk <= window, f"chunk {chunk} does not fit in window {window}"
		assert window <= self.config.context_length, "window must fit the rope tables"
		self.cache_window = window
		self.cache_chunk = chunk
		self.cache_len = 0                 # valid entries in the window, == rope pos of next token
		self.register_buffer('window_pos', torch.arange(window), persistent=False)
		for b in self.transformer.h:
			b.attn.kv_cache = KVCache(max_batch_size, window, self.config.n_head, self.config.head_n_embd, dtype)

	def reset_caches(self):
		self.cache_len = 0
		if getattr(self, 'ngram', None) is not None:
			self.ngram.reset()
		for b in self.transformer.h:
			b.attn.kv_cache.k_cache.zero_()
			b.attn.kv_cache.v_cache.zero_()

	def evict(self, n):
		n = min(n, self.cache_len)
		if n <= 0:
			return
		if n == self.cache_len:
			self.cache_len = 0    # nothing survives, so no phases to fix up (and rope has no
			return                # entry at `window` to de-rotate a full flush by)
		for b in self.transformer.h:
			b.attn.evict_cache(n, self.cache_len)
		self.cache_len -= n

	def next_positions(self, n, evict_tokens=None):
		"""Rope positions for the next n tokens, evicting first if they do not fit.

		Bookkeeping is deliberately outside forward(): cache_len is a Python int, and mutating
		it inside a compiled region would guard on its value and recompile every step.
		"""
		if self.cache_len + n > self.cache_window:
			e = evict_tokens or self.cache_chunk
			self.evict(max(e, self.cache_len + n - self.cache_window))
		pos = self.cache_len + torch.arange(n, device=self.window_pos.device)
		self.cache_len += n
		return pos

	def cache_mask(self, input_pos: Tensor) -> Tensor:
		"""Bool mask over [window | in-flight chunk] for queries at `input_pos`.

		Every valid cache entry is older than every query, so the window half only has to drop
		the slots that hold nothing yet (or hold evicted junk); the chunk half is plain causal.
		"""
		n = input_pos.shape[0]
		past = (self.window_pos < input_pos[0])[None, :].expand(n, -1)
		cur = input_pos[None, :] <= input_pos[:, None]
		return torch.cat([past, cur], dim=1).view(1, 1, n, -1)

	def forward(self, idx: Tensor, input_pos: Tensor=None, ngram: Tensor=None) -> Tensor:
		# `ngram` is computed outside this module and passed in as a plain tensor: the lookup uses a
		# sparse-gradient Embedding, which does not survive torch.compile, while this forward must.
		x = self.transformer.wte(idx)
		if ngram is not None:
			x = x + ngram

		# input_pos is what selects the streaming path: without it this is an ordinary
		# self-contained causal forward over idx, caches untouched.
		mask = self.cache_mask(input_pos) if input_pos is not None else None

		# boundary operator: stash the prelude output, then renormalise and re-inject it once
		# before the coda. Static Python control flow on config ints, so compile is unaffected.
		bo, e = self.config.bo_alpha, None
		fire = self.config.n_layer - self.config.bo_coda
		for i, layer in enumerate(self.transformer.h):
			x = layer(x, input_pos, mask)
			if bo:
				if i + 1 == self.config.bo_prelude:
					e = x
				if i + 1 == fire:
					x = norm(x) + bo * e

		x = norm(x)

		logits = self.lm_head(x)
		return logits

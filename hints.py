"""Causal retrieval hints for every target position of a token stream.

For target position t (the token being predicted is tokens[t]), each hint model reports what it
would predict from tokens[:t] alone, so a decoder can rebuild every hint from what it has
already decoded: hints cost no bits.

A hint model is a match model with minimum length m: it finds the most recent earlier
occurrence of the last m tokens, verifies it (rejecting hash collisions) and measures the true
match length L, then predicts the token that followed that occurrence. While its predictions
keep coming true it follows the same occurrence forward; on a miss it looks again.

Per target position and hint model:
  hint_tok[t, h]  predicted token (0 when there is no match)
  hint_bkt[t, h]  len_bucket(L) * N_DIST + dist_bucket(t - source), 0 when there is no match
                  (len_bucket >= 1 for any match, so 0 is free to mean "none")
Distance is how far back the copied occurrence sits; the edges are the transformer's view:
<=512 tokens is always in a 1024-token window's context, <=1024 sometimes, beyond never.

Frequency hints (compute_freq) complement these recency hints. For the context of the last k
tokens (k in FREQ_ORDERS) they report the context's most FREQUENT follower so far, with a bucket
of (log2 context count) x (that follower's share of the count). Counts only ever increase, so
updating the argmax on each increment keeps the exact most-frequent follower without rescans.
Counts live in checksummed 2-way hash tables; evictions and collisions make them approximate,
but deterministically so, which is all a decoder needs to reproduce them.

Frequency hints v2 (compute_freq2_chunk) extend this to orders 1-6 and to the 2nd and 3rd most
frequent followers (FREQ2_SPEC lists the (order, rank) columns). The top 3 are tracked exactly
under +1 increments: a follower outside the top 3 can only enter by passing the 3rd's count.
A rank-r follower's share is at most 1/r, so its share bins are scaled by r to use the range.

usage: python hints.py [--tokens 5k_enwik9]
       -> 5k_enwik9.hint_tok.npy,  5k_enwik9.hint_bkt.npy     (match hints)
          5k_enwik9.freq_tok.npy,  5k_enwik9.freq_bkt.npy     (frequency hints, orders 1-4, top 1)
       python hints.py --freq2
       -> 5k_enwik9.freq2_tok.npy, 5k_enwik9.freq2_bkt.npy    (frequency hints v2, FREQ2_SPEC)
"""
import argparse, time
import numpy as np
import numba as nb
import torch
from coding_data import load_tokens

MATCH_MIN = (3, 6, 12)
N_LEN_B = 16
DIST_EDGES = (512, 1024, 8192)
N_DIST = len(DIST_EDGES) + 1
N_BUCKETS = N_LEN_B * N_DIST
FREQ_ORDERS = (1, 2, 3, 4)
N_CNT_B = 8      # log2 context count: 1, 2-3, 4-7, ..., >=128
N_SHARE_B = 8    # top follower's share of the count, in eighths
assert N_CNT_B * N_SHARE_B == N_BUCKETS, "all hint models share one bucket count"
# (order, rank) per v2 column; the first four equal v1's columns in meaning
FREQ2_SPEC = ((1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (2, 2), (3, 2), (4, 2), (3, 3))
FREQ2_ORDERS = (1, 2, 3, 4, 5, 6)
TOPK = 3
U64 = np.uint64
MUL = U64(0x9E3779B97F4A7C15)

@nb.njit(inline='always')
def mix64(x):
	x = (x ^ (x >> U64(30))) * U64(0xBF58476D1CE4E5B9)
	x = (x ^ (x >> U64(27))) * U64(0x94D049BB133111EB)
	return x ^ (x >> U64(31))

@nb.njit(inline='always')
def len_bucket(L):
	if L <= 0:
		return 0
	b = 1
	while L > 1 and b < N_LEN_B - 1:   # log2-ish
		L >>= 1
		b += 1
	return b

@nb.njit(inline='always')
def dist_bucket(d):
	b = 0
	while b < len(DIST_EDGES) and d > DIST_EDGES[b]:
		b += 1
	return b

@nb.njit
def compute(toks, table_bits):
	n = toks.shape[0]
	H = len(MATCH_MIN)
	hint_tok = np.zeros((n, H), np.int16)
	hint_bkt = np.zeros((n, H), np.uint8)
	table = np.zeros((H, 1 << table_bits), np.int64)
	ptr = np.zeros(H, np.int64)
	length = np.zeros(H, np.int64)
	for t in range(1, n):
		# ---- hints for target t, from tokens[:t] ----
		for m in range(H):
			if length[m] > 0:
				hint_tok[t, m] = toks[ptr[m]]
				hint_bkt[t, m] = len_bucket(length[m]) * N_DIST + dist_bucket(t - ptr[m])
		# ---- update with tokens[t] ----
		y = toks[t]
		for m in range(H):
			if length[m] > 0:
				if toks[ptr[m]] == y:
					length[m] += 1
					ptr[m] += 1
				else:
					length[m] = 0
			k = MATCH_MIN[m]
			if t + 1 >= k:
				h = mix64(U64(m + 1) * MUL)
				for j in range(t + 1 - k, t + 1):
					h = mix64(h ^ U64(toks[j]) * MUL)
				idx = np.int64(h >> U64(64 - table_bits))
				if length[m] == 0:
					cand = table[m, idx]
					if cand > 0:
						L = 0
						while L < 65535 and cand - 1 - L >= 0 and toks[cand - 1 - L] == toks[t - L]:
							L += 1
						if L >= k:
							length[m] = L
							ptr[m] = cand
				table[m, idx] = t + 1
	return hint_tok, hint_bkt

@nb.njit
def compute_freq(toks, ctx_bits, pair_bits):
	n = toks.shape[0]
	H = len(FREQ_ORDERS)
	out_tok = np.zeros((n, H), np.int16)
	out_bkt = np.zeros((n, H), np.uint8)
	ctx_mask = (1 << ctx_bits) - 2
	ctx_chk = np.zeros(1 << ctx_bits, np.uint32)
	ctx_c = np.zeros(1 << ctx_bits, np.uint32)
	ctx_top = np.zeros(1 << ctx_bits, np.int16)
	ctx_topc = np.zeros(1 << ctx_bits, np.uint32)
	pair_mask = (1 << pair_bits) - 2
	pair_chk = np.zeros(1 << pair_bits, np.uint32)
	pair_c = np.zeros(1 << pair_bits, np.uint32)
	ctx_h = np.zeros(H, np.uint64)
	ctx_slot = np.full(H, -1, np.int64)
	for t in range(1, n):
		# ---- hints for target t, from tokens[:t] ----
		for o in range(H):
			k = FREQ_ORDERS[o]
			ctx_slot[o] = -1
			if t < k:
				continue
			h = mix64(U64(100 + k) * MUL)
			for j in range(t - k, t):
				h = mix64(h ^ U64(toks[j]) * MUL)
			ctx_h[o] = h
			i0 = np.int64(h & U64(ctx_mask))
			chk = np.uint32((h >> U64(32)) | U64(1))
			if ctx_chk[i0] == chk:
				ctx_slot[o] = i0
			elif ctx_chk[i0 + 1] == chk:
				ctx_slot[o] = i0 + 1
			s = ctx_slot[o]
			if s >= 0 and ctx_c[s] > 0:
				c = ctx_c[s]
				cb = 0
				while c > 1 and cb < N_CNT_B - 1:
					c >>= 1
					cb += 1
				share = min(float(ctx_topc[s]) / float(ctx_c[s]), 1.0)   # re-inserted contexts can overshoot
				sb = min(int(share * N_SHARE_B), N_SHARE_B - 1)
				out_tok[t, o] = ctx_top[s]
				out_bkt[t, o] = cb * N_SHARE_B + sb      # never 0: count 1 means share 1, the top bin
		# ---- update with tokens[t] ----
		y = toks[t]
		for o in range(H):
			if t < FREQ_ORDERS[o]:
				continue
			h = ctx_h[o]
			s = ctx_slot[o]
			if s < 0:
				i0 = np.int64(h & U64(ctx_mask))
				s = i0 if ctx_c[i0] <= ctx_c[i0 + 1] else i0 + 1
				ctx_chk[s] = np.uint32((h >> U64(32)) | U64(1))
				ctx_c[s] = 0
				ctx_top[s] = 0
				ctx_topc[s] = 0
			ph = mix64(h ^ (U64(y) + U64(1)) * U64(0xD6E8FEB86659FD93))
			j0 = np.int64(ph & U64(pair_mask))
			pchk = np.uint32((ph >> U64(32)) | U64(1))
			if pair_chk[j0] == pchk:
				j = j0
			elif pair_chk[j0 + 1] == pchk:
				j = j0 + 1
			else:
				j = j0 if pair_c[j0] <= pair_c[j0 + 1] else j0 + 1
				pair_chk[j] = pchk
				pair_c[j] = 0
			if pair_c[j] < 4000000000:
				pair_c[j] += np.uint32(1)
			if ctx_c[s] < 4000000000:
				ctx_c[s] += np.uint32(1)
			if ctx_top[s] == y or pair_c[j] > ctx_topc[s]:
				ctx_top[s] = y
				ctx_topc[s] = pair_c[j]
	return out_tok, out_bkt

@nb.njit
def compute_freq2_chunk(toks, t0, t1, ctx_chk, ctx_c, ctx_top, ctx_topc, pair_chk, pair_c,
                        spec_o, spec_r, out_tok, out_bkt):
	"""Targets t0..t1-1 into out_*[t - t0]. The tables carry state between chunks, so a run over
	consecutive chunks equals one run over the whole stream."""
	H = len(FREQ2_ORDERS)
	ctx_mask = ctx_chk.shape[0] - 2
	pair_mask = pair_chk.shape[0] - 2
	ctx_h = np.zeros(H, np.uint64)
	ctx_slot = np.full(H, -1, np.int64)
	n_cols = spec_o.shape[0]
	for t in range(max(t0, 1), t1):
		# ---- context lookups (from tokens[:t]) ----
		for o in range(H):
			k = FREQ2_ORDERS[o]
			ctx_slot[o] = -1
			if t < k:
				continue
			h = mix64(U64(200 + k) * MUL)
			for j in range(t - k, t):
				h = mix64(h ^ U64(toks[j]) * MUL)
			ctx_h[o] = h
			i0 = np.int64(h & U64(ctx_mask))
			chk = np.uint32((h >> U64(32)) | U64(1))
			if ctx_chk[i0] == chk:
				ctx_slot[o] = i0
			elif ctx_chk[i0 + 1] == chk:
				ctx_slot[o] = i0 + 1
		# ---- hints ----
		for col in range(n_cols):
			o = spec_o[col]
			r = spec_r[col]
			s = ctx_slot[o]
			if s < 0 or ctx_c[s] == 0 or ctx_topc[s, r - 1] == 0:
				continue
			c = ctx_c[s]
			cb = 0
			while c > 1 and cb < N_CNT_B - 1:
				c >>= 1
				cb += 1
			share = min(float(ctx_topc[s, r - 1]) / float(ctx_c[s]), 1.0)   # re-inserted contexts can overshoot
			sb = min(int(share * N_SHARE_B * r), N_SHARE_B - 1)
			out_tok[t - t0, col] = ctx_top[s, r - 1]
			out_bkt[t - t0, col] = cb * N_SHARE_B + sb     # never 0: rank r needs >= r observations
		# ---- update with tokens[t] ----
		y = toks[t]
		for o in range(H):
			if t < FREQ2_ORDERS[o]:
				continue
			h = ctx_h[o]
			s = ctx_slot[o]
			if s < 0:
				i0 = np.int64(h & U64(ctx_mask))
				s = i0 if ctx_c[i0] <= ctx_c[i0 + 1] else i0 + 1
				ctx_chk[s] = np.uint32((h >> U64(32)) | U64(1))
				ctx_c[s] = 0
				for q in range(TOPK):
					ctx_top[s, q] = 0
					ctx_topc[s, q] = 0
			ph = mix64(h ^ (U64(y) + U64(1)) * U64(0xD6E8FEB86659FD93))
			j0 = np.int64(ph & U64(pair_mask))
			pchk = np.uint32((ph >> U64(32)) | U64(1))
			if pair_chk[j0] == pchk:
				j = j0
			elif pair_chk[j0 + 1] == pchk:
				j = j0 + 1
			else:
				j = j0 if pair_c[j0] <= pair_c[j0 + 1] else j0 + 1
				pair_chk[j] = pchk
				pair_c[j] = 0
			if pair_c[j] < 4000000000:
				pair_c[j] += np.uint32(1)
			if ctx_c[s] < 4000000000:
				ctx_c[s] += np.uint32(1)
			cy = pair_c[j]
			# top-3 maintenance: an empty rank has count 0
			r = -1
			for q in range(TOPK):
				if ctx_topc[s, q] > 0 and ctx_top[s, q] == y:
					r = q
					break
			if r >= 0:
				ctx_topc[s, r] = cy
			elif cy > ctx_topc[s, TOPK - 1]:
				r = TOPK - 1
				ctx_top[s, r] = y
				ctx_topc[s, r] = cy
			while r > 0 and ctx_topc[s, r] > ctx_topc[s, r - 1]:
				tt = ctx_top[s, r]; ctx_top[s, r] = ctx_top[s, r - 1]; ctx_top[s, r - 1] = tt
				tc = ctx_topc[s, r]; ctx_topc[s, r] = ctx_topc[s, r - 1]; ctx_topc[s, r - 1] = tc
				r -= 1

def compute_freq2(toks, ctx_bits, pair_bits, tok_out, bkt_out, chunk=1 << 24):
	"""Runs compute_freq2_chunk over the stream, writing into (possibly memory-mapped) outputs
	chunk by chunk so the full output never has to sit in RAM."""
	ctx_chk = np.zeros(1 << ctx_bits, np.uint32)
	ctx_c = np.zeros(1 << ctx_bits, np.uint32)
	ctx_top = np.zeros((1 << ctx_bits, TOPK), np.int16)
	ctx_topc = np.zeros((1 << ctx_bits, TOPK), np.uint32)
	pair_chk = np.zeros(1 << pair_bits, np.uint32)
	pair_c = np.zeros(1 << pair_bits, np.uint32)
	spec_o = np.array([FREQ2_ORDERS.index(o) for o, _ in FREQ2_SPEC], np.int64)
	spec_r = np.array([r for _, r in FREQ2_SPEC], np.int64)
	n = len(toks)
	for a in range(0, n, chunk):
		b = min(n, a + chunk)
		ot = np.zeros((b - a, len(FREQ2_SPEC)), np.int16)
		ob = np.zeros((b - a, len(FREQ2_SPEC)), np.uint8)
		compute_freq2_chunk(toks, a, b, ctx_chk, ctx_c, ctx_top, ctx_topc, pair_chk, pair_c, spec_o, spec_r, ot, ob)
		tok_out[a:b] = ot
		bkt_out[a:b] = ob

def hint_paths(token_path):
	return f"{token_path}.hint_tok.npy", f"{token_path}.hint_bkt.npy"

def freq_paths(token_path):
	return f"{token_path}.freq_tok.npy", f"{token_path}.freq_bkt.npy"

def freq2_paths(token_path):
	return f"{token_path}.freq2_tok.npy", f"{token_path}.freq2_bkt.npy"

def n_hints(freq):
	"""freq: 0/False = match hints only, 1/True = + v1 frequency hints, 2 = + v2."""
	return len(MATCH_MIN) + {0: 0, 1: len(FREQ_ORDERS), 2: len(FREQ2_SPEC)}[int(freq)]

def _layout(token_path, n_hint):
	for level, extra in ((0, []), (1, [freq_paths(token_path)]), (2, [freq2_paths(token_path)])):
		if n_hint == n_hints(level):
			return [hint_paths(token_path)] + extra
	raise ValueError(f"no hint layout with {n_hint} hint models")

def load_hints_gpu(token_path, n_hint, device, n=None, chunk=1 << 23):
	"""[n, n_hint] int16 token and uint8 bucket tensors on `device`: the match hints, then the
	frequency hints of whichever version n_hint implies. Copied in row chunks straight from the
	memory-mapped files, so the host never holds the whole array."""
	parts = [(np.load(a, mmap_mode="r"), np.load(b, mmap_mode="r")) for a, b in _layout(token_path, n_hint)]
	n = len(parts[0][0]) if n is None else n
	tok = torch.empty((n, n_hint), dtype=torch.int16, device=device)
	bkt = torch.empty((n, n_hint), dtype=torch.uint8, device=device)
	col = 0
	for t_mm, b_mm in parts:
		w = t_mm.shape[1]
		for a in range(0, n, chunk):
			b = min(n, a + chunk)
			tok[a:b, col:col + w] = torch.from_numpy(np.ascontiguousarray(t_mm[a:b])).to(device)
			bkt[a:b, col:col + w] = torch.from_numpy(np.ascontiguousarray(b_mm[a:b])).to(device)
		col += w
	assert col == n_hint
	return tok, bkt

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--tokens", default="5k_enwik9")
	ap.add_argument("--table_bits", type=int, default=24)
	ap.add_argument("--ctx_bits", type=int, default=26)
	ap.add_argument("--pair_bits", type=int, default=28)
	ap.add_argument("--freq_only", action="store_true", help="skip the (unchanged) match hints")
	ap.add_argument("--freq2", action="store_true", help="only build the v2 frequency hints")
	# v2 shares one context table across six orders; at 26 bits orders 5-6 crowd out orders 3-4
	# (order-3 coverage 72.3% -> 67.3%). The decoder must rebuild with the same sizes.
	ap.add_argument("--freq2_ctx_bits", type=int, default=27)
	args = ap.parse_args()
	toks = np.asarray(load_tokens(args.tokens), dtype=np.int32)
	assert toks.max() < 32768, "hint_tok is stored as int16"
	if args.freq2:
		t0 = time.time()
		tok_path, bkt_path = freq2_paths(args.tokens)
		shape = (len(toks), len(FREQ2_SPEC))
		tok_out = np.lib.format.open_memmap(tok_path, mode="w+", dtype=np.int16, shape=shape)
		bkt_out = np.lib.format.open_memmap(bkt_path, mode="w+", dtype=np.uint8, shape=shape)
		compute_freq2(toks, args.freq2_ctx_bits, args.pair_bits, tok_out, bkt_out)
		tok_out.flush(); bkt_out.flush()
		print(f"frequency hints v2: {time.time() - t0:.0f}s | wrote {tok_path}, {bkt_path}")
		for col, (k, r) in enumerate(FREQ2_SPEC):
			has = bkt_out[1:, col] > 0
			right = (tok_out[1:, col] == toks[1:]) & has
			print(f"  order {k} rank {r}: hint on {has.mean():6.1%} of targets, correct on {right.sum() / max(has.sum(), 1):6.1%} of those")
		raise SystemExit
	if not args.freq_only:
		t0 = time.time()
		hint_tok, hint_bkt = compute(toks, args.table_bits)
		tok_path, bkt_path = hint_paths(args.tokens)
		np.save(tok_path, hint_tok)
		np.save(bkt_path, hint_bkt)
		print(f"match hints: {time.time() - t0:.0f}s | wrote {tok_path}, {bkt_path}")
		for m, k in enumerate(MATCH_MIN):
			has = hint_bkt[1:, m] > 0
			right = (hint_tok[1:, m] == toks[1:]) & has
			print(f"  min length {k:2d}: hint on {has.mean():6.1%} of targets, correct on {right.sum() / max(has.sum(), 1):6.1%} of those")
		print(f"  any hint: {(hint_bkt[1:] > 0).any(1).mean():6.1%} of targets")
		del hint_tok, hint_bkt
	t0 = time.time()
	freq_tok, freq_bkt = compute_freq(toks, args.ctx_bits, args.pair_bits)
	tok_path, bkt_path = freq_paths(args.tokens)
	np.save(tok_path, freq_tok)
	np.save(bkt_path, freq_bkt)
	print(f"frequency hints: {time.time() - t0:.0f}s | wrote {tok_path}, {bkt_path}")
	for o, k in enumerate(FREQ_ORDERS):
		has = freq_bkt[1:, o] > 0
		right = (freq_tok[1:, o] == toks[1:]) & has
		print(f"  order {k}: hint on {has.mean():6.1%} of targets, top follower correct on {right.sum() / max(has.sum(), 1):6.1%} of those")

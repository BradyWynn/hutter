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

usage: python hints.py [--tokens 5k_enwik9]   ->  5k_enwik9.hint_tok.npy, 5k_enwik9.hint_bkt.npy
"""
import argparse, time
import numpy as np
import numba as nb
from coding_data import load_tokens

MATCH_MIN = (3, 6, 12)
N_LEN_B = 16
DIST_EDGES = (512, 1024, 8192)
N_DIST = len(DIST_EDGES) + 1
N_BUCKETS = N_LEN_B * N_DIST
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

def hint_paths(token_path):
	return f"{token_path}.hint_tok.npy", f"{token_path}.hint_bkt.npy"

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--tokens", default="5k_enwik9")
	ap.add_argument("--table_bits", type=int, default=24)
	args = ap.parse_args()
	toks = np.asarray(load_tokens(args.tokens), dtype=np.int32)
	assert toks.max() < 32768, "hint_tok is stored as int16"
	t0 = time.time()
	hint_tok, hint_bkt = compute(toks, args.table_bits)
	tok_path, bkt_path = hint_paths(args.tokens)
	np.save(tok_path, hint_tok)
	np.save(bkt_path, hint_bkt)
	n = len(toks) - 1
	print(f"{len(toks):,} tokens | {time.time() - t0:.0f}s | wrote {tok_path}, {bkt_path}")
	for m, k in enumerate(MATCH_MIN):
		has = hint_bkt[1:, m] > 0
		right = (hint_tok[1:, m] == toks[1:]) & has
		print(f"  min length {k:2d}: hint on {has.mean():6.1%} of targets, correct on {right.sum() / max(has.sum(), 1):6.1%} of those")
	print(f"  any hint: {(hint_bkt[1:] > 0).any(1).mean():6.1%} of targets")

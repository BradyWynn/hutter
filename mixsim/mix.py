"""Prequential simulation: frozen transformer + online match / n-gram models + online mixer.

Everything online is strictly causal: at target position t each model predicts tokens[t]
from tokens[:t] only, the loss is charged, and only then do the models update. So the
decoder could rebuild every one of these predictions, and the bytes reported are an
honest ideal (pre-arithmetic-coder) payload for the mixed model.

Linear mixing p = sum_i w_i p_i(y) needs each model's probability of the ACTUAL token
only -- if every p_i is a normalised distribution, so is the mixture. That is why
dump_probs.py only has to save log p_T(actual) per token, not full distributions.

Models fed to the mixer (all normalised distributions over the vocab):
  T      frozen transformer (from dump_probs.py)
  M5/M12 match models: find the most recent earlier occurrence of the last m tokens
         (m = 5, 12), predict the token that followed it. p = c on the predicted token and
         (1-c)/(V-1) elsewhere, c learned online per match-length bucket. Uniform if no match.
  N2/N4/N6  hashed n-gram counts, interpolated Witten-Bell backoff up to order 2/4/6
         (one recursion, three taps). Hash tables carry 32-bit checksums with replacement, and a
         context's pair counts are keyed by a per-slot epoch, so evictions and collisions
         can only LOSE mass: sum_y p(y) <= 1. That is verified empirically below.

Mixer: one weight vector per context bucket (M5 match length x transformer confidence),
updated by exponentiated gradient on -log p_mix. Several learning rates and model subsets
run side by side.

usage: python mixsim/mix.py [--out mixsim/out] [--limit N]
"""
import argparse, math, os, sys, time
import numpy as np
import numba as nb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coding_data import load_tokens

U64 = np.uint64
MIX_MUL = U64(0x9E3779B97F4A7C15)

@nb.njit(inline='always')
def mix64(x):
	x = (x ^ (x >> U64(30))) * U64(0xBF58476D1CE4E5B9)
	x = (x ^ (x >> U64(27))) * U64(0x94D049BB133111EB)
	return x ^ (x >> U64(31))

@nb.njit(inline='always')
def window_hash(toks, end, k, salt):
	"""hash of toks[end-k : end] (k tokens ending just before `end`)"""
	h = mix64(U64(salt) * MIX_MUL + U64(k))
	for j in range(end - k, end):
		h = mix64(h ^ U64(toks[j]) * MIX_MUL)
	return h

@nb.njit(inline='always')
def pair_hash(ctx_hash, slot, epoch, y):
	"""Key of a (context, next token) count. Keyed by the context's slot AND that slot's
	epoch, not just the context hash: when a context is evicted and later re-inserted (in
	either way of its set) it must not find its old pair counts, or sum_y c(y) could exceed
	the fresh c(ctx) and the backoff distribution would sum past 1."""
	h = mix64(ctx_hash ^ (U64(slot) * U64(0xD6E8FEB86659FD93)))
	h = mix64(h ^ (U64(epoch) << U64(20)) ^ U64(y))
	return h

ORDERS = (1, 2, 3, 4, 6)          # n-gram orders in the backoff chain
TAPS = (1, 3, 4)                  # indices into the chain exposed to the mixer: N2, N4, N6
MATCH_MIN = (5, 12)
N_LEN_B = 16                      # match-length buckets for match confidence
N_CONF_B = 6                      # transformer-confidence buckets for mixer weight sets

@nb.njit(inline='always')
def len_bucket(L):
	if L <= 0:
		return 0
	b = 1
	while L > 1 and b < N_LEN_B - 1:   # log2-ish
		L >>= 1
		b += 1
	return b

@nb.njit
def run(toks, logp, pmax, V, ctx_bits, pair_bits, etas, subsets, n_models, norm_every):
	n = toks.shape[0]
	n_orders = len(ORDERS)
	# ---- n-gram tables (2-way set associative) ----
	ctx_mask = (1 << ctx_bits) - 2
	ctx_chk = np.zeros(1 << ctx_bits, np.uint32)
	ctx_c = np.zeros(1 << ctx_bits, np.uint32)
	ctx_d = np.zeros(1 << ctx_bits, np.uint32)
	ctx_ep = np.zeros(1 << ctx_bits, np.uint32)
	pair_mask = (1 << pair_bits) - 2
	pair_chk = np.zeros(1 << pair_bits, np.uint32)
	pair_c = np.zeros(1 << pair_bits, np.uint32)
	uni = np.zeros(V, np.float64)
	uni_tot = 0.0
	# ---- match models ----
	n_match = len(MATCH_MIN)
	mt_bits = 24
	mt = np.zeros((n_match, 1 << mt_bits), np.int64)
	m_ptr = np.zeros(n_match, np.int64)
	m_len = np.zeros(n_match, np.int64)
	m_hit = np.full((n_match, N_LEN_B), 0.5)
	m_tot = np.full((n_match, N_LEN_B), 1.0)
	# ---- mixers ----
	n_mix = subsets.shape[0]
	n_sets = N_LEN_B * N_CONF_B
	W = np.zeros((n_mix, n_sets, n_models))
	for k in range(n_mix):
		cnt = 0
		for i in range(n_models):
			cnt += subsets[k, i]
		for s in range(n_sets):
			z = 0.0
			for i in range(n_models):
				if subsets[k, i]:
					W[k, s, i] = 0.9 if i == 0 else 0.1 / max(cnt - 1, 1)
					z += W[k, s, i]
			for i in range(n_models):
				W[k, s, i] /= z
	nats = np.zeros(n_mix)
	# per M5-length-bucket diagnostics: tokens, T nats, best-mixer nats
	diag_cnt = np.zeros(N_LEN_B)
	diag_T = np.zeros(N_LEN_B)
	diag_mix = np.zeros((n_mix, N_LEN_B))
	m_correct = np.zeros((n_match, N_LEN_B))
	norm_sum = np.zeros(3)   # count, sum, max of sum_y p_N6(y)
	norm_sum[2] = 0.0

	p = np.zeros(n_models)
	chain = np.zeros(n_orders + 1)
	ctx_h = np.zeros(n_orders, np.uint64)
	ctx_slot = np.full(n_orders, -1, np.int64)
	inv_v = 1.0 / V

	for t in range(1, n):
		y = toks[t]
		# ================= predict =================
		p[0] = math.exp(logp[t])
		# match models
		for m in range(n_match):
			if m_len[m] > 0:
				b = len_bucket(m_len[m])
				c = m_hit[m, b] / m_tot[m, b]
				pred = toks[m_ptr[m]]
				p[1 + m] = c if y == pred else (1.0 - c) / (V - 1)
			else:
				p[1 + m] = inv_v
		# n-gram chain: order 0 (unigram, add-0.5), then Witten-Bell per order
		chain[0] = (uni[y] + 0.5) / (uni_tot + 0.5 * V)
		for o in range(n_orders):
			k = ORDERS[o]
			ctx_slot[o] = -1
			if t < k:
				chain[o + 1] = chain[o]
				continue
			h = window_hash(toks, t, k, 1)
			ctx_h[o] = h
			i0 = np.int64(h & U64(ctx_mask))
			chk = np.uint32((h >> U64(32)) | U64(1))
			slot = -1
			if ctx_chk[i0] == chk:
				slot = i0
			elif ctx_chk[i0 + 1] == chk:
				slot = i0 + 1
			ctx_slot[o] = slot
			if slot < 0 or ctx_c[slot] == 0:
				chain[o + 1] = chain[o]
				continue
			ph = pair_hash(h, slot, ctx_ep[slot], y)
			j0 = np.int64(ph & U64(pair_mask))
			pchk = np.uint32((ph >> U64(32)) | U64(1))
			cy = 0.0
			if pair_chk[j0] == pchk:
				cy = pair_c[j0]
			elif pair_chk[j0 + 1] == pchk:
				cy = pair_c[j0 + 1]
			c = float(ctx_c[slot]); d = float(ctx_d[slot])
			cy = min(cy, c)   # a false checksum match can't claim more than the context saw
			chain[o + 1] = (cy + d * chain[o]) / (c + d)
		for q in range(len(TAPS)):
			p[1 + n_match + q] = chain[TAPS[q] + 1]

		# empirical normalisation check of the top n-gram tap: sum over the whole vocab
		if norm_every > 0 and t % norm_every == 0:
			tot = 0.0
			for yy in range(V):
				pl = (uni[yy] + 0.5) / (uni_tot + 0.5 * V)
				for o in range(n_orders):
					slot = ctx_slot[o]
					if slot < 0 or ctx_c[slot] == 0:
						continue
					ph = pair_hash(ctx_h[o], slot, ctx_ep[slot], yy)
					j0 = np.int64(ph & U64(pair_mask))
					pchk = np.uint32((ph >> U64(32)) | U64(1))
					cy = 0.0
					if pair_chk[j0] == pchk:
						cy = pair_c[j0]
					elif pair_chk[j0 + 1] == pchk:
						cy = pair_c[j0 + 1]
					cy = min(cy, float(ctx_c[slot]))
					pl = (cy + float(ctx_d[slot]) * pl) / (float(ctx_c[slot]) + float(ctx_d[slot]))
				tot += pl
			norm_sum[0] += 1; norm_sum[1] += tot
			if tot > norm_sum[2]:
				norm_sum[2] = tot

		# mixers
		lb = len_bucket(m_len[0])
		pm = pmax[t]
		cb = min(int(pm * N_CONF_B), N_CONF_B - 1)
		s = lb * N_CONF_B + cb
		diag_cnt[lb] += 1
		diag_T[lb] -= logp[t]
		for k in range(n_mix):
			pmix = 0.0
			for i in range(n_models):
				pmix += W[k, s, i] * p[i]
			nats[k] -= math.log(pmix)
			diag_mix[k, lb] -= math.log(pmix)
			# exponentiated gradient on -log pmix
			z = 0.0
			eta = etas[k]
			for i in range(n_models):
				if subsets[k, i]:
					# ratio clipped: a model at the weight floor that suddenly predicts well
					# would otherwise give p_i/pmix ~ 1e6 and overflow the exp
					W[k, s, i] *= math.exp(eta * (min(p[i] / pmix, 50.0) - 1.0))
					z += W[k, s, i]
			# floor keeps every model revivable; renormalise after it so the mixture stays a
			# proper distribution
			z2 = 0.0
			for i in range(n_models):
				W[k, s, i] = max(W[k, s, i] / z, 1e-6) if subsets[k, i] else 0.0
				z2 += W[k, s, i]
			for i in range(n_models):
				W[k, s, i] /= z2

		# ================= update =================
		for m in range(n_match):
			if m_len[m] > 0:
				b = len_bucket(m_len[m])
				hit = toks[m_ptr[m]] == y
				m_tot[m, b] += 1
				if hit:
					m_hit[m, b] += 1
					m_correct[m, b] += 1
					m_len[m] += 1
					m_ptr[m] += 1
				else:
					m_len[m] = 0
			mm = MATCH_MIN[m]
			if t + 1 >= mm:
				h = window_hash(toks, t + 1, mm, 7 + m)
				idx = np.int64(h >> U64(64 - mt_bits))
				if m_len[m] == 0:
					cand = mt[m, idx]
					if cand > 0:
						# verify (rejects hash collisions) and measure the true match length
						L = 0
						while L < 65535 and cand - 1 - L >= 0 and toks[cand - 1 - L] == toks[t - L]:
							L += 1
						if L >= mm:
							m_len[m] = L
							m_ptr[m] = cand
				mt[m, idx] = t + 1
		uni[y] += 1.0
		uni_tot += 1.0
		for o in range(n_orders):
			k = ORDERS[o]
			if t < k:
				continue
			h = ctx_h[o]
			slot = ctx_slot[o]
			if slot < 0:
				# claim the less-used way; bump its epoch so stale pairs can't be found
				i0 = np.int64(h & U64(ctx_mask))
				slot = i0 if ctx_c[i0] <= ctx_c[i0 + 1] else i0 + 1
				ctx_chk[slot] = np.uint32((h >> U64(32)) | U64(1))
				ctx_c[slot] = 0
				ctx_d[slot] = 0
				ctx_ep[slot] = ctx_ep[slot] + np.uint32(1)
			ph = pair_hash(h, slot, ctx_ep[slot], y)
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
			if pair_c[j] == 0:
				ctx_d[slot] += 1
			if pair_c[j] < 4000000000:
				pair_c[j] += 1
				ctx_c[slot] += 1
	return nats, diag_cnt, diag_T, diag_mix, m_correct, m_tot, norm_sum

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", default="mixsim/out")
	ap.add_argument("--tokens", default="5k_enwik9")
	ap.add_argument("--limit", type=int, default=0)
	ap.add_argument("--ctx_bits", type=int, default=26)
	ap.add_argument("--pair_bits", type=int, default=27)
	ap.add_argument("--norm_every", type=int, default=20011)
	args = ap.parse_args()

	logp = np.load(os.path.join(args.out, "logp.npy"), mmap_mode="r")
	pmax = np.load(os.path.join(args.out, "pmax.npy"), mmap_mode="r")
	n = len(logp) if not args.limit else min(args.limit, len(logp))
	toks = np.asarray(load_tokens(args.tokens)[:n], dtype=np.int32)
	logp = np.ascontiguousarray(logp[:n], dtype=np.float32)
	pmax = np.ascontiguousarray(pmax[:n], dtype=np.float32)
	V = 10256
	names = ["T", "M5", "M12", "N2", "N4", "N6"]
	n_models = len(names)
	configs = [   # (label, models, eta)
		("T only",          ["T"], 0.0),
		("T+M",             ["T", "M5", "M12"], 0.02),
		("T+N",             ["T", "N2", "N4", "N6"], 0.02),
		("T+M+N eta=.005",  names, 0.005),
		("T+M+N eta=.02",   names, 0.02),
		("T+M+N eta=.08",   names, 0.08),
		("M+N (no T)",      ["M5", "M12", "N2", "N4", "N6"], 0.02),
	]
	subsets = np.array([[1 if nm in c[1] else 0 for nm in names] for c in configs], dtype=np.int64)
	etas = np.array([c[2] for c in configs])
	t0 = time.time()
	nats, dcnt, dT, dmix, mcor, mtot, norm = run(toks, logp, pmax, V, args.ctx_bits, args.pair_bits,
	                                             etas, subsets, n_models, args.norm_every)
	print(f"tokens {n:,} | {time.time() - t0:.0f}s")
	if norm[0]:
		print(f"n-gram normalisation check: {int(norm[0])} positions, mean sum_y p = {norm[1] / norm[0]:.4f}, max = {norm[2]:.7f}")
	base = nats[0]
	print(f"\n{'config':18s} {'nats/tok':>9s} {'MB':>9s} {'vs T':>9s}")
	for (label, _, _), x in zip(configs, nats):
		print(f"{label:18s} {x / (n - 1):9.4f} {x / math.log(2) / 8 / 1e6:9.2f} {(x - base) / math.log(2) / 8 / 1e6:+9.2f}")
	best = int(np.argmin(nats))
	print(f"\nby M5 match length (best mixer = {configs[best][0]}):")
	print(f"{'len bucket':>10s} {'share':>7s} {'T loss':>8s} {'mix loss':>9s} {'M5 hit':>7s} {'saved MB':>9s}")
	for b in range(len(dcnt)):
		if dcnt[b] == 0:
			continue
		lo = 0 if b == 0 else 2 ** (b - 1)
		hit = (mcor[0, b] / max(mtot[0, b] - 1, 1)) if b else float('nan')
		print(f"{('none' if b == 0 else f'>={lo}'):>10s} {dcnt[b] / (n - 1):7.2%} {dT[b] / dcnt[b]:8.3f} {dmix[best, b] / dcnt[b]:9.3f} {hit:7.2%} {(dT[b] - dmix[best, b]) / math.log(2) / 8 / 1e6:9.2f}")

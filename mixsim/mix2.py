"""mix.py, round two: exact n-gram bookkeeping, bigger tables, a two-layer mixer, a match-distance
feature and a transformer-based chain.

Same prequential contract as mix.py (predict tokens[t] from tokens[:t], charge the loss, then
update), and the same six base models: T, M5, M12, N2, N4, N6.

1. N-gram mass no longer leaks. In mix.py an evicted pair took its count with it but its
   context kept counting it in c and d, so sum_y p(y) fell to ~0.83: 17% of the n-gram
   distribution went to no token at all. Here every pair records the (slot, epoch) of the
   context that owns it, and evicting it subtracts its count from that context (if still
   alive). Live pairs then always satisfy sum_y c(y) == c, so each context's Witten-Bell
   distribution is exactly normalised, up to 32-bit checksum false matches. Replacement
   prefers orphaned pairs (owner context evicted) over low counts.

2. Match distance. A match's distance is how far back the copied occurrence sits. The dump
   gives the transformer 512..1024 tokens of context, so a match at <=512 is always inside
   its view, 512..1024 sometimes, and beyond that never: there the match model is the only
   thing that can see the repeat. Buckets <=512 / <=1024 / <=8192 / >8192 key the match
   confidence (with length) and a new layer-1 mixer F.

3. Transformer-based chain (modded-nanogpt PR #380). Start from P = p_T(y) and fold in one
   link per live evidence source, n-gram orders shortest first, then the match models:
       P <- (1 - lam) P + lam q
   q is the n-gram's maximum-likelihood c(y)/N (a distribution, since sum_y c(y) == N exactly)
   or [y == match prediction]. lam = sigmoid(theta) per (order, log2 N) and per (match,
   length bucket, distance bucket), learned online by SGD on -log P. Each link is a convex
   combination of distributions, so P stays a distribution. Several learning rates run side
   by side; one of them is also fed to the mixers as a seventh model, "chain".

4. Two-layer mixer, PAQ style. Layer-1 mixers each choose a weight vector by a different
   context and are trained by exponentiated gradient on their own log loss:
     A  M5 match length x M12 match length
     B  transformer confidence (pmax) x transformer entropy
     C  highest n-gram order with a live context x that context's count
     D  previous token
     E  M5 match length x pmax
     F  M5 match length x M5 distance x coarse pmax              (new)
   They run as two banks: over the six base models only, and over those plus the chain. Each
   bank has a final mixer over its layer-1 outputs, selected by M5 length x M5 distance.

usage: python mixsim/mix2.py [--out mixsim/out] [--limit N] [--ctx_bits 26] [--pair_bits 27]
"""
import argparse, math, os, sys, time
import numpy as np
import numba as nb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coding_data import load_tokens
from mix import U64, mix64, window_hash, pair_hash, len_bucket, ORDERS, TAPS, MATCH_MIN, N_LEN_B

N_PMAX_B = 12
N_ENT_B = 8
N_CNT_B = 8
N_DIST = 4
DIST_EDGES = (512, 1024, 8192)       # match distance buckets: <=512, <=1024, <=8192, beyond
N_CHAIN_B = 25                       # log2(N) bins for the chain's n-gram links
L1_NAMES = ("A match x match", "B conf x entropy", "C ngram order x count", "D prev token",
            "E match x conf", "F match x dist x conf")

@nb.njit(inline='always')
def dist_bucket(d):
	b = 0
	while b < len(DIST_EDGES) and d > DIST_EDGES[b]:
		b += 1
	return b

@nb.njit(inline='always')
def sigmoid(x):
	return 1.0 / (1.0 + math.exp(-x))

@nb.njit(inline='always')
def eg_update(W, s, p, pmix, eta, n_in):
	"""Exponentiated-gradient step on -log(sum_i W[s,i] p_i), with a floor and renormalisation
	so the weights stay a proper convex combination."""
	z = 0.0
	for i in range(n_in):
		W[s, i] *= math.exp(eta * (min(p[i] / pmix, 50.0) - 1.0))
		z += W[s, i]
	z2 = 0.0
	for i in range(n_in):
		W[s, i] = max(W[s, i] / z, 1e-6)
		z2 += W[s, i]
	for i in range(n_in):
		W[s, i] /= z2

@nb.njit
def run(toks, logp, pmax_b, ent_b, V, ctx_bits, pair_bits, eta1, eta2, chain_lrs, chain_feed, norm_every):
	n = toks.shape[0]
	n_orders = len(ORDERS)
	n_base = 6                       # T, M5, M12, N2, N4, N6
	n_models = n_base + 1            # + chain
	# ---- n-gram tables (2-way set associative) ----
	ctx_mask = (1 << ctx_bits) - 2
	ctx_chk = np.zeros(1 << ctx_bits, np.uint32)
	ctx_c = np.zeros(1 << ctx_bits, np.uint32)
	ctx_d = np.zeros(1 << ctx_bits, np.uint32)
	ctx_ep = np.zeros(1 << ctx_bits, np.uint32)
	pair_mask = (1 << pair_bits) - 2
	pair_chk = np.zeros(1 << pair_bits, np.uint32)
	pair_c = np.zeros(1 << pair_bits, np.uint32)
	pair_own = np.zeros(1 << pair_bits, np.uint32)     # owning context slot
	pair_oep = np.zeros(1 << pair_bits, np.uint32)     # owner's epoch when the pair was made
	uni = np.zeros(V, np.float64)
	uni_tot = 0.0
	# ---- match models ----
	n_match = len(MATCH_MIN)
	mt_bits = 24
	mt = np.zeros((n_match, 1 << mt_bits), np.int64)
	m_ptr = np.zeros(n_match, np.int64)
	m_len = np.zeros(n_match, np.int64)
	m_hit = np.full((n_match, N_LEN_B, N_DIST), 0.5)
	m_tot = np.full((n_match, N_LEN_B, N_DIST), 1.0)
	# ---- chain: one theta table per learning rate ----
	n_lr = chain_lrs.shape[0]
	th_ng = np.full((n_lr, n_orders, N_CHAIN_B), math.log(0.02 / 0.98))
	th_m = np.full((n_lr, n_match, N_LEN_B, N_DIST), math.log(0.05 / 0.95))
	max_links = n_orders + n_match
	lq = np.zeros(max_links)                 # link q
	lkind = np.zeros(max_links, np.int64)    # 0 = n-gram, 1 = match
	la = np.zeros(max_links, np.int64)       # order / match index
	lb_ = np.zeros(max_links, np.int64)      # count bin / length bucket
	lc = np.zeros(max_links, np.int64)       # - / distance bucket
	lam = np.zeros(max_links)
	Pc = np.zeros(max_links + 1)
	nats_chain = np.zeros(n_lr)
	# ---- layer-1 mixer banks: [bank][mixer] -> weight table ----
	sizes = np.array([N_LEN_B * 4, N_PMAX_B * N_ENT_B, (n_orders + 1) * N_CNT_B, V, N_LEN_B * N_PMAX_B,
	                  N_LEN_B * N_DIST * 3])
	n_l1 = sizes.shape[0]
	n_bank = 2
	bank_in = np.array([n_base, n_models])
	W1 = [np.zeros((sizes[k % n_l1], n_models)) for k in range(n_bank * n_l1)]
	for bk in range(n_bank):
		for k in range(n_l1):
			Wk = W1[bk * n_l1 + k]
			for s in range(sizes[k]):
				Wk[s, 0] = 0.9
				for i in range(1, bank_in[bk]):
					Wk[s, i] = 0.1 / (bank_in[bk] - 1)
	n_fin_sets = N_LEN_B * N_DIST
	W2 = np.full((n_bank, n_fin_sets, n_l1), 1.0 / n_l1)
	# ---- accounting ----
	nats_l1 = np.zeros((n_bank, n_l1))
	nats_fin = np.zeros(n_bank)
	diag_cnt = np.zeros(N_LEN_B)
	diag_T = np.zeros(N_LEN_B)
	diag_fin = np.zeros((n_bank, N_LEN_B))
	ddist_cnt = np.zeros(N_DIST)             # M5 matched tokens by distance bucket
	ddist_T = np.zeros(N_DIST)
	ddist_fin = np.zeros((n_bank, N_DIST))
	ddist_hit = np.zeros(N_DIST)
	norm_sum = np.zeros(3)

	p = np.zeros(n_models)
	q = np.zeros(n_l1)
	chain = np.zeros(n_orders + 1)
	qml = np.zeros(n_orders)                 # per order: c(y)/N, or -1 if the context is not live
	cbin = np.zeros(n_orders, np.int64)
	ctx_h = np.zeros(n_orders, np.uint64)
	ctx_slot = np.full(n_orders, -1, np.int64)
	sel = np.zeros(n_l1, np.int64)
	mdb = np.zeros(n_match, np.int64)
	inv_v = 1.0 / V

	for t in range(1, n):
		y = np.int64(toks[t])
		# ================= predict =================
		p[0] = math.exp(logp[t])
		for m in range(n_match):
			if m_len[m] > 0:
				b = len_bucket(m_len[m])
				db = dist_bucket(t - m_ptr[m])
				mdb[m] = db
				c = m_hit[m, b, db] / m_tot[m, b, db]
				p[1 + m] = c if y == toks[m_ptr[m]] else (1.0 - c) / (V - 1)
			else:
				mdb[m] = 0
				p[1 + m] = inv_v
		chain[0] = (uni[y] + 0.5) / (uni_tot + 0.5 * V)
		top_order = 0
		top_cnt = 0
		for o in range(n_orders):
			k = ORDERS[o]
			ctx_slot[o] = -1
			qml[o] = -1.0
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
			top_order = o + 1
			top_cnt = ctx_c[slot]
			ph = pair_hash(h, slot, ctx_ep[slot], y)
			j0 = np.int64(ph & U64(pair_mask))
			pchk = np.uint32((ph >> U64(32)) | U64(1))
			cy = 0.0
			if pair_chk[j0] == pchk and pair_c[j0] > 0:
				cy = pair_c[j0]
			elif pair_chk[j0 + 1] == pchk and pair_c[j0 + 1] > 0:
				cy = pair_c[j0 + 1]
			c = float(ctx_c[slot]); d = float(ctx_d[slot])
			cy = min(cy, c)
			chain[o + 1] = (cy + d * chain[o]) / (c + d)
			qml[o] = cy / c
			nb_ = 0
			cc = ctx_c[slot]
			while cc > 1 and nb_ < N_CHAIN_B - 1:
				cc >>= 1
				nb_ += 1
			cbin[o] = nb_
		for qq in range(len(TAPS)):
			p[1 + n_match + qq] = chain[TAPS[qq] + 1]

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
					if pair_chk[j0] == pchk and pair_c[j0] > 0:
						cy = pair_c[j0]
					elif pair_chk[j0 + 1] == pchk and pair_c[j0 + 1] > 0:
						cy = pair_c[j0 + 1]
					cy = min(cy, float(ctx_c[slot]))
					pl = (cy + float(ctx_d[slot]) * pl) / (float(ctx_c[slot]) + float(ctx_d[slot]))
				tot += pl
			norm_sum[0] += 1; norm_sum[1] += tot
			if tot > norm_sum[2]:
				norm_sum[2] = tot

		# ---- chain: links (n-gram orders shortest first, then matches) on top of p_T ----
		K = 0
		for o in range(n_orders):
			if qml[o] >= 0.0:
				lq[K] = qml[o]; lkind[K] = 0; la[K] = o; lb_[K] = cbin[o]
				K += 1
		for m in range(n_match):
			if m_len[m] > 0:
				lq[K] = 1.0 if y == toks[m_ptr[m]] else 0.0
				lkind[K] = 1; la[K] = m; lb_[K] = len_bucket(m_len[m]); lc[K] = mdb[m]
				K += 1
		for r in range(n_lr):
			Pc[0] = p[0]
			for j in range(K):
				if lkind[j] == 0:
					lam[j] = sigmoid(th_ng[r, la[j], lb_[j]])
				else:
					lam[j] = sigmoid(th_m[r, la[j], lb_[j], lc[j]])
				Pc[j + 1] = (1.0 - lam[j]) * Pc[j] + lam[j] * lq[j]
			PK = Pc[K]
			nats_chain[r] -= math.log(PK)
			if r == chain_feed:
				p[n_base] = PK
			# d(-log PK)/d theta_j = -(q_j - P_j) * prod_{i>j}(1 - lam_i) / PK * lam_j (1 - lam_j);
			# its magnitude is at most 1, so plain SGD is stable
			A = 1.0
			lr = chain_lrs[r]
			for jj in range(K):
				j = K - 1 - jj
				g = -(lq[j] - Pc[j]) * A / PK * lam[j] * (1.0 - lam[j])
				A *= 1.0 - lam[j]
				if lkind[j] == 0:
					th_ng[r, la[j], lb_[j]] -= lr * g
				else:
					th_m[r, la[j], lb_[j], lc[j]] -= lr * g

		# ---- layer 1 ----
		lb = len_bucket(m_len[0])
		db0 = mdb[0]
		lb12 = min(len_bucket(m_len[1]), 3)
		cb = 0
		while top_cnt > 1 and cb < N_CNT_B - 1:
			top_cnt >>= 1
			cb += 1
		sel[0] = lb * 4 + lb12
		sel[1] = np.int64(pmax_b[t]) * N_ENT_B + np.int64(ent_b[t])
		sel[2] = top_order * N_CNT_B + cb
		sel[3] = np.int64(toks[t - 1])
		sel[4] = lb * N_PMAX_B + np.int64(pmax_b[t])
		sel[5] = (lb * N_DIST + db0) * 3 + np.int64(pmax_b[t]) // 4
		fs = lb * N_DIST + db0
		diag_cnt[lb] += 1
		diag_T[lb] -= logp[t]
		if lb > 0:
			ddist_cnt[db0] += 1
			ddist_T[db0] -= logp[t]
			if y == toks[m_ptr[0]]:
				ddist_hit[db0] += 1
		for bk in range(n_bank):
			nin = bank_in[bk]
			for k in range(n_l1):
				Wk = W1[bk * n_l1 + k]
				s = sel[k]
				pm = 0.0
				for i in range(nin):
					pm += Wk[s, i] * p[i]
				q[k] = pm
				nats_l1[bk, k] -= math.log(pm)
				eg_update(Wk, s, p, pm, eta1, nin)
			# ---- final layer ----
			pf = 0.0
			for k in range(n_l1):
				pf += W2[bk, fs, k] * q[k]
			nats_fin[bk] -= math.log(pf)
			diag_fin[bk, lb] -= math.log(pf)
			if lb > 0:
				ddist_fin[bk, db0] -= math.log(pf)
			eg_update(W2[bk], fs, q, pf, eta2, n_l1)

		# ================= update =================
		for m in range(n_match):
			if m_len[m] > 0:
				b = len_bucket(m_len[m])
				db = mdb[m]
				m_tot[m, b, db] += 1
				if toks[m_ptr[m]] == y:
					m_hit[m, b, db] += 1
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
				# claim the less-used way; the epoch bump orphans the old owner's pairs
				i0 = np.int64(h & U64(ctx_mask))
				slot = i0 if ctx_c[i0] <= ctx_c[i0 + 1] else i0 + 1
				ctx_chk[slot] = np.uint32((h >> U64(32)) | U64(1))
				ctx_c[slot] = 0
				ctx_d[slot] = 0
				ctx_ep[slot] = ctx_ep[slot] + np.uint32(1)
				ctx_slot[o] = slot
			ph = pair_hash(h, slot, ctx_ep[slot], y)
			j0 = np.int64(ph & U64(pair_mask))
			pchk = np.uint32((ph >> U64(32)) | U64(1))
			if pair_chk[j0] == pchk and pair_c[j0] > 0:
				j = j0
			elif pair_chk[j0 + 1] == pchk and pair_c[j0 + 1] > 0:
				j = j0 + 1
			else:
				# victim: an empty or orphaned way first, else the lower count
				orph0 = pair_c[j0] == 0 or ctx_ep[pair_own[j0]] != pair_oep[j0]
				orph1 = pair_c[j0 + 1] == 0 or ctx_ep[pair_own[j0 + 1]] != pair_oep[j0 + 1]
				if orph0:
					j = j0
				elif orph1:
					j = j0 + 1
				else:
					j = j0 if pair_c[j0] <= pair_c[j0 + 1] else j0 + 1
				# give the evicted pair's mass back: its live owner forgets it entirely
				if pair_c[j] > 0:
					ow = pair_own[j]
					if ctx_ep[ow] == pair_oep[j]:
						ctx_c[ow] -= pair_c[j]
						ctx_d[ow] -= np.uint32(1)
				pair_chk[j] = pchk
				pair_c[j] = 0
				pair_own[j] = np.uint32(slot)
				pair_oep[j] = ctx_ep[slot]
			if pair_c[j] == 0:
				ctx_d[slot] += np.uint32(1)
			if pair_c[j] < 4000000000:
				pair_c[j] += np.uint32(1)
				ctx_c[slot] += np.uint32(1)
	return (nats_chain, nats_l1, nats_fin, diag_cnt, diag_T, diag_fin,
	        ddist_cnt, ddist_T, ddist_fin, ddist_hit, norm_sum, th_m[chain_feed])

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", default="mixsim/out")
	ap.add_argument("--tokens", default="5k_enwik9")
	ap.add_argument("--limit", type=int, default=0)
	ap.add_argument("--ctx_bits", type=int, default=26)
	ap.add_argument("--pair_bits", type=int, default=27)
	ap.add_argument("--eta1", type=float, default=0.02)
	ap.add_argument("--eta2", type=float, default=0.05)
	ap.add_argument("--norm_every", type=int, default=20011)
	args = ap.parse_args()
	assert args.ctx_bits <= 32, "pair_own stores context slots as uint32"

	logp_m = np.load(os.path.join(args.out, "logp.npy"), mmap_mode="r")
	n = len(logp_m) if not args.limit else min(args.limit, len(logp_m))
	toks = np.asarray(load_tokens(args.tokens)[:n], dtype=np.uint16)
	logp = np.ascontiguousarray(logp_m[:n], dtype=np.float32)
	# mixer-context buckets precomputed in chunks, as uint8, to keep RAM down
	pmax_b = np.zeros(n, np.uint8); ent_b = np.zeros(n, np.uint8)
	pm_m = np.load(os.path.join(args.out, "pmax.npy"), mmap_mode="r")
	en_m = np.load(os.path.join(args.out, "ent.npy"), mmap_mode="r")
	for a in range(0, n, 1 << 24):
		b = min(n, a + (1 << 24))
		pm = np.nan_to_num(pm_m[a:b].astype(np.float32)); en = np.nan_to_num(en_m[a:b].astype(np.float32))
		pmax_b[a:b] = np.minimum(pm * N_PMAX_B, N_PMAX_B - 1).astype(np.uint8)
		ent_b[a:b] = np.minimum(en, N_ENT_B - 1).astype(np.uint8)     # 1-nat buckets
	V = 10256
	chain_lrs = np.array([0.01, 0.05, 0.2])
	chain_feed = 1
	t0 = time.time()
	(nats_chain, nats_l1, nats_fin, dcnt, dT, dfin, ddc, ddT, ddf, ddh, norm, th_m) = run(
		toks, logp, pmax_b, ent_b, V, args.ctx_bits, args.pair_bits, args.eta1, args.eta2,
		chain_lrs, chain_feed, args.norm_every)
	print(f"tokens {n:,} | ctx_bits {args.ctx_bits} pair_bits {args.pair_bits} | {time.time() - t0:.0f}s")
	if norm[0]:
		print(f"n-gram normalisation check: {int(norm[0])} positions, mean sum_y p = {norm[1] / norm[0]:.5f}, max = {norm[2]:.7f}")
	T_nats = -logp[1:].astype(np.float64).sum()
	mb = lambda x: x / math.log(2) / 8 / 1e6
	row = lambda name, x: print(f"{name:38s} {x / (n - 1):9.4f} {mb(x):9.2f} {mb(x - T_nats):+8.2f}")
	print(f"\n{'config':38s} {'nats/tok':>9s} {'MB':>9s} {'vs T':>8s}")
	row("T only", T_nats)
	for lr, x in zip(chain_lrs, nats_chain):
		row(f"chain alone, lr={lr}" + (" (fed to bank 2)" if lr == chain_lrs[chain_feed] else ""), x)
	for bk, bname in enumerate(("bank 1: base models", "bank 2: base + chain")):
		print(f"-- {bname}")
		for name, x in zip(L1_NAMES, nats_l1[bk]):
			row("  layer-1 " + name, x)
		row("  2-layer final", nats_fin[bk])
	DN = ("<=512", "<=1024", "<=8192", ">8192")
	print(f"\nM5-matched tokens by match distance:")
	print(f"{'distance':>9s} {'share':>7s} {'M5 hit':>7s} {'T loss':>8s} {'bank1':>8s} {'bank2':>8s} {'saved MB (bank2)':>17s}")
	for b in range(len(ddc)):
		if ddc[b] == 0:
			continue
		print(f"{DN[b]:>9s} {ddc[b] / (n - 1):7.2%} {ddh[b] / ddc[b]:7.2%} {ddT[b] / ddc[b]:8.3f} {ddf[0, b] / ddc[b]:8.3f} {ddf[1, b] / ddc[b]:8.3f} {mb(ddT[b] - ddf[1, b]):17.2f}")
	print(f"\nby M5 match length (bank 2 final):")
	print(f"{'len bucket':>10s} {'share':>7s} {'T loss':>8s} {'mix loss':>9s} {'saved MB':>9s}")
	for b in range(len(dcnt)):
		if dcnt[b] == 0:
			continue
		lo = 0 if b == 0 else 2 ** (b - 1)
		print(f"{('none' if b == 0 else f'>={lo}'):>10s} {dcnt[b] / (n - 1):7.2%} {dT[b] / dcnt[b]:8.3f} {dfin[1, b] / dcnt[b]:9.3f} {mb(dT[b] - dfin[1, b]):9.2f}")
	print(f"\nlearned chain lambda for M5 links (lr={chain_lrs[chain_feed]}), rows = length bucket, cols = distance:")
	for b in range(1, th_m.shape[1]):
		lam = 1 / (1 + np.exp(-th_m[0, b]))
		if not np.allclose(lam, 0.05):
			print(f"  >={2 ** (b - 1):<5d} " + " ".join(f"{v:6.3f}" for v in lam))

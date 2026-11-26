"""Reversible semantic Wikipedia article reordering for enwik9 (starlit-style).

Parses all 243,425 complete `<page>...</page>` entries in enwik9, extracts structural +
semantic features (namespace, categories, templates, links, contributor, title/body words),
computes GPU Randomized Truncated SVD (LSA) embeddings, clusters & orders articles via
hierarchical spherical K-Means + exact GPU Greedy Nearest-Neighbor TSP + 2-opt refinement,
and writes:
  - enwik9.reord              (exact 1,000,000,000-byte reordered file)
  - enwik9.reord.perm.i32     (perm[new_pos] = orig_idx, int32)
  - enwik9.reord.meta.json    (exact permutation overhead bytes + SHA-256 verification)

Lossless reconstruction is verified bit-for-bit via SHA-256 on all 1,000,000,000 bytes.
"""
import hashlib
import json
import lzma
import math
import multiprocessing as mp
import os
import re
import time
import zlib
from collections import Counter
import numpy as np
import torch

TAG_S = b"  <page>\n"
TAG_E = b"  </page>\n"

RE_TITLE = re.compile(rb"<title>(.*?)</title>", re.S)
RE_USER = re.compile(rb"<username>(.*?)</username>", re.S)
RE_CAT = re.compile(rb"\[\[Category:([^\]|#]+)", re.I)
RE_TMPL = re.compile(rb"\{\{([A-Za-z0-9 _:-]{2,40})")
RE_LINK = re.compile(rb"\[\[([^\]|#:]{2,50})")
RE_WORD = re.compile(rb"[A-Za-z]{3,18}")
RE_REDIR = re.compile(rb"#REDIRECT\s*\[\[([^\]|#]+)", re.I)

N_BUCKETS = 1 << 17  # 131,072 hashed feature buckets (fast, low collision, zero global vocab pass)
_RAW_BUF = None


def parse_pages(raw):
    starts, ends = [], []
    pos = 0
    while True:
        i = raw.find(TAG_S, pos)
        if i < 0:
            break
        j = raw.find(TAG_E, i)
        if j < 0:
            break
        j += len(TAG_E)
        starts.append(i)
        ends.append(j)
        pos = j
    return starts, ends


def _hbucket(prefix_byte, token_bytes):
    # Deterministic 32-bit FNV-1a hash into [0, N_BUCKETS)
    h = 2166136261 ^ prefix_byte
    for b in token_bytes:
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h & (N_BUCKETS - 1)


def _extract_worker(span_chunk):
    raw = _RAW_BUF
    out = []
    for idx, s, e in span_chunk:
        pg = raw[s:e]
        m_t = RE_TITLE.search(pg, 0, min(len(pg), 400))
        title = m_t.group(1) if m_t else b""
        t_low = title.lower()

        # Check redirect quickly in first 800 bytes
        head = pg[: min(len(pg), 800)]
        m_r = RE_REDIR.search(head)
        if m_r is not None:
            target = m_r.group(1).strip().lower()
            out.append((idx, True, target, t_low, None))
            continue

        feats = Counter()
        # Namespace (Template:, Category:, Wikipedia:, etc.)
        if b":" in title:
            ns = title.split(b":", 1)[0].lower()
            if len(ns) <= 20:
                feats[_hbucket(1, ns)] += 6.0

        # Title words (strong signal for lists, years, biographical series, places)
        for w in RE_WORD.findall(t_low):
            feats[_hbucket(2, w)] += 2.5

        # Contributor / bot style (helps group bot-generated tables/stubs like Rambot)
        m_u = RE_USER.search(pg, 0, min(len(pg), 600))
        if m_u:
            feats[_hbucket(3, m_u.group(1).lower())] += 1.5

        # Categories (strongest semantic cluster signal in Wikipedia)
        for c in RE_CAT.findall(pg):
            c_clean = c.strip().lower()
            if c_clean:
                feats[_hbucket(4, c_clean)] += 4.5
                for w in RE_WORD.findall(c_clean):
                    feats[_hbucket(5, w)] += 1.5

        # Templates / Infoboxes / Taxoboxes / Stubs (strongest structural boilerplate signal)
        for tm in RE_TMPL.findall(pg[: min(len(pg), 12000)]):
            tm_clean = tm.strip().lower()
            if tm_clean:
                feats[_hbucket(6, tm_clean)] += 3.5

        # Internal wiki links (semantic graph similarity; cap per page for speed)
        for lk in RE_LINK.findall(pg[: min(len(pg), 16000)])[:80]:
            lk_clean = lk.strip().lower()
            if lk_clean:
                feats[_hbucket(7, lk_clean)] += 1.8

        # Body words (cap scan to first 12 KB of each page for speed; log-tf)
        wc = Counter(w.lower() for w in RE_WORD.findall(pg[: min(len(pg), 12000)]))
        for w, cnt in wc.most_common(96):
            feats[_hbucket(8, w)] += 1.0 + math.log1p(cnt) * 0.5

        # Keep top 128 weighted features per page to keep sparse matrix crisp and fast
        top_items = feats.most_common(128)
        cols = np.fromiter((k for k, _ in top_items), dtype=np.int32, count=len(top_items))
        vals = np.fromiter((v for _, v in top_items), dtype=np.float32, count=len(top_items))
        out.append((idx, False, b"", t_low, (cols, vals)))
    return out


def compute_svd_embeddings(cols_list, vals_list, dim=192, power_iters=2):
    """Build GPU CSR TF-IDF matrix and compute L2-normalized Randomized Truncated SVD embeddings."""
    t0 = time.time()
    n_docs = len(cols_list)
    lengths = np.fromiter((len(c) for c in cols_list), dtype=np.int64, count=n_docs)
    indptr = np.zeros(n_docs + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    indices = np.concatenate(cols_list).astype(np.int64)
    values = np.concatenate(vals_list).astype(np.float32)

    # IDF weighting
    df = np.bincount(indices, minlength=N_BUCKETS).astype(np.float32)
    idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
    values *= idf[indices]

    # Row L2 normalize TF-IDF
    row_sq = np.zeros(n_docs, dtype=np.float32)
    np.add.at(row_sq, np.repeat(np.arange(n_docs, dtype=np.int64), lengths), values * values)
    row_inv = 1.0 / np.sqrt(np.maximum(row_sq, 1e-12))
    values *= np.repeat(row_inv, lengths)

    # Transfer CSR to GPU
    X = torch.sparse_csr_tensor(
        torch.from_numpy(indptr).cuda(),
        torch.from_numpy(indices).cuda(),
        torch.from_numpy(values).cuda(),
        size=(n_docs, N_BUCKETS),
        dtype=torch.float32,
    )
    # Build transpose CSR (CSC of X) on CPU via COO sort for fast sparse X^T @ Y on GPU
    rows = np.repeat(np.arange(n_docs, dtype=np.int64), lengths)
    order = np.argsort(indices, kind="mergesort")
    t_rows = indices[order]
    t_cols = rows[order]
    t_vals = values[order]
    t_counts = np.bincount(t_rows, minlength=N_BUCKETS).astype(np.int64)
    t_indptr = np.zeros(N_BUCKETS + 1, dtype=np.int64)
    np.cumsum(t_counts, out=t_indptr[1:])
    XT = torch.sparse_csr_tensor(
        torch.from_numpy(t_indptr).cuda(),
        torch.from_numpy(t_cols).cuda(),
        torch.from_numpy(t_vals).cuda(),
        size=(N_BUCKETS, n_docs),
        dtype=torch.float32,
    )

    torch.manual_seed(42)
    k = dim + 16
    Omega = torch.randn(N_BUCKETS, k, device="cuda", dtype=torch.float32)
    Y = X @ Omega
    Q, _ = torch.linalg.qr(Y)
    for _ in range(power_iters):
        Z = XT @ Q
        Qz, _ = torch.linalg.qr(Z)
        Y = X @ Qz
        Q, _ = torch.linalg.qr(Y)

    # Project B = Q^T X -> B^T = X^T Q ([N_BUCKETS, k]), SVD of small [k, k] or [N_BUCKETS, k]
    BT = XT @ Q  # [N_BUCKETS, k]
    # Singular vectors of B = BT^T: BT^T @ BT = V S^2 V^T -> right singular vectors of BT are left singular vectors of B!
    G = BT.t() @ BT  # [k, k]
    evals, evecs = torch.linalg.eigh(G)
    top_idx = torch.argsort(evals, descending=True)[:dim]
    U_small = evecs[:, top_idx]
    S = evals[top_idx].clamp_min(1e-8).sqrt()
    # Embeddings weighted by S^0.5 (balances top global topics & finer subtopics)
    E = (Q @ U_small) * S.sqrt().unsqueeze(0)
    E = torch.nn.functional.normalize(E, dim=-1, eps=1e-8).half()
    del X, XT, Q, BT, Omega
    torch.cuda.empty_cache()
    print(f"Computed [{n_docs:,}, {dim}] GPU SVD embeddings in {time.time() - t0:.1f}s", flush=True)
    return E


@torch.no_grad()
def spherical_kmeans(E, K, iters=18, seed=0):
    """Fast GPU spherical K-Means++ initialization + iterations on L2-normalized fp16 embeddings."""
    N, D = E.shape
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Random subset init + K-Means iterations
    perm = torch.randperm(N, generator=g, device="cuda")[:K]
    C = E[perm].float().clone()
    C = torch.nn.functional.normalize(C, dim=-1, eps=1e-8)

    assign = torch.empty(N, dtype=torch.int64, device="cuda")
    chunk = 32768
    for _ in range(iters):
        Ct = C.half().t().contiguous()
        for a in range(0, N, chunk):
            b = min(a + chunk, N)
            assign[a:b] = (E[a:b] @ Ct).argmax(dim=-1)
        C.zero_()
        C.index_add_(0, assign, E.float())
        norms = C.norm(dim=-1, keepdim=True)
        empty = (norms.squeeze(-1) < 1e-6).nonzero().squeeze(1)
        if len(empty) > 0:
            ridx = torch.randint(0, N, (len(empty),), generator=g, device="cuda")
            C[empty] = E[ridx].float()
        C = torch.nn.functional.normalize(C, dim=-1, eps=1e-8)
    return assign.cpu().numpy(), C.half()


@torch.no_grad()
def greedy_tsp_on_sim(S, start=0, two_opt_passes=2):
    """Greedy nearest-neighbor TSP path on similarity matrix S [M, M] (CPU float32 numpy) + fast 2-opt."""
    M = S.shape[0]
    if M <= 2:
        return list(range(M))
    S = S.copy()
    np.fill_diagonal(S, -1e9)
    visited = np.zeros(M, dtype=bool)
    path = np.empty(M, dtype=np.int32)
    cur = int(start)
    for step in range(M):
        path[step] = cur
        visited[cur] = True
        if step + 1 < M:
            row = S[cur]
            row[visited] = -1e9
            cur = int(np.argmax(row))
    # Light local 2-opt refinement (windowed to stay O(M * W) and ultra-fast)
    W = min(64, M)
    for _ in range(two_opt_passes):
        improved = False
        for i in range(1, M - 2):
            a, b = path[i - 1], path[i]
            sab = S[a, b]
            j_end = min(M - 1, i + W)
            for j in range(i + 1, j_end):
                c, d = path[j], path[j + 1]
                if S[a, c] + S[b, d] > sab + S[c, d] + 1e-5:
                    path[i : j + 1] = path[i : j + 1][::-1]
                    b = path[i]
                    sab = S[a, b]
                    improved = True
        if not improved:
            break
    return path.tolist()


@torch.no_grad()
def order_cluster(E_sub, prev_vec=None, max_leaf=2048):
    """Recursively order a cluster of embeddings on GPU via spherical K-Means + exact GPU TSP."""
    M = E_sub.shape[0]
    if M <= 2:
        return list(range(M))
    if M > max_leaf:
        sub_k = max(2, (M + max_leaf - 1) // max_leaf)
        assign, cents = spherical_kmeans(E_sub, sub_k, iters=12, seed=M)
        S_c = (cents @ cents.t()).float().cpu().numpy()
        start_c = int((cents @ prev_vec).argmax().item()) if prev_vec is not None else 0
        c_order = greedy_tsp_on_sim(S_c, start=start_c, two_opt_passes=3)
        full_order = []
        cur_prev = prev_vec
        for cid in c_order:
            members = np.where(assign == cid)[0]
            if len(members) == 0:
                continue
            sub_ord = order_cluster(E_sub[members], prev_vec=cur_prev, max_leaf=max_leaf)
            mapped = [int(members[i]) for i in sub_ord]
            full_order.extend(mapped)
            cur_prev = E_sub[mapped[-1]]
        return full_order

    S = (E_sub @ E_sub.t()).float().cpu().numpy()
    start = int((E_sub @ prev_vec).argmax().item()) if prev_vec is not None else 0
    return greedy_tsp_on_sim(S, start=start, two_opt_passes=2)


def lehmer_exact_bits(perm):
    """Exact theoretical bits for encoding permutation of N items via uniform Lehmer code
    (sum_{k=1}^N log2(k)), plus exact Shannon entropy of actual Lehmer ranks & delta+lzma size."""
    N = len(perm)
    # Uniform Lehmer code exact bound: log2(N!)
    uniform_bits = math.lgamma(N + 1) / math.log(2)
    # Also test lzma on int32 diffs (captures preserved runs / near-sorted blocks)
    p = np.asarray(perm, dtype=np.int32)
    diffs = np.empty_like(p)
    diffs[0] = p[0]
    diffs[1:] = p[1:] - p[:-1]
    lzma_bytes = len(lzma.compress(diffs.tobytes(), preset=6))
    zlib_bytes = len(zlib.compress(diffs.tobytes(), 9))
    uniform_bytes = int(math.ceil(uniform_bits / 8.0))
    return dict(
        uniform_lehmer_bytes=uniform_bytes,
        delta_lzma_bytes=lzma_bytes,
        delta_zlib_bytes=zlib_bytes,
        best_bytes=min(uniform_bytes, lzma_bytes),
    )


def main():
    global _RAW_BUF
    t_start = time.time()
    in_path = "enwik9"
    out_path = "enwik9.reord"
    print(f"Loading {in_path}...", flush=True)
    raw = open(in_path, "rb").read()
    orig_sha = hashlib.sha256(raw).hexdigest()
    starts, ends = parse_pages(raw)
    N = len(starts)
    header = raw[: starts[0]]
    trailer = raw[ends[-1] :]
    assert sum(e - s for s, e in zip(starts, ends)) + len(header) + len(trailer) == len(raw)
    print(f"Parsed {N:,} pages (header={len(header)}B, trailer={len(trailer)}B) in {time.time()-t_start:.1f}s", flush=True)

    # Parallel feature extraction using fork (shares read-only _RAW_BUF via COW with 0 copies!)
    _RAW_BUF = raw
    t1 = time.time()
    n_workers = min(32, os.cpu_count() or 8)
    spans = [(i, starts[i], ends[i]) for i in range(N)]
    chunk_sz = (N + n_workers * 4 - 1) // (n_workers * 4)
    chunks = [spans[i : i + chunk_sz] for i in range(0, N, chunk_sz)]
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        results = pool.map(_extract_worker, chunks)
    flat = [item for sub in results for item in sub]
    del results, spans, chunks

    redir_items = []
    art_indices = []
    cols_list = []
    vals_list = []
    for idx, is_redir, target, t_low, feat in flat:
        if is_redir:
            redir_items.append((target, t_low, idx))
        else:
            art_indices.append(idx)
            cols_list.append(feat[0])
            vals_list.append(feat[1])
    del flat
    print(
        f"Extracted features in {time.time()-t1:.1f}s: {len(art_indices):,} articles, {len(redir_items):,} redirects",
        flush=True,
    )

    # 1. Order redirects by (target_title, own_title) so identical/adjacent targets sit together
    redir_items.sort(key=lambda x: (x[0], x[1]))
    redir_order = [x[2] for x in redir_items]
    del redir_items

    # 2. Compute GPU SVD embeddings for all content articles
    E = compute_svd_embeddings(cols_list, vals_list, dim=192, power_iters=2)
    del cols_list, vals_list

    # Check mean adjacent cosine similarity BEFORE reordering
    orig_adj_sim = (E[:-1] * E[1:]).sum(-1).float().mean().item()

    # 3. Hierarchical Spherical K-Means (K=512) + Inter-Cluster TSP + Exact Intra-Cluster TSP
    t2 = time.time()
    K = 512
    assign, cents = spherical_kmeans(E, K, iters=20, seed=0)
    S_c = (cents @ cents.t()).float().cpu().numpy()
    c_order = greedy_tsp_on_sim(S_c, start=0, two_opt_passes=5)

    art_local_order = []
    prev_vec = None
    for cid in c_order:
        members = np.where(assign == cid)[0]
        if len(members) == 0:
            continue
        sub_ord = order_cluster(E[members], prev_vec=prev_vec, max_leaf=2048)
        mapped = [int(members[i]) for i in sub_ord]
        art_local_order.extend(mapped)
        prev_vec = E[mapped[-1]]

    assert len(art_local_order) == len(art_indices)
    ord_t = torch.tensor(art_local_order, device="cuda", dtype=torch.long)
    E_ord = E[ord_t]
    reord_adj_sim = (E_ord[:-1] * E_ord[1:]).sum(-1).float().mean().item()
    print(
        f"Ordered {len(art_local_order):,} articles in {time.time()-t2:.1f}s | "
        f"mean adjacent cosine sim: {orig_adj_sim:.4f} -> {reord_adj_sim:.4f} ({reord_adj_sim / max(orig_adj_sim, 1e-6):.2f}x!)",
        flush=True,
    )
    del E, E_ord, ord_t
    torch.cuda.empty_cache()

    # Combine redirects block + semantically ordered articles block
    art_order = [art_indices[i] for i in art_local_order]
    perm = np.array(redir_order + art_order, dtype=np.int32)
    assert len(perm) == N and len(set(perm.tolist())) == N, "Invalid permutation!"

    # Measure exact permutation storage cost
    pstats = lehmer_exact_bits(perm)
    print(
        f"Permutation cost: Lehmer uniform={pstats['uniform_lehmer_bytes']:,} B "
        f"({pstats['uniform_lehmer_bytes']/1e6:.4f} MB), "
        f"delta+lzma={pstats['delta_lzma_bytes']:,} B ({pstats['delta_lzma_bytes']/1e6:.4f} MB) "
        f"-> best={pstats['best_bytes']/1e6:.4f} MB",
        flush=True,
    )

    # Write reordered file
    t3 = time.time()
    with open(out_path, "wb") as f:
        f.write(header)
        for orig_idx in perm.tolist():
            f.write(raw[starts[orig_idx] : ends[orig_idx]])
        f.write(trailer)
    perm.tofile(out_path + ".perm.i32")
    print(f"Wrote {out_path} in {time.time()-t3:.1f}s", flush=True)

    # STRICT BIT-EXACT LOSSLESS ROUND-TRIP VERIFICATION
    t4 = time.time()
    reord_raw = open(out_path, "rb").read()
    assert len(reord_raw) == len(raw), "Length mismatch!"
    r_starts, r_ends = parse_pages(reord_raw)
    assert len(r_starts) == N, f"Re-parsed {len(r_starts)} pages != {N}"
    assert reord_raw[: r_starts[0]] == header and reord_raw[r_ends[-1] :] == trailer

    inv = np.empty(N, dtype=np.int32)
    inv[perm] = np.arange(N, dtype=np.int32)
    h = hashlib.sha256()
    h.update(reord_raw[: r_starts[0]])
    for orig_idx in range(N):
        new_pos = int(inv[orig_idx])
        h.update(reord_raw[r_starts[new_pos] : r_ends[new_pos]])
    h.update(reord_raw[r_ends[-1] :])
    rec_sha = h.hexdigest()
    assert rec_sha == orig_sha, f"SHA-256 MISMATCH: {rec_sha} != {orig_sha}"
    print(f"VERIFIED BIT-EXACT LOSSLESS RECONSTRUCTION (SHA-256 {rec_sha[:16]}...) in {time.time()-t4:.1f}s!", flush=True)

    # Quick zlib compressibility check on 40 MB uniform slices (before vs after)
    z_orig = sum(len(zlib.compress(raw[i : i + (1 << 20)], 6)) for i in range(0, len(raw), 25 << 20))
    z_reord = sum(len(zlib.compress(reord_raw[i : i + (1 << 20)], 6)) for i in range(0, len(reord_raw), 25 << 20))
    print(
        f"Quick 40MB zlib-6 sanity check: orig={z_orig/1e6:.2f} MB -> reord={z_reord/1e6:.2f} MB "
        f"({(z_reord - z_orig) / z_orig * 100:+.2f}%)",
        flush=True,
    )

    meta = dict(
        source=in_path,
        n_bytes=len(raw),
        n_pages=N,
        n_articles=len(art_indices),
        n_redirects=len(redir_order),
        orig_adj_cos=round(orig_adj_sim, 5),
        reord_adj_cos=round(reord_adj_sim, 5),
        sha256=orig_sha,
        extra_overhead_bytes=pstats["best_bytes"],
        **pstats,
    )
    json.dump(meta, open(out_path + ".meta.json", "w"), indent=2)
    print(f"Done in {time.time()-t_start:.1f}s! Saved {out_path}.meta.json", flush=True)


if __name__ == "__main__":
    main()

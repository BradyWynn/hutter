"""Benchmark zero-archive-cost output-side online hash dictionaries on top of a frozen Transformer.

Evaluates strictly causal prequential compression over a contiguous slice of enwik9:
  1. Baseline: frozen v8+ Transformer with T=2048 sliding-window inference
  2. Design 1 (Dense Online Engram-Head): M hash tables of dense d-dim vectors added before lm_head
     with exact closed-form gradient g = W_eff^T (p - e_y) (zero trunk backprop)
  3. Design 2 (Sparse Top-K Online Logit Bucket): M hash tables storing up to K (token_id, delta)
     entries per context hash, updated strictly token-by-token (S=1) in compiled C using the exact
     sparse-delta softmax identity (zero matmuls, zero trunk backprop)

Usage:
  python online_hash_bench.py model_ema.pt --data enwik9.split4096 --n-eval 8388608
"""
import argparse
import ctypes
import json
import math
import os
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
import matchfeat


def load_ckpt_sd(path):
    raw = torch.load(path, map_location="cuda", weights_only=False)
    if isinstance(raw, dict) and "ema" in raw and raw["ema"] is not None:
        return raw["ema"]
    if isinstance(raw, dict) and "model" in raw:
        return raw["model"]
    return raw


def build_or_load_cache(args):
    os.makedirs(args.cache_dir, exist_ok=True)
    ckpt_tag = os.path.basename(args.ckpt).replace("/", "_")
    cache_dir = os.path.join(args.cache_dir, f"cache_{ckpt_tag}_{args.data}_{args.start_offset}_{args.n_eval}")
    done_flag = os.path.join(cache_dir, "done")

    toks_all = np.fromfile(args.data + ".i16", dtype=np.int16)
    n_total = len(toks_all)
    n_bytes_global = os.path.getsize("enwik9")
    bpt_global = n_bytes_global / n_total
    vocab_size = 256 + len(json.load(open(args.data + ".json"))["merges"])

    sd = load_ckpt_sd(args.ckpt)
    n_embd = sd["wte.weight"].shape[1]
    n_match = sd["match_gate"].shape[0] if "match_gate" in sd else 0
    if "logit_scale" in sd:
        scale = sd["logit_scale"].float()
        W_eff = (sd["wte.weight"].float() * scale).cpu().numpy().astype(np.float32)
        b_head = (sd["lm_head.bias"].float() * scale).cpu().numpy().astype(np.float32)
    else:
        W_eff = sd["lm_head.weight"].float().cpu().numpy().astype(np.float32)
        b_head = sd["lm_head.bias"].float().cpu().numpy().astype(np.float32)

    if os.path.exists(done_flag):
        print(f"Loading cached hidden states from {cache_dir}")
        H = np.load(os.path.join(cache_dir, "H.npy"), mmap_mode="r")
        mp_arr = np.load(os.path.join(cache_dir, "mp.npy"))
        mb_arr = np.load(os.path.join(cache_dir, "mb.npy"))
        ctx_toks = np.load(os.path.join(cache_dir, "ctx_toks.npy"))
        N = len(ctx_toks) - 16
        y_all = ctx_toks[16:].astype(np.int64)
        W_t = torch.from_numpy(W_eff).cuda()
        b_t = torch.from_numpy(b_head).cuda()
        Z = np.empty(N, dtype=np.float32)
        base_nll = np.empty(N, dtype=np.float32)
        with torch.no_grad():
            torch.set_float32_matmul_precision("highest")
            for a in range(0, N, 65536):
                b_end = min(a + 65536, N)
                hb = torch.from_numpy(np.array(H[a:b_end])).cuda().float()
                lg = hb @ W_t.t() + b_t
                if n_match > 0:
                    lg.scatter_add_(
                        1,
                        torch.from_numpy(mp_arr[a:b_end]).cuda().long(),
                        torch.from_numpy(mb_arr[a:b_end]).cuda().float(),
                    )
                z = torch.logsumexp(lg, dim=-1)
                yb = torch.from_numpy(y_all[a:b_end]).cuda()
                Z[a:b_end] = z.cpu().numpy()
                base_nll[a:b_end] = (z - lg.gather(1, yb[:, None]).squeeze(1)).cpu().numpy()
        return H, Z, base_nll, mp_arr, mb_arr, ctx_toks, W_eff, b_head, bpt_global, n_embd, n_match, vocab_size

    os.makedirs(cache_dir, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    data_gpu = torch.from_numpy(toks_all).cuda()
    match_k = tuple(int(x) for x in args.match_k.split(",") if x) if n_match else ()
    assert len(match_k) == n_match, f"ckpt has n_match={n_match}, got --match-k={match_k}"
    match_tables = matchfeat.cached(data_gpu, match_k, args.data) if match_k else ()

    T, STRIDE, N_EVAL = args.T, args.stride, args.n_eval
    cfg = GPTConfig(
        context_length=T,
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_match=n_match,
        match_codes=matchfeat.N_CODE,
    )
    model = GPT(cfg).cuda().eval()
    model.load_state_dict(sd)
    for p in model.parameters():
        p.requires_grad_(False)
    model.build_swa_mask()

    @torch.compile
    def run_window(x, mp, ml):
        with torch.autocast("cuda", torch.bfloat16):
            return model(x, mp, ml, return_features=True)

    start0 = max(args.start_offset, T - STRIDE)
    n_win = N_EVAL // STRIDE
    assert n_win * STRIDE == N_EVAL

    H_out = np.lib.format.open_memmap(os.path.join(cache_dir, "H.npy"), mode="w+", dtype=np.float16, shape=(N_EVAL, n_embd))
    Z_out = np.empty(N_EVAL, dtype=np.float32)
    nll_out = np.empty(N_EVAL, dtype=np.float32)
    mp_out = np.empty((N_EVAL, max(n_match, 1)), dtype=np.int16)
    mb_out = np.empty((N_EVAL, max(n_match, 1)), dtype=np.float16)

    ctx_pad = 16
    ctx_toks = toks_all[start0 - ctx_pad : start0 + N_EVAL].copy()
    np.save(os.path.join(cache_dir, "ctx_toks.npy"), ctx_toks)

    B = args.batch_size
    offs = torch.arange(T + 1, device="cuda")
    W_t = torch.from_numpy(W_eff).cuda()
    b_t = torch.from_numpy(b_head).cuda()
    t0 = time.time()
    with torch.no_grad():
        for w0 in range(0, n_win, B):
            wb = min(B, n_win - w0)
            starts = [start0 + (w0 + j) * STRIDE - (T - STRIDE) - 1 for j in range(wb)]
            st_t = torch.tensor(starts, dtype=torch.long, device="cuda")
            seq = data_gpu[st_t[:, None] + offs].long()
            if n_match:
                mp_np = np.stack([match_tables[0][s + 1 : s + 1 + T] for s in starts])
                ml_np = np.stack([match_tables[1][s + 1 : s + 1 + T] for s in starts])
                mp_t = torch.from_numpy(mp_np).cuda()
                ml_t = torch.from_numpy(ml_np).cuda()
            else:
                mp_t = ml_t = None

            _, hn, bonus = run_window(seq[:, :-1], mp_t, ml_t)
            hn_h = hn[:, -STRIDE:, :].reshape(-1, n_embd).half()
            y_s = seq[:, 1:][:, -STRIDE:].reshape(-1)
            # Compute exact fp32 Z and base_nll from fp16 hn/bonus so C sparse-delta matches to <1e-5 on cold build too
            torch.set_float32_matmul_precision("highest")
            lg_s = hn_h.float() @ W_t.t() + b_t
            torch.set_float32_matmul_precision("high")
            if n_match:
                mp_s = mp_t[:, -STRIDE:, :].reshape(-1, n_match)
                bo_h = bonus[:, -STRIDE:, :].reshape(-1, n_match).half()
                lg_s.scatter_add_(1, mp_s.long(), bo_h.float())

            z_s = torch.logsumexp(lg_s, dim=-1)
            y_lg = lg_s.gather(1, y_s[:, None]).squeeze(1)
            nll_s = z_s - y_lg

            i0 = w0 * STRIDE
            i1 = i0 + wb * STRIDE
            H_out[i0:i1] = hn_h.cpu().numpy()
            Z_out[i0:i1] = z_s.cpu().numpy()
            nll_out[i0:i1] = nll_s.cpu().numpy()
            if n_match:
                mp_out[i0:i1] = mp_s.cpu().numpy()
                mb_out[i0:i1] = bo_h.cpu().numpy()
            else:
                mp_out[i0:i1] = 0
                mb_out[i0:i1] = 0

            if (w0 // B) % 100 == 0 or w0 + wb == n_win:
                done = i1
                bpb_so_far = nll_out[:done].mean() / math.log(2) / bpt_global
                print(f"  cached {done:,}/{N_EVAL:,} toks ({time.time()-t0:.1f}s) | baseline bpb={bpb_so_far:.4f}", flush=True)

    H_out.flush()
    np.save(os.path.join(cache_dir, "Z.npy"), Z_out)
    np.save(os.path.join(cache_dir, "base_nll.npy"), nll_out)
    np.save(os.path.join(cache_dir, "mp.npy"), mp_out)
    np.save(os.path.join(cache_dir, "mb.npy"), mb_out)
    open(done_flag, "w").close()
    print(f"Cache complete in {time.time()-t0:.1f}s")
    return H_out, Z_out, nll_out, mp_out, mb_out, ctx_toks, W_eff, b_head, bpt_global, n_embd, n_match, vocab_size


# ---------------------------------------------------------------------------
# Design 2: Exact S=1 Token-by-Token Sparse Top-K Online Logit Dictionary (C)
# ---------------------------------------------------------------------------
C_SRC = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static inline float dot_f16_f32(const uint16_t *h16, const float *w, int D) {
    const _Float16 *hf = (const _Float16 *)h16;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int i = 0;
    for (; i + 3 < D; i += 4) {
        s0 += (float)hf[i]   * w[i];
        s1 += (float)hf[i+1] * w[i+1];
        s2 += (float)hf[i+2] * w[i+2];
        s3 += (float)hf[i+3] * w[i+3];
    }
    float sum = (s0 + s1) + (s2 + s3);
    for (; i < D; i++) sum += (float)hf[i] * w[i];
    return sum;
}

#define KMAX 8
#define MMAX 8

typedef struct {
    int16_t tok[KMAX];
    float   val[KMAX];
    uint16_t cnt;
    uint16_t tag;
} Bucket;

void run_sparse_bucket(
    int N,
    int ctx_pad,
    int D,
    int n_match,
    const int16_t *ctx_toks,     // length N + ctx_pad
    const uint16_t *H16,         // [N, D] float16
    const float *Z_base,         // [N]
    const int16_t *mp,           // [N, max(n_match, 1)]
    const uint16_t *mb16,        // [N, max(n_match, 1)] float16
    const float *W_eff,          // [V, D]
    const float *b_head,         // [V]
    int M,
    const int *orders,
    const int *table_bits,
    const float *lrs,
    int K,
    float decay,
    float count_pow,
    float count_c0,
    int use_tag,
    float *out_nll               // [N]
) {
    Bucket *tables[MMAX];
    uint64_t masks[MMAX];
    for (int m = 0; m < M; m++) {
        size_t sz = (size_t)1ULL << table_bits[m];
        tables[m] = (Bucket *)calloc(sz, sizeof(Bucket));
        masks[m] = sz - 1;
        for (size_t r = 0; r < sz; r++) {
            for (int k = 0; k < KMAX; k++) tables[m][r].tok[k] = -1;
        }
    }

    int16_t cand_tok[MMAX * KMAX + 2];
    float   cand_delta[MMAX * KMAX + 2];
    float   cand_base[MMAX * KMAX + 2];
    float   cand_p[MMAX * KMAX + 2];
    uint64_t rows[MMAX];
    uint16_t tags[MMAX];
    int match_stride = n_match > 0 ? n_match : 1;

    for (int t = 0; t < N; t++) {
        int pos = ctx_pad + t;
        int16_t y = ctx_toks[pos];

        int n_cand = 0;
        for (int m = 0; m < M; m++) {
            int ord = orders[m];
            uint64_t h = 0x9e3779b97f4a7c15ULL ^ (uint64_t)(m + 1);
            if (ord > 0) {
                for (int j = 1; j <= ord; j++) {
                    h = mix64(h ^ (uint64_t)(uint16_t)ctx_toks[pos - j]);
                }
            } else if (ord == -1) {
                h = mix64(h ^ (uint64_t)(uint16_t)ctx_toks[pos - 1]);
                h = mix64(h ^ (uint64_t)(uint16_t)ctx_toks[pos - 3]);
            } else if (ord == -2) {
                h = mix64(h ^ (uint64_t)(uint16_t)ctx_toks[pos - 1]);
                h = mix64(h ^ (uint64_t)(uint16_t)ctx_toks[pos - 4]);
            } else if (ord == -3) {
                h = mix64(h ^ (uint64_t)(uint16_t)ctx_toks[pos - 1]);
                h = mix64(h ^ (uint64_t)(uint16_t)ctx_toks[pos - 2]);
                h = mix64(h ^ (uint64_t)(uint16_t)ctx_toks[pos - 4]);
            }
            rows[m] = h & masks[m];
            tags[m] = (uint16_t)((h >> 48) | 1);

            Bucket *bk = &tables[m][rows[m]];
            if (use_tag && bk->cnt > 0 && bk->tag != tags[m]) continue;
            for (int k = 0; k < K; k++) {
                int16_t tk = bk->tok[k];
                if (tk < 0) continue;
                float v = bk->val[k];
                if (v == 0.f) continue;
                int found = -1;
                for (int c = 0; c < n_cand; c++) {
                    if (cand_tok[c] == tk) { found = c; break; }
                }
                if (found >= 0) {
                    cand_delta[found] += v;
                } else {
                    cand_tok[n_cand] = tk;
                    cand_delta[n_cand] = v;
                    n_cand++;
                }
            }
        }

        int y_idx = -1;
        for (int c = 0; c < n_cand; c++) {
            if (cand_tok[c] == y) { y_idx = c; break; }
        }
        if (y_idx < 0) {
            y_idx = n_cand;
            cand_tok[n_cand] = y;
            cand_delta[n_cand] = 0.f;
            n_cand++;
        }

        const uint16_t *h_row = H16 + (size_t)t * (size_t)D;
        const int16_t *mp_row = mp + (size_t)t * (size_t)match_stride;
        const _Float16 *mb_row = (const _Float16 *)(mb16 + (size_t)t * (size_t)match_stride);
        float z0 = Z_base[t];
        float max_l = z0;

        for (int c = 0; c < n_cand; c++) {
            int16_t tk = cand_tok[c];
            float lg = dot_f16_f32(h_row, W_eff + (size_t)tk * (size_t)D, D) + b_head[tk];
            for (int j = 0; j < n_match; j++) {
                if (mp_row[j] == tk) lg += (float)mb_row[j];
            }
            cand_base[c] = lg;
            float lnew = lg + cand_delta[c];
            if (lnew > max_l) max_l = lnew;
        }

        double sum_exp = exp((double)(z0 - max_l));
        for (int c = 0; c < n_cand; c++) {
            if (cand_delta[c] != 0.f) {
                double e_new = exp((double)(cand_base[c] + cand_delta[c] - max_l));
                double e_old = exp((double)(cand_base[c] - max_l));
                sum_exp += (e_new - e_old);
            }
        }
        if (sum_exp < 1e-30) sum_exp = 1e-30;
        float z_new = max_l + (float)log(sum_exp);

        float y_lnew = cand_base[y_idx] + cand_delta[y_idx];
        out_nll[t] = z_new - y_lnew;

        for (int c = 0; c < n_cand; c++) {
            cand_p[c] = expf((cand_base[c] + cand_delta[c]) - z_new);
        }

        for (int m = 0; m < M; m++) {
            Bucket *bk = &tables[m][rows[m]];
            if (use_tag && bk->cnt > 0 && bk->tag != tags[m]) {
                for (int k = 0; k < K; k++) { bk->tok[k] = -1; bk->val[k] = 0.f; }
                bk->cnt = 0;
            }
            bk->tag = tags[m];
            float step_lr = lrs[m] / powf((float)bk->cnt + count_c0, count_pow);
            if (bk->cnt < 65535) bk->cnt++;

            int y_slot = -1, empty_slot = -1, min_slot = 0;
            float min_abs = 1e30f;

            for (int k = 0; k < K; k++) {
                int16_t tk = bk->tok[k];
                if (tk < 0) {
                    if (empty_slot < 0) empty_slot = k;
                    continue;
                }
                if (tk == y) y_slot = k;
                float p_tk = 0.f;
                for (int c = 0; c < n_cand; c++) {
                    if (cand_tok[c] == tk) { p_tk = cand_p[c]; break; }
                }
                float grad = p_tk - (tk == y ? 1.f : 0.f);
                bk->val[k] = bk->val[k] * (1.f - decay) - step_lr * grad;
                float av = fabsf(bk->val[k]);
                if (av < min_abs) { min_abs = av; min_slot = k; }
            }
            if (y_slot < 0) {
                int slot = (empty_slot >= 0) ? empty_slot : min_slot;
                bk->tok[slot] = y;
                bk->val[slot] = step_lr * (1.f - cand_p[y_idx]);
            }
        }
    }

    for (int m = 0; m < M; m++) free(tables[m]);
}
"""


def compile_c_ext(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    c_path = os.path.join(cache_dir, "online_hash.c")
    so_path = os.path.join(cache_dir, "online_hash.so")
    with open(c_path, "w") as f:
        f.write(C_SRC)
    subprocess.check_call(["gcc", "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC", c_path, "-o", so_path, "-lm"])
    lib = ctypes.CDLL(so_path)
    lib.run_sparse_bucket.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
        ctypes.c_void_p,
    ]
    return lib


def bench_design1_dense_engram(
    H_mm, Z_np, mp_np, mb_np, ctx_toks, W_eff_np, b_head_np, bpt_global, n_match,
    orders=(1, 2, 3, 4), bits=(18, 21, 21, 21), dim=64,
    chunk_s=64, lr=0.25, decay=1e-4, count_pow=0.25, count_c0=2.0,
):
    """Design 1: Dense low-rank online hash embeddings in top-d SVD subspace of W_eff (or full D)."""
    t0 = time.time()
    N = len(Z_np)
    ctx_pad = len(ctx_toks) - N
    D = W_eff_np.shape[1]
    W_t = torch.from_numpy(W_eff_np).cuda()
    b_t = torch.from_numpy(b_head_np).cuda()

    if dim < D:
        _, _, Vh = torch.linalg.svd(W_t, full_matrices=False)
        P = Vh[:dim, :].contiguous()
        W_sub = (W_t @ P.t()).contiguous()
    else:
        dim = D
        W_sub = W_t.contiguous()

    M = len(orders)
    tables = [torch.zeros((1 << b, dim), dtype=torch.float16, device="cuda") for b in bits]
    counts = [torch.zeros((1 << b,), dtype=torch.float32, device="cuda") for b in bits]
    ram_mb = sum(t.numel() * 2 + c.numel() * 4 for t, c in zip(tables, counts)) / 1e6

    hashes = []
    for m, (ord_, b) in enumerate(zip(orders, bits)):
        h = np.full(N, np.uint64(0x9e3779b97f4a7c15 ^ (m + 1)), dtype=np.uint64)
        for j in range(1, ord_ + 1):
            tk = ctx_toks[ctx_pad - j : ctx_pad - j + N].astype(np.uint16).astype(np.uint64)
            h ^= tk
            h ^= h >> np.uint64(30); h *= np.uint64(0xbf58476d1ce4e5b9)
            h ^= h >> np.uint64(27); h *= np.uint64(0x94d049bb133111eb)
            h ^= h >> np.uint64(31)
        hashes.append((h & np.uint64((1 << b) - 1)).astype(np.int64))

    y_all = ctx_toks[ctx_pad:].astype(np.int64)
    nll_out = np.empty(N, dtype=np.float32)

    macro = 32768
    for a in range(0, N, macro):
        b_end = min(a + macro, N)
        H_blk = torch.from_numpy(np.array(H_mm[a:b_end])).cuda().float()
        y_blk = torch.from_numpy(y_all[a:b_end]).cuda()
        h_blks = [torch.from_numpy(hashes[m][a:b_end]).cuda() for m in range(M)]

        lg_base = H_blk @ W_t.t() + b_t
        if n_match > 0:
            mp_blk = torch.from_numpy(mp_np[a:b_end]).cuda().long()
            mb_blk = torch.from_numpy(mb_np[a:b_end]).cuda().float()
            lg_base.scatter_add_(1, mp_blk, mb_blk)

        nll_blk = torch.empty(b_end - a, dtype=torch.float32, device="cuda")
        for s0 in range(0, b_end - a, chunk_s):
            s1 = min(s0 + chunk_s, b_end - a)
            e_sum = torch.zeros((s1 - s0, dim), dtype=torch.float32, device="cuda")
            idxs = [h_blks[m][s0:s1] for m in range(M)]
            for m in range(M):
                e_sum.add_(tables[m][idxs[m]].float())
            lg = lg_base[s0:s1] + e_sum @ W_sub.t()
            y_s = y_blk[s0:s1]
            logp = F.log_softmax(lg, dim=-1)
            nll_blk[s0:s1] = -logp.gather(1, y_s[:, None]).squeeze(1)

            p = logp.exp()
            p.scatter_add_(1, y_s[:, None], torch.full((s1 - s0, 1), -1.0, device="cuda"))
            g = p @ W_sub

            for m in range(M):
                idx = idxs[m]
                c = counts[m][idx]
                step_lr = (lr / M) / ((c + count_c0).pow(count_pow))
                upd = -(step_lr[:, None] * g)
                if decay > 0:
                    tables[m][idx] = (tables[m][idx].float() * (1.0 - decay)).half()
                tables[m].index_add_(0, idx, upd.half())
                counts[m].index_add_(0, idx, torch.ones_like(c))
        nll_out[a:b_end] = nll_blk.cpu().numpy()

    dt = time.time() - t0
    bpb_all = nll_out.mean() / math.log(2) / bpt_global
    bpb_last = nll_out[N // 2 :].mean() / math.log(2) / bpt_global
    del tables, counts
    torch.cuda.empty_cache()
    return bpb_all, bpb_last, ram_mb, dt, nll_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="model_ema.pt")
    ap.add_argument("--data", default="enwik9.split4096")
    ap.add_argument("--match-k", default="2,3,5,8,13,27")
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--n-eval", type=int, default=8 * 1024 * 1024)
    ap.add_argument("--start-offset", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cache-dir", default="/tmp/hutter_online_hash")
    args = ap.parse_args()

    H, Z, base_nll, mp, mb, ctx_toks, W_eff, b_head, bpt, n_embd, n_match, _ = build_or_load_cache(args)
    N = len(Z)
    ctx_pad = len(ctx_toks) - N
    base_bpb = base_nll.mean() / math.log(2) / bpt
    base_bpb_q1 = base_nll[: N // 4].mean() / math.log(2) / bpt
    base_bpb_last = base_nll[N // 2 :].mean() / math.log(2) / bpt
    print(f"\n=== BASELINE (frozen {args.ckpt}, {N:,} toks = {N * bpt / 1e6:.1f} MB) ===")
    print(f"  Full stream bpb: {base_bpb:.4f} ({base_bpb * 125:.2f} MB equiv) | Q1: {base_bpb_q1:.4f} | 2nd half: {base_bpb_last:.4f}")

    lib = compile_c_ext(args.cache_dir)
    print("\n=== DESIGN 2: SPARSE TOP-K ONLINE LOGIT BUCKET (EXACT S=1 TOKEN-BY-TOKEN IN C) ===")
    H_contig = np.ascontiguousarray(H)
    Z_c = np.ascontiguousarray(Z, dtype=np.float32)
    mp_c = np.ascontiguousarray(mp, dtype=np.int16)
    mb_c = np.ascontiguousarray(mb, dtype=np.float16)
    W_c = np.ascontiguousarray(W_eff, dtype=np.float32)
    b_c = np.ascontiguousarray(b_head, dtype=np.float32)
    ctx_c = np.ascontiguousarray(ctx_toks, dtype=np.int16)
    out_nll = np.empty(N, dtype=np.float32)

    ord0 = np.array([1], dtype=np.int32)
    bit0 = np.array([14], dtype=np.int32)
    lr0 = np.array([0.0], dtype=np.float32)
    n_chk = min(N, 65536)
    lib.run_sparse_bucket(
        n_chk, ctx_pad, n_embd, n_match,
        ctx_c.ctypes.data, H_contig.ctypes.data, Z_c.ctypes.data,
        mp_c.ctypes.data, mb_c.ctypes.data, W_c.ctypes.data, b_c.ctypes.data,
        1, ord0.ctypes.data, bit0.ctypes.data, lr0.ctypes.data,
        8, 0.0, 0.0, 1.0, 1, out_nll.ctypes.data,
    )
    err_c = float(np.abs(out_nll[:n_chk] - base_nll[:n_chk]).max())
    print(f"  [Sanity check] C sparse-delta max NLL error at lr=0: {err_c:.6f} (mean {np.abs(out_nll[:n_chk]-base_nll[:n_chk]).mean():.6f})")
    assert err_c < 1e-3, f"C reconstruction mismatch: {err_c}"

    configs_d2 = [
        ("4-ord (1,2,3,4) 22b K=8 lr=0.25",      [1, 2, 3, 4],       [18, 22, 22, 22],       [0.10, 0.20, 0.25, 0.30], 8, 1e-4, 0.35, 1.0, 1),
        ("4-ord (1,2,3,4) 22b K=8 lr=0.60",      [1, 2, 3, 4],       [18, 22, 22, 22],       [0.25, 0.50, 0.65, 0.80], 8, 1e-4, 0.35, 1.0, 1),
        ("4-ord (1,2,3,4) 22b K=8 lr=1.20",      [1, 2, 3, 4],       [18, 22, 22, 22],       [0.40, 0.90, 1.20, 1.50], 8, 1e-4, 0.35, 1.0, 1),
        ("6-ord (1..6)    22b K=8 lr=0.60",      [1, 2, 3, 4, 5, 6], [18, 22, 22, 22, 22, 22], [0.20, 0.45, 0.65, 0.80, 0.90, 1.00], 8, 1e-4, 0.35, 1.0, 1),
        ("6-ord (1..6)    23b K=8 lr=1.00",      [1, 2, 3, 4, 5, 6], [18, 22, 23, 23, 23, 23], [0.30, 0.70, 1.00, 1.20, 1.40, 1.50], 8, 0.0,  0.35, 1.0, 1),
        ("6-ord+skips     22b K=8 lr=0.60",      [1, 2, 3, 4, 5, -1, -2], [18, 22, 22, 22, 22, 21, 21], [0.20, 0.45, 0.65, 0.80, 0.90, 0.35, 0.30], 8, 1e-4, 0.35, 1.0, 1),
        ("6-ord (1..6)    23b K=8 fast-local",   [1, 2, 3, 4, 5, 6], [18, 22, 23, 23, 23, 23], [0.30, 0.80, 1.20, 1.60, 1.80, 2.00], 8, 0.002, 0.20, 1.0, 1),
    ]
    for name, orders, bits, lrs, K, dec, cpow, c0, utag in configs_d2:
        ram_gb = sum((1 << b) * 52 for b in bits) / 1e9
        ord_a = np.array(orders, dtype=np.int32)
        bit_a = np.array(bits, dtype=np.int32)
        lr_a = np.array(lrs, dtype=np.float32)
        t0 = time.time()
        lib.run_sparse_bucket(
            N, ctx_pad, n_embd, n_match,
            ctx_c.ctypes.data, H_contig.ctypes.data, Z_c.ctypes.data,
            mp_c.ctypes.data, mb_c.ctypes.data, W_c.ctypes.data, b_c.ctypes.data,
            len(orders), ord_a.ctypes.data, bit_a.ctypes.data, lr_a.ctypes.data,
            K, dec, cpow, c0, utag, out_nll.ctypes.data,
        )
        dt = time.time() - t0
        bpb_all = out_nll.mean() / math.log(2) / bpt
        bpb_last = out_nll[N // 2 :].mean() / math.log(2) / bpt
        us_tok = dt / N * 1e6
        print(f"  {name:38s} | RAM={ram_gb:.2f}GB | {dt:5.1f}s ({us_tok:.2f}us/tok) | bpb={bpb_all:.4f} ({bpb_all-base_bpb:+.4f}, {(bpb_all-base_bpb)*125:+.2f}MB) | 2ndH={bpb_last:.4f} ({bpb_last-base_bpb_last:+.4f})")

    print("\n=== DESIGN 1: DENSE ONLINE ENGRAM-HEAD (GPU, EXACT CLOSED-FORM GRAD) ===")
    bpb0, _, _, _, _ = bench_design1_dense_engram(
        H, Z, mp, mb, ctx_toks, W_eff, b_head, bpt, n_match,
        orders=(1,), bits=(14,), dim=n_embd, chunk_s=256, lr=0.0, decay=0.0,
    )
    print(f"  [Sanity check] GPU dense lr=0 bpb={bpb0:.5f} vs base={base_bpb:.5f} (diff={bpb0-base_bpb:+.5f})")
    for name, orders, bits, dim, chunk_s, lr, dec in [
        ("Dense d=64  (1,2,3,4) 20b S=64 lr=0.5", (1, 2, 3, 4), (16, 20, 20, 20), 64, 64, 0.5, 0.0),
        ("Dense d=64  (1,2,3,4) 20b S=64 lr=2.0", (1, 2, 3, 4), (16, 20, 20, 20), 64, 64, 2.0, 0.0),
        ("Dense d=256 (1,2,3,4) 19b S=64 lr=0.5", (1, 2, 3, 4), (16, 19, 19, 19), 256, 64, 0.5, 0.0),
        ("Dense d=256 (1,2,3,4) 19b S=64 lr=2.0", (1, 2, 3, 4), (16, 19, 19, 19), 256, 64, 2.0, 0.0),
        ("Dense d=256 (1,2,3,4) 19b S=16 lr=2.0", (1, 2, 3, 4), (16, 19, 19, 19), 256, 16, 2.0, 0.0),
    ]:
        bpb_all, bpb_last, ram_mb, dt, _ = bench_design1_dense_engram(
            H, Z, mp, mb, ctx_toks, W_eff, b_head, bpt, n_match,
            orders=orders, bits=bits, dim=dim, chunk_s=chunk_s, lr=lr, decay=dec,
        )
        print(f"  {name:38s} | RAM={ram_mb/1000:.2f}GB | {dt:5.1f}s | bpb={bpb_all:.4f} ({bpb_all-base_bpb:+.4f}, {(bpb_all-base_bpb)*125:+.2f}MB) | 2ndH={bpb_last:.4f} ({bpb_last-base_bpb_last:+.4f})")


if __name__ == "__main__":
    main()

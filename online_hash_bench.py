"""Benchmark zero-archive-cost output-side online hash dictionaries on top of a frozen Transformer.

Evaluates strictly causal prequential compression over a contiguous slice of enwik9.split4096:
  1. Baseline: frozen Transformer (model_ema_v2_59k.pt) with T=2048 sliding-window inference
  2. Design 1 (Dense Online Engram-Head): M hash tables of dense d-dim vectors added before lm_head
     with exact closed-form gradient g = W_eff^T (p - e_y) (zero trunk backprop)
  3. Design 2 (Sparse Top-K Online Logit Bucket): M hash tables storing up to K (token_id, delta)
     entries per context hash, updated strictly token-by-token (S=1) in compiled C using the exact
     sparse-delta softmax identity (zero matmuls, zero trunk backprop)
  4. Design 3 (Hybrid): Design 1 + Design 2 combined
"""
import ctypes
import math
import os
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
import matchfeat

SCRATCH = "/home/brady/.gemini/antigravity-cli/brain/d12bab96-3c3e-4e36-9db9-f039f2dbf177/scratch"
os.makedirs(SCRATCH, exist_ok=True)

CKPT = "model_ema_v2_59k.pt"
DATA = "enwik9.split4096"
MATCH_K = (2, 5, 13, 27)
T = 2048
STRIDE = 1024  # every scored token gets 1024..2047 tokens of Transformer context
N_EVAL = 8 * 1024 * 1024  # 8,388,608 contiguous tokens (~23.4 MB of enwik9)
START_OFFSET = 0  # start from beginning of enwik9 (true cold start)
N_BYTES_GLOBAL = 1_000_000_000


def build_or_load_cache():
    cache_dir = os.path.join(SCRATCH, f"cache_{CKPT}_{START_OFFSET}_{N_EVAL}")
    done_flag = os.path.join(cache_dir, "done")
    toks_all = np.fromfile(DATA + ".i16", dtype=np.int16)
    N_total = len(toks_all)
    bpt_global = N_BYTES_GLOBAL / N_total

    sd = torch.load(CKPT, map_location="cuda", weights_only=True)
    scale = sd["logit_scale"].float()
    W_eff = (sd["wte.weight"].float() * scale).cpu().numpy().astype(np.float32)
    b_head = (sd["lm_head.bias"].float() * scale).cpu().numpy().astype(np.float32)

    if os.path.exists(done_flag):
        print(f"Loading cached hidden states from {cache_dir}")
        H = np.load(os.path.join(cache_dir, "H.npy"), mmap_mode="r")
        mp_arr = np.load(os.path.join(cache_dir, "mp.npy"))
        mb_arr = np.load(os.path.join(cache_dir, "mb.npy"))
        ctx_toks = np.load(os.path.join(cache_dir, "ctx_toks.npy"))
        # Recompute exact float32 Z and base_nll from cached H (fp16), W_eff (fp32), b_head (fp32), mb (fp16)
        # so sparse-delta softmax in C matches Z_base to machine precision (<1e-5)!
        N = len(mp_arr)
        ctx_pad = len(ctx_toks) - N
        y_all = ctx_toks[ctx_pad:].astype(np.int64)
        W_t = torch.from_numpy(W_eff).cuda()
        b_t = torch.from_numpy(b_head).cuda()
        Z = np.empty(N, dtype=np.float32)
        base_nll = np.empty(N, dtype=np.float32)
        with torch.no_grad():
            for a in range(0, N, 65536):
                b_end = min(a + 65536, N)
                hb = torch.from_numpy(np.array(H[a:b_end])).cuda().float()
                lg = hb @ W_t.t() + b_t
                lg.scatter_add_(1, torch.from_numpy(mp_arr[a:b_end]).cuda().long(), torch.from_numpy(mb_arr[a:b_end]).cuda().float())
                z = torch.logsumexp(lg, dim=-1)
                yb = torch.from_numpy(y_all[a:b_end]).cuda()
                Z[a:b_end] = z.cpu().numpy()
                base_nll[a:b_end] = (z - lg.gather(1, yb[:, None]).squeeze(1)).cpu().numpy()
        return H, Z, base_nll, mp_arr, mb_arr, ctx_toks, W_eff, b_head, bpt_global

    os.makedirs(cache_dir, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    data_gpu = torch.from_numpy(toks_all).cuda()
    match_tables = matchfeat.cached(data_gpu, MATCH_K, DATA)

    cfg = GPTConfig(context_length=T, vocab_size=4096, n_match=len(MATCH_K), match_codes=matchfeat.N_CODE)
    model = GPT(cfg).cuda().eval()
    model.load_state_dict(sd)
    for p in model.parameters():
        p.requires_grad_(False)

    # Extract norm(x) right before lm_head, plus exact logits_base, Z_base, match bonus
    @torch.compile
    def run_window(x, mp, ml):
        with torch.autocast("cuda", torch.bfloat16):
            T_ = x.shape[1]
            h = model.wte(x)
            mp_l, ml_l = mp.long(), ml.long()
            has = (ml_l > 0).to(h.dtype)
            gate = model.match_gate[model.match_col, ml_l] * has
            h = h + (model.wte(mp_l) * gate.unsqueeze(-1)).sum(2)
            h = F.rms_norm(h, (h.size(-1),), eps=1e-6)
            x0 = h
            cos, sin = model.cos[:, :T_], model.sin[:, :T_]
            v0 = None
            p = 0
            for loop in range(model.c.n_loop):
                if model.loop_emb is not None:
                    h = h + model.loop_emb[loop].to(h.dtype)
                for b in model.h:
                    h = model.resid_lambdas[p].to(h.dtype) * h + model.x0_lambdas[p].to(h.dtype) * x0
                    v_lam = model.v_lambdas[p].to(h.dtype) if p > 0 else None
                    h, v0 = b(h, cos, sin, v0=v0, v_lambda=v_lam)
                    p += 1
            hn = F.rms_norm(h, (h.size(-1),), eps=1e-6)
            logits = model.lm_head(hn) * model.logit_scale.to(hn.dtype)
            bonus = (model.match_logit[model.match_col, ml_l] * has).to(logits.dtype)
            logits = logits.scatter_add(-1, mp_l, bonus)
            return hn, logits, bonus

    # We score targets at positions [start0 .. start0 + N_EVAL - 1], where start0 >= T - STRIDE
    # so every scored target has at least STRIDE=1024 context tokens!
    start0 = max(START_OFFSET, T - STRIDE)
    n_win = N_EVAL // STRIDE
    assert n_win * STRIDE == N_EVAL

    H_out = np.lib.format.open_memmap(os.path.join(cache_dir, "H.npy"), mode="w+", dtype=np.float16, shape=(N_EVAL, 256))
    Z_out = np.empty(N_EVAL, dtype=np.float32)
    nll_out = np.empty(N_EVAL, dtype=np.float32)
    mp_out = np.empty((N_EVAL, 4), dtype=np.int16)
    mb_out = np.empty((N_EVAL, 4), dtype=np.float16)

    # Save tokens from start0 - 16 through start0 + N_EVAL for exact n-gram hashing
    ctx_pad = 16
    ctx_toks = toks_all[start0 - ctx_pad : start0 + N_EVAL].copy()
    np.save(os.path.join(cache_dir, "ctx_toks.npy"), ctx_toks)

    B = 16
    offs = torch.arange(T + 1, device="cuda")
    t0 = time.time()
    with torch.no_grad():
        for w0 in range(0, n_win, B):
            wb = min(B, n_win - w0)
            # For window k, we want to score targets at [start0 + k*STRIDE .. start0 + (k+1)*STRIDE - 1].
            # Window input x is tokens [s .. s + T - 1], predicting targets [s + 1 .. s + T].
            # Setting s = start0 + k*STRIDE - (T - STRIDE) - 1 puts those targets in the LAST STRIDE positions!
            starts = [start0 + (w0 + j) * STRIDE - (T - STRIDE) - 1 for j in range(wb)]
            st_t = torch.tensor(starts, dtype=torch.long, device="cuda")
            seq = data_gpu[st_t[:, None] + offs].long()
            mp_np = np.stack([match_tables[0][s + 1 : s + 1 + T] for s in starts])
            ml_np = np.stack([match_tables[1][s + 1 : s + 1 + T] for s in starts])
            mp_t = torch.from_numpy(mp_np).cuda()
            ml_t = torch.from_numpy(ml_np).cuda()

            hn, logits, bonus = run_window(seq[:, :-1], mp_t, ml_t)
            # Take last STRIDE positions (each has 1024..2047 tokens of Transformer context)
            hn_s = hn[:, -STRIDE:, :].reshape(-1, 256)
            lg_s = logits[:, -STRIDE:, :].reshape(-1, 4096).float()
            y_s = seq[:, 1:][:, -STRIDE:].reshape(-1)
            mp_s = mp_t[:, -STRIDE:, :].reshape(-1, 4)
            bo_s = bonus[:, -STRIDE:, :].reshape(-1, 4)

            z_s = torch.logsumexp(lg_s, dim=-1)
            y_lg = lg_s.gather(1, y_s[:, None]).squeeze(1)
            nll_s = z_s - y_lg

            i0 = w0 * STRIDE
            i1 = i0 + wb * STRIDE
            H_out[i0:i1] = hn_s.half().cpu().numpy()
            Z_out[i0:i1] = z_s.cpu().numpy()
            nll_out[i0:i1] = nll_s.cpu().numpy()
            mp_out[i0:i1] = mp_s.cpu().numpy()
            mb_out[i0:i1] = bo_s.half().cpu().numpy()

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
    return H_out, Z_out, nll_out, mp_out, mb_out, ctx_toks, W_eff, b_head, bpt_global


# ---------------------------------------------------------------------------
# Design 2: Exact S=1 Token-by-Token Sparse Top-K Online Logit Dictionary (C)
# ---------------------------------------------------------------------------
C_SRC = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Fast 64-bit splitmix / FNV mixing
static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static inline float dot256_f16_f32(const uint16_t *h16, const float *w) {
    // IEEE-754 float16 -> float32 via _Float16 (GCC 14 native AVX2/F16C!)
    const _Float16 *hf = (const _Float16 *)h16;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    for (int i = 0; i < 256; i += 4) {
        s0 += (float)hf[i]   * w[i];
        s1 += (float)hf[i+1] * w[i+1];
        s2 += (float)hf[i+2] * w[i+2];
        s3 += (float)hf[i+3] * w[i+3];
    }
    return (s0 + s1) + (s2 + s3);
}

#define KMAX 8
#define MMAX 8

typedef struct {
    int16_t tok[KMAX];
    float   val[KMAX];
    uint16_t cnt;
    uint16_t tag; // 16-bit hash tag to reject collisions!
} Bucket;

// Run exact S=1 token-by-token sparse-bucket online logit dictionary!
// orders: array of M context specs:
//   order > 0: exact n-gram of length `order` (tokens t-order..t-1)
//   order == -1: skip-bigram (t-3, t-1)
//   order == -2: skip-bigram (t-4, t-1)
//   order == -3: skip-trigram (t-4, t-2, t-1)
void run_sparse_bucket(
    int N,
    int ctx_pad,
    const int16_t *ctx_toks,     // length N + ctx_pad; target y_t is ctx_toks[ctx_pad + t]
    const uint16_t *H16,         // [N, 256] float16
    const float *Z_base,         // [N]
    const int16_t *mp,           // [N, 4]
    const uint16_t *mb16,        // [N, 4] float16
    const float *W_eff,          // [4096, 256]
    const float *b_head,         // [4096]
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

    for (int t = 0; t < N; t++) {
        int pos = ctx_pad + t;
        int16_t y = ctx_toks[pos];

        // 1. Compute M context hashes (strictly causal: uses only ctx_toks[< pos])
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
            if (use_tag && bk->cnt > 0 && bk->tag != tags[m]) {
                // Hash collision with a different context -> skip reading
                continue;
            }
            for (int k = 0; k < K; k++) {
                int16_t tk = bk->tok[k];
                if (tk < 0) continue;
                float v = bk->val[k];
                if (v == 0.f) continue;
                // Merge into unique candidate list
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

        // Ensure true target y is in candidate list (so we always get exact nll and grad!)
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

        // 2. Compute exact base logits ONLY for the <= n_cand tokens
        const uint16_t *h_row = H16 + (size_t)t * 256;
        const int16_t *mp_row = mp + (size_t)t * 4;
        const _Float16 *mb_row = (const _Float16 *)(mb16 + (size_t)t * 4);
        float z0 = Z_base[t];
        float max_l = z0;

        for (int c = 0; c < n_cand; c++) {
            int16_t tk = cand_tok[c];
            float lg = dot256_f16_f32(h_row, W_eff + (size_t)tk * 256) + b_head[tk];
            for (int j = 0; j < 4; j++) {
                if (mp_row[j] == tk) lg += (float)mb_row[j];
            }
            cand_base[c] = lg;
            float lnew = lg + cand_delta[c];
            if (lnew > max_l) max_l = lnew;
        }

        // 3. Exact sparse-delta partition function Z_new
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

        // Exact NLL for true target y
        float y_lnew = cand_base[y_idx] + cand_delta[y_idx];
        out_nll[t] = z_new - y_lnew;

        // Exact probabilities for candidates
        for (int c = 0; c < n_cand; c++) {
            cand_p[c] = expf((cand_base[c] + cand_delta[c]) - z_new);
        }

        // 4. Online causal SGD update for each table
        for (int m = 0; m < M; m++) {
            Bucket *bk = &tables[m][rows[m]];
            if (use_tag && bk->cnt > 0 && bk->tag != tags[m]) {
                // Overwrite collided bucket cleanly
                for (int k = 0; k < K; k++) { bk->tok[k] = -1; bk->val[k] = 0.f; }
                bk->cnt = 0;
            }
            bk->tag = tags[m];
            float step_lr = lrs[m] / powf((float)bk->cnt + count_c0, count_pow);
            if (bk->cnt < 65535) bk->cnt++;

            int y_slot = -1;
            int empty_slot = -1;
            int min_slot = 0;
            float min_abs = 1e30f;

            for (int k = 0; k < K; k++) {
                int16_t tk = bk->tok[k];
                if (tk < 0) {
                    if (empty_slot < 0) empty_slot = k;
                    continue;
                }
                if (tk == y) y_slot = k;
                // Find prob of tk in cand list
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


def compile_c_ext():
    c_path = os.path.join(SCRATCH, "online_hash.c")
    so_path = os.path.join(SCRATCH, "online_hash.so")
    with open(c_path, "w") as f:
        f.write(C_SRC)
    subprocess.check_call(["gcc", "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC", c_path, "-o", so_path, "-lm"])
    lib = ctypes.CDLL(so_path)
    lib.run_sparse_bucket.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
        ctypes.c_void_p,
    ]
    return lib


def bench_design1_dense_engram(H_mm, Z_np, mp_np, mb_np, ctx_toks, W_eff_np, b_head_np, bpt_global,
                               orders=(1, 2, 3, 4), bits=(18, 21, 21, 21), dim=64,
                               chunk_s=64, lr=0.25, decay=1e-4, count_pow=0.25, count_c0=2.0):
    """Design 1: Dense low-rank online hash embeddings in top-d SVD subspace of W_eff (or full d=256)."""
    t0 = time.time()
    N = len(Z_np)
    ctx_pad = len(ctx_toks) - N
    W_t = torch.from_numpy(W_eff_np).cuda()  # [4096, 256]
    b_t = torch.from_numpy(b_head_np).cuda()  # [4096]

    if dim < 256:
        # Project onto top-dim singular vectors of W_eff (fixed, zero extra archive cost since W_eff is shipped!)
        U, S, Vh = torch.linalg.svd(W_t, full_matrices=False)
        P = Vh[:dim, :].contiguous()  # [dim, 256]
        W_sub = (W_t @ P.t()).contiguous()  # [4096, dim]
    else:
        W_sub = W_t.contiguous()  # [4096, 256]

    # Allocate zero-initialized tables on GPU (float16) + visit counters (int32)
    M = len(orders)
    tables = [torch.zeros((1 << b, dim), dtype=torch.float16, device="cuda") for b in bits]
    counts = [torch.zeros((1 << b,), dtype=torch.float32, device="cuda") for b in bits]
    ram_mb = sum(t.numel() * 2 + c.numel() * 4 for t, c in zip(tables, counts)) / 1e6

    # Precompute causal hashes on CPU (numpy vectorized, fast!)
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

    # Stream in macro-blocks of 32,768 tokens to GPU, updating every chunk_s tokens
    macro = 32768
    for a in range(0, N, macro):
        b_end = min(a + macro, N)
        H_blk = torch.from_numpy(np.array(H_mm[a:b_end])).cuda().float()
        mp_blk = torch.from_numpy(mp_np[a:b_end]).cuda().long()
        mb_blk = torch.from_numpy(mb_np[a:b_end]).cuda().float()
        y_blk = torch.from_numpy(y_all[a:b_end]).cuda()
        h_blks = [torch.from_numpy(hashes[m][a:b_end]).cuda() for m in range(M)]

        # Reconstruct base logits for macro-block once
        lg_base = H_blk @ W_t.t() + b_t
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

            # Exact gradient w.r.t. e_sum: W_sub^T (p - e_y)
            p = logp.exp()
            p.scatter_add_(1, y_s[:, None], torch.full((s1 - s0, 1), -1.0, device="cuda"))
            g = p @ W_sub  # [chunk_s, dim]

            # Sparse causal update
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
    H, Z, base_nll, mp, mb, ctx_toks, W_eff, b_head, bpt = build_or_load_cache()
    N = len(Z)
    ctx_pad = len(ctx_toks) - N
    base_bpb = base_nll.mean() / math.log(2) / bpt
    base_bpb_q1 = base_nll[: N // 4].mean() / math.log(2) / bpt
    base_bpb_last = base_nll[N // 2 :].mean() / math.log(2) / bpt
    print(f"\n=== BASELINE (frozen {CKPT}, {N:,} toks = {N * bpt / 1e6:.1f} MB) ===")
    print(f"  Full stream bpb: {base_bpb:.4f} ({base_bpb * 125:.2f} MB equiv) | Q1: {base_bpb_q1:.4f} | 2nd half: {base_bpb_last:.4f}")

    # --- Design 2: Exact S=1 Sparse Top-K Online Logit Bucket (C) ---
    lib = compile_c_ext()
    print("\n=== DESIGN 2: SPARSE TOP-K ONLINE LOGIT BUCKET (EXACT S=1 TOKEN-BY-TOKEN IN C) ===")
    H_contig = np.ascontiguousarray(H)
    Z_c = np.ascontiguousarray(Z, dtype=np.float32)
    mp_c = np.ascontiguousarray(mp, dtype=np.int16)
    mb_c = np.ascontiguousarray(mb, dtype=np.float16)
    W_c = np.ascontiguousarray(W_eff, dtype=np.float32)
    b_c = np.ascontiguousarray(b_head, dtype=np.float32)
    ctx_c = np.ascontiguousarray(ctx_toks, dtype=np.int16)
    out_nll = np.empty(N, dtype=np.float32)

    # Sanity check: with lr=0, both Design 2 (C) and Design 1 (GPU) must match base_nll exactly!
    ord0 = np.array([1], dtype=np.int32)
    bit0 = np.array([14], dtype=np.int32)
    lr0 = np.array([0.0], dtype=np.float32)
    lib.run_sparse_bucket(
        min(N, 65536), ctx_pad,
        ctx_c.ctypes.data, H_contig.ctypes.data, Z_c.ctypes.data,
        mp_c.ctypes.data, mb_c.ctypes.data, W_c.ctypes.data, b_c.ctypes.data,
        1, ord0.ctypes.data, bit0.ctypes.data, lr0.ctypes.data,
        8, 0.0, 0.0, 1.0, 1, out_nll.ctypes.data,
    )
    err_c = float(np.abs(out_nll[:65536] - base_nll[:65536]).max())
    print(f"  [Sanity check] C sparse-delta max NLL error at lr=0: {err_c:.6f} (mean {np.abs(out_nll[:65536]-base_nll[:65536]).mean():.6f})")
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
        ram_gb = sum((1 << b) * 52 for b in bits) / 1e9  # sizeof(Bucket) ~ 52B
        ord_a = np.array(orders, dtype=np.int32)
        bit_a = np.array(bits, dtype=np.int32)
        lr_a = np.array(lrs, dtype=np.float32)
        t0 = time.time()
        lib.run_sparse_bucket(
            N, ctx_pad,
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

    # --- Design 1: Dense Low-Rank Online Engram-Head (GPU) ---
    print("\n=== DESIGN 1: DENSE ONLINE ENGRAM-HEAD (GPU, EXACT CLOSED-FORM GRAD) ===")
    bpb0, _, _, _, nll0 = bench_design1_dense_engram(
        H, Z, mp, mb, ctx_toks, W_eff, b_head, bpt,
        orders=(1,), bits=(14,), dim=256, chunk_s=256, lr=0.0, decay=0.0,
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
            H, Z, mp, mb, ctx_toks, W_eff, b_head, bpt,
            orders=orders, bits=bits, dim=dim, chunk_s=chunk_s, lr=lr, decay=dec,
        )
        print(f"  {name:38s} | RAM={ram_mb/1000:.2f}GB | {dt:5.1f}s | bpb={bpb_all:.4f} ({bpb_all-base_bpb:+.4f}, {(bpb_all-base_bpb)*125:+.2f}MB) | 2ndH={bpb_last:.4f} ({bpb_last-base_bpb_last:+.4f})")


if __name__ == "__main__":
    main()

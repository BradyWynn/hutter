"""Causal match features over a byte or token stream.

Two modes are supported (selected via MATCH_MODE env var, default "v2"):

  - v2 ("pool_8w24_hyb1", default):
      Multi-scale 8-way associative hash tables (TABLE_BITS_V2=24, 8 slots/bucket = 32B/bucket,
      ~3.2 GB total RAM for 6 scales) + global and per-scale active-match continuation.
      At each position i, all verified candidates (L >= k) across all scales' 8 slots and active
      cursors are pooled and deduplicated by predicted token (keeping max L, tie-break most recent):
        - Col 0: Global longest (active-continued) match across all scales
        - Col 1: Most recent valid match (L >= min(ks)) — gives explicit recency + col0/col1 agree
        - Col 2..M-1: Next distinct candidates ranked by (L desc, j desc)
      On enwik9.split4096 this lifts 6-column oracle coverage from 45.4% -> 58.0% (+12.6%!) and
      best-match accuracy from 39.80% -> 40.22% at 0 extra archive bytes and ~3-5 min single-core CPU.
      Built via a tiny compiled C shared library (exact sequential decoder semantics, O(1)/token).

  - v1 ("legacy_1w26"):
      Original GPU-vectorized 1-way direct-mapped table (TABLE_BITS=26, 1.6 GB RAM), one independent
      column per context length k.

In both modes, every prediction is strictly causal (costs 0 archive bits) and encodes:
  code = 1 + LEN_BIN[min(L, LCAP)] * N_DIST + dist_bucket(i - j)   (0 = no match)
"""
import ctypes
import hashlib
import math
import os
import subprocess
import tempfile
import time
import numpy as np
import torch

P = 0x100000001B3  # FNV-64 prime
M = 0x9E3779B97F4A7C15
TABLE_BITS = 26       # legacy v1 default (1-way)
TABLE_BITS_V2 = 24    # v2 default (8-way, same 3.22 GB RAM, 32B = half cache line)
WAYS_V2 = 8
LCAP = 128
DIST_EDGES = (512, 1024, 2048, 8192, 1 << 16, 1 << 20)
N_DIST = len(DIST_EDGES) + 1
N_CODE = 256  # code byte range; codes in use stay below this (checked in build)
MATCH_MODE = os.environ.get("MATCH_MODE", "v2")


def len_bins(lcap=LCAP):
    """LEN_BIN[L] for L = 0..lcap (index 0 unused)."""
    return [0] + [L - 1 if L <= 24 else 24 + int(4 * math.log2(L / 24)) for L in range(1, lcap + 1)]


def dist_bucket(d):
    return sum(d > e for e in DIST_EDGES)


def _i64(x):
    return x - (1 << 64) if x >= 1 << 63 else x


_V2_C_SRC = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t P_MUL = 0x100000001B3ULL;
static const uint64_t M_MUL = 0x9E3779B97F4A7C15ULL;

typedef struct {
    int32_t pred;
    int32_t L;
    int32_t j;
} Cand;

static inline void insert_cand(Cand *top, int *ntop, int max_k, int32_t pred, int32_t L, int32_t j) {
    for (int t = 0; t < *ntop; t++) {
        if (top[t].pred == pred) {
            if (L > top[t].L || (L == top[t].L && j > top[t].j)) {
                top[t].L = L;
                top[t].j = j;
                while (t > 0 && (top[t].L > top[t - 1].L || (top[t].L == top[t - 1].L && top[t].j > top[t - 1].j))) {
                    Cand tmp = top[t];
                    top[t] = top[t - 1];
                    top[t - 1] = tmp;
                    t--;
                }
            }
            return;
        }
    }
    int n = *ntop;
    if (n == max_k) {
        if (L < top[n - 1].L || (L == top[n - 1].L && j <= top[n - 1].j)) return;
        n--;
    }
    int pos = n;
    while (pos > 0 && (L > top[pos - 1].L || (L == top[pos - 1].L && j > top[pos - 1].j))) {
        top[pos] = top[pos - 1];
        pos--;
    }
    top[pos].pred = pred;
    top[pos].L = L;
    top[pos].j = j;
    if (*ntop < max_k) (*ntop)++;
}

static inline int calc_dist_bucket(int64_t d, const int64_t *edges, int n_edges) {
    int b = 0;
    for (int e = 0; e < n_edges; e++) b += (d > edges[e]);
    return b;
}

#define DEFINE_BUILD_V2(FN_NAME, T_DATA) \
int FN_NAME( \
    const T_DATA *data, int64_t N, \
    const int32_t *ks, int n_cols, \
    int table_bits, int ways, int lcap, \
    const int32_t *lut, const int64_t *edges, int n_edges, \
    T_DATA *out_pred, uint8_t *out_code \
) { \
    if (n_cols <= 0 || n_cols > 16) return -1; \
    const uint64_t n_bkt = 1ULL << table_bits; \
    const uint64_t shift = 64 - table_bits; \
    const int n_dist = n_edges + 1; \
    const int max_top = n_cols + 1; \
    int min_k = ks[0]; \
    for (int c = 1; c < n_cols; c++) if (ks[c] < min_k) min_k = ks[c]; \
    uint32_t *tabs[16]; \
    uint64_t pks[16], hs[16], bkts[16]; \
    int64_t act_j[16]; \
    int32_t act_L[16]; \
    for (int c = 0; c < n_cols; c++) { \
        tabs[c] = (uint32_t *)malloc(n_bkt * (size_t)ways * sizeof(uint32_t)); \
        if (!tabs[c]) { for (int j = 0; j < c; j++) free(tabs[j]); return -2; } \
        memset(tabs[c], 0xFF, n_bkt * (size_t)ways * sizeof(uint32_t)); \
        uint64_t pk = 1; \
        for (int t = 0; t < ks[c]; t++) pk *= P_MUL; \
        pks[c] = pk; \
        hs[c] = 0; \
        act_j[c] = -1; \
        act_L[c] = 0; \
    } \
    memset(out_pred, 0, (size_t)N * (size_t)n_cols * sizeof(T_DATA)); \
    memset(out_code, 0, (size_t)N * (size_t)n_cols * sizeof(uint8_t)); \
    int64_t g_act_j = -1; \
    int32_t g_act_L = 0; \
    Cand top[20]; \
    Cand out[16]; \
    for (int64_t i = 0; i < N; i++) { \
        for (int c = 0; c < n_cols; c++) { \
            int k = ks[c]; \
            if (i < k) continue; \
            if (i == k) { \
                uint64_t h = 0; \
                for (int t = 0; t < k; t++) h = h * P_MUL + (uint64_t)(int64_t)data[t]; \
                hs[c] = h; \
            } else { \
                hs[c] = hs[c] * P_MUL + (uint64_t)(int64_t)data[i - 1] - pks[c] * (uint64_t)(int64_t)data[i - 1 - k]; \
            } \
            uint64_t bkt = (hs[c] * M_MUL) >> shift; \
            bkts[c] = bkt; \
            __builtin_prefetch(tabs[c] + bkt * (size_t)ways, 1, 1); \
        } \
        if (g_act_j >= 0) { \
            if (data[g_act_j] == data[i - 1]) { g_act_j++; g_act_L++; } \
            else { g_act_j = -1; g_act_L = 0; } \
        } \
        int ntop = 0; \
        Cand rec_best = {0, 0, -1}; \
        if (g_act_j >= 0 && g_act_L >= min_k) { \
            int32_t pr = (int32_t)data[g_act_j]; \
            insert_cand(top, &ntop, max_top, pr, g_act_L, (int32_t)g_act_j); \
        } \
        T_DATA d_m1 = (i > 0) ? data[i - 1] : 0; \
        for (int c = 0; c < n_cols; c++) { \
            int k = ks[c]; \
            if (i < k) continue; \
            uint32_t *slots = tabs[c] + bkts[c] * (size_t)ways; \
            if (act_j[c] >= 0) { \
                if (data[act_j[c]] == d_m1) { act_j[c]++; act_L[c]++; } \
                else { act_j[c] = -1; act_L[c] = 0; } \
            } \
            if (act_j[c] >= 0 && act_L[c] >= k) { \
                insert_cand(top, &ntop, max_top, (int32_t)data[act_j[c]], act_L[c], (int32_t)act_j[c]); \
            } \
            T_DATA d_mk = data[i - k]; \
            int64_t best_j = -1; \
            int32_t best_L = 0; \
            for (int w = 0; w < ways; w++) { \
                uint32_t sj = slots[w]; \
                if (sj == 0xFFFFFFFFU) break; \
                if ((int64_t)sj < k) continue; \
                if (data[sj - 1] != d_m1 || data[sj - k] != d_mk) continue; \
                int max_l = ((int64_t)sj < lcap) ? (int)sj : lcap; \
                int L = 1; \
                while (L < max_l && data[i - 1 - L] == data[sj - 1 - L]) L++; \
                if (L >= k) { \
                    int32_t pr = (int32_t)data[sj]; \
                    insert_cand(top, &ntop, max_top, pr, L, (int32_t)sj); \
                    if ((int32_t)sj > rec_best.j) { \
                        rec_best.pred = pr; \
                        rec_best.L = L; \
                        rec_best.j = (int32_t)sj; \
                    } \
                    if (L > best_L) { best_L = L; best_j = (int64_t)sj; } \
                } \
            } \
            if (best_L > act_L[c]) { act_j[c] = best_j; act_L[c] = best_L; } \
            for (int w = ways - 1; w > 0; w--) slots[w] = slots[w - 1]; \
            slots[0] = (uint32_t)i; \
        } \
        if (ntop > 0 && top[0].L > g_act_L) { \
            g_act_j = top[0].j; \
            g_act_L = top[0].L; \
        } \
        if (ntop > 0) { \
            int nout = 0; \
            out[nout++] = top[0]; \
            if (n_cols >= 2) { \
                Cand r = (rec_best.j >= 0) ? rec_best : top[0]; \
                out[nout++] = r; \
                for (int t = 1; t < ntop && nout < n_cols; t++) { \
                    if (top[t].pred == r.pred && r.pred != top[0].pred) continue; \
                    out[nout++] = top[t]; \
                } \
            } \
            T_DATA *op = out_pred + i * (int64_t)n_cols; \
            uint8_t *oc = out_code + i * (int64_t)n_cols; \
            for (int c = 0; c < nout; c++) { \
                op[c] = (T_DATA)out[c].pred; \
                int capL = out[c].L > lcap ? lcap : out[c].L; \
                int db = calc_dist_bucket(i - (int64_t)out[c].j, edges, n_edges); \
                oc[c] = (uint8_t)(1 + lut[capL] * n_dist + db); \
            } \
        } \
    } \
    for (int c = 0; c < n_cols; c++) free(tabs[c]); \
    return 0; \
}

DEFINE_BUILD_V2(build_v2_i16, int16_t)
DEFINE_BUILD_V2(build_v2_u8, uint8_t)
"""

_V2_LIB = None


def _get_v2_lib():
    global _V2_LIB
    if _V2_LIB is not None:
        return _V2_LIB
    h = hashlib.sha1(_V2_C_SRC.encode()).hexdigest()[:12]
    so_path = os.path.join(tempfile.gettempdir(), f"matchfeat_v2_{h}.so")
    if not os.path.exists(so_path):
        c_path = so_path + ".c"
        tmp_so = so_path + f".{os.getpid()}.tmp.so"
        with open(c_path, "w") as f:
            f.write(_V2_C_SRC)
        subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", c_path, "-o", tmp_so])
        os.replace(tmp_so, so_path)
    lib = ctypes.CDLL(so_path)
    for fn_name, ptr_t in (("build_v2_i16", ctypes.POINTER(ctypes.c_int16)), ("build_v2_u8", ctypes.POINTER(ctypes.c_uint8))):
        fn = getattr(lib, fn_name)
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ptr_t, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
            ptr_t, ctypes.POINTER(ctypes.c_uint8),
        ]
    _V2_LIB = lib
    return lib


def build_v2(data, ks, table_bits=TABLE_BITS_V2, ways=WAYS_V2, lcap=LCAP, verbose=True):
    """Build v2 pooled 8-way + active-continuation hybrid match features via compiled C library.
    Returns CPU numpy arrays (pred, code) of shapes [N, len(ks)]."""
    assert all(k <= lcap for k in ks)
    lut_list = len_bins(lcap)
    assert 1 + max(lut_list) * N_DIST + N_DIST - 1 < N_CODE
    t0 = time.time()
    arr = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
    arr = np.ascontiguousarray(arr)
    assert arr.dtype in (np.int16, np.uint8), f"unsupported dtype {arr.dtype}"
    N = len(arr)
    n_cols = len(ks)
    pred = np.empty((N, n_cols), dtype=arr.dtype)
    code = np.empty((N, n_cols), dtype=np.uint8)
    ks_a = np.asarray(ks, dtype=np.int32)
    lut_a = np.asarray(lut_list, dtype=np.int32)
    edges_a = np.asarray(DIST_EDGES, dtype=np.int64)

    lib = _get_v2_lib()
    if arr.dtype == np.int16:
        fn = lib.build_v2_i16
        ptr_t = ctypes.POINTER(ctypes.c_int16)
    else:
        fn = lib.build_v2_u8
        ptr_t = ctypes.POINTER(ctypes.c_uint8)
    rc = fn(
        arr.ctypes.data_as(ptr_t), N,
        ks_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), n_cols,
        int(table_bits), int(ways), int(lcap),
        lut_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        edges_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), len(DIST_EDGES),
        pred.ctypes.data_as(ptr_t),
        code.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    assert rc == 0, f"build_v2 C kernel failed with rc={rc}"
    if verbose:
        dt = time.time() - t0
        chunk = 1 << 24
        u_right = 0
        for c in range(n_cols):
            n_hit = n_right = 0
            for a in range(0, N, chunk):
                has = code[a : a + chunk, c] > 0
                n_hit += int(has.sum())
                n_right += int((has & (pred[a : a + chunk, c] == arr[a : a + chunk])).sum())
            print(
                f"match v2 col {c} (k-pool {ks}): hit {n_hit / N:6.2%}, "
                f"acc {n_right / max(n_hit, 1):6.2%}, net {n_right / N:6.2%}"
            )
        for a in range(0, N, chunk):
            b = min(a + chunk, N)
            has = code[a:b] > 0
            eq = (pred[a:b] == arr[a:b, None]) & has
            u_right += int(eq.any(axis=1).sum())
        print(f"match v2 built in {dt:.1f}s ({ways}w@{table_bits}b) | UNION right: {u_right / N:6.3%}")
    return pred, code


def build_v1(data, ks, table_bits=TABLE_BITS, lcap=LCAP, chunk=1 << 25, verbose=True):
    """Legacy v1 GPU-vectorized 1-way direct-mapped match features."""
    assert all(k <= lcap for k in ks)
    dev = data.device
    lut = torch.tensor(len_bins(lcap), device=dev)
    assert 1 + int(lut.max()) * N_DIST + N_DIST - 1 < N_CODE
    edges = torch.tensor(DIST_EDGES, device=dev)
    N = len(data)
    shift, mask = 64 - table_bits, (1 << table_bits) - 1
    pbits = max(1, (chunk - 1).bit_length())
    assert table_bits + pbits <= 62
    pred = torch.zeros(N, len(ks), dtype=data.dtype, device=dev)
    code = torch.zeros(N, len(ks), dtype=torch.uint8, device=dev)
    for c, k in enumerate(ks):
        t0 = time.time()
        table = torch.full((1 << table_bits,), -1, dtype=torch.long, device=dev)
        for a in range(k, N, chunk):
            b = min(a + chunk, N)
            i = torch.arange(a, b, device=dev)
            h = torch.zeros(b - a, dtype=torch.long, device=dev)
            for t in range(k):
                h = h * P + data[a - k + t : b - k + t].long()
            bucket = ((h * _i64(M)) >> shift) & mask
            del h
            key = (bucket << pbits) | (i - a)
            del bucket
            key = key.sort().values
            sb, sp = key >> pbits, key & ((1 << pbits) - 1)
            del key
            first = torch.ones_like(sb, dtype=torch.bool)
            first[1:] = sb[1:] != sb[:-1]
            prev_sorted = torch.empty_like(sp)
            prev_sorted[1:] = sp[:-1] + a
            prev_sorted[first] = table[sb[first]]
            last = torch.ones_like(first)
            last[:-1] = first[1:]
            table[sb[last]] = sp[last] + a
            j = torch.empty_like(prev_sorted)
            j[sp] = prev_sorted
            del sb, sp, first, last, prev_sorted

            act = (j >= 0).nonzero().squeeze(1)
            L = torch.zeros(b - a, dtype=torch.long, device=dev)
            for t in range(1, lcap + 1):
                if len(act) == 0:
                    break
                ja = j[act] - t
                ok = (ja >= 0) & (data[act + a - t] == data[ja.clamp_min(0)])
                act = act[ok]
                L[act] += 1
            hit = L >= k
            pred[a:b, c] = torch.where(hit, data[j.clamp_min(0)], 0)
            db = ((i - j)[:, None] > edges).sum(1)
            code[a:b, c] = torch.where(hit, 1 + lut[L] * N_DIST + db, 0).to(torch.uint8)
            del db
            del i, j, L, hit
        if verbose:
            n_hit = n_right = 0
            for a in range(0, N, chunk):
                has = code[a : a + chunk, c] > 0
                n_hit += has.sum().item()
                n_right += (has & (pred[a : a + chunk, c] == data[a : a + chunk])).sum().item()
            print(
                f"match k={k:2d}: {time.time() - t0:5.1f}s | hit on {n_hit / N:6.1%} of bytes,"
                f" correct on {n_right / max(n_hit, 1):6.1%} of those"
            )
    return pred, code


def build(data, ks, table_bits=None, lcap=LCAP, mode=None, ways=WAYS_V2, verbose=True):
    mode = mode or MATCH_MODE
    if mode == "v1":
        tb = TABLE_BITS if table_bits is None else table_bits
        return build_v1(data, ks, table_bits=tb, lcap=lcap, verbose=verbose)
    tb = TABLE_BITS_V2 if table_bits is None else table_bits
    return build_v2(data, ks, table_bits=tb, ways=ways, lcap=lcap, verbose=verbose)


def cached(data, ks, prefix, table_bits=None, lcap=LCAP, mode=None, ways=WAYS_V2):
    """build(), stored on disk as two .npy files and returned as read-only numpy memmaps."""
    mode = mode or MATCH_MODE
    if mode == "v1":
        tb = TABLE_BITS if table_bits is None else table_bits
        key = repr((len(data), str(data.dtype), tuple(ks), tb, lcap, P, M, DIST_EDGES, N_CODE))
    else:
        tb = TABLE_BITS_V2 if table_bits is None else table_bits
        key = repr(("v2_pool_hyb1", len(data), str(data.dtype), tuple(ks), tb, ways, lcap, P, M, DIST_EDGES, N_CODE))
    stem = f"{prefix}.match_{hashlib.sha1(key.encode()).hexdigest()[:10]}"
    paths = (stem + ".pred.npy", stem + ".code.npy")
    if not all(os.path.exists(p) for p in paths):
        tables = build(data, ks, table_bits=tb, lcap=lcap, mode=mode, ways=ways)
        for t, path in zip(tables, paths):
            tmp = path + ".tmp"
            if isinstance(t, torch.Tensor):
                dt = t[:1].cpu().numpy().dtype
                out = np.lib.format.open_memmap(tmp, mode="w+", dtype=dt, shape=tuple(t.shape))
                for a in range(0, len(t), 1 << 26):
                    out[a : a + (1 << 26)] = t[a : a + (1 << 26)].cpu().numpy()
            else:
                out = np.lib.format.open_memmap(tmp, mode="w+", dtype=t.dtype, shape=t.shape)
                for a in range(0, len(t), 1 << 26):
                    out[a : a + (1 << 26)] = t[a : a + (1 << 26)]
            out.flush()
            del out
            os.replace(tmp, path)
        del tables
        print(f"match tables ({mode}) written to {stem}.{{pred,code}}.npy")
    return tuple(np.load(p, mmap_mode="r") for p in paths)


def reference_v2(data, ks, table_bits=TABLE_BITS_V2, ways=WAYS_V2, lcap=LCAP):
    """Exact pure-Python token-by-token decoder loop for v2, for verifying build_v2()."""
    arr = data.cpu().tolist() if isinstance(data, torch.Tensor) else list(data)
    N = len(arr)
    n_cols = len(ks)
    min_k = min(ks)
    lut = len_bins(lcap)
    mask64 = (1 << 64) - 1
    shift = 64 - table_bits

    tabs = [{} for _ in range(n_cols)]
    act_j = [-1] * n_cols
    act_L = [0] * n_cols
    g_act_j, g_act_L = -1, 0

    pr_out = [[0] * n_cols for _ in range(N)]
    cd_out = [[0] * n_cols for _ in range(N)]

    for i in range(N):
        if g_act_j >= 0:
            if arr[g_act_j] == arr[i - 1]:
                g_act_j += 1
                g_act_L += 1
            else:
                g_act_j, g_act_L = -1, 0

        cands = {}  # pred -> (L, j)
        rec_best = None  # (j, L, pred)

        def add_cand(p, L, j):
            cur = cands.get(p)
            if cur is None or (L, j) > cur:
                cands[p] = (L, j)

        if g_act_j >= 0 and g_act_L >= min_k:
            add_cand(arr[g_act_j], g_act_L, g_act_j)

        for c, k in enumerate(ks):
            if i < k:
                continue
            h = 0
            for t in range(k):
                h = (h * P + (arr[i - k + t] & mask64)) & mask64
            bkt = ((h * M) & mask64) >> shift
            slots = tabs[c].get(bkt, [])

            if act_j[c] >= 0:
                if arr[act_j[c]] == arr[i - 1]:
                    act_j[c] += 1
                    act_L[c] += 1
                else:
                    act_j[c], act_L[c] = -1, 0
            if act_j[c] >= 0 and act_L[c] >= k:
                add_cand(arr[act_j[c]], act_L[c], act_j[c])

            best_j, best_L = -1, 0
            for sj in slots:
                L = 0
                while L < lcap and sj - L - 1 >= 0 and arr[i - L - 1] == arr[sj - L - 1]:
                    L += 1
                if L >= k:
                    p = arr[sj]
                    add_cand(p, L, sj)
                    if rec_best is None or sj > rec_best[0]:
                        rec_best = (sj, L, p)
                    if L > best_L:
                        best_L, best_j = L, sj
            if best_L > act_L[c]:
                act_j[c], act_L[c] = best_j, best_L

            tabs[c][bkt] = ([i] + slots)[:ways]

        ranked = sorted(((L, j, p) for p, (L, j) in cands.items()), reverse=True)
        if ranked and ranked[0][0] > g_act_L:
            g_act_L, g_act_j = ranked[0][0], ranked[0][1]

        if ranked:
            out = [ranked[0]]
            if n_cols >= 2:
                r = (rec_best[1], rec_best[0], rec_best[2]) if rec_best is not None else ranked[0]
                out.append(r)
                for item in ranked[1:]:
                    if len(out) >= n_cols:
                        break
                    if item[2] == r[2] and r[2] != ranked[0][2]:
                        continue
                    out.append(item)
            for c, (L, j, p) in enumerate(out):
                pr_out[i][c] = p
                cd_out[i][c] = 1 + lut[min(L, lcap)] * N_DIST + dist_bucket(i - j)
    return pr_out, cd_out


def reference(data, ks, table_bits=TABLE_BITS, lcap=LCAP):
    """Legacy v1 byte-by-byte decoder loop, for checking build_v1() on small inputs."""
    data = data.cpu().tolist()
    lut = len_bins(lcap)
    out = []
    for k in ks:
        table, pr, ln = {}, [0] * len(data), [0] * len(data)
        for i in range(k, len(data)):
            h = 0
            for t in range(k):
                h = (h * P + data[i - k + t]) & ((1 << 64) - 1)
            bkt = ((h * M) & ((1 << 64) - 1)) >> (64 - table_bits)
            j = table.get(bkt, -1)
            table[bkt] = i
            if j < 0:
                continue
            L = 0
            while L < lcap and j - L - 1 >= 0 and data[i - L - 1] == data[j - L - 1]:
                L += 1
            if L >= k:
                pr[i], ln[i] = data[j], 1 + lut[L] * N_DIST + dist_bucket(i - j)
        out.append((pr, ln))
    return out


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    words = [bytes(torch.randint(97, 123, (int(n),), generator=g).tolist()) for n in torch.randint(2, 9, (300,), generator=g)]
    text = b" ".join(words[int(w)] for w in torch.randint(0, 300, (8000,), generator=g))

    # 1. Verify legacy v1 GPU/CPU build
    data_u8 = torch.frombuffer(bytearray(text[:40000]), dtype=torch.uint8)
    ks_v1, bits = (3, 6, 16), 10
    pred, length = build_v1(data_u8, ks_v1, table_bits=bits, lcap=40, chunk=1000, verbose=False)
    for c, (pr, ln) in enumerate(reference(data_u8, ks_v1, table_bits=bits, lcap=40)):
        assert pred[:, c].tolist() == pr and length[:, c].tolist() == ln, f"v1 mismatch at k={ks_v1[c]}"
    print("build_v1() matches reference() bit-for-bit")

    # 2. Verify v2 C builder against pure-Python reference_v2 on BOTH uint8 and int16 (with heavy collisions)
    ks_v2 = (2, 3, 5, 8, 13, 27)
    for name, arr in (
        ("uint8", data_u8[:20000]),
        ("int16", torch.randint(0, 250, (20000,), generator=g, dtype=torch.int16)),
    ):
        p2, c2 = build_v2(arr, ks_v2, table_bits=9, ways=8, lcap=40, verbose=False)
        rp2, rc2 = reference_v2(arr, ks_v2, table_bits=9, ways=8, lcap=40)
        assert p2.tolist() == rp2 and c2.tolist() == rc2, f"v2 mismatch on {name}!"
    print("build_v2() matches reference_v2() bit-for-bit on both uint8 and int16!")

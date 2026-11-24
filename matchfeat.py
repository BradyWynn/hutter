"""Causal match features over a byte or token stream, built on the GPU in chunks.

The stream is any integer tensor (uint8 bytes, or int16 token ids); "byte" below means one
element of it, and lengths and distances are counted in elements.

For each context length k and each position i (the byte being predicted is data[i]):
  bucket = hash(data[i-k:i])          top TABLE_BITS bits of a 64-bit polynomial hash
  j      = table[bucket]; table[bucket] = i    (most recent earlier position in that bucket)
  L      = length of the common suffix of data[:i] and data[:j], capped at LCAP
  if j exists and L >= k:  pred = data[j], code = 1 + LEN_BIN[L] * N_DIST + dist_bucket(i - j)
  else (no match / collision): pred = code = 0
LEN_BIN is exact for L <= 24 and quarter-octave above; dist_bucket counts how many of
DIST_EDGES the distance exceeds. With a 1024-byte context, <= 512 is always inside the
model's window, <= 1024 sometimes, and beyond it only the match feature can see the source.

This is exactly a sequential hash table a decoder can run byte by byte, collisions and
overwrites included, so the features cost no bits. A C decoder needs, per k:
  uint64 h = 0; for t in 0..k-1: h = h * P + data[i-k+t];   bucket = (h * M) >> (64 - TABLE_BITS)
with uint64 wraparound, and a uint32 table of 2^TABLE_BITS entries initialised to "empty".

The build is vectorised: within a chunk, sorting (bucket, position) puts each position next to
the previous one in its bucket; the first one of each bucket in the chunk reads the table left
by earlier chunks, and the last one writes it.
"""
import math
import time
import torch

P = 0x100000001B3  # FNV-64 prime
M = 0x9E3779B97F4A7C15
TABLE_BITS = 26
LCAP = 128
DIST_EDGES = (512, 1024, 2048, 8192, 1 << 16, 1 << 20)
N_DIST = len(DIST_EDGES) + 1
N_CODE = 256  # code byte range; codes in use stay below this (checked in build)


def len_bins(lcap=LCAP):
    """LEN_BIN[L] for L = 0..lcap (index 0 unused)."""
    return [0] + [L - 1 if L <= 24 else 24 + int(4 * math.log2(L / 24)) for L in range(1, lcap + 1)]


def dist_bucket(d):
    return sum(d > e for e in DIST_EDGES)


def _i64(x):
    return x - (1 << 64) if x >= 1 << 63 else x


def build(data, ks, table_bits=TABLE_BITS, lcap=LCAP, chunk=1 << 25, verbose=True):
    """data: uint8 or int16 tensor (any device). Returns (pred, code), [N, len(ks)] each:
    pred in data's dtype, code uint8."""
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

            # verify and measure the match length, shrinking to the still-matching subset
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
            for a in range(0, N, chunk):  # chunked: a full-length bool sum upcasts to int64
                has = code[a : a + chunk, c] > 0
                n_hit += has.sum().item()
                n_right += (has & (pred[a : a + chunk, c] == data[a : a + chunk])).sum().item()
            print(
                f"match k={k:2d}: {time.time() - t0:5.1f}s | hit on {n_hit / N:6.1%} of bytes,"
                f" correct on {n_right / max(n_hit, 1):6.1%} of those"
            )
    return pred, code


def cached(data, ks, prefix, table_bits=TABLE_BITS, lcap=LCAP):
    """build(), stored on disk as two .npy files and returned as read-only numpy memmaps, so
    the [N, len(ks)] tables (~8 GB for enwik9 with 4 columns) stay off the GPU. The file name
    encodes every setting that changes the output; delete the files after editing build()."""
    import hashlib
    import os
    import numpy as np

    key = repr((len(data), str(data.dtype), tuple(ks), table_bits, lcap, P, M, DIST_EDGES, N_CODE))
    stem = f"{prefix}.match_{hashlib.sha1(key.encode()).hexdigest()[:10]}"
    paths = (stem + ".pred.npy", stem + ".code.npy")
    if not all(os.path.exists(p) for p in paths):
        tables = build(data, ks, table_bits=table_bits, lcap=lcap)
        for t, path in zip(tables, paths):
            tmp = path + ".tmp"
            out = np.lib.format.open_memmap(tmp, mode="w+", dtype=t[:1].cpu().numpy().dtype, shape=tuple(t.shape))
            for a in range(0, len(t), 1 << 26):
                out[a : a + (1 << 26)] = t[a : a + (1 << 26)].cpu().numpy()
            out.flush()
            del out
            os.replace(tmp, path)
        del tables
        print(f"match tables written to {stem}.{{pred,code}}.npy")
    return tuple(np.load(p, mmap_mode="r") for p in paths)


def reference(data, ks, table_bits=TABLE_BITS, lcap=LCAP):
    """Byte-by-byte decoder-style loop, for checking build() on small inputs."""
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
    # small CPU check against the reference loop: tiny table and chunks force collisions,
    # overwrites and matches that cross chunk boundaries
    g = torch.Generator().manual_seed(0)
    words = [bytes(torch.randint(97, 123, (int(n),), generator=g).tolist()) for n in torch.randint(2, 9, (300,), generator=g)]
    text = b" ".join(words[int(w)] for w in torch.randint(0, 300, (8000,), generator=g))
    data = torch.frombuffer(bytearray(text[:40000]), dtype=torch.uint8)
    ks, bits = (3, 6, 16), 10
    pred, length = build(data, ks, table_bits=bits, lcap=40, chunk=1000)
    for c, (pr, ln) in enumerate(reference(data, ks, table_bits=bits, lcap=40)):
        assert pred[:, c].tolist() == pr and length[:, c].tolist() == ln, f"mismatch at k={ks[c]}"
    print("build() matches the reference loop")

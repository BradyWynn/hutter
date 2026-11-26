"""Byte-level BPE for enwik9, with or without GPT-2 style pre-splitting.

  python bpe_tok.py nosplit 1024      -> enwik9.nosplit1024.json, enwik9.nosplit1024.i16
  python bpe_tok.py split 1024        -> enwik9.split1024.json,   enwik9.split1024.i16

Bytes are read as latin-1, so every byte is exactly one character and token ids 0-255 are the
raw bytes. The file is cut into chunks that end at the first newline at least CHUNK bytes in;
merges never cross a chunk edge, and a decoder must cut the same way. "split" additionally
applies the GPT-2 regex inside each chunk (on bytes, not UTF-8 characters, which only matters
for the ~0.4% of non-ASCII bytes). Merges are trained on the first TRAIN_MB of the file.

The .i16 file holds the token ids as int16 (vocab must stay below 32768); the .json file holds
the merges in order, which is all a decoder needs to rebuild the vocab.
"""
import json, sys, time
import numpy as np
from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers

CHUNK = 1 << 16
CHUNK_NOSPLIT = 4096
TRAIN_MB = 64
TRAIN_MB_NOSPLIT = 24
GPT2 = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
ALPHABET = [chr(b) for b in range(256)]


def chunks(text, chunk_size=CHUNK):
    i = 0
    while i < len(text):
        j = text.find("\n", i + chunk_size)
        j = len(text) if j < 0 else j + 1
        yield text[i:j]
        i = j


def pre_tokenizer(scheme):
    return pre_tokenizers.Split(Regex(GPT2), behavior="isolated") if scheme == "split" else None


def load(prefix):
    """Rebuild the tokenizer from a saved .json (merges in order)."""
    meta = json.load(open(prefix + ".json"))
    merges = [tuple(m) for m in meta["merges"]]
    vocab = {c: i for i, c in enumerate(ALPHABET)}
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    if (pt := pre_tokenizer(meta["scheme"])) is not None:
        tok.pre_tokenizer = pt
    return tok, vocab


def main(scheme, vocab_size, path="enwik9"):
    assert scheme in ("split", "nosplit") and 256 < vocab_size < 32768
    prefix = f"{path}.{scheme}{vocab_size}"
    text = open(path, "rb").read().decode("latin-1")

    t0 = time.time()
    tok = Tokenizer(models.BPE())
    if (pt := pre_tokenizer(scheme)) is not None:
        tok.pre_tokenizer = pt
    tr = trainers.BpeTrainer(
        vocab_size=vocab_size, initial_alphabet=ALPHABET, limit_alphabet=256,
        min_frequency=2, show_progress=False,
    )
    chunk_size = CHUNK_NOSPLIT if scheme == "nosplit" else CHUNK
    all_chunks = list(chunks(text, chunk_size))
    if scheme == "split":
        # Full-file BPE training for exact global optimal merges (~30-45s on 1 GB in parallel Rust)
        sample_chunks = all_chunks
        sample_mb = len(text) / (1 << 20)
        sample_mode = "full"
    else:
        n_sample = max(1, min(len(all_chunks), (TRAIN_MB_NOSPLIT << 20) // chunk_size))
        sample_idx = np.linspace(0, len(all_chunks) - 1, n_sample, dtype=np.int64)
        sample_chunks = [all_chunks[int(i)] for i in sample_idx]
        sample_mb = sum(len(c) for c in sample_chunks) / (1 << 20)
        sample_mode = "uniform"
    tok.train_from_iterator(sample_chunks, tr)
    merges = json.loads(tok.to_str())["model"]["merges"]
    merges = [m.split(" ", 1) if isinstance(m, str) else list(m) for m in merges]
    json.dump(
        dict(scheme=scheme, chunk=chunk_size, train_mb=round(sample_mb, 2), train_sample=sample_mode, merges=merges),
        open(prefix + ".json", "w"),
    )
    print(f"trained {len(merges)} merges on {sample_mb:.1f} MB ({sample_mode}, {len(sample_chunks)}/{len(all_chunks)} chunks) in {time.time() - t0:.0f}s", flush=True)

    tok, vocab = load(prefix)
    assert len(vocab) == vocab_size, (len(vocab), vocab_size)
    lens = np.zeros(vocab_size, dtype=np.int64)
    for s, i in vocab.items():
        lens[i] = len(s)
    by_id = [None] * vocab_size
    for s, i in vocab.items():
        by_id[i] = s

    t0 = time.time()
    n_tok, batch, checked = 0, [], 0
    with open(prefix + ".i16", "wb") as out:
        def flush():
            nonlocal n_tok, checked
            ids = [e.ids for e in tok.encode_batch(batch)]
            arr = np.fromiter((i for x in ids for i in x), dtype=np.int16)
            if checked < 3:  # exact round trip on the first few batches
                assert "".join(by_id[i] for i in arr) == "".join(batch), "round trip failed"
                checked += 1
            assert lens[arr].sum() == sum(len(c) for c in batch), "decoded length mismatch"
            arr.tofile(out)
            n_tok += len(arr)
            batch.clear()

        size = 0
        for c in all_chunks:
            batch.append(c)
            size += len(c)
            if size >= 32 << 20:
                flush()
                size = 0
        if batch:
            flush()
    print(
        f"encoded {len(text):,} bytes -> {n_tok:,} tokens ({len(text) / n_tok:.3f} bytes/token)"
        f" in {time.time() - t0:.0f}s | wrote {prefix}.i16, {prefix}.json"
    )


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "enwik9")

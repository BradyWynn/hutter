"""Exact token coverage and reversible BPE dictionary validation for size estimates."""
from functools import lru_cache
import os
import numpy as np
import torch


def load_tokens(path='5k_enwik9'):
    if os.path.getsize(path) % 2:
        raise ValueError('Token file must contain whole uint16 values')
    return np.memmap(path, dtype='<u2', mode='r')


class DataLoaderLite:
    """Token zero is sent separately; pos is the next target, never wraps."""
    def __init__(self, B, T, tokens=None):
        if B <= 0 or T <= 0:
            raise ValueError('B and T must be positive')
        self.B, self.T = B, T
        self.tokens = load_tokens() if tokens is None else tokens
        self.n = len(self.tokens)
        if not self.n:
            raise ValueError('Empty token stream')
        self.pos = 1

    def next_batch(self, max_tokens=None):
        remaining = self.n - self.pos
        if max_tokens is not None:
            remaining = min(remaining, max_tokens)
        if remaining <= 0:
            raise StopIteration
        # Complete rows first, then a separate short row: no padding or wraparound.
        rows = min(self.B, remaining // self.T)
        length = self.T if rows else remaining
        rows = max(rows, 1)
        count = rows * length
        starts = self.pos - 1 + np.arange(rows) * length
        idx = starts[:, None] + np.arange(length)
        x = torch.from_numpy(self.tokens[idx].astype(np.int64))
        y = torch.from_numpy(self.tokens[idx + 1].astype(np.int64))
        self.pos += count
        return x, y


def doc_start_tokens(first_path='first', second_path='second', vocab_size=10256, marker=b'<page>'):
    """Token ids whose bytes contain `marker`. In enwik9 every <page> tag lands inside exactly
    one token (which also carries the end of the previous page), so these ids mark the first
    token of each new document."""
    merges = vocab_size - 256
    first = np.fromfile(first_path, dtype='<i4', count=merges)
    second = np.fromfile(second_path, dtype='<i4', count=merges)
    pieces = [bytes([i]) for i in range(256)]
    for a, b in zip(first, second):
        pieces.append(pieces[a] + pieces[b])
    return [i for i, piece in enumerate(pieces) if marker in piece]


def validate_dictionary(token_path, source_path, first_path, second_path, vocab_size):
    paths = (token_path, source_path, first_path, second_path)
    signature = tuple((os.path.abspath(p), os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in paths)
    return _validate_dictionary(signature, vocab_size)


@lru_cache(maxsize=8)
def _validate_dictionary(signature, vocab_size):
    token_path, source_path, first_path, second_path = [s[0] for s in signature]
    if not 256 <= vocab_size <= 65536:
        raise ValueError('BPE vocabulary must be between 256 and 65536')
    merges = vocab_size - 256
    for path in (first_path, second_path):
        if os.path.getsize(path) % 4 or os.path.getsize(path) < merges * 4:
            raise ValueError('Dictionary files must contain enough int32 merge entries')
    first = np.fromfile(first_path, dtype='<i4', count=merges)
    second = np.fromfile(second_path, dtype='<i4', count=merges)
    pieces = [bytes([i]) for i in range(256)]
    for i, (a, b) in enumerate(zip(first, second)):
        if not (0 <= a < 256 + i and 0 <= b < 256 + i):
            raise ValueError(f'Invalid BPE merge {i}')
        pieces.append(pieces[a] + pieces[b])
    tokens = load_tokens(token_path)
    if not len(tokens):
        raise ValueError('Empty token stream')
    decoded_bytes = 0
    with open(source_path, 'rb') as source:
        for start in range(0, len(tokens), 65536):
            chunk = tokens[start:start + 65536]
            if int(chunk.max()) >= vocab_size:
                raise ValueError('Token ID outside configured vocabulary')
            decoded = b''.join(map(pieces.__getitem__, chunk.tolist()))
            if source.read(len(decoded)) != decoded:
                raise ValueError(f'Dictionary does not reconstruct source near token {start}')
            decoded_bytes += len(decoded)
        if source.read(1):
            raise ValueError('Token stream does not reconstruct the entire source')
    # Defined representation: two little-endian int32 arrays, truncated to used vocabulary.
    return {'dictionary_bytes': merges * 8, 'source_bytes': decoded_bytes,
            'num_tokens': len(tokens)}

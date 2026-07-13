import numpy as np
from tqdm import tqdm

with open('enwik9', 'rb') as f:
    data = f.read()
tokens = np.frombuffer(data, dtype=np.uint8)

# unconditional (unigram) probability of each token id
counts = np.bincount(tokens)
probs = counts / len(tokens)

# avoid log(0) for any token id that never appears (shouldn't matter since
# such tokens can never be sampled as targets, but guard anyway)
eps = np.finfo(np.float64).tiny
log_probs = np.log(probs + eps)

ctx_len = 128
n_samples = 100000

scores = []
for _ in tqdm(range(n_samples)):
    i = np.random.randint(0, len(tokens) - ctx_len)
    y = tokens[i + ctx_len]
    scores.append(-log_probs[y])

t = np.asarray(scores)
mean = t.mean()
se = t.std(ddof=1) / np.sqrt(len(t))
lo, hi = mean - 1.96 * se, mean + 1.96 * se
print(f"[sampled]   {mean:.3f} nats  [{lo:.3f}, {hi:.3f}]")

# closed-form check: expected loss under the unigram model is just its entropy
entropy = -(probs[probs > 0] * log_probs[probs > 0]).sum()
print(f"[closed-form entropy] {entropy:.3f} nats")
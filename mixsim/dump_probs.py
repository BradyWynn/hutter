"""Run a frozen checkpoint over the whole token stream and save, per target position:
  logp[t]  = log p_T(tokens[t] | context)   float32 (nats)
  pmax[t]  = max_v p_T(v | context)          float16 (model confidence, a mixer feature)
  ent[t]   = entropy of p_T(. | context)     float16 (nats)
Index 0 has no prediction (token zero is sent raw) and is stored as NaN.

Context: overlapping windows of length T with stride T/2. Each window scores only its last
T/2 targets, so every token is predicted from at least T/2 tokens of left context --
roughly what a streaming coder with a T-token KV window sees. The first window scores
everything (positions there simply have all the context that exists).

usage: python mixsim/dump_probs.py log/model_xxx.pt [--out mixsim/out] [--limit N]
"""
import argparse, os, sys, time
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig
from coding_data import load_tokens, doc_start_tokens
import hints as hint_lib

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--tokens", default="5k_enwik9")
ap.add_argument("--out", default="mixsim/out")
ap.add_argument("--limit", type=int, default=0, help="only the first N tokens (0 = all)")
ap.add_argument("--batch", type=int, default=64, help="windows per forward")
args = ap.parse_args()

torch.set_float32_matmul_precision('high')
dev = "cuda"
ck = torch.load(args.ckpt, map_location="cpu")
cfg = GPTConfig(**ck["model_config"])
model = GPT(cfg).to(dev)
model.load_state_dict(ck["model"])
model.eval()
T, H = cfg.context_length, cfg.context_length // 2

toks_np = load_tokens(args.tokens)
n = len(toks_np) if not args.limit else min(args.limit, len(toks_np))
toks = torch.from_numpy(np.asarray(toks_np[:n], dtype=np.int32)).to(dev)
# a doc-masked model is scored with the same masking: segment ids are a running count of
# document-start tokens within each window, exactly as train.py builds them
doc_mask = bool(ck.get("run_config", {}).get("doc_mask", False))
if doc_mask:
	starts_ids = doc_start_tokens(vocab_size=cfg.vocab_size)
	is_start = torch.zeros(max(int(toks.max()), max(starts_ids)) + 1, dtype=torch.int32, device=dev)
	is_start[starts_ids] = 1
# a hint-trained model gets the same causal hints it trained with, indexed by target position
if cfg.n_hint:
	hint_tok, hint_bkt = hint_lib.hints_gpu(args.tokens, cfg.n_hint, dev, n)
assert n > T

# window starts: 0, H, 2H, ...; the last one is pulled back so it ends exactly at n-1
starts = list(range(0, n - 1 - T + 1, H))
if starts[-1] != n - 1 - T:
	starts.append(n - 1 - T)
starts = torch.tensor(starts, device=dev)

logp = np.full(n, np.nan, dtype=np.float32)
pmax = np.full(n, np.nan, dtype=np.float16)
ent = np.full(n, np.nan, dtype=np.float16)
offs = torch.arange(T + 1, device=dev)

def score_from(x, y, ht, hb, seg, k0):
	"""Scores for target slots k0..T-1 only: the softmax over the vocab is the expensive
	part, so don't compute it for positions an earlier window already scored."""
	with torch.autocast("cuda", torch.bfloat16):
		logits = model(x, hint_tok=ht, hint_bkt=hb, doc_seg=seg)[:, k0:]
	y = y[:, k0:]
	lsm = F.log_softmax(logits.float(), dim=-1)
	lp = lsm.gather(-1, y.unsqueeze(-1)).squeeze(-1)
	pm = lsm.max(-1).values.exp()
	en = -(lsm.exp() * lsm).sum(-1)
	return lp, pm, en
score_tail = torch.compile(lambda x, y, ht, hb, seg: score_from(x, y, ht, hb, seg, H))

t0 = time.time()
scored_to = 0          # every target position < scored_to+1 is done
with torch.no_grad():
	for i in range(0, len(starts), args.batch):
		s = starts[i:i + args.batch]
		idx = s[:, None] + offs
		w = toks[idx].long()
		x, y = w[:, :-1], w[:, 1:]
		ht = hint_tok[idx[:, 1:]].long() if cfg.n_hint else None
		hb = hint_bkt[idx[:, 1:]].long() if cfg.n_hint else None
		seg = torch.cumsum(is_start[x], dim=1) if doc_mask else None
		sub = lambda a, sl: None if a is None else a[sl]
		if i == 0:      # first window scores all of its targets; the rest score the last H
			full = [a.cpu().numpy() for a in score_from(x[:1], y[:1], sub(ht, slice(0, 1)), sub(hb, slice(0, 1)), sub(seg, slice(0, 1)), 0)]
			tail = [a.cpu().numpy() for a in score_tail(x[1:], y[1:], sub(ht, slice(1, None)), sub(hb, slice(1, None)), sub(seg, slice(1, None)))] if len(s) > 1 else None
		else:
			full, tail = None, [a.cpu().numpy() for a in score_tail(x, y, ht, hb, seg)]
		for j, st in enumerate(s.tolist()):
			# targets of this window are positions st+1 .. st+T; keep the ones not yet scored
			# (the last H, or fewer for the pulled-back final window)
			first = max(scored_to + 1, st + 1)
			if i == 0 and j == 0:
				vals, k = full, 0
			else:
				vals, k = tail, (j - 1 if i == 0 else j)
				first = max(first, st + 1 + H)   # tail slots only cover the last H targets
			k0 = first - (st + 1) - (0 if vals is full else H)
			logp[first:st + T + 1] = vals[0][k, k0:]
			pmax[first:st + T + 1] = vals[1][k, k0:]
			ent[first:st + T + 1] = vals[2][k, k0:]
			scored_to = st + T
		if (i // args.batch) % 200 == 0:
			print(f"{scored_to / n:6.1%}  {time.time() - t0:6.0f}s", flush=True)

assert scored_to == n - 1 and not np.isnan(logp[1:]).any()
os.makedirs(args.out, exist_ok=True)
np.save(os.path.join(args.out, "logp.npy"), logp)
np.save(os.path.join(args.out, "pmax.npy"), pmax)
np.save(os.path.join(args.out, "ent.npy"), ent)
nats = -logp[1:].astype(np.float64).sum()
print(f"tokens {n} | mean loss {nats / (n - 1):.4f} nats | ideal payload {nats / np.log(2) / 8 / 1e6:.2f} MB | {time.time() - t0:.0f}s")

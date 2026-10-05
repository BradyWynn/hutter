"""Online-learn only the hashed n-gram table on top of a frozen, trained body.

Loads a train.py checkpoint, freezes every GPT weight, and codes the token stream as B equal
segments walked in lockstep, one non-overlapping context_length window per segment per step
(the windowed train.py predictor, doc-masked the same way). Each window is scored before the
table trains on it, so the loss is a prequential code length.

Arms:
  trained  the checkpoint's own table, frozen        (what the trained model codes at)
  zero     table zeroed, frozen                      (body alone; the table is a no-op at zero)
  online   table zeroed, then SparseRowAdam on it as the stream is coded

usage: python ngram_online.py log/model_xxx.pt [--arms trained,zero,online] [--lr 1.0]
"""
import argparse
import json
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from model import GPT, GPTConfig, HashedNgram
from sparse_adam import SparseRowAdam
from coding_data import load_tokens, doc_start_tokens

LOG2E = math.log2(math.e)

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--arms", default="trained,zero,online")
parser.add_argument("--lr", type=float, default=1.0)
parser.add_argument("--B", type=int, default=32)
parser.add_argument("--log_interval", type=int, default=500)
args = parser.parse_args()

device = "cuda"
torch.set_float32_matmul_precision('high')

ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
rc = ck["run_config"]
raw_model = GPT(GPTConfig(**ck["model_config"])).to(device)
raw_model.load_state_dict(ck["model"])
raw_model.requires_grad_(False)
raw_model.eval()

ngram = HashedNgram(rc["ngram_rows"], rc["n_embd"], tuple(rc["ngram_orders"]), heads=rc["ngram_heads"],
                    table_dtype=torch.float16 if rc["ngram_table_fp16"] else torch.float32).to(device)
trained_table = ck["ngram"]["table.weight"]

tokens = load_tokens(rc["token_path"])
num_tokens = len(tokens)
tok = torch.from_numpy(np.asarray(tokens, dtype=np.int16)).to(device)
starts_ids = doc_start_tokens(vocab_size=rc["vocab_size"]) if rc["doc_mask"] else None
is_start = None
if starts_ids is not None:
	is_start = torch.zeros(max(int(tok.max()), max(starts_ids)) + 1, dtype=torch.int32, device=device)
	is_start[starts_ids] = 1

B, T = args.B, rc["context_length"]
seg_len = num_tokens // B
n_steps = (seg_len - 1) // T
seg_starts = torch.arange(B, device=device)[:, None] * seg_len
offsets = torch.arange(T + 1, device=device)
coverage = B * n_steps * T
print(f"{num_tokens} tokens, {B} segments x {n_steps} windows of {T}: {coverage} targets "
      f"({coverage / num_tokens:.4%} coverage)")

def token_nll(x, y, ng, doc_seg):
	with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
		logits = raw_model(x, ngram=ng, doc_seg=doc_seg)
	V = logits.size(-1)
	return F.cross_entropy(logits.reshape(-1, V).float(), y.flatten(), reduction='none').view(y.shape)

loss_fn = torch.compile(token_nll, mode='max-autotune-no-cudagraphs')

def run_arm(arm):
	with torch.no_grad():
		ngram.table.weight.copy_(trained_table if arm == "trained" else torch.zeros_like(trained_table))
	learn = arm == "online"
	opt = SparseRowAdam([ngram.table.weight], lr=args.lr,
	                    state_dtype=torch.bfloat16 if rc["ngram_state_bf16"] else torch.float32) if learn else None
	total = torch.zeros((), device=device, dtype=torch.float64)
	window = torch.zeros((), device=device, dtype=torch.float64)
	curve = []
	t0 = time.time()
	for step in range(n_steps):
		w = tok[seg_starts + step * T + offsets].long()
		x, y = w[:, :-1], w[:, 1:]
		seg = None if is_start is None else torch.cumsum(is_start[x], dim=1)
		if learn:
			nll = loss_fn(x, y, ngram(x, use_state=False), seg)
			nll.mean().backward()
			opt.step()
			ngram.zero_grad(set_to_none=True)
			nll = nll.detach()
		else:
			with torch.no_grad():
				nll = loss_fn(x, y, ngram(x, use_state=False), seg)
		s = nll.sum(dtype=torch.float64)
		total += s
		window += s
		if (step + 1) % args.log_interval == 0 or step == n_steps - 1:
			n_win = ((step % args.log_interval) + 1) * B * T
			win_loss = window.item() / n_win
			mean_loss = total.item() / ((step + 1) * B * T)
			curve.append((step + 1, win_loss))
			window.zero_()
			print(f"[{arm}] step {step + 1:6d}/{n_steps} | window loss {win_loss:.5f} | running {mean_loss:.5f} | "
			      f"{(step + 1) * B * T / (time.time() - t0) / 1e6:.2f}M tok/s")
	loss = total.item() / coverage
	mb = loss * LOG2E * num_tokens / 8 / 1e6   # extrapolated to the whole stream, as train.py's "bytes"
	print(f"[{arm}] FINAL loss {loss:.5f} nats/token | {loss * LOG2E:.4f} bits/token | ~{mb:.2f} MB payload "
	      f"| {time.time() - t0:.0f}s")
	return {"loss": loss, "payload_mb": mb, "curve": curve}

results = {arm: run_arm(arm) for arm in args.arms.split(",")}
print("\nsummary:")
for arm, r in results.items():
	print(f"  {arm:8s} {r['loss']:.5f} nats/token  ~{r['payload_mb']:.2f} MB")
out = args.checkpoint.replace(".pt", f"_ngram_online_lr{args.lr}.json")
with open(out, "w") as f:
	json.dump({"args": vars(args), "results": results}, f, indent=1)
print(f"wrote {out}")

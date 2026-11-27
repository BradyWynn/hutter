"""Pure-Neural Chunk-Level Test-Time Training (TTT) Benchmark on Exact Archive-Quantized ECQAT Model.

Evaluates strictly causal online adaptation over evenly spaced contiguous segments spanning all 1 GB
of enwik9 (starting from exact snapped archive-precision weights of an ECQAT checkpoint, 0 archive bytes).

Strict causality:
  For each chunk of `stride` tokens (e.g. 512 or 1024 tokens):
    1. Forward pass on window [pos - (T - stride) : pos + stride] -> SCORE [pos : pos + stride] BEFORE any update!
    2. Compute loss on the just-decoded chunk [pos : pos + stride] and take 1 online gradient step
       on a tiny adapter subset with elastic anchor decay toward base weights w_0:
         w <- w - lr * g - anchor * (w - w_0)
  (Every scored token is strictly predicted BEFORE it is ever seen by an update.)

Modes benchmarked:
  - head:       lm_head.bias + logit_scale + match_* (~5.4K params, ZERO trunk backprop, <1% CPU overhead)
  - coda_head:  head + final block h[-1] (pass_scale[-1], pass_shift[-1], attn_gate, proj, qkv, shared_*) (~5-8% CPU)
  - adarms_head: head + all-layer pass_scale/pass_shift/attn_gate/resid/x0/v/skip (~17K params, light backprop)
  - full_light: head + all-layer adarms + all blocks (upper bound on chunk-level TTT gain)
"""
import argparse
import copy
import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from ecqat import ECQATWrapper
from eval_hutter import vocab_info
import matchfeat


@torch.no_grad()
def load_exact_ecqat_model(ckpt_path, cfg, rate_model="mix2", delta_mode="rowcol", use_wht=False):
    """Load an ECQATWrapper checkpoint and snap every single parameter to exact archive precision."""
    base = GPT(cfg).cuda()
    ec = ECQATWrapper(base, rate_model=rate_model, delta_mode=delta_mode, use_wht=use_wht).cuda()
    ec.load_state_dict(torch.load(ckpt_path, map_location="cuda", weights_only=True))

    qm = copy.deepcopy(ec.model).eval()
    qm.set_qat(group_size=0)
    q_ids = set()

    def get_target(qm_root, key):
        if key == "wte":
            return qm_root.wte, "weight"
        if key == "lm_head":
            return qm_root.lm_head, "weight"
        parts = key.split("_")
        li = int(parts[0][1:])
        tag = parts[1]
        blk = qm_root.h[li]
        if tag in ("qkv", "proj", "fc", "out"):
            return getattr(blk, tag), "weight"
        if tag == "sfc":
            return blk.moe.shared_fc, "weight"
        if tag == "sout":
            return blk.moe.shared_out, "weight"
        if tag == "efc":
            return blk.moe, "w_fc"
        if tag == "eout":
            return blk.moe, "w_out"
        raise ValueError(key)

    for key, tier, _, _, wht_dim in ec.specs:
        mod, attr = get_target(qm, key)
        p = getattr(mod, attr)
        q_ids.add(id(p))
        dr_fp16 = ec.log_deltas_r[key].detach().exp().clamp(1e-5, 10.0).to(torch.float16)
        if ec.delta_mode == "rowcol":
            dc_fp16 = ec.log_deltas_c[key].detach().exp().clamp(1e-3, 1e3).to(torch.float16)
            delta = (dr_fp16.float() * dc_fp16.float()).clamp(1e-5, 10.0)
        else:
            delta = dr_fp16.float()
        q = (p.detach().float() / delta).round()
        wq = q * delta
        p.copy_(wq.to(p.dtype))

    if qm.c.tie_weights:
        qm.lm_head.weight = qm.wte.weight
        q_ids.add(id(qm.lm_head.weight))

    other = [p for p in qm.parameters() if id(p) not in q_ids]
    other_bufs = [b.moe.router_bias for b in qm.h if b.moe is not None]
    for p in other + other_bufs:
        p.copy_(p.to(torch.float16).to(p.dtype))
    return qm


def select_ttt_params(model, mode):
    """Select parameter list for TTT adaptation."""
    head_names = {
        "lm_head.bias", "logit_scale",
        "match_gate", "match_logit", "match_ctx.weight", "match_ctx.bias",
        "match_hit_emb", "match_miss_emb",
        "match_hit_scale", "match_miss_scale", "match_agree_scale",
        "match_hit_bias", "match_miss_bias",
    }
    if mode == "head":
        ps = [p for n, p in model.named_parameters() if n in head_names]
    elif mode == "adarms_head":
        adarms = {"pass_scale", "pass_shift", "resid_lambdas", "x0_lambdas", "v_lambdas", "skip_lambdas"}
        ps = [p for n, p in model.named_parameters() if (n in head_names or n in adarms or "attn_gate" in n)]
    elif mode == "coda_head":
        last_prefix = f"h.{len(model.h) - 1}."
        ps = [
            p for n, p in model.named_parameters()
            if (n in head_names or n in {"pass_scale", "pass_shift"} or (n.startswith(last_prefix) and "moe.w_" not in n))
        ]
    elif mode == "coda_full":
        last_prefix = f"h.{len(model.h) - 1}."
        ps = [
            p for n, p in model.named_parameters()
            if (n in head_names or n in {"pass_scale", "pass_shift"} or n.startswith(last_prefix))
        ]
    elif mode == "dense_all":
        ps = [p for n, p in model.named_parameters() if ("moe.w_" not in n and "wte." not in n and "lm_head.weight" not in n)]
    elif mode == "all":
        ps = list(model.parameters())
    else:
        raise ValueError(mode)
    return ps


def eval_ttt_batched(
    base_model,
    data,
    match_tables,
    seg_starts,
    seg_len,
    T,
    stride,
    tok_lens_t,
    mode,
    lr,
    anchor,
    opt_type="adam",
    grad_clip=1.0,
):
    """Evaluate strictly causal TTT across `len(seg_starts)` independent contiguous segments
    (sequentially per segment, resetting weights to `base_model` at the start of each segment)."""
    dev = data.device
    offs = torch.arange(T + 1, device=dev)
    n_chunks = (seg_len - T) // stride
    assert n_chunks > 0

    total_nats = 0.0
    total_bytes = 0
    t0 = time.time()

    # Work on one reusable clone, reset from saved base state each segment
    model = copy.deepcopy(base_model).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if mode == "none":
        with torch.no_grad():
            for s0 in seg_starts:
                # Batch all chunks of this segment for fast baseline scoring
                ch_starts = torch.arange(s0, s0 + n_chunks * stride, stride, device=dev)
                for a in range(0, len(ch_starts), 16):
                    st = ch_starts[a : a + 16]
                    w = data[st[:, None] + offs].long()
                    mp = torch.from_numpy(np.stack([match_tables[0][int(s) + 1 : int(s) + 1 + T] for s in st.cpu().tolist()])).to(dev)
                    ml = torch.from_numpy(np.stack([match_tables[1][int(s) + 1 : int(s) + 1 + T] for s in st.cpu().tolist()])).to(dev)
                    with torch.autocast("cuda", torch.bfloat16):
                        logits = model(w[:, :-1], mp, ml)
                    targ = w[:, 1:]
                    loss_s = F.cross_entropy(logits[:, -stride:].reshape(-1, logits.size(-1)).float(), targ[:, -stride:].reshape(-1), reduction="sum")
                    total_nats += loss_s.item()
                    total_bytes += tok_lens_t[targ[:, -stride:]].sum().item()
        bpb = total_nats / math.log(2) / total_bytes
        return bpb, bpb * 125.0, time.time() - t0, 0

    ttt_ps = select_ttt_params(model, mode)
    for p in ttt_ps:
        p.requires_grad_(True)
    n_ttt = sum(p.numel() for p in ttt_ps)
    base_copies = [p.detach().clone() for p in ttt_ps]

    for s0 in seg_starts:
        # Reset adapted params to exact archive base weights
        with torch.no_grad():
            for p, b0 in zip(ttt_ps, base_copies):
                p.copy_(b0)

        if opt_type == "adam":
            opt = torch.optim.Adam(ttt_ps, lr=lr, betas=(0.9, 0.99), eps=1e-8)
        elif opt_type == "sgd":
            opt = torch.optim.SGD(ttt_ps, lr=lr, momentum=0.5)
        else:
            raise ValueError(opt_type)

        for ci in range(n_chunks):
            s = int(s0 + ci * stride)
            w = data[s : s + T + 1].unsqueeze(0).long()
            mp = torch.from_numpy(match_tables[0][s + 1 : s + 1 + T][None]).to(dev)
            ml = torch.from_numpy(match_tables[1][s + 1 : s + 1 + T][None]).to(dev)

            # 1. STRICTLY CAUSAL FORWARD & SCORE BEFORE ANY UPDATE ON THIS CHUNK
            with torch.autocast("cuda", torch.bfloat16):
                logits = model(w[:, :-1], mp, ml)
            targ = w[:, 1:]
            logits_s = logits[:, -stride:].reshape(-1, logits.size(-1)).float()
            targ_s = targ[:, -stride:].reshape(-1)
            loss_sum = F.cross_entropy(logits_s, targ_s, reduction="sum")
            total_nats += loss_sum.item()
            total_bytes += tok_lens_t[targ_s].sum().item()

            # 2. ONLINE UPDATE AFTER CHUNK TOKENS ARE DECODED
            loss_mean = loss_sum / stride
            loss_mean.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ttt_ps, grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if anchor > 0:
                with torch.no_grad():
                    for p, b0 in zip(ttt_ps, base_copies):
                        p.lerp_(b0, anchor)

    bpb = total_nats / math.log(2) / total_bytes
    return bpb, bpb * 125.0, time.time() - t0, n_ttt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/v10_moe30m_matchv2_100k_ecqat35k/ecqat_best.pt")
    ap.add_argument("--data", default="enwik9.split4096")
    ap.add_argument("--n-segs", type=int, default=16)
    ap.add_argument("--seg-len", type=int, default=65536)  # 64K toks (~185KB) per segment -> 1M scored toks across all 1GB!
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--n-experts", type=int, default=16)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    tok_lens, _, vocab = vocab_info(args.data)
    tok_lens_t = torch.from_numpy(tok_lens).cuda()
    data = torch.from_numpy(np.fromfile(args.data + ".i16", dtype=np.int16)).cuda()
    N = len(data)
    mt = matchfeat.cached(data, (2, 3, 5, 8, 13, 27), args.data)

    T = 2048
    cfg = GPTConfig(
        context_length=T, vocab_size=vocab, n_layer=8, n_loop=1,
        n_experts=args.n_experts, n_experts_active=2, shared_h=256, expert_h=256,
        n_match=6, match_codes=matchfeat.N_CODE, fp8=False,
    )
    model = load_exact_ecqat_model(args.ckpt, cfg)
    seg_starts = np.linspace(0, N - args.seg_len - 1, args.n_segs, dtype=np.int64)
    n_chunks = (args.seg_len - T) // args.stride
    print(f"Loaded exact archive-snapped {args.ckpt} | {args.n_segs} segs x {n_chunks} chunks (stride={args.stride}, ~{args.n_segs * n_chunks * args.stride / 1e6:.2f}M scored toks)")

    base_bpb, base_mb, dt, _ = eval_ttt_batched(model, data, mt, seg_starts, args.seg_len, T, args.stride, tok_lens_t, "none", 0.0, 0.0)
    print(f"\n{'Mode':14s} | {'Opt':4s} | {'LR':7s} | {'Anchor':6s} | {'Params':>9s} | {'bpb':>8s} | {'Payload MB':>10s} | {'Delta MB':>9s} | {'Time':>6s}")
    print("-" * 96)
    print(f"{'baseline':14s} | {'-':4s} | {'-':7s} | {'-':6s} | {0:9,d} | {base_bpb:8.5f} | {base_mb:10.3f} | {0.0:+9.3f} | {dt:5.1f}s")

    configs = [
        ("head",        "adam", 3e-3, 0.02),
        ("head",        "adam", 1e-2, 0.05),
        ("adarms_head", "adam", 1e-3, 0.02),
        ("adarms_head", "adam", 3e-3, 0.05),
        ("coda_head",   "adam", 5e-4, 0.02),
        ("coda_head",   "adam", 1e-3, 0.03),
        ("coda_full",   "adam", 5e-4, 0.02),
        ("dense_all",   "adam", 3e-4, 0.02),
        ("dense_all",   "adam", 5e-4, 0.03),
        ("all",         "adam", 3e-4, 0.02),
    ]
    for mode, opt_t, lr, anc in configs:
        bpb, mb, dt, np_ttt = eval_ttt_batched(
            model, data, mt, seg_starts, args.seg_len, T, args.stride, tok_lens_t,
            mode=mode, lr=lr, anchor=anc, opt_type=opt_t,
        )
        print(f"{mode:14s} | {opt_t:4s} | {lr:7.1e} | {anc:6.3f} | {np_ttt:9,d} | {bpb:8.5f} | {mb:10.3f} | {mb - base_mb:+9.3f} | {dt:5.1f}s", flush=True)


if __name__ == "__main__":
    main()

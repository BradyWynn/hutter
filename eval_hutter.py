"""True strided full-context Hutter-MB evaluation harness.

Can be imported by train.py for periodic checkpoint evaluation, or run standalone:
  python eval_hutter.py model_ema.pt --data enwik9.split4096 --windows 512 --stride 1024

Reports:
  - full_bpb:        full-window [0..T-1] mean bpb (comparable to legacy eval_ce)
  - strided_bpb:     full-context [T-stride..T-1] bpb across evenly spaced windows over all 1 GB
  - q_strided_bpb:   same, after exact archive quantization (Int6 emb + Int4 blocks + fp16 misc)
  - weight_mb:       entropy-coded quantized weight MB (Shannon bound achievable by arith coder)
  - vocab_mb:        zlib-9 size of compact BPE merge table
  - hutter_mb:       projected total Hutter Prize archive MB (payload + weights + vocab + code)
"""
import argparse
import json
import math
import os
import struct
import zlib
import numpy as np
import torch
import torch.nn.functional as F
from model import GPT, GPTConfig, quantize_stats, quantize_model_copy
import matchfeat


def vocab_info(data_prefix):
    """Return (tok_lens: np.int32[V], vocab_bytes_zlib: int, vocab_size: int)."""
    if data_prefix == "enwik9":
        return np.ones(256, dtype=np.int32), 0, 256
    meta = json.load(open(data_prefix + ".json"))
    merges = [tuple(m) for m in meta["merges"]]
    vocab = {chr(b): b for b in range(256)}
    merge_ids = bytearray()
    for a, b in merges:
        ia, ib = vocab[a], vocab[b]
        merge_ids.extend(struct.pack("<HH", ia, ib))
        vocab.setdefault(a + b, len(vocab))
    lens = np.zeros(len(vocab), dtype=np.int32)
    for s, i in vocab.items():
        lens[i] = len(s)
    vocab_bytes = len(zlib.compress(bytes(merge_ids), 9))
    return lens, vocab_bytes, len(vocab)


def make_strided_starts(n_pos, T, stride, n_windows):
    """Return `n_windows` evenly spaced window starts spanning all of enwik9."""
    lo = 0
    hi = n_pos - T - 1
    assert hi > lo, f"data length {n_pos} too short for T={T}"
    return np.linspace(lo, hi, n_windows, dtype=np.int64)


@torch.no_grad()
def eval_strided(
    model,
    data,
    match_tables,
    starts,
    T,
    stride=1024,
    batch_size=16,
    tok_lens=None,
    n_bytes_total=1_000_000_000,
):
    """Evaluate `model` (eager BF16) on evenly spaced strided windows across enwik9.
    Returns both full-window [0..T-1] and full-context [T-stride..T-1] metrics."""
    dev = data.device
    offs = torch.arange(T + 1, device=dev)
    tok_lens_t = torch.from_numpy(tok_lens).to(dev) if tok_lens is not None else None
    n_pos = len(data)
    use_match = len(match_tables) > 0 and model.n_match > 0

    full_nats = 0.0
    full_toks = 0
    strided_nats = 0.0
    strided_toks = 0
    strided_bytes = 0

    was_training = model.training
    model.eval()
    for a in range(0, len(starts), batch_size):
        s_np = starts[a : a + batch_size]
        st = torch.from_numpy(s_np).to(dev)
        w = data[st[:, None] + offs].long()
        if use_match:
            mp = torch.from_numpy(
                np.stack([match_tables[0][int(s) + 1 : int(s) + 1 + T] for s in s_np])
            ).to(dev, non_blocking=True)
            ml = torch.from_numpy(
                np.stack([match_tables[1][int(s) + 1 : int(s) + 1 + T] for s in s_np])
            ).to(dev, non_blocking=True)
        else:
            mp = ml = None

        with torch.autocast("cuda", torch.bfloat16):
            logits = model(w[:, :-1], mp, ml)
        targets = w[:, 1:]
        loss_all = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), reduction="none").view_as(targets)

        full_nats += loss_all.sum().item()
        full_toks += targets.numel()
        loss_s = loss_all[:, -stride:]
        targ_s = targets[:, -stride:]
        strided_nats += loss_s.sum().item()
        strided_toks += targ_s.numel()
        if tok_lens_t is not None:
            strided_bytes += tok_lens_t[targ_s].sum().item()
    if was_training:
        model.train()

    full_loss = full_nats / full_toks
    strided_loss = strided_nats / strided_toks
    scale_global = n_pos / n_bytes_total / math.log(2)
    full_bpb = full_loss * scale_global
    strided_bpb_global = strided_loss * scale_global
    strided_bpb_local = (strided_nats / math.log(2) / strided_bytes) if strided_bytes > 0 else strided_bpb_global
    return dict(
        full_loss=full_loss,
        full_bpb=full_bpb,
        strided_loss=strided_loss,
        strided_bpb=strided_bpb_local,
        strided_bpb_global=strided_bpb_global,
        payload_mb=strided_bpb_local * n_bytes_total / 8.0 / 1e6,
    )


@torch.no_grad()
def extra_overhead_bytes(data_prefix):
    """Detect any lossless transform overhead (e.g. enwik9.reord.meta.json permutation bytes)."""
    base = data_prefix.split(".split")[0].split(".nosplit")[0]
    meta_path = base + ".meta.json"
    if os.path.exists(meta_path):
        return int(json.load(open(meta_path)).get("extra_overhead_bytes", 0))
    return 0


def eval_hutter_suite(
    model,
    ema,
    data,
    match_tables,
    starts,
    T,
    stride=1024,
    batch_size=16,
    tok_lens=None,
    n_bytes_total=1_000_000_000,
    vocab_bytes=0,
    extra_bytes=0,
    code_overhead_mb=0.05,
    block_bits=4,
    emb_bits=6,
    expert_bits=4,
    qat_group=64,
):
    """Evaluate raw model, EMA, and exact archive-quantized EMA, returning projected Hutter MB."""
    out = {}
    m_res = eval_strided(model, data, match_tables, starts, T, stride, batch_size, tok_lens, n_bytes_total)
    out["eval_loss"] = m_res["full_loss"]
    out["eval_bpb"] = m_res["full_bpb"]
    out["strided_bpb"] = m_res["strided_bpb"]

    target_m = ema if ema is not None else model
    if ema is not None:
        e_res = eval_strided(ema, data, match_tables, starts, T, stride, batch_size, tok_lens, n_bytes_total)
        out["ema_eval_loss"] = e_res["full_loss"]
        out["ema_eval_bpb"] = e_res["full_bpb"]
        out["ema_strided_bpb"] = e_res["strided_bpb"]
        out["ema_strided_bpb_global"] = e_res["strided_bpb_global"]
        base_strided = e_res["strided_bpb"]
    else:
        base_strided = m_res["strided_bpb"]

    qm = quantize_model_copy(target_m, block_bits=block_bits, emb_bits=emb_bits, expert_bits=expert_bits, group_size=qat_group)
    q_res = eval_strided(qm, data, match_tables, starts, T, stride, batch_size, tok_lens, n_bytes_total)
    qs = quantize_stats(target_m, block_bits=block_bits, emb_bits=emb_bits, expert_bits=expert_bits, group_size=qat_group)
    del qm

    vocab_mb = vocab_bytes / 1e6
    extra_mb = extra_bytes / 1e6
    weight_mb = qs["entropy_mb"]
    payload_mb = q_res["payload_mb"]
    hutter_mb = payload_mb + weight_mb + vocab_mb + extra_mb + code_overhead_mb

    out["q_strided_bpb"] = q_res["strided_bpb"]
    out["q_drop_bpb"] = q_res["strided_bpb"] - base_strided
    out["q_payload_mb"] = payload_mb
    out["weight_mb"] = weight_mb
    out["weight_raw_mb"] = qs["raw_mb"]
    out["vocab_mb"] = vocab_mb
    out["extra_mb"] = extra_mb
    out["hutter_mb"] = hutter_mb
    out["block_bpw"] = qs["block_bpw"]
    out["emb_bpw"] = qs["emb_bpw"]
    if qs["expert_params"] > 0:
        out["expert_bpw"] = qs["expert_bpw"]
        out["expert_mb"] = qs["expert_entropy_mb"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="Path to model.pt / model_ema.pt / ckpt.pt")
    ap.add_argument("--data", default="enwik9.split4096")
    ap.add_argument("--windows", type=int, default=512)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--block-bits", type=int, default=4)
    ap.add_argument("--emb-bits", type=int, default=6)
    ap.add_argument("--qat-group", type=int, default=64)
    ap.add_argument("--match-k", default="2,3,5,8,13,27")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    n_bytes = os.path.getsize("enwik9")
    tok_lens, vocab_bytes, vocab_size = vocab_info(args.data)
    if args.data == "enwik9":
        data = torch.frombuffer(bytearray(open("enwik9", "rb").read()), dtype=torch.uint8).cuda()
    else:
        data = torch.from_numpy(np.fromfile(args.data + ".i16", dtype=np.int16)).cuda()

    raw = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    sd = raw["ema"] if (isinstance(raw, dict) and "ema" in raw and raw["ema"] is not None) else (
        raw["model"] if (isinstance(raw, dict) and "model" in raw) else raw
    )
    n_match = sd["match_gate"].shape[0] if "match_gate" in sd else 0
    match_k = tuple(int(x) for x in args.match_k.split(",") if x) if n_match else ()
    assert len(match_k) == n_match, f"ckpt has n_match={n_match} but --match-k={match_k}"
    match_tables = matchfeat.cached(data, match_k, args.data) if match_k else ()

    cfg = GPTConfig(
        context_length=2048,
        vocab_size=vocab_size,
        n_match=n_match,
        match_codes=matchfeat.N_CODE,
        qat_block_bits=args.block_bits,
        qat_emb_bits=args.emb_bits,
    )
    model = GPT(cfg).cuda().eval()
    model.load_state_dict(sd)

    starts = make_strided_starts(len(data), cfg.context_length, args.stride, args.windows)
    x_bytes = extra_overhead_bytes(args.data)
    res = eval_hutter_suite(
        model, None, data, match_tables, starts, cfg.context_length,
        stride=args.stride, batch_size=args.batch_size, tok_lens=tok_lens,
        n_bytes_total=n_bytes, vocab_bytes=vocab_bytes, extra_bytes=x_bytes,
        block_bits=args.block_bits, emb_bits=args.emb_bits, qat_group=args.qat_group,
    )
    print(f"=== Hutter Eval: {args.ckpt} ({args.windows} strided windows, stride={args.stride}) ===")
    print(f"  Full-window bpb (legacy):  {res['eval_bpb']:.5f}")
    print(f"  Strided bpb (unquant):     {res['strided_bpb']:.5f}")
    print(f"  Strided bpb (quantized):   {res['q_strided_bpb']:.5f} (drop {res['q_drop_bpb']:+.5f})")
    print(f"  Payload MB:                {res['q_payload_mb']:.3f} MB")
    print(f"  Weights MB (entropy):      {res['weight_mb']:.3f} MB (raw {res['weight_raw_mb']:.3f} MB | emb {res['emb_bpw']:.2f} bpw, blk {res['block_bpw']:.2f} bpw)")
    print(f"  Vocab + perm + code:       {res['vocab_mb']:.4f} + {res['extra_mb']:.4f} + 0.0500 MB")
    print(f"  PROJECTED HUTTER TOTAL:    {res['hutter_mb']:.3f} MB")


if __name__ == "__main__":
    main()

"""Entropy-Constrained QAT (ECQAT) — Exact 1:1 Hutter Prize Rate-Distortion Optimization.

Branches from a converged v10 checkpoint and directly minimizes exact Hutter total nats:
    L(w, delta, b) = CE_nats(x, y) + (RATE_MULT / N_tok) * sum_t Rate_nats(w_t; delta_t, b_t)

Where:
  - Each quantized tensor gets a learnable per-row step size `delta = exp(log_delta)`
    with exact LSQ STE gradient: `W_ste = (u + (u.round() - u).detach()) * delta` (`u = W / delta`).
  - Rate is modeled by a continuous Ballé-style zero-centered Laplace CDF per tensor with
    learnable scale `b = exp(log_b)`, matching empirical Shannon entropy closely.
  - Because `1 / N_tok` is the exact 1:1 exchange rate between weight nats and enwik9 nats
    (no magic tuning knob), the network selectively allocates higher precision to high-impact
    rows/layers and coarser precision/zeros to redundant MoE expert rows.
  - Uses AdamW (not Muon) so coordinate-wise rate gradients stay exact.
  - Evaluates true archive Hutter MB using exact empirical Shannon entropy of integer symbols
    `q = round(W / fp16(delta))` + fp16 `delta` bytes + zlib-9 fp16 misc + 2048-win strided NLL.
"""
import copy
import math
import os
import time
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from model import GPT, GPTConfig, FP8Linear, QATLinear, MoEMLP
from eval_hutter import vocab_info, extra_overhead_bytes, make_strided_starts, eval_strided
import matchfeat


def laplace_rate_nats(u, log_b):
    """Exact Ballé-style Laplace CDF rate (-log P(|u|; b)) in nats, C1 smooth everywhere."""
    b = log_b.exp().clamp(0.05, 50.0)
    au = u.abs()
    half = 0.5 / b
    # For |u| <= 0.5: P = 1 - exp(-0.5/b) * cosh(u/b)
    # For |u| >  0.5: P = sinh(0.5/b) * exp(-|u|/b)
    # Compute stably in log space:
    p_inner = (1.0 - torch.exp(-half) * torch.cosh(au.clamp_max(0.5) / b)).clamp_min(1e-12)
    nll_inner = -torch.log(p_inner)
    # log(sinh(h)) = h + log((1 - exp(-2h)) / 2)
    log_sinh = half + torch.log((1.0 - torch.exp(-2.0 * half)).clamp_min(1e-12) * 0.5)
    nll_outer = au / b - log_sinh
    return torch.where(au <= 0.5, nll_inner, nll_outer).sum()


def ste_quant_row(w, log_delta):
    """Per-row learnable step size quantization with exact LSQ STE gradient."""
    delta = log_delta.exp().clamp(1e-5, 10.0)
    u = w.float() / delta
    u_ste = u + (u.round() - u).detach()
    return (u_ste * delta).to(w.dtype), u


class ECQATWrapper(nn.Module):
    """Wraps a GPT model with per-row learnable step sizes `log_delta` and per-tensor
    Laplace rate scales `log_b` for all quantized weight matrices (emb, dense blocks, MoE experts)."""

    def __init__(self, model, init_emb_bits=6, init_blk_bits=5, init_exp_bits=4):
        super().__init__()
        self.model = model
        # Disable fixed g=64 fake_quant inside model; ECQAT hooks provide learned-delta fake_quant
        self.model.set_qat(group_size=0)

        self.log_deltas = nn.ParameterDict()
        self.log_bs = nn.ParameterDict()
        self.specs = []  # (key, tier, module, attr_name, delta_shape)

        def register_mat(key, tier, mod, attr, bits):
            w = getattr(mod, attr).detach().float()
            qmax = float((1 << (bits - 1)) - 1)
            # Per-output-row delta: [out, 1] for 2D, [E, out, 1] for 3D MoE
            d_init = (w.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-4)
            u_init = w / d_init
            b_init = u_init.abs().mean().clamp(0.2, 20.0)
            self.log_deltas[key] = nn.Parameter(d_init.log())
            self.log_bs[key] = nn.Parameter(b_init.log().reshape(()))
            self.specs.append((key, tier, mod, attr))

        register_mat("wte", "emb", model.wte, "weight", init_emb_bits)
        if not model.c.tie_weights:
            register_mat("lm_head", "emb", model.lm_head, "weight", init_emb_bits)
        for li, blk in enumerate(model.h):
            for name in ("qkv", "proj", "fc", "out"):
                sub = getattr(blk, name, None)
                if isinstance(sub, (FP8Linear, QATLinear)):
                    register_mat(f"h{li}_{name}", "blk", sub, "weight", init_blk_bits)
            if blk.moe is not None:
                if blk.moe.shared_fc is not None:
                    register_mat(f"h{li}_sfc", "blk", blk.moe.shared_fc, "weight", init_blk_bits)
                    register_mat(f"h{li}_sout", "blk", blk.moe.shared_out, "weight", init_blk_bits)
                register_mat(f"h{li}_efc", "exp", blk.moe, "w_fc", init_exp_bits)
                register_mat(f"h{li}_eout", "exp", blk.moe, "w_out", init_exp_bits)

    def forward_with_rate(self, x, mp, ml):
        """Temporarily replace module weights with STE-quantized views, run forward, compute exact Laplace rate."""
        saved = []
        total_rate_nats = torch.zeros((), device=x.device, dtype=torch.float32)
        for key, tier, mod, attr in self.specs:
            orig_p = getattr(mod, attr)
            wq, u = ste_quant_row(orig_p, self.log_deltas[key])
            total_rate_nats = total_rate_nats + laplace_rate_nats(u, self.log_bs[key])
            saved.append((mod, attr, orig_p))
            # Swap parameter in _parameters dict so autograd flows cleanly through wq -> orig_p & log_delta
            mod._parameters[attr] = wq
        try:
            # Keep tied lm_head pointing to quantized wte
            if self.model.c.tie_weights:
                self.model.lm_head._parameters["weight"] = self.model.wte._parameters["weight"]
            with torch.autocast("cuda", torch.bfloat16):
                logits = self.model(x, mp, ml)
        finally:
            for mod, attr, orig_p in saved:
                mod._parameters[attr] = orig_p
            if self.model.c.tie_weights:
                self.model.lm_head._parameters["weight"] = self.model.wte._parameters["weight"]
        return logits, total_rate_nats

    @torch.no_grad()
    def exact_archive_eval(self, data, match_tables, starts, T, stride, batch_size, tok_lens, n_bytes, vocab_bytes, extra_bytes=0, code_overhead_mb=0.05):
        """Snap every weight to exact fp16(delta) * round(W / fp16(delta)), snap misc to fp16,
        compute TRUE empirical Shannon entropy per tier + exact strided NLL -> honest hutter_mb."""
        qm = copy.deepcopy(self.model).eval().requires_grad_(False)
        qm.set_qat(group_size=0)

        tier_q = {"emb": [], "blk": [], "exp": []}
        tier_scale_bytes = {"emb": 0, "blk": 0, "exp": 0}
        q_ids = set()

        # Map specs onto qm
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

        for key, tier, _, _ in self.specs:
            mod, attr = get_target(qm, key)
            p = getattr(mod, attr)
            q_ids.add(id(p))
            delta_fp16 = self.log_deltas[key].detach().exp().clamp(1e-5, 10.0).to(torch.float16)
            q = (p.detach().float() / delta_fp16.float()).round()
            p.copy_((q * delta_fp16.float()).to(p.dtype))
            tier_q[tier].append(q.to(torch.int32).cpu().numpy().ravel())
            tier_scale_bytes[tier] += delta_fp16.numel() * 2

        if qm.c.tie_weights:
            qm.lm_head.weight = qm.wte.weight
            q_ids.add(id(qm.lm_head.weight))

        # Snap all remaining misc params & router_bias to fp16
        other = [p for p in qm.parameters() if id(p) not in q_ids]
        other_bufs = [b.moe.router_bias for b in qm.h if b.moe is not None]
        for p in other + other_bufs:
            p.copy_(p.to(torch.float16).to(p.dtype))
        other_fp16 = b"".join(p.detach().to(torch.float16).cpu().numpy().tobytes() for p in (other + other_bufs))
        other_zlib = len(zlib.compress(other_fp16, 9))

        tier_bpw = {}
        tier_mb = {}
        tier_zero = {}
        total_ent_bytes = other_zlib
        for tier in ("emb", "blk", "exp"):
            if not tier_q[tier]:
                tier_bpw[tier] = 0.0
                tier_mb[tier] = 0.0
                tier_zero[tier] = 0.0
                continue
            arr = np.concatenate(tier_q[tier])
            _, counts = np.unique(arr, return_counts=True)
            probs = counts / counts.sum()
            h_bits = float(-(probs * np.log2(probs)).sum())
            ent_bytes = int(math.ceil(len(arr) * h_bits / 8.0)) + tier_scale_bytes[tier]
            tier_bpw[tier] = h_bits
            tier_mb[tier] = ent_bytes / 1e6
            tier_zero[tier] = float((arr == 0).mean())
            total_ent_bytes += ent_bytes

        r = eval_strided(qm, data, match_tables, starts, T, stride=stride, batch_size=batch_size, tok_lens=tok_lens, n_bytes_total=n_bytes)
        del qm

        wt_mb = total_ent_bytes / 1e6
        vocab_mb = vocab_bytes / 1e6
        extra_mb = extra_bytes / 1e6
        hutter_mb = r["payload_mb"] + wt_mb + vocab_mb + extra_mb + code_overhead_mb
        return dict(
            eval_bpb=r["full_bpb"],
            q_strided_bpb=r["strided_bpb"],
            q_payload_mb=r["payload_mb"],
            weight_mb=wt_mb,
            emb_bpw=tier_bpw["emb"],
            block_bpw=tier_bpw["blk"],
            expert_bpw=tier_bpw["exp"],
            emb_mb=tier_mb["emb"],
            block_mb=tier_mb["blk"],
            expert_mb=tier_mb["exp"],
            expert_zero_frac=tier_zero["exp"],
            scale_mb=sum(tier_scale_bytes.values()) / 1e6,
            hutter_mb=hutter_mb,
        )


def main():
    DATA = os.environ.get("DATA", "enwik9.split4096")
    MATCH_K = tuple(int(x) for x in os.environ.get("MATCH_K", "2,3,5,8,13,27").split(","))
    INIT_FROM = os.environ["INIT_FROM"]
    OUT_DIR = os.environ.get("OUT_DIR", "runs/ecqat")
    RUN_NAME = os.environ.get("RUN_NAME", "v10-moe17m-ecqat-15k")
    os.makedirs(OUT_DIR, exist_ok=True)

    B, T = 16, 2048
    STEPS = int(os.environ.get("STEPS", "15000"))
    LOG = int(os.environ.get("LOG", "100"))
    CKPT = int(os.environ.get("CKPT", "1500"))
    EVAL_WINDOWS = int(os.environ.get("EVAL_WINDOWS", "2048"))
    WARMUP = int(os.environ.get("WARMUP", "500"))
    LR_W = float(os.environ.get("LR_W", "3e-4"))
    LR_DELTA = float(os.environ.get("LR_DELTA", "3e-3"))
    LR_B = float(os.environ.get("LR_B", "5e-3"))
    RATE_MULT = float(os.environ.get("RATE_MULT", "1.0"))  # 1.0 = exact 1:1 Hutter MDL!
    EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.998"))

    N_LAYER = int(os.environ.get("N_LAYER", "8"))
    N_LOOP = int(os.environ.get("N_LOOP", "1"))
    N_EXPERTS = int(os.environ.get("N_EXPERTS", "8"))
    TOP_K = int(os.environ.get("TOP_K", "2"))
    SHARED_H = int(os.environ.get("SHARED_H", "256"))
    EXPERT_H = int(os.environ.get("EXPERT_H", "256"))

    torch.set_float32_matmul_precision("high")
    N_BYTES = os.path.getsize("enwik9")
    TOK_LENS, VOCAB_BYTES, VOCAB = vocab_info(DATA)
    EXTRA_BYTES = extra_overhead_bytes(DATA)
    data = torch.from_numpy(np.fromfile(DATA + ".i16", dtype=np.int16)).cuda()
    N = len(data)
    match_tables = matchfeat.cached(data, MATCH_K, DATA)
    torch.cuda.empty_cache()
    offs = torch.arange(T + 1, device="cuda")
    eval_starts = make_strided_starts(N, T, T // 2, EVAL_WINDOWS)

    cfg = GPTConfig(
        context_length=T, vocab_size=VOCAB, n_layer=N_LAYER, n_loop=N_LOOP,
        n_experts=N_EXPERTS, n_experts_active=TOP_K, shared_h=SHARED_H, expert_h=EXPERT_H,
        n_match=len(MATCH_K), match_codes=matchfeat.N_CODE, fp8=False,
    )
    base_model = GPT(cfg).cuda()
    base_model.load_state_dict(torch.load(INIT_FROM, map_location="cuda", weights_only=True))
    base_model.build_swa_mask()

    ec = ECQATWrapper(base_model, init_emb_bits=6, init_blk_bits=5, init_exp_bits=4).cuda()
    ema_ec = copy.deepcopy(ec).requires_grad_(False)
    ema_p, mod_p = list(ema_ec.parameters()), list(ec.parameters())

    print(f"Loaded {INIT_FROM} | params={sum(p.numel() for p in base_model.parameters()):,} | rate_weight={RATE_MULT}/N ({RATE_MULT/N:.3e})")
    ev0 = ec.exact_archive_eval(data, match_tables, eval_starts, T, T // 2, B, TOK_LENS, N_BYTES, VOCAB_BYTES, EXTRA_BYTES)
    print(
        f"step 0 | init q_strided {ev0['q_strided_bpb']:.4f} | payload {ev0['q_payload_mb']:.2f}MB | "
        f"wt {ev0['weight_mb']:.2f}MB (emb {ev0['emb_bpw']:.2f}, blk {ev0['block_bpw']:.2f}, exp {ev0['expert_bpw']:.2f}bpw, scales {ev0['scale_mb']:.2f}MB) | "
        f"HUTTER {ev0['hutter_mb']:.2f} MB",
        flush=True,
    )
    if os.environ.get("EVAL_ONLY", "0") == "1":
        return

    opt = torch.optim.AdamW(
        [
            dict(params=list(ec.model.parameters()), lr=LR_W, weight_decay=0.0),
            dict(params=list(ec.log_deltas.parameters()), lr=LR_DELTA, weight_decay=0.0),
            dict(params=list(ec.log_bs.parameters()), lr=LR_B, weight_decay=0.0),
        ],
        fused=True,
    )
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]

    # Batches helper (fresh seed)
    gen = torch.Generator().manual_seed(9876)
    ring = [
        [torch.empty(B, dtype=torch.long).pin_memory()]
        + [torch.empty(B, T, a.shape[1], dtype=torch.from_numpy(np.empty(0, a.dtype)).dtype).pin_memory() for a in match_tables]
        for _ in range(8)
    ]
    events = [None] * 8
    bi = 0

    def next_batch():
        nonlocal bi
        i, bi = bi, (bi + 1) % 8
        if events[i] is not None:
            events[i].synchronize()
        bufs = ring[i]
        torch.randint(0, N - T, (B,), generator=gen, out=bufs[0])
        for a, buf in zip(match_tables, bufs[1:]):
            bn = buf.numpy()
            for r, s in enumerate(bufs[0].tolist()):
                bn[r] = a[s + 1 : s + 1 + T]
        out = [b.cuda(non_blocking=True) for b in bufs]
        events[i] = torch.cuda.Event()
        events[i].record()
        return out[0], out[1:]

    @torch.compile(mode="default")
    def step_fn(x, y, mp, ml):
        logits, rate_nats = ec.forward_with_rate(x, mp, ml)
        ce = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
        mdl_loss = ce + (RATE_MULT / N) * rate_nats
        return mdl_loss, ce, rate_nats

    def lr_mult(step):
        if WARMUP > 0 and step < WARMUP:
            return (step + 1) / WARMUP
        return max(0.0, 1.0 - (step - WARMUP) / max(STEPS - WARMUP, 1))

    wandb.init(
        project="hutter",
        name=RUN_NAME,
        config=dict(
            method="ecqat", INIT_FROM=INIT_FROM, STEPS=STEPS, LR_W=LR_W, LR_DELTA=LR_DELTA,
            LR_B=LR_B, RATE_MULT=RATE_MULT, WARMUP=WARMUP, EMA_DECAY=EMA_DECAY,
            EVAL_WINDOWS=EVAL_WINDOWS, DATA=DATA, **vars(cfg),
        ),
    )
    wandb.log(ev0, step=0)
    best_hutter = ev0["hutter_mb"]
    t0 = time.time()

    for step in range(STEPS):
        starts, match = next_batch()
        w = data[starts[:, None] + offs].long()
        mp, ml = match
        mdl_loss, ce, rate_nats = step_fn(w[:, :-1], w[:, 1:], mp, ml)
        mdl_loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(ec.parameters(), 1.0)
        lrm = lr_mult(step)
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * lrm
        opt.step()
        ec.zero_grad(set_to_none=True)
        ec.model.update_router_biases(gamma=5e-4)

        torch._foreach_lerp_(ema_p, mod_p, 1.0 - EMA_DECAY)
        for eb, mb in zip(ema_ec.model.h, ec.model.h):
            if mb.moe is not None:
                eb.moe.router_bias.copy_(mb.moe.router_bias)

        if (step + 1) % LOG == 0 or step == STEPS - 1:
            ce_v = ce.item()
            rate_mb = rate_nats.item() / math.log(2) / 8e6
            bpb = ce_v / math.log(2) * N / N_BYTES
            dt = (time.time() - t0) / LOG
            t0 = time.time()
            print(
                f"step {step + 1} | bpb {bpb:.4f} | laplace_wt {rate_mb:.2f}MB | {dt * 1000:.1f}ms/step",
                flush=True,
            )
            wandb.log(
                dict(bpb=bpb, loss=ce_v, laplace_wt_mb=rate_mb, ms_per_step=dt * 1000, grad_norm=gnorm.item(), lr_mult=lrm),
                step=step + 1,
            )

        if (step + 1) % CKPT == 0 or step == STEPS - 1:
            ev = ema_ec.exact_archive_eval(data, match_tables, eval_starts, T, T // 2, B, TOK_LENS, N_BYTES, VOCAB_BYTES, EXTRA_BYTES)
            print(
                f"step {step + 1} | q_strided {ev['q_strided_bpb']:.4f} | payload {ev['q_payload_mb']:.2f}MB | "
                f"wt {ev['weight_mb']:.2f}MB (emb {ev['emb_bpw']:.2f}, blk {ev['block_bpw']:.2f}, exp {ev['expert_bpw']:.2f}bpw, z={ev['expert_zero_frac']*100:.1f}%) | "
                f"HUTTER {ev['hutter_mb']:.2f} MB",
                flush=True,
            )
            wandb.log(ev, step=step + 1)
            if ev["hutter_mb"] < best_hutter:
                best_hutter = ev["hutter_mb"]
                torch.save(ema_ec.state_dict(), os.path.join(OUT_DIR, "ecqat_best.pt"))
            torch.save(ema_ec.state_dict(), os.path.join(OUT_DIR, "ecqat_last.pt"))

    print(f"Done! Best HUTTER TOTAL: {best_hutter:.3f} MB")
    wandb.finish()


if __name__ == "__main__":
    main()

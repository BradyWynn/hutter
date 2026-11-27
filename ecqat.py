"""Entropy-Constrained QAT (ECQAT v2) — Exact 1:1 Hutter Prize Rate-Distortion Optimization.

Branches from a converged v10 checkpoint and directly minimizes exact Hutter total nats:
    L(w, delta, rate_params) = CE_nats(x, y) + (RATE_MULT / N_tok) * sum_t Rate_nats(w_t; ...)

Key upgrades over v1 (all configurable via env vars for clean ablations):
  - RATE_MODEL=mix2 (default) | laplace:
      2-component zero-centered Ballé Laplace mixture per tensor ((1-pi)*Lap(b1) + pi*Lap(b2)),
      modeling both the sharp spike at q=0 and wide tails accurately.
  - DELTA_MODE=rowcol (default) | row:
      Rank-1 factorized per-row x per-column step size delta = delta_r[out,1] * delta_c[1,in]
      (+0.08 MB fp16 scales total) to jointly adapt to output-row and input-channel sensitivity.
  - WHT=0 (default) | 1:
      Exact orthogonal Walsh-Hadamard transform along in_features (decoded ONCE at CPU startup
      in <5ms, 0% inference FLOPs overhead, 0 extra archive bytes).
  - Exact per-tensor empirical Shannon entropy + honest zlib-9 frequency table overhead
    (also reports legacy 3-tier pooled weight_mb for apples-to-apples comparison).
"""
import copy
import math
import os
import struct
import time
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from model import GPT, GPTConfig, FP8Linear, QATLinear, MoEMLP
from eval_hutter import vocab_info, extra_overhead_bytes, stream_bytes_for, make_strided_starts, eval_strided
import matchfeat


_H_CACHE = {}


def hadamard_matrix(n, device, dtype=torch.float32):
    """Exact orthogonal normalized Sylvester Hadamard matrix H_n (H @ H == I, H == H.T)."""
    assert n > 0 and (n & (n - 1)) == 0, f"WHT dim {n} must be a power of 2"
    key = (n, str(device), dtype)
    if key not in _H_CACHE:
        h = torch.ones((1, 1), dtype=torch.float32)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        h = (h / math.sqrt(n)).to(device=device, dtype=dtype).contiguous()
        _H_CACHE[key] = h
    return _H_CACHE[key]


def _laplace_log_prob(au, b):
    """Exact Ballé-style zero-centered Laplace bin log-probability log P(|u|; b), C1 smooth."""
    half = 0.5 / b
    p_inner = (1.0 - torch.exp(-half) * torch.cosh(au.clamp_max(0.5) / b)).clamp_min(1e-12)
    lp_inner = torch.log(p_inner)
    log_sinh = half + torch.log((1.0 - torch.exp(-2.0 * half)).clamp_min(1e-12) * 0.5)
    lp_outer = log_sinh - au / b
    return torch.where(au <= 0.5, lp_inner, lp_outer)


def laplace_rate_nats(u, log_b):
    b = log_b.exp().clamp(0.05, 50.0)
    return (-_laplace_log_prob(u.abs(), b)).sum()


def mix2_laplace_rate_nats(u, log_b1, log_b2, logit_pi):
    """2-component zero-centered Laplace mixture rate (-log(w1*P(u;b1) + w2*P(u;b2))) in nats."""
    au = u.abs()
    b1 = log_b1.exp().clamp(0.03, 50.0)
    b2 = log_b2.exp().clamp(0.05, 50.0)
    lp1 = F.logsigmoid(logit_pi) + _laplace_log_prob(au, b1)
    lp2 = F.logsigmoid(-logit_pi) + _laplace_log_prob(au, b2)
    return (-torch.logaddexp(lp1, lp2)).sum()


def ste_quant(w, log_delta_r, log_delta_c=None):
    """Per-row (or factorized per-row x per-col) learnable step size quantization with LSQ STE."""
    if log_delta_c is not None:
        delta = (log_delta_r + log_delta_c).exp().clamp(1e-5, 10.0)
    else:
        delta = log_delta_r.exp().clamp(1e-5, 10.0)
    u = w.float() / delta
    u_ste = u + (u.round() - u).detach()
    return (u_ste * delta).to(w.dtype), u


def _encode_sym_table_bytes(uniq_vals, counts):
    """Honest byte cost for storing a tensor's empirical integer histogram for rANS."""
    if len(uniq_vals) == 0:
        return 0
    # Store (min_sym: i16, n_bins: u16) + uint16 quantized frequencies (12-bit rANS table) zlib-compressed
    lo, hi = int(uniq_vals[0]), int(uniq_vals[-1])
    span = hi - lo + 1
    if span <= 4096:
        tot = float(counts.sum())
        dense = np.zeros(span, dtype=np.uint16)
        # Quantize probabilities onto 12-bit rANS grid (minimum 1 for any observed symbol)
        q_cnt = np.maximum(1, np.rint(counts / tot * 4096.0).astype(np.int64))
        dense[uniq_vals - lo] = np.minimum(q_cnt, 65535).astype(np.uint16)
        raw = struct.pack("<hH", max(-32768, min(32767, lo)), span) + dense.tobytes()
    else:
        raw = struct.pack("<I", len(uniq_vals)) + uniq_vals.astype(np.int16).tobytes() + np.minimum(counts, 65535).astype(np.uint16).tobytes()
    return len(zlib.compress(raw, 9))


class ECQATWrapper(nn.Module):
    """Wraps a GPT model with learnable step sizes, optional orthogonal WHT weight-space rotation,
    and per-tensor Laplace / 2-component Laplace mixture rate models."""

    def __init__(
        self,
        model,
        init_emb_bits=6,
        init_blk_bits=5,
        init_exp_bits=4,
        rate_model="mix2",
        delta_mode="rowcol",
        use_wht=False,
    ):
        super().__init__()
        assert rate_model in ("laplace", "mix2")
        assert delta_mode in ("row", "rowcol")
        self.model = model
        self.rate_model = rate_model
        self.delta_mode = delta_mode
        self.use_wht = bool(use_wht)
        self.model.set_qat(group_size=0)

        self.log_deltas_r = nn.ParameterDict()
        self.log_deltas_c = nn.ParameterDict()
        self.log_bs = nn.ParameterDict()
        self.log_b1s = nn.ParameterDict()
        self.logit_pis = nn.ParameterDict()
        self.specs = []  # (key, tier, module, attr_name, wht_dim)

        def register_mat(key, tier, mod, attr, bits):
            p = getattr(mod, attr)
            w = p.detach().float()
            in_dim = w.shape[-1]
            wht_dim = in_dim if (self.use_wht and (in_dim & (in_dim - 1)) == 0) else 0
            if wht_dim:
                H = hadamard_matrix(wht_dim, w.device, torch.float32)
                w_rot = w @ H
                p.data.copy_(w_rot.to(p.dtype))
                w = w_rot
            qmax = float((1 << (bits - 1)) - 1)
            d_init = (w.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-4)
            u_init = w / d_init
            b_init = u_init.abs().mean().clamp(0.2, 20.0)
            self.log_deltas_r[key] = nn.Parameter(d_init.log())
            if self.delta_mode == "rowcol":
                c_shape = list(w.shape)
                c_shape[-2] = 1
                self.log_deltas_c[key] = nn.Parameter(torch.zeros(c_shape, device=w.device, dtype=torch.float32))
            self.log_bs[key] = nn.Parameter(b_init.log().reshape(()))
            if self.rate_model == "mix2":
                # Narrow zero-spike component + wide tail component
                self.log_b1s[key] = nn.Parameter(torch.tensor(math.log(0.35), device=w.device, dtype=torch.float32))
                self.logit_pis[key] = nn.Parameter(torch.tensor(-1.0, device=w.device, dtype=torch.float32))
            self.specs.append((key, tier, mod, attr, wht_dim))

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

    def delta_parameters(self):
        ps = list(self.log_deltas_r.parameters())
        if self.delta_mode == "rowcol":
            ps.extend(self.log_deltas_c.parameters())
        return ps

    def rate_parameters(self):
        ps = list(self.log_bs.parameters())
        if self.rate_model == "mix2":
            ps.extend(self.log_b1s.parameters())
            ps.extend(self.logit_pis.parameters())
        return ps

    def forward_with_rate(self, x, mp, ml):
        saved = []
        total_rate_nats = torch.zeros((), device=x.device, dtype=torch.float32)
        for key, tier, mod, attr, wht_dim in self.specs:
            orig_p = getattr(mod, attr)
            ldc = self.log_deltas_c[key] if self.delta_mode == "rowcol" else None
            wq, u = ste_quant(orig_p, self.log_deltas_r[key], ldc)
            if self.rate_model == "mix2":
                total_rate_nats = total_rate_nats + mix2_laplace_rate_nats(
                    u, self.log_b1s[key], self.log_bs[key], self.logit_pis[key]
                )
            else:
                total_rate_nats = total_rate_nats + laplace_rate_nats(u, self.log_bs[key])
            if wht_dim:
                H = hadamard_matrix(wht_dim, wq.device, torch.float32)
                wq = (wq.float() @ H).to(orig_p.dtype)
            saved.append((mod, attr, orig_p))
            mod._parameters[attr] = wq
        try:
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
    def exact_archive_eval(
        self,
        data,
        match_tables,
        starts,
        T,
        stride,
        batch_size,
        tok_lens,
        n_bytes,
        vocab_bytes,
        extra_bytes=0,
        code_overhead_mb=0.05,
        n_bytes_stream=None,
    ):
        """Snap every weight to exact fp16(delta) * round(W / fp16(delta)) (+ inverse WHT if enabled),
        snap misc to fp16, compute TRUE per-tensor empirical Shannon entropy + rANS table overhead
        + fp16 scale bytes + zlib-9 fp16 misc + exact strided NLL -> honest hutter_mb."""
        qm = copy.deepcopy(self.model).eval().requires_grad_(False)
        qm.set_qat(group_size=0)

        tier_q = {"emb": [], "blk": [], "exp": []}
        tier_pt_ent_bytes = {"emb": 0, "blk": 0, "exp": 0}
        tier_tbl_bytes = {"emb": 0, "blk": 0, "exp": 0}
        tier_scale_raw = {"emb": bytearray(), "blk": bytearray(), "exp": bytearray()}
        tier_scale_raw_bytes = {"emb": 0, "blk": 0, "exp": 0}
        tier_params = {"emb": 0, "blk": 0, "exp": 0}
        tier_zeros = {"emb": 0, "blk": 0, "exp": 0}
        q_ids = set()
        scale_bits = int(os.environ.get("SCALE_BITS", "16"))

        def snap_scale(log_d, lo_c, hi_c):
            if scale_bits == 8:
                ld = log_d.detach().float().clamp(math.log(lo_c), math.log(hi_c))
                mn = float(ld.min().item())
                mx = float(ld.max().item())
                span = max(mx - mn, 1e-6)
                q8 = torch.round((ld - mn) / span * 255.0).clamp(0, 255).to(torch.uint8)
                d_rec = (mn + q8.float() * (span / 255.0)).exp()
                raw_b = struct.pack("<ff", mn, span) + q8.cpu().numpy().tobytes()
                return d_rec, raw_b
            d16 = log_d.detach().exp().clamp(lo_c, hi_c).to(torch.float16)
            return d16.float(), d16.cpu().numpy().tobytes()

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

        for key, tier, _, _, wht_dim in self.specs:
            mod, attr = get_target(qm, key)
            p = getattr(mod, attr)
            q_ids.add(id(p))
            dr_f, dr_b = snap_scale(self.log_deltas_r[key], 1e-5, 10.0)
            tier_scale_raw[tier].extend(dr_b)
            tier_scale_raw_bytes[tier] += len(dr_b)
            if self.delta_mode == "rowcol":
                dc_f, dc_b = snap_scale(self.log_deltas_c[key], 1e-3, 1e3)
                delta = (dr_f * dc_f).clamp(1e-5, 10.0)
                tier_scale_raw[tier].extend(dc_b)
                tier_scale_raw_bytes[tier] += len(dc_b)
            else:
                delta = dr_f

            q = (p.detach().float() / delta).round()
            wq = q * delta
            if wht_dim:
                H = hadamard_matrix(wht_dim, wq.device, torch.float32)
                wq = wq @ H
            p.copy_(wq.to(p.dtype))

            q_np = q.to(torch.int32).cpu().numpy().ravel()
            tier_q[tier].append(q_np)
            tier_params[tier] += len(q_np)
            tier_zeros[tier] += int((q_np == 0).sum())
            uniq, counts = np.unique(q_np, return_counts=True)
            probs = counts / counts.sum()
            h_bits = float(-(probs * np.log2(probs)).sum())
            tier_pt_ent_bytes[tier] += int(math.ceil(len(q_np) * h_bits / 8.0))
            tier_tbl_bytes[tier] += _encode_sym_table_bytes(uniq, counts)

        if qm.c.tie_weights:
            qm.lm_head.weight = qm.wte.weight
            q_ids.add(id(qm.lm_head.weight))

        other = [p for p in qm.parameters() if id(p) not in q_ids]
        other_bufs = [b.moe.router_bias for b in qm.h if b.moe is not None]
        for p in other + other_bufs:
            p.copy_(p.to(torch.float16).to(p.dtype))
        other_fp16 = b"".join(p.detach().to(torch.float16).cpu().numpy().tobytes() for p in (other + other_bufs))
        other_zlib = len(zlib.compress(other_fp16, 9))

        tier_scale_bytes = {
            t: min(tier_scale_raw_bytes[t], len(zlib.compress(bytes(tier_scale_raw[t]), 9))) if tier_scale_raw_bytes[t] else 0
            for t in ("emb", "blk", "exp")
        }
        tier_bpw = {}
        tier_mb = {}
        tier_zero = {}
        total_pt_bytes = other_zlib
        total_pool_bytes = other_zlib
        for tier in ("emb", "blk", "exp"):
            if not tier_q[tier]:
                tier_bpw[tier] = 0.0
                tier_mb[tier] = 0.0
                tier_zero[tier] = 0.0
                continue
            arr = np.concatenate(tier_q[tier])
            _, counts = np.unique(arr, return_counts=True)
            probs = counts / counts.sum()
            h_pool = float(-(probs * np.log2(probs)).sum())
            pool_bytes = int(math.ceil(len(arr) * h_pool / 8.0)) + tier_scale_raw_bytes[tier]
            pt_bytes = tier_pt_ent_bytes[tier] + tier_tbl_bytes[tier] + tier_scale_bytes[tier]
            tier_bpw[tier] = (tier_pt_ent_bytes[tier] * 8.0) / max(tier_params[tier], 1)
            tier_mb[tier] = pt_bytes / 1e6
            tier_zero[tier] = tier_zeros[tier] / max(tier_params[tier], 1)
            total_pt_bytes += pt_bytes
            total_pool_bytes += pool_bytes

        r = eval_strided(
            qm, data, match_tables, starts, T,
            stride=stride, batch_size=batch_size, tok_lens=tok_lens,
            n_bytes_total=n_bytes, n_bytes_stream=n_bytes_stream,
        )
        del qm

        wt_mb = total_pt_bytes / 1e6
        wt_pool_mb = total_pool_bytes / 1e6
        vocab_mb = vocab_bytes / 1e6
        extra_mb = extra_bytes / 1e6
        hutter_mb = r["payload_mb"] + wt_mb + vocab_mb + extra_mb + code_overhead_mb
        hutter_s12_mb = r["payload_mb"] + 2.0 * wt_mb + 2.0 * vocab_mb + extra_mb + 2.0 * code_overhead_mb
        return dict(
            eval_bpb=r["full_bpb"],
            q_strided_bpb=r["strided_bpb"],
            q_payload_mb=r["payload_mb"],
            weight_mb=wt_mb,
            weight_pooled_mb=wt_pool_mb,
            table_mb=sum(tier_tbl_bytes.values()) / 1e6,
            emb_bpw=tier_bpw["emb"],
            block_bpw=tier_bpw["blk"],
            expert_bpw=tier_bpw["exp"],
            emb_mb=tier_mb["emb"],
            block_mb=tier_mb["blk"],
            expert_mb=tier_mb["exp"],
            expert_zero_frac=tier_zero["exp"],
            scale_mb=sum(tier_scale_bytes.values()) / 1e6,
            hutter_mb=hutter_mb,
            hutter_s12_mb=hutter_s12_mb,
        )


def main():
    DATA = os.environ.get("DATA", "enwik9.split4096")
    MATCH_K = tuple(int(x) for x in os.environ.get("MATCH_K", "2,3,5,8,13,27").split(","))
    INIT_FROM = os.environ["INIT_FROM"]
    OUT_DIR = os.environ.get("OUT_DIR", "runs/ecqat")
    RUN_NAME = os.environ.get("RUN_NAME", "v10-moe17m-ecqat")
    os.makedirs(OUT_DIR, exist_ok=True)

    B = int(os.environ.get("B", "16"))
    ACCUM = int(os.environ.get("ACCUM", "1"))
    T = int(os.environ.get("T", "2048"))
    STEPS = int(os.environ.get("STEPS", "15000"))
    LOG = int(os.environ.get("LOG", "100"))
    CKPT = int(os.environ.get("CKPT", "1500"))
    EVAL_WINDOWS = int(os.environ.get("EVAL_WINDOWS", "2048"))
    WARMUP = int(os.environ.get("WARMUP", "500"))
    LR_W = float(os.environ.get("LR_W", "3e-4"))
    LR_DELTA = float(os.environ.get("LR_DELTA", "3e-3"))
    LR_B = float(os.environ.get("LR_B", "5e-3"))
    RATE_MULT = float(os.environ.get("RATE_MULT", "1.0"))
    RATE_WARMUP = int(os.environ.get("RATE_WARMUP", "0"))
    KD_ALPHA = float(os.environ.get("KD_ALPHA", "0.0"))
    KD_TEMP = float(os.environ.get("KD_TEMP", "1.0"))
    EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.998"))
    RATE_MODEL = os.environ.get("RATE_MODEL", "mix2")
    DELTA_MODE = os.environ.get("DELTA_MODE", "rowcol")
    USE_WHT = bool(int(os.environ.get("WHT", "0")))

    N_LAYER = int(os.environ.get("N_LAYER", "8"))
    N_LOOP = int(os.environ.get("N_LOOP", "1"))
    N_EMBD = int(os.environ.get("N_EMBD", "256"))
    N_HEAD = int(os.environ.get("N_HEAD", "4"))
    N_KV_HEAD = int(os.environ.get("N_KV_HEAD", "2"))
    N_EXPERTS = int(os.environ.get("N_EXPERTS", "8"))
    TOP_K = int(os.environ.get("TOP_K", "2"))
    SHARED_H = int(os.environ.get("SHARED_H", "256"))
    EXPERT_H = int(os.environ.get("EXPERT_H", "256"))

    torch.set_float32_matmul_precision("high")
    N_BYTES = os.path.getsize("enwik9")
    N_BYTES_STREAM = stream_bytes_for(DATA, N_BYTES)
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
        n_embd=N_EMBD, n_head=N_HEAD, n_kv_head=N_KV_HEAD,
        n_experts=N_EXPERTS, n_experts_active=TOP_K, shared_h=SHARED_H, expert_h=EXPERT_H,
        n_match=len(MATCH_K), match_codes=matchfeat.N_CODE, fp8=False,
    )
    base_model = GPT(cfg).cuda()
    base_model.load_state_dict(torch.load(INIT_FROM, map_location="cuda", weights_only=True))
    base_model.build_swa_mask()

    teacher = None
    if KD_ALPHA > 0.0:
        teacher = copy.deepcopy(base_model).eval().requires_grad_(False)
        teacher.build_swa_mask()

    ec = ECQATWrapper(
        base_model,
        init_emb_bits=6,
        init_blk_bits=5,
        init_exp_bits=4,
        rate_model=RATE_MODEL,
        delta_mode=DELTA_MODE,
        use_wht=USE_WHT,
    ).cuda()
    ema_ec = copy.deepcopy(ec).requires_grad_(False)
    ema_p, mod_p = list(ema_ec.parameters()), list(ec.parameters())

    print(
        f"Loaded {INIT_FROM} | params={sum(p.numel() for p in base_model.parameters()):,} | "
        f"rate_model={RATE_MODEL}, delta_mode={DELTA_MODE}, wht={int(USE_WHT)}, "
        f"kd_alpha={KD_ALPHA}, rate_warmup={RATE_WARMUP}, scale_bits={os.environ.get('SCALE_BITS', '16')} | "
        f"rate_weight={RATE_MULT}/N ({RATE_MULT/N:.3e})"
    )
    ev0 = ec.exact_archive_eval(data, match_tables, eval_starts, T, T // 2, B, TOK_LENS, N_BYTES, VOCAB_BYTES, EXTRA_BYTES, n_bytes_stream=N_BYTES_STREAM)
    print(
        f"step 0 | init q_strided {ev0['q_strided_bpb']:.4f} | payload {ev0['q_payload_mb']:.2f}MB | "
        f"wt {ev0['weight_mb']:.2f}MB (pool {ev0['weight_pooled_mb']:.2f}MB, emb {ev0['emb_bpw']:.2f}, blk {ev0['block_bpw']:.2f}, exp {ev0['expert_bpw']:.2f}bpw, scales {ev0['scale_mb']:.2f}MB, tbl {ev0['table_mb']*1e3:.1f}KB) | "
        f"S2 {ev0['hutter_mb']:.2f} MB | S1+S2 {ev0['hutter_s12_mb']:.2f} MB",
        flush=True,
    )
    if os.environ.get("EVAL_ONLY", "0") == "1":
        return

    opt = torch.optim.AdamW(
        [
            dict(params=list(ec.model.parameters()), lr=LR_W, weight_decay=0.0),
            dict(params=ec.delta_parameters(), lr=LR_DELTA, weight_decay=0.0),
            dict(params=ec.rate_parameters(), lr=LR_B, weight_decay=0.0),
        ],
        fused=True,
    )
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]

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
    def teacher_fn(x, mp, ml):
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            return teacher(x, mp, ml)

    @torch.compile(mode="default")
    def step_fn(x, y, mp, ml, t_logits, rate_w):
        logits, rate_nats = ec.forward_with_rate(x, mp, ml)
        logits_f = logits.flatten(0, 1).float()
        ce = F.cross_entropy(logits_f, y.flatten())
        if t_logits is not None:
            t_logp = F.log_softmax(t_logits.flatten(0, 1).float() / KD_TEMP, -1)
            s_logp = F.log_softmax(logits_f / KD_TEMP, -1)
            kl = F.kl_div(s_logp, t_logp, log_target=True, reduction="batchmean")
            data_loss = (1.0 - KD_ALPHA) * ce + KD_ALPHA * (KD_TEMP ** 2) * kl
        else:
            data_loss = ce
        mdl_loss = data_loss + rate_w * rate_nats
        return mdl_loss, ce, rate_nats

    def lr_mult(step):
        if WARMUP > 0 and step < WARMUP:
            return (step + 1) / WARMUP
        return max(0.0, 1.0 - (step - WARMUP) / max(STEPS - WARMUP, 1))

    wandb.init(
        project="hutter",
        name=RUN_NAME,
        config=dict(
            method="ecqat_v2", INIT_FROM=INIT_FROM, STEPS=STEPS, LR_W=LR_W, LR_DELTA=LR_DELTA,
            LR_B=LR_B, RATE_MULT=RATE_MULT, RATE_WARMUP=RATE_WARMUP, KD_ALPHA=KD_ALPHA, KD_TEMP=KD_TEMP,
            RATE_MODEL=RATE_MODEL, DELTA_MODE=DELTA_MODE, SCALE_BITS=int(os.environ.get("SCALE_BITS", "16")),
            USE_WHT=USE_WHT, WARMUP=WARMUP, EMA_DECAY=EMA_DECAY,
            EVAL_WINDOWS=EVAL_WINDOWS, DATA=DATA, MATCH_MODE=matchfeat.MATCH_MODE, **vars(cfg),
        ),
    )
    wandb.log(ev0, step=0)
    best_key = "hutter_s12_mb" if RATE_MULT >= 1.5 else "hutter_mb"
    best_hutter = ev0[best_key]
    rate_w_t = torch.zeros((), device="cuda", dtype=torch.float32)
    t0 = time.time()

    for step in range(STEPS):
        rw = (RATE_MULT / N) * (min(1.0, (step + 1) / RATE_WARMUP) if RATE_WARMUP > 0 else 1.0)
        rate_w_t.fill_(rw)
        ce_acc = 0.0
        rate_acc = 0.0
        for _ in range(ACCUM):
            starts, match = next_batch()
            w = data[starts[:, None] + offs].long()
            mp, ml = match
            x_in = w[:, :-1]
            t_logits = teacher_fn(x_in, mp, ml) if teacher is not None else None
            mdl_loss, ce, rate_nats = step_fn(x_in, w[:, 1:], mp, ml, t_logits, rate_w_t)
            (mdl_loss / ACCUM).backward()
            ce_acc += ce.detach() / ACCUM
            rate_acc += rate_nats.detach() / ACCUM
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
            ce_v = ce_acc.item()
            rate_mb = rate_acc.item() / math.log(2) / 8e6
            bpb = ce_v / math.log(2) * N / N_BYTES
            dt = (time.time() - t0) / LOG
            t0 = time.time()
            print(
                f"step {step + 1} | bpb {bpb:.4f} | model_wt {rate_mb:.2f}MB | {dt * 1000:.1f}ms/step",
                flush=True,
            )
            wandb.log(
                dict(bpb=bpb, loss=ce_v, laplace_wt_mb=rate_mb, ms_per_step=dt * 1000, grad_norm=gnorm.item(), lr_mult=lrm),
                step=step + 1,
            )

        if (step + 1) % CKPT == 0 or step == STEPS - 1:
            ev = ema_ec.exact_archive_eval(data, match_tables, eval_starts, T, T // 2, B, TOK_LENS, N_BYTES, VOCAB_BYTES, EXTRA_BYTES, n_bytes_stream=N_BYTES_STREAM)
            print(
                f"step {step + 1} | q_strided {ev['q_strided_bpb']:.4f} | payload {ev['q_payload_mb']:.2f}MB | "
                f"wt {ev['weight_mb']:.2f}MB (pool {ev['weight_pooled_mb']:.2f}MB, emb {ev['emb_bpw']:.2f}, blk {ev['block_bpw']:.2f}, exp {ev['expert_bpw']:.2f}bpw, z={ev['expert_zero_frac']*100:.1f}%) | "
                f"S2 {ev['hutter_mb']:.2f} MB | S1+S2 {ev['hutter_s12_mb']:.2f} MB",
                flush=True,
            )
            wandb.log(ev, step=step + 1)
            if ev[best_key] < best_hutter:
                best_hutter = ev[best_key]
                torch.save(ema_ec.state_dict(), os.path.join(OUT_DIR, "ecqat_best.pt"))
            torch.save(ema_ec.state_dict(), os.path.join(OUT_DIR, "ecqat_last.pt"))

    print(f"Done! Best {best_key}: {best_hutter:.3f} MB")
    wandb.finish()


if __name__ == "__main__":
    main()

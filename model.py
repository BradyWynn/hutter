from dataclasses import dataclass
import math
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


@dataclass
class GPTConfig:
    context_length: int = 2048
    vocab_size: int = 4096
    n_layer: int = 7
    n_loop: int = 2
    n_head: int = 4
    n_kv_head: int = 2
    n_embd: int = 256
    conv_k: int = 4
    tie_weights: bool = True
    swiglu: bool = True        # SwiGLU MLP (hidden=704) instead of ReLU^2 (hidden=1024)
    rope_frac: float = 0.5     # fraction of head_dim rotated by RoPE (0.5 = half-truncated RoPE)
    unet_skips: bool = True    # mirrored encoder->decoder U-Net skip connections across passes
    attn_gate: bool = True     # per-head sigmoid attention output gating
    pass_adarms: bool = True   # per-pass AdaRMS scale+shift after norm(x) (replaces loop_emb)
    swa_window: int = 256      # local sliding-window size in tokens (0 disables SWA)
    swa_full_every: int = 4    # every Nth pass uses full causal attention
    qat_group: int = 0         # symmetric 4-bit [-7, 7] STE QAT group size (0 disables QAT)
    fp8: bool = True
    n_match: int = (
        0  # match feature columns (one per context length k, see matchfeat.py)
    )
    match_codes: int = (
        256  # match code byte: 0 = no match, else (length bin, distance bin)
    )


E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2


def quant(t, dt):
    m = torch.finfo(dt).max
    s = m / t.abs().amax().float().clamp_min(1e-12)
    return (t.float() * s).clamp(-m, m).to(dt), s.reciprocal()


def fake_quant_4bit(w, group_size=64):
    """Per-group symmetric 4-bit [-7, 7] STE quantization with fp16 group scales.
    Forward uses exact dequantized int4 + fp16-scale weights so archive export has 0.0 drop."""
    if not group_size:
        return w
    orig_shape = w.shape
    wg = w.float().reshape(-1, group_size)
    scale = (wg.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(1e-7).to(torch.float16).float()
    q = (wg / scale).round().clamp(-7.0, 7.0)
    wq = (q * scale).reshape(orig_shape).to(w.dtype)
    return w + (wq - w).detach()


class FP8MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        xq, sx = quant(x, E4)
        wq, sw = quant(w, E4)
        ctx.save_for_backward(xq, sx, wq, sw)
        return torch._scaled_mm(xq, wq.t(), sx, sw, out_dtype=torch.bfloat16)

    @staticmethod
    def backward(ctx, g):
        xq, sx, wq, sw = ctx.saved_tensors
        gq, sg = quant(g, E5)
        dx = torch._scaled_mm(
            gq, wq.t().contiguous().t(), sg, sw, out_dtype=torch.bfloat16
        )
        dw = torch._scaled_mm(
            gq.t().contiguous(),
            xq.t().contiguous().t(),
            sg,
            sx,
            out_dtype=torch.float32,
        )
        return dx, dw


class FP8Linear(nn.Linear):
    def __init__(self, in_features, out_features, qat_group=0):
        super().__init__(in_features, out_features)
        self.qat_group = qat_group

    def forward(self, x):
        w = fake_quant_4bit(self.weight, self.qat_group) if self.qat_group else self.weight
        y = FP8MM.apply(x.reshape(-1, x.shape[-1]).to(torch.bfloat16), w)
        return y.view(*x.shape[:-1], -1) + self.bias.to(torch.bfloat16)


class QATLinear(nn.Linear):
    def __init__(self, in_features, out_features, qat_group=0):
        super().__init__(in_features, out_features)
        self.qat_group = qat_group

    def forward(self, x):
        w = fake_quant_4bit(self.weight, self.qat_group) if self.qat_group else self.weight
        return F.linear(x, w.to(x.dtype), self.bias.to(x.dtype))


def norm(x):
    return F.rms_norm(x, (x.size(-1),), eps=1e-6)


def rope(x, cos, sin):
    rd = cos.shape[-1] * 2
    if rd == 0:
        return x
    xr = x[..., :rd]
    d = rd // 2
    x1, x2 = xr[..., :d], xr[..., d:]
    out = torch.cat([x1 * cos + x2 * sin, x2 * cos - x1 * sin], -1).type_as(x)
    return torch.cat([out, x[..., rd:]], -1) if rd < x.shape[-1] else out


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        assert c.n_head % c.n_kv_head == 0
        self.c = c
        self.head_dim = c.n_embd // c.n_head
        self.kv_dim = c.n_kv_head * self.head_dim
        L = (lambda i, o: FP8Linear(i, o, c.qat_group)) if c.fp8 else (lambda i, o: QATLinear(i, o, c.qat_group))
        if c.conv_k > 1:
            self.conv = nn.Conv1d(c.n_embd, c.n_embd, c.conv_k, groups=c.n_embd)
            self.conv.weight.data.zero_()
            self.conv.bias.data.zero_()
        else:
            self.conv = None
        self.qkv = L(c.n_embd, c.n_embd + 2 * self.kv_dim)
        if c.attn_gate:
            self.attn_gate = nn.Linear(c.n_embd, c.n_head)
            self.attn_gate.weight.data.zero_()
            self.attn_gate.bias.data.zero_()
        else:
            self.attn_gate = None
        self.proj = L(c.n_embd, c.n_embd)
        if c.swiglu:
            mlp_h = ((int(8 * c.n_embd / 3) + 63) // 64) * 64  # 704 for n_embd=256 (~same params as 4*C)
            self.fc = L(c.n_embd, 2 * mlp_h)
            self.out = L(mlp_h, c.n_embd)
        else:
            self.fc = L(c.n_embd, 4 * c.n_embd)
            self.out = L(4 * c.n_embd, c.n_embd)
        self.proj.weight.data.zero_()
        self.out.weight.data.zero_()
        self.scale = (2 * GPT.num_passes(c)) ** -0.5

    def forward(
        self,
        x,
        cos,
        sin,
        v0=None,
        v_lambda=None,
        attn_scale=None,
        attn_shift=None,
        mlp_scale=None,
        mlp_shift=None,
        swa_block_mask=None,
        swa_dense_mask=None,
    ):
        B, T, C = x.shape
        h = norm(x)
        if attn_scale is not None:
            h = h * (1.0 + attn_scale.to(h.dtype)) + attn_shift.to(h.dtype)
        if self.conv is not None:
            hc = F.pad(h.transpose(1, 2), (self.c.conv_k - 1, 0))
            h = h + self.conv(hc).transpose(1, 2)
        qkv = self.qkv(h)
        q, k, v = qkv.split((C, self.kv_dim, self.kv_dim), dim=-1)
        q = q.view(B, T, self.c.n_head, self.head_dim)
        k = k.view(B, T, self.c.n_kv_head, self.head_dim)
        v = v.view(B, T, self.c.n_kv_head, self.head_dim)
        if v0 is None:
            v0 = v
        elif v_lambda is not None:
            v = (1 - v_lambda) * v + v_lambda * v0
        q, k = rope(norm(q), cos, sin), rope(norm(k), cos, sin)
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        gqa = self.c.n_kv_head != self.c.n_head
        if swa_block_mask is not None and torch.compiler.is_compiling():
            y = flex_attention(qt, kt, vt, block_mask=swa_block_mask, enable_gqa=gqa)
        elif swa_dense_mask is not None:
            y = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=swa_dense_mask, enable_gqa=gqa)
        else:
            y = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, enable_gqa=gqa)
        yt = y.transpose(1, 2)
        if self.attn_gate is not None:
            gate = 2.0 * torch.sigmoid(self.attn_gate(h).to(yt.dtype))
            yt = yt * gate.unsqueeze(-1)
        x = x + self.scale * self.proj(yt.reshape(B, T, C))
        m = norm(x)
        if mlp_scale is not None:
            m = m * (1.0 + mlp_scale.to(m.dtype)) + mlp_shift.to(m.dtype)
        if self.c.swiglu:
            g, u = self.fc(m).chunk(2, dim=-1)
            x = x + self.scale * self.out(F.silu(g) * u)
        else:
            x = x + self.scale * self.out(F.relu(self.fc(m)).square())
        return x, v0


class GPT(nn.Module):
    @staticmethod
    def num_passes(c):
        if c.n_loop > 1 and c.n_layer > 2:
            # Prelude (h[0]) -> Core (h[1:-1]) x n_loop -> Coda (h[-1])
            return 2 + (c.n_layer - 2) * c.n_loop
        return c.n_layer * c.n_loop

    def __init__(self, c=GPTConfig()):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.h = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size)
        if c.tie_weights:
            nn.init.normal_(self.wte.weight, std=0.02)
            self.lm_head.weight = self.wte.weight
            self.logit_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.lm_head.weight.data.zero_()
            self.logit_scale = None
        n_pass = self.num_passes(c)
        self.n_pass = n_pass
        # Per-pass residual/x0/v0 gates for passes p = 1..n_pass-1 (pass 0 starts directly at x0)
        self.resid_lambdas = nn.Parameter(torch.ones(max(n_pass - 1, 1)))
        self.x0_lambdas = nn.Parameter(torch.zeros(max(n_pass - 1, 1)))
        self.v_lambdas = nn.Parameter(torch.full((max(n_pass - 1, 1),), -2.0))
        # Per-pass AdaRMS conditioning applied in normalized space after norm(x)
        if c.pass_adarms:
            self.pass_scale = nn.Parameter(torch.zeros(n_pass, 2, c.n_embd))
            self.pass_shift = nn.Parameter(torch.zeros(n_pass, 2, c.n_embd))
            self.loop_emb = None
        else:
            self.pass_scale = None
            self.pass_shift = None
            self.loop_emb = nn.Parameter(torch.zeros(c.n_loop, c.n_embd)) if c.n_loop > 1 else None
        # Mirrored U-Net skip connections: encoder outputs p=0..n_skip-1 -> decoder inputs p=n_pass-1..n_pass-n_skip
        self.n_skip = (n_pass - 1) // 2 if c.unet_skips and n_pass >= 3 else 0
        self.skip_lambdas = nn.Parameter(torch.zeros(self.n_skip, c.n_embd)) if self.n_skip else None
        self.n_match = c.n_match
        if c.n_match:
            # match features (matchfeat.py), cheap in parameters since weights ship in the
            # archive: input adds wte(pred) * match_gate[k, code] + prev hit/miss embeddings,
            # and the predicted token's logit gets a context-gated copy bonus.
            self.match_gate = nn.Parameter(torch.zeros(c.n_match, c.match_codes))
            self.match_logit = nn.Parameter(torch.zeros(c.n_match, c.match_codes))
            self.match_ctx = nn.Linear(c.n_embd, c.n_match)
            self.match_ctx.weight.data.zero_()
            self.match_ctx.bias.data.zero_()
            self.match_hit_emb = nn.Parameter(torch.zeros(c.n_match, c.n_embd))
            self.match_miss_emb = nn.Parameter(torch.zeros(c.n_match, c.n_embd))
            self.match_hit_scale = nn.Parameter(torch.zeros(c.n_match))
            self.match_miss_scale = nn.Parameter(torch.zeros(c.n_match))
            self.match_agree_scale = nn.Parameter(torch.zeros(c.n_match))
            self.match_hit_bias = nn.Parameter(torch.zeros(c.n_match))
            self.match_miss_bias = nn.Parameter(torch.zeros(c.n_match))
            self.register_buffer("match_col", torch.arange(c.n_match), persistent=False)
        d = c.n_embd // c.n_head
        rope_dim = (int(round(d * c.rope_frac)) // 2) * 2
        f = torch.outer(
            torch.arange(c.context_length).float(),
            1.0 / 10000 ** (torch.arange(0, rope_dim, 2).float() / max(rope_dim, 1)),
        )
        self.register_buffer("cos", f.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", f.sin()[None, :, None, :], persistent=False)
        self._swa_block_mask = None
        if c.swa_window > 0 and c.swa_full_every > 1:
            idx = torch.arange(c.context_length)
            swa_mask = (idx[:, None] >= idx[None, :]) & (idx[:, None] - idx[None, :] < c.swa_window)
            self.register_buffer("swa_mask", swa_mask, persistent=False)
        else:
            self.swa_mask = None

    def build_swa_mask(self):
        """Build block-sparse flex_attention BlockMask on the model's device for fast compiled SWA."""
        c = self.c
        if c.swa_window > 0 and c.swa_full_every > 1:
            W = c.swa_window
            self._swa_block_mask = create_block_mask(
                lambda b, h, q_idx, kv_idx: (q_idx >= kv_idx) & (q_idx - kv_idx < W),
                B=None,
                H=None,
                Q_LEN=c.context_length,
                KV_LEN=c.context_length,
                device=self.cos.device,
            )

    def _is_full_attn(self, p):
        c = self.c
        if c.swa_window <= 0 or c.swa_full_every <= 1:
            return True
        return ((p + 1) % c.swa_full_every == 0) or (p == self.num_passes(c) - 1)

    def _run_block(self, b, p, x, x0, v0, cos, sin, swa_bm, swa_dm, skips):
        if p > 0:
            x = (
                self.resid_lambdas[p - 1].to(x.dtype) * x
                + self.x0_lambdas[p - 1].to(x.dtype) * x0
            )
        if self.n_skip and p >= self.n_pass - self.n_skip:
            si = self.n_pass - 1 - p
            x = x + self.skip_lambdas[si].to(x.dtype) * skips[si]
        v_lam = torch.sigmoid(self.v_lambdas[p - 1]).to(x.dtype) if p > 0 else None
        if self.pass_scale is not None:
            a_s, m_s = self.pass_scale[p, 0], self.pass_scale[p, 1]
            a_b, m_b = self.pass_shift[p, 0], self.pass_shift[p, 1]
        else:
            a_s = m_s = a_b = m_b = None
        full = self._is_full_attn(p)
        x, v0 = b(
            x,
            cos,
            sin,
            v0=v0,
            v_lambda=v_lam,
            attn_scale=a_s,
            attn_shift=a_b,
            mlp_scale=m_s,
            mlp_shift=m_b,
            swa_block_mask=None if full else swa_bm,
            swa_dense_mask=None if full else swa_dm,
        )
        if self.n_skip and p < self.n_skip:
            skips.append(x)
        return x, v0

    def forward(self, idx, mpred=None, mcode=None):
        """mpred, mcode: [B, T, n_match] match features for the byte AFTER each input position
        (the one this position predicts)."""
        T = idx.shape[1]
        wte = fake_quant_4bit(self.wte.weight, self.c.qat_group) if self.c.qat_group else self.wte.weight
        x = F.embedding(idx, wte)
        if self.n_match:
            mp, ml = mpred.long(), mcode.long()
            has_b = ml > 0
            has = has_b.to(x.dtype)  # forces the len=0 rows to contribute nothing
            gate = self.match_gate[self.match_col, ml] * has
            x = x + (F.embedding(mp, wte) * gate.unsqueeze(-1)).sum(2)
            # Strictly causal previous-step match hit/miss: mp[:, t-1] predicted idx[:, t]
            if T > 1:
                p_has = has_b[:, :-1]
                p_eq = mp[:, :-1] == idx[:, 1:, None]
                prev_hit = F.pad((p_has & p_eq).to(x.dtype), (0, 0, 1, 0))
                prev_miss = F.pad((p_has & ~p_eq).to(x.dtype), (0, 0, 1, 0))
            else:
                prev_hit = torch.zeros_like(has)
                prev_miss = torch.zeros_like(has)
            x = (
                x
                + prev_hit @ self.match_hit_emb.to(x.dtype)
                + prev_miss @ self.match_miss_emb.to(x.dtype)
            )
            if self.n_match > 1:
                same = (mp.unsqueeze(-1) == mp.unsqueeze(-2)) & has_b.unsqueeze(-1) & has_b.unsqueeze(-2)
                agree = (same.sum(-1).to(x.dtype) - has).clamp_min(0.0) / (self.n_match - 1)
            else:
                agree = torch.zeros_like(has)
        x = norm(x)
        x0 = x
        cos, sin = self.cos[:, :T], self.sin[:, :T]
        swa_bm = self._swa_block_mask if (self._swa_block_mask is not None and T == self.c.context_length) else None
        swa_dm = self.swa_mask[:T, :T] if self.swa_mask is not None else None
        v0 = None
        skips = []
        p = 0
        if self.c.n_loop > 1 and self.c.n_layer > 2:
            # Prelude (h[0]) -> Core (h[1:-1]) x n_loop -> Coda (h[-1])
            x, v0 = self._run_block(self.h[0], p, x, x0, v0, cos, sin, swa_bm, swa_dm, skips)
            p += 1
            for loop in range(self.c.n_loop):
                if self.loop_emb is not None:
                    x = x + self.loop_emb[loop].to(x.dtype)
                for b in self.h[1:-1]:
                    x, v0 = self._run_block(b, p, x, x0, v0, cos, sin, swa_bm, swa_dm, skips)
                    p += 1
            x, v0 = self._run_block(self.h[-1], p, x, x0, v0, cos, sin, swa_bm, swa_dm, skips)
        else:
            for loop in range(self.c.n_loop):
                if self.loop_emb is not None:
                    x = x + self.loop_emb[loop].to(x.dtype)
                for b in self.h:
                    x, v0 = self._run_block(b, p, x, x0, v0, cos, sin, swa_bm, swa_dm, skips)
                    p += 1
        head_w = wte if self.c.tie_weights else (
            fake_quant_4bit(self.lm_head.weight, self.c.qat_group) if self.c.qat_group else self.lm_head.weight
        )
        hn = norm(x)
        logits = F.linear(hn, head_w.to(x.dtype), self.lm_head.bias.to(x.dtype))
        if self.logit_scale is not None:
            logits = logits * self.logit_scale.to(logits.dtype)
        if self.n_match:
            ctx_gate = self.match_ctx(hn).to(logits.dtype)
            mod = F.softplus(
                0.54132485
                + ctx_gate
                + self.match_hit_scale.to(logits.dtype) * prev_hit.to(logits.dtype)
                + self.match_miss_scale.to(logits.dtype) * prev_miss.to(logits.dtype)
                + self.match_agree_scale.to(logits.dtype) * agree.to(logits.dtype)
            )
            base_bonus = self.match_logit[self.match_col, ml].to(logits.dtype)
            bonus = (
                base_bonus * mod
                + self.match_hit_bias.to(logits.dtype) * prev_hit.to(logits.dtype)
                + self.match_miss_bias.to(logits.dtype) * prev_miss.to(logits.dtype)
            ) * has.to(logits.dtype)
            logits = logits.scatter_add(-1, mp, bonus)
        return logits


@torch.no_grad()
def quantize_4bit_stats(model):
    """Estimate raw packed and entropy-coded archive size for 4-bit QAT weights + fp16 misc params."""
    gs = model.c.qat_group or 64
    q_weights = [model.wte.weight]
    if not model.c.tie_weights:
        q_weights.append(model.lm_head.weight)
    q_weights += [m.weight for m in model.h.modules() if isinstance(m, (FP8Linear, QATLinear))]
    q_ids = {id(p) for p in q_weights}
    other = [p for p in model.parameters() if id(p) not in q_ids]

    q_all = []
    scale_bytes = 0
    for w in q_weights:
        wg = w.detach().float().reshape(-1, gs)
        s = (wg.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(1e-7).to(torch.float16)
        q = (wg / s.float()).round().clamp(-7, 7).to(torch.int8).cpu().numpy().ravel()
        q_all.append(q)
        scale_bytes += s.numel() * 2
    q_cat = np.concatenate(q_all)
    # Shannon entropy of the 15-level int4 symbols (achievable with an arithmetic coder)
    _, counts = np.unique(q_cat, return_counts=True)
    probs = counts / counts.sum()
    h_bits = float(-(probs * np.log2(probs)).sum())
    raw_q_bytes = (len(q_cat) + 1) // 2 + scale_bytes
    ent_q_bytes = int(math.ceil(len(q_cat) * h_bits / 8.0)) + scale_bytes
    other_fp16 = b"".join(p.detach().to(torch.float16).cpu().numpy().tobytes() for p in other)
    other_raw = len(other_fp16)
    other_zlib = len(zlib.compress(other_fp16, 9))
    return dict(
        q_params=int(len(q_cat)),
        other_params=sum(p.numel() for p in other),
        bits_per_qweight=h_bits,
        raw_mb=(raw_q_bytes + other_raw) / 1e6,
        entropy_mb=(ent_q_bytes + other_zlib) / 1e6,
    )

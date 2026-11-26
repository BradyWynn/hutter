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
    xsa: bool = True           # Exclusive Self-Attention: project out self-v from attn output
    n_experts: int = 0         # >0 enables DeepSeek-style Shared + Routed Fine-Grained MoE MLP
    n_experts_active: int = 2  # top_k routed experts active per token
    shared_h: int = 256        # shared SwiGLU expert hidden dim (0 disables shared expert)
    expert_h: int = 256        # per-routed-expert SwiGLU hidden dim
    engram_rows: int = 0       # >0 enables 4-head learned bigram+trigram Engram lookup table (rows per head)
    engram_dim: int = 32       # embedding dim per hash head (4 heads -> 4*engram_dim total)
    qat_group: int = 0         # symmetric per-group STE QAT group size (0 disables QAT)
    qat_block_bits: int = 4    # bit width for dense attention / shared MLP weights
    qat_expert_bits: int = 4   # bit width for routed MoE expert weights
    qat_emb_bits: int = 6      # bit width for wte / lm_head weights (6 -> [-31, 31])
    fp8: bool = False          # pure BF16 is faster and cleaner than FP8MM at d=256
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


def fake_quant(w, bits=4, group_size=64):
    """Per-group symmetric N-bit STE quantization with fp16 group scales.
    Forward uses exact dequantized intN + fp16-scale weights so archive export has 0.0 drop."""
    if not group_size or not bits:
        return w
    qmax = float((1 << (bits - 1)) - 1)
    orig_shape = w.shape
    wf = w.float().reshape(-1)
    n = wf.numel()
    rem = n % group_size
    if rem:
        wf = F.pad(wf, (0, group_size - rem))
    wg = wf.reshape(-1, group_size)
    scale = (wg.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-7).to(torch.float16).float()
    q = (wg / scale).round().clamp(-qmax, qmax)
    wq = (q * scale).reshape(-1)[:n].reshape(orig_shape).to(w.dtype)
    return w + (wq - w).detach()


def fake_quant_4bit(w, group_size=64):
    return fake_quant(w, bits=4, group_size=group_size)


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
    def __init__(self, in_features, out_features, qat_group=0, qat_bits=4):
        super().__init__(in_features, out_features)
        self.qat_group = qat_group
        self.qat_bits = qat_bits

    def forward(self, x):
        if self.qat_group and self.qat_bits:
            # During late-cooldown QAT, run exact BF16 matmul on exact dequantized weights
            # (avoids extra per-tensor FP8 re-quantization noise on top of per-group IntN).
            w = fake_quant(self.weight, self.qat_bits, self.qat_group)
            return F.linear(x, w.to(x.dtype), self.bias.to(x.dtype))
        y = FP8MM.apply(x.reshape(-1, x.shape[-1]).to(torch.bfloat16), self.weight)
        return y.view(*x.shape[:-1], -1) + self.bias.to(torch.bfloat16)


class QATLinear(nn.Linear):
    def __init__(self, in_features, out_features, qat_group=0, qat_bits=4):
        super().__init__(in_features, out_features)
        self.qat_group = qat_group
        self.qat_bits = qat_bits

    def forward(self, x):
        w = fake_quant(self.weight, self.qat_bits, self.qat_group) if (self.qat_group and self.qat_bits) else self.weight
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


class Engram(nn.Module):
    """DeepSeek-Engram style multi-head learned N-gram (bigram + trigram) lookup table with
    context-aware gating. Injected after block 0 so attention context suppresses hash collisions.
    Costs <1% CPU FLOPs while adding millions of O(1) static memorization parameters."""

    # Independent odd multipliers for 2-head bigram + 2-head trigram polynomial hashing
    P1, M1 = 1000003, 0x9E3779B97F4A7C15
    P2, M2 = 1000033, 0xBF58476D1CE4E5B9

    def __init__(self, c):
        super().__init__()
        assert c.engram_rows > 0 and (c.engram_rows & (c.engram_rows - 1)) == 0, "engram_rows must be power of 2"
        self.rows = c.engram_rows
        self.mask = c.engram_rows - 1
        self.dim = c.engram_dim
        self.qat_group = c.qat_group
        self.qat_bits = c.qat_expert_bits
        # 4 tables: [bigram_h1, bigram_h2, trigram_h1, trigram_h2] packed as [4 * rows, dim]
        self.table = nn.Parameter(torch.empty(4 * c.engram_rows, c.engram_dim))
        nn.init.normal_(self.table, std=0.02)
        in_d = 4 * c.engram_dim
        self.key_proj = nn.Linear(in_d, c.n_embd, bias=False)
        self.val_proj = nn.Linear(in_d, c.n_embd, bias=False)
        nn.init.normal_(self.key_proj.weight, std=0.02)
        self.val_proj.weight.data.zero_()  # start as exact identity
        self.inv_sqrt_d = c.n_embd ** -0.5

    def forward(self, idx, h):
        # Strictly causal n-gram hashes from input tokens idx[:, :T]
        t0 = idx.to(torch.int64)
        t1 = F.pad(t0[:, :-1], (1, 0))
        t2 = F.pad(t0[:, :-2], (2, 0))
        m = self.mask
        # Signed int64 wraparound hashing (matches C uint64 after & mask on right-shifted bits)
        m1 = self.M1 - (1 << 64) if self.M1 >= (1 << 63) else self.M1
        m2 = self.M2 - (1 << 64) if self.M2 >= (1 << 63) else self.M2
        hb1 = (((t1 * self.P1 + t0) * m1) >> 32) & m
        hb2 = ((((t1 * self.P2 + t0) * m2) >> 32) & m) + self.rows
        ht1 = ((((t2 * self.P1 + t1) * self.P1 + t0) * m1) >> 32 & m) + 2 * self.rows
        ht2 = ((((t2 * self.P2 + t1) * self.P2 + t0) * m2) >> 32 & m) + 3 * self.rows

        tbl = fake_quant(self.table, self.qat_bits, self.qat_group) if (self.qat_group and self.qat_bits) else self.table
        eb1 = F.embedding(hb1, tbl)
        eb2 = F.embedding(hb2, tbl)
        et1 = F.embedding(ht1, tbl)
        et2 = F.embedding(ht2, tbl)
        # Zero out incomplete prefix positions (t=0 for bigrams, t<2 for trigrams)
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)[None, :, None]
        m_bi = (pos >= 1).to(eb1.dtype)
        m_tri = (pos >= 2).to(et1.dtype)
        e = torch.cat([eb1 * m_bi, eb2 * m_bi, et1 * m_tri, et2 * m_tri], dim=-1).to(h.dtype)

        ek = self.key_proj(e)
        ev = self.val_proj(e)
        # Context-aware collision-suppressing sigmoid gate
        gate = torch.sigmoid((norm(h).float() * norm(ek).float()).sum(dim=-1, keepdim=True) * self.inv_sqrt_d).to(h.dtype)
        return gate * ev


class MoEMLP(nn.Module):
    """DeepSeek-V3 style Shared + Fine-Grained Routed SwiGLU MoE using native torch._grouped_mm,
    with auxiliary-loss-free bias load balancing and per-group STE QAT."""

    def __init__(self, c, L):
        super().__init__()
        self.n_experts = c.n_experts
        self.top_k = c.n_experts_active
        self.expert_h = c.expert_h
        self.qat_group = c.qat_group
        self.qat_bits = c.qat_expert_bits

        if c.shared_h > 0:
            self.shared_fc = L(c.n_embd, 2 * c.shared_h)
            self.shared_out = L(c.shared_h, c.n_embd)
            self.shared_out.weight.data.zero_()
        else:
            self.shared_fc = None
            self.shared_out = None

        self.router = nn.Linear(c.n_embd, c.n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        # Persistent load-balancing bias (32B/layer in fp16) + non-persistent step load counter
        self.register_buffer("router_bias", torch.zeros(c.n_experts))
        self.register_buffer("last_counts", torch.zeros(c.n_experts, dtype=torch.int64), persistent=False)

        self.w_fc = nn.Parameter(torch.empty(c.n_experts, 2 * c.expert_h, c.n_embd))
        self.w_out = nn.Parameter(torch.zeros(c.n_experts, c.n_embd, c.expert_h))
        bound = c.n_embd ** -0.5
        nn.init.uniform_(self.w_fc, -bound, bound)

    def forward(self, m):
        B, T, C = m.shape
        y_shared = None
        if self.shared_fc is not None:
            sg, su = self.shared_fc(m).chunk(2, dim=-1)
            y_shared = self.shared_out(F.silu(sg) * su)

        x_flat = m.reshape(-1, C)  # [N, C]
        scores = torch.sigmoid(self.router(x_flat).float())  # [N, E]
        sel_scores = scores + self.router_bias.unsqueeze(0)
        _, top_idx = sel_scores.topk(self.top_k, dim=-1)  # [N, K]
        top_w = scores.gather(-1, top_idx)
        top_w = (top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)).to(m.dtype)

        flat_e = top_idx.reshape(-1)  # [N*K]
        order = flat_e.argsort(stable=False)
        sorted_e = flat_e[order]
        tok_ids = order // self.top_k
        sorted_x = x_flat[tok_ids]

        # Pure-GPU expert histogram (avoids bincount host sync)
        counts = torch.zeros(self.n_experts, dtype=torch.int32, device=m.device)
        counts.scatter_add_(0, sorted_e, torch.ones_like(sorted_e, dtype=torch.int32))
        offs = counts.cumsum(0, dtype=torch.int32)
        if self.training:
            self.last_counts.add_(counts.to(torch.int64))

        w_fc = fake_quant(self.w_fc, self.qat_bits, self.qat_group) if (self.qat_group and self.qat_bits) else self.w_fc
        w_out = fake_quant(self.w_out, self.qat_bits, self.qat_group) if (self.qat_group and self.qat_bits) else self.w_out

        # _grouped_mm requires explicit bf16/fp16 (not auto-cast by torch.autocast)
        mm_dt = torch.bfloat16 if m.is_cuda else m.dtype
        h = torch._grouped_mm(sorted_x.to(mm_dt), w_fc.to(mm_dt).transpose(-1, -2), offs=offs)
        g, u = h.chunk(2, dim=-1)
        y_sorted = torch._grouped_mm((F.silu(g) * u).to(mm_dt), w_out.to(mm_dt).transpose(-1, -2), offs=offs)

        y_routed = torch.zeros(x_flat.shape, dtype=mm_dt, device=m.device)
        y_routed.index_add_(0, tok_ids, y_sorted * top_w.reshape(-1, 1)[order].to(mm_dt))
        y_routed = y_routed.view(B, T, C).to(m.dtype)
        return (y_shared + y_routed) if y_shared is not None else y_routed


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        assert c.n_head % c.n_kv_head == 0
        self.c = c
        self.head_dim = c.n_embd // c.n_head
        self.kv_dim = c.n_kv_head * self.head_dim
        L = (
            (lambda i, o: FP8Linear(i, o, c.qat_group, c.qat_block_bits))
            if c.fp8
            else (lambda i, o: QATLinear(i, o, c.qat_group, c.qat_block_bits))
        )
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
        self.proj.weight.data.zero_()
        if c.n_experts > 0:
            self.moe = MoEMLP(c, L)
            self.fc = None
            self.out = None
        else:
            self.moe = None
            if c.swiglu:
                mlp_h = ((int(8 * c.n_embd / 3) + 63) // 64) * 64  # 704 for n_embd=256 (~same params as 4*C)
                self.fc = L(c.n_embd, 2 * mlp_h)
                self.out = L(mlp_h, c.n_embd)
            else:
                self.fc = L(c.n_embd, 4 * c.n_embd)
                self.out = L(4 * c.n_embd, c.n_embd)
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
        if self.c.xsa:
            # Exclusive Self-Attention: project out each token's own v direction
            v_e = v.repeat_interleave(self.c.n_head // self.c.n_kv_head, dim=2) if gqa else v
            vn = F.normalize(v_e.float(), dim=-1, eps=1e-6).to(yt.dtype)
            yt = yt - (yt * vn).sum(dim=-1, keepdim=True) * vn
        if self.attn_gate is not None:
            gate = 2.0 * torch.sigmoid(self.attn_gate(h).to(yt.dtype))
            yt = yt * gate.unsqueeze(-1)
        x = x + self.scale * self.proj(yt.reshape(B, T, C))
        m = norm(x)
        if mlp_scale is not None:
            m = m * (1.0 + mlp_scale.to(m.dtype)) + mlp_shift.to(m.dtype)
        if self.moe is not None:
            x = x + self.scale * self.moe(m)
        elif self.c.swiglu:
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
        self.qat_group = c.qat_group
        self.qat_block_bits = c.qat_block_bits
        self.qat_expert_bits = c.qat_expert_bits
        self.qat_emb_bits = c.qat_emb_bits
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.engram = Engram(c) if c.engram_rows > 0 else None
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

    def set_qat(self, group_size=64, block_bits=None, emb_bits=None, expert_bits=None):
        """Enable/update mixed-precision per-group STE QAT dynamically (e.g. for late cooldown)."""
        self.qat_group = int(group_size)
        if block_bits is not None:
            self.qat_block_bits = int(block_bits)
        if emb_bits is not None:
            self.qat_emb_bits = int(emb_bits)
        if expert_bits is not None:
            self.qat_expert_bits = int(expert_bits)
        for m in self.h.modules():
            if isinstance(m, (FP8Linear, QATLinear)):
                m.qat_group = self.qat_group
                m.qat_bits = self.qat_block_bits
            elif isinstance(m, MoEMLP):
                m.qat_group = self.qat_group
                m.qat_bits = self.qat_expert_bits
        if self.engram is not None:
            self.engram.qat_group = self.qat_group
            self.engram.qat_bits = self.qat_expert_bits

    @torch.no_grad()
    def update_router_biases(self, gamma=1e-3, clamp=4.0):
        """DeepSeek-V3 auxiliary-loss-free load balancing update across all MoE layers.
        Returns max(expert_load / mean_load) across layers for monitoring."""
        max_ratio = 1.0
        for b in self.h:
            if b.moe is not None:
                cnt = b.moe.last_counts.float()
                tot = cnt.sum()
                if tot > 0:
                    target = tot / b.moe.n_experts
                    err = target - cnt
                    b.moe.router_bias.add_(err.sign() * gamma).clamp_(-clamp, clamp)
                    ratio = (cnt.max() / target).item()
                    if ratio > max_ratio:
                        max_ratio = ratio
                b.moe.last_counts.zero_()
        return max_ratio

    def forward(self, idx, mpred=None, mcode=None, return_features=False):
        """mpred, mcode: [B, T, n_match] match features for the byte AFTER each input position
        (the one this position predicts)."""
        T = idx.shape[1]
        wte = (
            fake_quant(self.wte.weight, self.qat_emb_bits, self.qat_group)
            if (self.qat_group and self.qat_emb_bits)
            else self.wte.weight
        )
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
            if self.engram is not None:
                x = x + self.engram(idx, x)
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
                    if p == 0 and self.engram is not None:
                        x = x + self.engram(idx, x)
                    p += 1
        head_w = wte if self.c.tie_weights else (
            fake_quant(self.lm_head.weight, self.qat_emb_bits, self.qat_group)
            if (self.qat_group and self.qat_emb_bits)
            else self.lm_head.weight
        )
        hn = norm(x)
        logits = F.linear(hn, head_w.to(x.dtype), self.lm_head.bias.to(x.dtype))
        if self.logit_scale is not None:
            logits = logits * self.logit_scale.to(logits.dtype)
        bonus = None
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
        if return_features:
            return logits, hn, bonus
        return logits


def _quant_group_stats(weights, bits, gs):
    if not weights:
        return dict(params=0, bpw=0.0, raw_bytes=0, ent_bytes=0)
    qmax = (1 << (bits - 1)) - 1
    q_all = []
    scale_bytes = 0
    total_params = 0
    for w in weights:
        wf = w.detach().float().reshape(-1)
        n = wf.numel()
        total_params += n
        rem = n % gs
        if rem:
            wf = F.pad(wf, (0, gs - rem))
        wg = wf.reshape(-1, gs)
        s = (wg.abs().amax(dim=-1, keepdim=True) / float(qmax)).clamp_min(1e-7).to(torch.float16)
        q = (wg / s.float()).round().clamp(-qmax, qmax).to(torch.int16).cpu().numpy().ravel()[:n]
        q_all.append(q)
        scale_bytes += s.numel() * 2
    q_cat = np.concatenate(q_all)
    _, counts = np.unique(q_cat, return_counts=True)
    probs = counts / counts.sum()
    h_bits = float(-(probs * np.log2(probs)).sum())
    raw_bytes = int(math.ceil(len(q_cat) * bits / 8.0)) + scale_bytes
    ent_bytes = int(math.ceil(len(q_cat) * h_bits / 8.0)) + scale_bytes
    return dict(params=int(total_params), bpw=h_bits, raw_bytes=raw_bytes, ent_bytes=ent_bytes)


@torch.no_grad()
def quantize_stats(model, block_bits=None, emb_bits=None, expert_bits=None, group_size=None):
    """Estimate raw packed and entropy-coded archive size for 3-tier mixed-precision weights + fp16 misc."""
    gs = group_size or model.qat_group or model.c.qat_group or 64
    b_bits = block_bits if block_bits is not None else (model.qat_block_bits or model.c.qat_block_bits or 4)
    e_bits = emb_bits if emb_bits is not None else (model.qat_emb_bits or model.c.qat_emb_bits or 6)
    x_bits = expert_bits if expert_bits is not None else (model.qat_expert_bits or model.c.qat_expert_bits or 4)
    emb_weights = [model.wte.weight]
    if not model.c.tie_weights:
        emb_weights.append(model.lm_head.weight)
    block_weights = [m.weight for m in model.h.modules() if isinstance(m, (FP8Linear, QATLinear))]
    expert_weights = []
    for m in model.h.modules():
        if isinstance(m, MoEMLP):
            expert_weights.extend([m.w_fc, m.w_out])
    if model.engram is not None:
        expert_weights.append(model.engram.table)
    q_ids = {id(p) for p in emb_weights + block_weights + expert_weights}
    other = [p for p in model.parameters() if id(p) not in q_ids]
    other_bufs = [b.moe.router_bias for b in model.h if b.moe is not None]

    es = _quant_group_stats(emb_weights, e_bits, gs)
    bs = _quant_group_stats(block_weights, b_bits, gs)
    xs = _quant_group_stats(expert_weights, x_bits, gs)
    other_fp16 = b"".join(p.detach().to(torch.float16).cpu().numpy().tobytes() for p in (other + other_bufs))
    other_raw = len(other_fp16)
    other_zlib = len(zlib.compress(other_fp16, 9))
    q_params = es["params"] + bs["params"] + xs["params"]
    avg_bpw = (es["params"] * es["bpw"] + bs["params"] * bs["bpw"] + xs["params"] * xs["bpw"]) / max(q_params, 1)
    return dict(
        q_params=q_params,
        emb_params=es["params"],
        block_params=bs["params"],
        expert_params=xs["params"],
        other_params=sum(p.numel() for p in other) + sum(b.numel() for b in other_bufs),
        emb_bits=e_bits,
        block_bits=b_bits,
        expert_bits=x_bits,
        group_size=gs,
        emb_bpw=es["bpw"],
        block_bpw=bs["bpw"],
        expert_bpw=xs["bpw"],
        bits_per_qweight=avg_bpw,
        emb_entropy_mb=es["ent_bytes"] / 1e6,
        block_entropy_mb=bs["ent_bytes"] / 1e6,
        expert_entropy_mb=xs["ent_bytes"] / 1e6,
        other_zlib_mb=other_zlib / 1e6,
        raw_mb=(es["raw_bytes"] + bs["raw_bytes"] + xs["raw_bytes"] + other_raw) / 1e6,
        entropy_mb=(es["ent_bytes"] + bs["ent_bytes"] + xs["ent_bytes"] + other_zlib) / 1e6,
    )


def quantize_4bit_stats(model):
    return quantize_stats(model, block_bits=4, emb_bits=4, expert_bits=4, group_size=model.qat_group or model.c.qat_group or 64)


@torch.no_grad()
def quantize_model_copy(model, block_bits=None, emb_bits=None, expert_bits=None, group_size=None):
    """Return an eval copy of `model` with exact archive-precision weights:
    per-group Int{emb_bits} embeddings, Int{block_bits} dense linears, Int{expert_bits} MoE experts, and fp16 misc."""
    import copy
    gs = group_size or model.qat_group or model.c.qat_group or 64
    b_bits = block_bits if block_bits is not None else (model.qat_block_bits or model.c.qat_block_bits or 4)
    e_bits = emb_bits if emb_bits is not None else (model.qat_emb_bits or model.c.qat_emb_bits or 6)
    x_bits = expert_bits if expert_bits is not None else (model.qat_expert_bits or model.c.qat_expert_bits or 4)
    qm = copy.deepcopy(model).eval().requires_grad_(False)
    qm.set_qat(group_size=gs, block_bits=b_bits, emb_bits=e_bits, expert_bits=x_bits)
    q_ids = {id(qm.wte.weight)}
    if not qm.c.tie_weights:
        q_ids.add(id(qm.lm_head.weight))
    for m in qm.h.modules():
        if isinstance(m, (FP8Linear, QATLinear)):
            q_ids.add(id(m.weight))
        elif isinstance(m, MoEMLP):
            q_ids.add(id(m.w_fc))
            q_ids.add(id(m.w_out))
            m.router_bias.copy_(m.router_bias.to(torch.float16).to(m.router_bias.dtype))
    if qm.engram is not None:
        q_ids.add(id(qm.engram.table))
    for p in qm.parameters():
        if id(p) not in q_ids:
            p.copy_(p.to(torch.float16).to(p.dtype))
    return qm

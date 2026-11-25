# Hutter Prize Neural Compression — `rich-voice-20` Analysis & Next Steps

> Summary of W&B run `rich-voice-20` (`bradywynn/hutter/4gg1avnk`, finished 2026-11-25) and prioritized research/engineering roadmap for beating the current Hutter Prize #1 record (`fx2-cmix-T`: **100,424,672 bytes** / ~100.42 MB; next prize threshold ~99.42 MB).

---

## 1. Analysis of `rich-voice-20` (`4gg1avnk`)

### Configuration & Results
| Metric | Value | Notes |
|---|---|---|
| **Final `eval_bpb` / `ema_eval_bpb`** | **`0.83056` / `0.83058`** | `eval_loss = 1.60894` nats/tok on `enwik9.split4096` (`357,813,851` toks, `2.795` B/tok) |
| **Implied payload size** | **`103.82 MB`** | `0.83056 × 10⁹ / 8` |
| **Model size** | `6,268,130` params | `L=7`, `n_loop=2` (Prelude + 5 Core × 2 + Coda = 12 passes), `d=256`, `n_head=4`, `n_kv_head=2`, SwiGLU (`h=704`), `conv_k=4`, Half-RoPE, U-Net skips, `attn_gate`, `pass_adarms`, `swa_window=256` (full every 4th + last), `tie_weights=True`, `fp8=True`, `qat_group=0` |
| **Match features** | `n_match=6` | `MATCH_K = [2, 3, 5, 8, 13, 27]`, `TABLE_BITS=26`, context-gated copy bonus + prev hit/miss + agreement |
| **Optimizer & Schedule** | NorMuon + AdamW | `LR_MUON=0.015`, `WD_MUON=0.005`, `WD_ADAM=0.005`, `EMA_DECAY=0.999`, `COOLDOWN=0.50` (const 0–50k, linear 50k–100k) |
| **Est. 4-bit weight cost** | `~2.6–3.3 MB` | `quantize_4bit_stats` (~3.1 bpw entropy-coded) + vocab/code |
| **Compute** | `2.41 h` (`81.3 ms/step`) | 1× RTX 5070 Ti (16 GB), `100,000` steps × `B=16 × T=2048` = `3.28B` toks = **`9.16` epochs** of `enwik9` |

### Progression Across Recent W&B Runs
| Run | Architecture / Change | Steps | `ema_eval_bpb` | Payload Equiv. |
|---|---|---:|---:|---:|
| `worldly-leaf-3` (`p9jerr74`) | 85.4M byte-level (old v1, `T=1024`, 4-col match) | 30k | `0.8855` | `110.69 MB` |
| `smooth-galaxy-12` (`de5isjbh`) | 6.12M v6 + `qat_group=64` from scratch | 92.5k | `0.9270` | `115.88 MB` |
| `dulcet-waterfall-16` (`zmi50tl3`) | 6.12M v6 (4-col match, const LR) | 104.6k | `0.8801` | `110.01 MB` |
| `swift-energy-17` (`rrrwcnd6`) | 6.15M v7 (+AdaRMS, U-Net skips, Half-RoPE, attn gate, const LR) | 67.7k | `0.8712` | `108.90 MB` |
| `devoted-cosmos-18` (`p3c2kj4t`) | v7 + 35k linear cooldown from 67k EMA | +35k | `0.8444` | `105.55 MB` |
| `frosty-silence-19` (`8d2iwazm`) | 6.27M v8 (+SwiGLU, +SWA=256, +6-col match, `COOLDOWN=0.3`) | 68.2k | `0.8539` | `106.74 MB` |
| **`rich-voice-20`** (`4gg1avnk`) | **6.27M v8 + `COOLDOWN=0.5`** | **100k** | **`0.8306`** | **`103.82 MB`** |

### Training Dynamics of `rich-voice-20`
| Window | Mean Train `bpb` | End `eval_bpb` | End `ema_eval_bpb` | `eval - ema` | `lr_mult` | Mean `grad_norm` |
|---|---:|---:|---:|---:|---:|---:|
| 0k–10k | `0.9912` | `0.9357` | `0.9069` | `+0.0288` | `1.00` | `0.1263` |
| 10k–20k | `0.9289` | `0.9094` | `0.8812` | `+0.0282` | `1.00` | `0.1090` |
| 20k–30k | `0.9144` | `0.8971` | `0.8694` | `+0.0277` | `1.00` | `0.1177` |
| 30k–40k | `0.9065` | `0.8917` | `0.8631` | `+0.0286` | `1.00` | `0.1283` |
| 40k–50k | `0.8808` | `0.8864` | `0.8586` | `+0.0277` | `1.00` | `0.1249` |
| 50k–60k | `0.8661` | `0.8767` | `0.8532` | `+0.0234` | `0.80` | `0.1227` |
| 60k–70k | `0.8663` | `0.8651` | `0.8467` | `+0.0184` | `0.60` | `0.1229` |
| 70k–80k | `0.8640` | `0.8544` | `0.8411` | `+0.0134` | `0.40` | `0.1240` |
| 80k–90k | `0.8466` | `0.8424` | `0.8355` | `+0.0069` | `0.20` | `0.1243` |
| 90k–100k | `0.8424` | `0.8306` | `0.8306` | `0.0000` | `0.00` | `0.1276` |

### 4 Key Diagnostics
1. **Severely undertrained—never plateaued:** During cooldown (50k→100k), `ema_eval_bpb` dropped `-0.0280 bpb` (**-3.50 MB**) nearly linearly all the way to step 100k (`-0.0049` in the last 10k alone; `-0.00119` at 98k→99k, `-0.00071` at 99k→100k). At 9.16 epochs / 2.4h, **2–4+ MB** are likely still on the table purely from training longer.
2. **Large constant-LR variance (`eval - ema = +0.028 bpb` / +3.5 MB):** Raw weights stayed `+0.028 bpb` behind EMA throughout 0–50k, closing to `0.0000` only as LR hit 0. This explains why `COOLDOWN=0.50` beat `0.30`. Also `grad_norm` stayed `~0.11–0.13`, so `clip_grad_norm_(..., 1)` in `train.py` never fired.
3. **Two opposing offsets between `eval_bpb` and real Hutter archive size:**
   - **Favorable:** `eval_ce` in `train.py` averages across all `T=2048` positions of each window (early tokens have short context). Strided/KV-cached decoding where every token sees `1024–2047` tokens of context typically improves bpb by **~0.01–0.02 bpb (~1.5–2.5 MB)**. *(Caveat: `eval_batches` is only 32 fixed batches = 0.29% of `enwik9`, and 90k–100k mean train bpb was `0.8424`, so verify whether those 32 batches skew slightly easy.)*
   - **Cost:** `qat_group=0`, so 4-bit quantization loss and weight storage (~2.6–3.3 MB) haven't been paid yet. Still, estimated purely static total is already **~104.5–106.5 MB**—roughly matching **`tufazip v2` (105.39 MB on 2×B200 for 7.5 days)** and only **~4–6 MB** from **Hutter #1 (`fx2-cmix-T`: 100.42 MB)**.
4. **Single-CPU-core Hutter budget:** Official rules require **1 CPU core, <10 GB RAM, ~50h**. At `357.8M` tokens × `~24M FLOPs/tok` ≈ `8.7 PFLOPs` (KV-cached), `rich-voice-20` is already close to the single-core AVX-512/VNNI ceiling. Track decode FLOPs alongside total MB.

---

## 2. External Context & Hutter MDL Framing

Unlike **Q Labs Slowrun** and **Pretraining under Infinite Compute** (which optimize held-out generalization and therefore use aggressive ~30× weight decay to fight memorization), the Hutter Prize has **zero train/test split**—it is pure Minimum Description Length (MDL):

$$\text{Total Bytes} = \text{bits}(\text{weights}) + \text{NLL}(\text{enwik9} \mid \text{weights}) + \text{code/vocab/overhead}$$

Memorizing `enwik9` into weights is desirable whenever 1 bit of weights saves >1 bit of payload. At 6.3M params (~2.6 MB) compressing 1,000 MB, the model is **~400× smaller than the data**, so the real bottlenecks are **capacity per archived byte**, **target variance**, **data locality**, and **zero-archive-cost online memory**.

Key external references:
- **Hutter #1 (`fx2-cmix-T`, Vladimir Ivanov, July 2026 — `100,424,672` bytes):** Replaces `fx2-cmix`'s slow online 0.4M LSTM with an offline-pretrained ~6M Transformer, combined with `cmix` context models and `starlit`-style Wikipedia article reordering.
- **`tufazip v2` (Tufa Labs — `105,388,353` bytes):** Pure neural Transformer (12 layers, 163M active params expanded from a 4.33 MB quantized Cycle2 2-block offline starter prior, 4K vocab, 192-tok context + 35M-tok replay buffer) trained deterministically online. Requires 2×B200 for ~7.5 days (not CPU-viable directly, but shows the massive power of online adaptation + depth-shared starter priors).
- **`modded-nanogpt` (2025–2026) & OpenAI `Parameter Golf` (16 MB challenge, 2026):** Source of compact-model SOTA tricks—Exclusive Self-Attention (XSA), Polar Express Muon orthogonalization, SmearGate, compact Value Embeddings, depth recurrence, late-cooldown mixed-precision QAT/GPTQ, and chunk-level LoRA Test-Time Training (TTT).
- **Q Labs `Slowrun` & `Pretraining under Infinite Compute`:** Multi-epoch training, looped Transformers, and **distilling a large model/ensemble into a much smaller student** (retaining most of the teacher's loss asymptote).

---

## 3. Prioritized Roadmap: What to Try Next

### Tier 1 — Big Structural Levers (~2–5 MB each)

1. **Zero-Archive-Cost Online Adaptation (`tufazip` + `online_hash_bench.py` + Parameter Golf TTT)**
   - **Sparse/Dense Online Logit Hash (`online_hash_bench.py`):** Already implemented with an exact `S=1` token-by-token C sparse top-K logit bucket (microseconds/tok on CPU, 0 archive bytes). Benchmark it on a `rich-voice-20`-class checkpoint first; next step is co-training/residual-training the Transformer with it.
   - **Pure-Neural CPU-Viable TTT (TufaZip-lite / Parameter Golf):** During decode, update *only* a tiny subset—e.g. `lm_head` bias, a rank-8 LoRA on the coda block, and/or per-pass `pass_scale`/`pass_shift` AdaRMS vectors—every 256–1024 tokens. Costs **0 archive bytes**, `<10%` of forward FLOPs, strictly deterministic on CPU.
   - **Match-Cache Upgrade (if keeping `matchfeat.py`):** Treat `matchfeat` as an O(1) infinite-context neural retrieval cache (cleaner than `cmix`). High-value general upgrades: **2-way hash buckets** (most recent + longest) and **active-match continuation** (follow `j+1` when `pred[t-1] == tok[t]`, zero extra hashing).

2. **Reversible Semantic Wikipedia Article Reordering (`starlit`-style, from `fx2-cmix-T`)**
   - Cluster `enwik9` articles by semantic similarity (embeddings + TSP approximation) before tokenization; store the reversible article permutation (~0.4–0.6 MB compressed).
   - No `cmix` heuristics needed, yet it massively boosts local context for `T=2048` attention, increases short-distance long-length `matchfeat` hits, and makes online adaptation far more effective. Historically worth **~2–4 MB**.

3. **Giant Offline Teacher → Small Student Distillation (`Infinite Compute` + `Slowrun`)**
   - `train.py` already supports `TEACHER` / `KD_ALPHA` / `KD_TEMP`, and early byte-level 85M `worldly-leaf-3` hit `0.8855 bpb` in just 30k steps on v1.
   - Train a **60M–150M v8 teacher** (or 2-seed ensemble) directly on `enwik9.split4096`—the teacher is **never shipped** (0 archive bytes)—then distill via `KL(teacher || student)` into the 6M–10M student. Dense 4096-way soft targets dramatically cut gradient variance for a capacity-bottlenecked student.

4. **Longer Runs + Late-Cooldown Mixed-Precision QAT + MDL Weight-Budget Tuning**
   - **Scale steps 3–5×:** Run `300k–500k` steps (~30–50 epochs) with WSD + cosine or `1-sqrt` cooldown.
   - **Never QAT from scratch:** `smooth-galaxy-12` (`qat_group=64` from step 0) lagged `~0.04 bpb` behind FP8. Instead, train in FP8/BF16 and turn on **QAT only in the final 15–20% cooldown** with **mixed precision** (e.g., **Int6 for tied `wte`/`lm_head`**, **Int4 or Int3.5/VQ for block weights**) plus a light entropy/L2 penalty to minimize `bits_per_qweight`.
   - **Find the MDL sweet spot:** At `~2.6 MB` of weights, adding +1 MB of weights is net-positive whenever it saves `>0.008 bpb` (`>1 MB` payload). Test **8M–12M params** (`~3.5–4.5 MB` quantized, similar to `tufazip`'s 4.33 MB starter), balanced against single-core CPU FLOPs.

---

### Tier 2 — Clean Architecture, Optimizer & Pipeline Upgrades (`modded-nanogpt` 2026 / `Parameter Golf` / `Slowrun`)

Your `model.py` and `muon.py` already have NorMuon + cautious WD, Prelude-Core×2-Coda looping, U-Net skips, per-pass AdaRMS, Half-RoPE, QK-Norm, GQA+SWA, per-head `attn_gate`, SwiGLU, and `conv_k=4`. High-leverage 2026 additions not yet implemented:

1. **Exclusive Self-Attention (XSA)** (`modded-nanogpt` 2026 & `Parameter Golf`, Apple Mar 2026):
   - Project out the component of each token's attention output along its own `v` vector (`y = y - proj_v(y)`), optionally with a learnable per-(pass, head) gate. **0 parameters**, minimal compute, prevents attention from duplicating self-copying already handled by residuals/`conv_k=4`.
2. **Polar Express + Muon Momentum Schedule (`muon.py`):**
   - Replace fixed `(3.4445, -4.7750, 2.0315)` Newton-Schulz coefficients in `ns5` with **Polar Express** per-step minimax coefficients, and schedule Muon momentum (`0.85 → 0.95` warmup, `0.95 → 0.85` during cooldown).
3. **SmearGate + Compact GQA Value Embeddings:**
   - **SmearGate:** Learned gated blend of previous-token embedding into `x0` (~0 params).
   - **Value Embedding:** Since `kv_dim=128`, a low-rank/shared token-indexed value embedding `[4096, 64] → 128` costs only `~0.12 MB` in 4-bit and enriches attention values across passes beyond `v0` mixing.
4. **Deeper Weight-Tied Looping (`Slowrun` / `TufaZip Cycle2`):**
   - Test `n_layer=6, n_loop=3` (**14 passes**, fewer archived bytes than `7×2`) or `n_layer=7, n_loop=3` (**17 passes**, 0 extra matrix bytes). Watch CPU decode FLOPs.
5. **Fix BPE Training Sample & Test `VOCAB=8192` (`bpe_tok.py`):**
   - `bpe_tok.py` currently trains merges only on `text[: 30 << 20]` (first 30 MB of `enwik9`). Sample evenly across all 1 GB of `enwik9` for free compression gains.
   - Test `VOCAB=8192` (~0.45 MB extra in 4-bit for tied `wte`/`lm_head`): ~15–20% fewer tokens → **proportionally faster single-core CPU decoding** *and* longer effective byte context (~6.8 KB vs ~5.7 KB in `T=2048`).
6. **True Hutter-Score Eval Harness (Build First):**
   - Evaluate strided full-context windows (`[T//2:]`) across evenly spaced chunks of `enwik9` + exact post-quantization entropy-coded weight bytes + vocab/code overhead, logging projected **total Hutter MB** directly to W&B.

---

## 4. Suggested Execution Order on Vast.ai

1. **Harness & free fixes first:** Add true strided Hutter-MB eval; fix `bpe_tok.py` to sample across all of `enwik9`; implement XSA, Polar Express + momentum schedule, and late-cooldown mixed-precision QAT.
2. **Fast ablations (20k–30k steps each):** Validate XSA + Polar Express, `n_loop=3`, `VOCAB=8192`, and `online_hash_bench.py` on a cooled-down checkpoint.
3. **Big levers:** Add reversible `starlit`-style article reordering → train a 100M v8 teacher on reordered `enwik9` → run a 300k–500k distilled student + late-cooldown mixed-precision QAT + CPU-fast online logit/TTT adaptation.

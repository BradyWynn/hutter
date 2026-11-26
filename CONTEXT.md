# Hutter Prize Neural Compression — Project Context & Session Findings

> Up-to-date handoff document for agents working in `/workspace/hutter` (updated 2026-11-25).
> Replaces the older `NEXT_STEPS.md`. Read this alongside `train.py` and `model.py` before running experiments.

---

## 1. Goal, Rules & MDL Objective

- **Target:** Beat the Hutter Prize #1 record (`fx2-cmix-T`, July 2026: **100,424,672 bytes / 100.42 MB**; next prize threshold **~99.42 MB**) on `enwik9` (`1,000,000,000` bytes).
- **Official runtime rules:** Single CPU core, `<10 GB` RAM, `~50 hours` (`180,000 s`) decompress time.
- **Pure Minimum Description Length (MDL) — zero train/test split:**
  $$\text{Total Archive Bytes} = \text{NLL}(\text{enwik9} \mid \text{weights}) + \text{bits}(\text{weights}) + \text{vocab} + \text{overhead}$$
  Memorizing `enwik9` into weights is always a net win whenever `+1 MB` of compressed weights saves `>1 MB` of payload (`>0.008 bpb`), **provided inference stays inside the 50h single-core CPU budget**.

---

## 2. Single Most Important Metric: `hutter_mb`

Always judge runs by **`hutter_mb`** logged by `eval_hutter.py` (and `train.py` every `CKPT` steps):

$$\text{hutter\_mb} = \underbrace{\text{q\_strided\_bpb} \times 125.0}_{\text{Payload MB (full 1024–2047 tok ctx, post-quant)}} + \underbrace{\text{weight\_mb}}_{\text{Entropy-coded Int6 emb + Int4 blk + fp16 misc}} + \underbrace{\text{vocab\_mb} + \text{extra\_mb} + 0.05}_{\text{BPE merges + transforms + code}}$$

- **Why `EVAL_WINDOWS=2048` matters:** `enwik9` has massive local entropy variance (dense prose `~1.1 bpb` vs US Census/tables `~0.2 bpb`). Earlier `512`-window evals had `±1.5 MB` sampling variance; `2048` evenly spaced windows (`2.1M` scored full-context tokens across all 1 GB) takes only ~3s/model on GPU and cuts noise sharply.
- **Supporting metrics to watch alongside `hutter_mb`:**
  - `q_drop_bpb` (`q_strided_bpb - ema_strided_bpb`): PTQ penalty (`~+0.012–0.015 bpb` before QAT, snaps to **`0.0000`** once late-cooldown QAT activates at `QAT_START=0.85`).
  - `ema_eval_bpb`: Legacy full-window `[0..2047]` bpb, kept strictly for comparison with historical run `rich-voice-20` (`0.8306` at 100k steps, unquantized).

---

## 3. Environment, Hardware & Repo Map

### Hardware & Vast.ai Notes
- **GPU:** 1× **RTX 5070 Ti** (16 GB VRAM, Blackwell `sm_120`, Driver `580.126.09`, CUDA `13.0`), 72 vCPUs (`Xeon E5-2686 v4 @ 2.30 GHz`), 125 GB RAM.
- **Python env:** `/workspace/hutter/.venv` (Python 3.11.18): `torch==2.14.1+cu130`, `triton==3.8.0`, `numpy==2.4.6`, `tokenizers==0.23.3`, `tqdm==4.70.1`, `wandb==0.31.2` (logged in as `bradywynn` on Forge W&B, project `hutter`).
- **Disk & persistence:** Container overlay is **32 GB total** (`~16 GB` free). Each 6-column `matchfeat` `.npy` cache is `~6.3 GB` — delete unused `.match_*.npy` files when switching tokenizers. **`workspace_is_volume` is `false`**, so checkpoints/changes are lost if the instance is destroyed/recycled unless synced off-box.
- **Compile speed tip:** On this 2.3 GHz Xeon CPU, cold `mode="max-autotune"` takes ~15 min. Use `COMPILE_MODE=reduce-overhead MUON_COMPILE_MODE=default` for fast compiles (~2 min cold, ~15s cached) with identical CUDA graph + Triton + `flex_attention` runtime speed.

### Key Files
- `model.py` — v10 `GPT` architecture: Prelude-Core×`n_loop`-Coda (or unlooped `n_loop=1`), `d=256`, GQA (`4Q/2KV`), SWA=256 (full every 4th + last via `flex_attention`), SwiGLU or **DeepSeek-V3 `MoEMLP`** (shared + fine-grained routed SwiGLU experts via native Blackwell `torch._grouped_mm` + aux-loss-free `router_bias`), `conv_k=4`, Half-RoPE, U-Net skips, per-pass AdaRMS, per-head `attn_gate`, **Exclusive Self-Attention (`xsa=True`)**, 6-col context-gated match head (`matchfeat.py`), tied `wte`/`lm_head`, **3-tier dynamic mixed-precision STE QAT** (`set_qat(group_size=64, block_bits=5, emb_bits=6, expert_bits=4)`), `fp8=False` default, `return_features=True`, `quantize_stats()`, and bit-exact `quantize_model_copy()`. *(Also contains optional `Engram` class, tested and ruled out at `engram_rows=0` default.)*
- `muon.py` — NorMuon + cautious WD + **5-step Polar Express** orthogonalization + 0-D GPU tensor `beta` (dynamic momentum with zero `torch.compile` recompiles) + native 3D `[E, M, N]` MoE expert stack support.
- `train.py` — Training loop with env-var overrides (`DATA`, `MATCH_K`, `N_LAYER`, `N_LOOP`, `N_EXPERTS`, `TOP_K`, `SHARED_H`, `EXPERT_H`, `QAT_START`, `QAT_EMB_BITS`, `QAT_BLOCK_BITS`, `QAT_EXPERT_BITS`, `STEPS`, `CKPT`, `EVAL_WINDOWS`, `FP8`, `RUN_NAME`, `OUT_DIR`, `COMPILE_MODE`), WSD schedule (`COOLDOWN=0.50`), Muon momentum schedule (`0.85 → 0.95 → 0.85`), aux-loss-free `update_router_biases()`, late-cooldown QAT (separate compiled graph `loss_fn_qat`), and full `eval_hutter_suite` logging.
- `eval_hutter.py` — Standalone + importable strided full-context Hutter-MB evaluator (`python eval_hutter.py <ckpt> --data enwik9.split4096 --windows 2048`).
- `bpe_tok.py` — Byte-level BPE tokenizer (`split` trains on full 1 GB in ~55s; `nosplit` uses 4 KB newline chunks).
- `matchfeat.py` — GPU-built causal 64-bit FNV polynomial hash match tables (`TABLE_BITS=26`, `MATCH_K=(2,3,5,8,13,27)` for `split4096`), cached as disk memmaps.
- `online_hash_bench.py` — Updated to v8/v9 `GPT` (`return_features=True`), dynamic `D`/`n_match=6` C sparse-bucket kernel + GPU dense-engram benchmark.
- `reorder_enwik9.py` — Fast GPU-SVD + hierarchical K-Means/TSP reversible article reorderer (SHA-256 verified; see §4 for why naive page reordering broke even).

---

## 4. Empirical Findings From This Session

### What Worked (Keep)
1. **Uniform/full-file BPE (`bpe_tok.py`):** Replacing the old first-30-MB prefix (`357.8M` toks, `2.795 B/tok`) with uniform 64 MB sampling cut `enwik9.split4096` to **`350,034,900` tokens (`2.857 B/tok`)** — **7.78M fewer tokens (-2.2%)** for free. (Training on all 1,000 MB only trims another `0.08%`.)
2. **`split4096` is our best tokenizer so far** (beat both `split8192` and `nosplit4096` on `hutter_mb`).
3. **Late-cooldown mixed-precision QAT (`QAT_START=0.85`, Int6 tied `wte`/`lm_head`, Int4 blocks, `g=64`):** Completely erases the `+1.65 MB` PTQ penalty (`q_drop_bpb: +0.0132 → 0.0000`) in the final 15% of training, yielding **`3.29 MB`** total entropy-coded weights (`emb 5.65 bpw`, `blk 3.48 bpw`). Never QAT from step 0.
4. **Pure BF16 (`fp8=False`) beats FP8 at `d=256`:** Both **faster** (`83.5 ms/step` vs `87.1 ms/step` because `FP8MM` quantization overhead outweighs matmul speedup on small `256×704` matrices) AND **cleaner** (`-0.0051 bpb` / `-0.63 MB` less noise than per-tensor FP8 quant).
5. **v9 upgrades (XSA + Polar Express + dynamic Muon momentum + BF16):** Completed 100k anchor run **`v9-split4096-100k`** (`vcm3zsc0`, `2.65h`, `83.5 ms/step`) — reached **`ema_eval_bpb = 0.8343` / `ema_strided = 0.8208` (`102.60 MB` unquantized payload)** by step 85k (ahead of `rich-voice-20`'s pace), and finished at **`107.10 MB` true quantized 2048-win `hutter_mb`** (`q_strided = 0.8300`, `q_drop = 0.0000`, `wt = 3.29 MB`).
6. **Important QAT precision finding from 100k run:** As the tiny 6.27M model saturated (steps 50k→85k), the Int4 block PTQ gap grew sharply (`+0.0162` → `+0.0291 bpb` / `+3.64 MB`). Late QAT (85k→100k) recovered `~2.5 MB` (`109.59 → 107.10 MB`), but Int4 blocks still left `~1.5–1.9 MB` of capacity damage vs where BF16 was headed (`~0.815`). Because `5.13M` block params going from Int4 → Int5 only costs `+0.65 MB` (`+1.29 MB` for Int6), **Int4 over-quantizes tiny dense models**; test **Int5/Int6 for dense/shared layers** and **Int4 for larger MoE routed experts**!

### What We Tested & Ruled Out (Do NOT Repeat Without Addressing Root Cause)

| Experiment (20k steps, v9) | Result vs `split4096` baseline | Why It Lost |
|---|---|---|
| **`nosplit4096`** (`ysub4tg6`, `MATCH_K=1,2,3,5,9,18`) | **`+3.35 MB` worse** (`113.78` vs `110.43 MB` on 512-win) | Cut tokens `-20.5%` (`278.1M`, `3.596 B/tok`), **but** cross-boundary merges misaligned repeated phrases, crushing both `matchfeat` accuracy (`k=5` acc `55.5%` vs `k=8` `78.2%` at similar byte span) and in-context attention copying (`-2.14 MB` strided gain vs `-3.25 MB`). |
| **`split8192`** (`eq7sv33t`, `7.32M` params) | **`+0.43 MB` worse** (`110.86` vs `110.43 MB` on 512-win) | Saved only `-0.36 MB` strided payload (`0.8537` vs `0.8566`), which did not pay back `+0.78 MB` extra Int6 tied embeddings (`4.07` vs `3.29 MB`), and trained 8% slower (`90.3` vs `83.6 ms/step`). *(Note: looked better on legacy full-window `ema_eval_bpb`, proving why `hutter_mb` is essential.)* |
| **Bolted-on online n-gram hash tables** (`online_hash_bench.py` on `v9_split4096_20k`, first `8.39M` toks) | **`-0.30 MB` max** (Design 2 C sparse `lr=0.25`); **`-0.17 MB`** (Design 1 dense `lr=0.5`); all higher LRs hurt | End-to-end 6-col context-gated `matchfeat` already captures almost all exact n-gram repetition; unlearned additive hash deltas overlap with it and hurt calibration at higher LRs. |
| **Naive semantic page reordering** (`reorder_enwik9.py`, `6cf4mt2j`, 192-dim GPU SVD + K-Means/TSP, `0.50 MB` Lehmer perm) | **`+0.45 MB` net** (`112.34` vs `111.89 MB` on **2048-win**: payload `-0.05` to `-0.42 MB` gross, perm `+0.50 MB`) | Three reasons: (1) Static model with no online learning only sees ~5.8 KB in `T=2048`; (2) Original `enwik9` already has strong creation-order locality (`adj_cos = 0.308`); (3) Permuting pages scrambles strictly monotonic `<page><id>` numbers, costing `~0.5 MB` inside XML `<id>` tags. Only worth revisiting *with* `starlit`-style XML `<id>` delta-encoding + online adaptation. |
| **Learned N-Gram Engram Table** (`v10-moe17m-engram8m-20k`, 4-head bigram+trigram `65536×32` = `+8.45M` params, context-gated after Block 0) | **`+2.82 MB` worse** at step 12k (`117.41` vs `114.59 MB`; killed early) | Saved only **`-1.20 MB`** payload (`-0.0096 bpb`) at **`+4.05 MB`** Int4 weight cost (**`0.30 MB/MB`** efficiency vs **`1.5+ MB/MB`** for MoE experts!). Static n-gram lookup tables are ~5× less byte-efficient than contextual MoE MLP experts when `conv_k=4` + 6-col `matchfeat` are already present. |

### Recent Checkpoints on Disk
- `runs/v9_split4096_20k/` (`nb76hzxp`): 20k `split4096` v9 (`111.89 MB` on 2048-win).
- `runs/v9_split4096_100k/` (`vcm3zsc0`): 100k pure-BF16 v9 anchor (`6.27M` params, `12` looped passes, `83.5 ms/step`, `2.65h`). Final 2048-win: `q_strided=0.8300` (`103.75 MB` payload) + `wt=3.29 MB` = **`107.10 MB`**.
- `runs/v10_moe17m_20k/` (`fhm2rmi4`): 20k unlooped MoE (`16.85M` params, `L=8×1`, `1s256 + 8r256`, `top_k=2`, `0.72×` active CPU MACs of v9, `6/5/4`-bit QAT). Final 2048-win: `q_strided=0.8195` + `wt=8.67 MB` = **`111.17 MB`** (`-0.72 MB` ahead of `v9-20k`).
- `runs/v10_moe17m_100k/` (`k428b6dj`, **NEW BEST — 3.93 MB FROM HUTTER #1!**): 100k unlooped MoE (`16.85M` params, `L=8×1`, `1s256 + 8r256`, `top_k=2`, `0.72×` CPU MACs, `6/5/4`-bit QAT at `85k–100k`, `~101 ms/step`, `3.17h`).
  - Step 85k (pre-QAT BF16): `ema_eval_bpb=0.7776`, `ema_strided=0.7650` (`95.63 MB` payload), PTQ `q_strided=0.7778` (`+0.0128` drop, less than half of v9's `+0.0291`), `HUTTER=105.95 MB`.
  - **Step 100k (final quantized, 2048-win):** `eval_bpb=0.7778`, `ema_eval_bpb=0.7777` (vs `rich-voice-20`'s `0.8306`!), **`q_strided=0.7649` (`95.62 MB` payload, `q_drop=0.0000`)**, `wt=8.67 MB` (`emb 5.65 bpw`, `blk 4.55 bpw`, `exp 3.49 bpw`), **`HUTTER=104.35 MB`** — **beats `tufazip v2` (`105.39 MB`) by `-1.04 MB`, beats `v9-100k` (`107.10 MB`) by `-2.75 MB`, and was still dropping `-0.20 MB` every 2.5k steps at step 100k!**
  - Single-core CPU benchmark (`taskset -c 0`, Broadwell 2.3 GHz AVX2 FP32): multi-stream batching (`B=32–64`) gave **6.5×–7.7× speedup** (`23–28h` for 350M toks even in FP32 on Broadwell; `~4–5h` est. on modern INT8/VNNI core).

---

## 5. Strategic Direction & Next Steps (Path From `104.35 MB` → Sub-`100.42 MB`)

### Proven Core Insight
Unlooped Fine-Grained MoE (`v10-moe17m-100k`: `16.85M` params, `8.67 MB` weights, `0.72×` active CPU MACs of v9) returned **`~1.5 MB` payload savings per `1 MB` of weights** — jumping from `107.10 MB → 104.35 MB` (`-2.75 MB` net, beating `tufazip v2`'s `105.39 MB`) with **28% less CPU decode compute** and only **2.34 epochs/expert**. By contrast, static Engram lookup tables only returned `0.30 MB/MB`.

Multi-stream batched CPU decoding (`B=32–64` interleaved chunks on 1 core) gives a **6.5×–7.7× speedup** (`23–28h` for all 350M tokens even in FP32 on a 2.3 GHz Broadwell core; `~4–6h` est. on modern INT8/VNNI), so we still have substantial CPU and weight-budget headroom.

### Highest-Leverage Next Experiments (Only `3.93 MB` From Hutter #1!)

1. **Compress MoE Expert Weights Below `3.49 bpw` (Fast ~25 min QAT branch from `v10-moe17m-100k`):**
   - Branch directly from `runs/v10_moe17m_100k/model_ema.pt` (15k late-QAT steps) to test **3-bit routed expert QAT (`QAT_EXPERT_BITS=3`)** and/or **light L1/entropy regularization during late QAT** (`3.49 → ~2.3–2.6 bpw`). Could immediately shave `~1.5–2.5 MB` off `104.35 MB` with zero extra CPU FLOPs.
2. **Find the MoE Parameter Sweet Spot (`20k` fast ablation, ~40 min):**
   - Test **`v10-moe30m`** (`L=8×1`, `1s256 + 16r256`, `top_k=2`, `29.5M` params, **same `0.72×` CPU MACs**; use `B=128` streams on CPU so each expert still gets ~16 tokens/GEMM). Pair with compressed expert bits (`3b` or L1-sparsified `4b`) to see if `30M` beats `17M` on total `hutter_mb`.
3. **Longer Schedule (`250k–400k` steps) + Giant Offline Teacher Distillation:**
   - `v10-moe17m-100k` only saw `2.34` epochs/expert and was still dropping `-0.20 MB` every 2.5k steps at step 100k. Running `250k–400k` steps (~8–12h) and/or distilling from an offline 100M+ teacher (`TEACHER` / `KD_ALPHA` already wired in `train.py`) gives `~2–4 MB` at 0 archive bytes and 0 CPU FLOPs.
4. **End-to-End `matchfeat.py` Upgrades (0 archive bytes, `O(1)` CPU):**
   - Keep the same 6 `k` columns (more `k`s have diminishing returns), and upgrade retrieval quality via **active-match continuation** (`j+1` when `pred[t-1] == tok[t]`) and **2-way hash buckets** (most recent + longest/previous).
5. **Optional Later Levers:**
   - Reversible Wikipedia XML metadata/caps preprocessor (`starlit`-style delta-`<id>` + tag/caps factoring) and/or CPU-viable chunk-level neural TTT (updating `lm_head.bias` / AdaRMS / rank-8 coda LoRA every 512–1024 tokens).

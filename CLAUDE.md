# MiniGPT — Claude Code Context

**Status:** Modern Llama-style architecture (RoPE + SwiGLU + RMSNorm, ~30M) implemented and gradient-checked. Hand-written backward passes for all components pass `scripts/validate_gradients.py` on the NumPy backend. Tuned for an RTX 3060 12GB rig. (Previously a GPT-1 (2018) faithful replica — see git history / `docs/GPT1_IMPLEMENTATION_SPEC.md` for the prior design.)

---

## Current State

### What MiniGPT is now

A modern decoder-only Transformer (Llama-style) implemented with a thin `xp` abstraction over NumPy (CPU) and CuPy (GPU). The hand-written backward pass remains — no PyTorch, no autograd. Old GPT-1 checkpoints (LayerNorm/GELU/learned-pos, 40K vocab) are **not** compatible.

### Architecture (modern, Llama-style)

| Component | Implementation | Location |
|---|---|---|
| Normalization | **RMSNorm** (γ only, `eps=1e-5`), compact Jacobian-vector-product backward; pre-norm + a final RMSNorm before the head | `src/minigpt/model.py::RMSNorm` |
| Block order | **Pre-norm**: `h = x + Attention(RMSNorm1(x)); y = h + SwiGLU(RMSNorm2(h))` | `src/minigpt/model.py::TransformerBlock` |
| Activation | **SwiGLU**: `down( silu(gate(x)) · up(x) )`, 3 matrices, no bias; `silu`/gate fused via `cupy.fuse` | `src/minigpt/model.py::FeedForward, _fused_silu, _fused_swiglu` |
| Positional encoding | **RoPE** (rotary), `theta=10000`, applied to Q/K per head; backward is the inverse rotation. No `W_pos`. | `src/minigpt/model.py::build_rope_tables, apply_rope` |
| Attention | **Full MHA**, **no biases**; `n_kv_heads == n_heads` (no GQA) | `src/minigpt/model.py::MultiHeadAttention` |
| Output projection | **Tied to token embedding**: `logits = RMSNorm_final(x) @ W_emb.T`; gradient accumulates from the output-projection path (`dW_emb_out`) and the input-lookup path (`dX_emb` scatter-added by the trainer). | `src/minigpt/model.py::MiniTransformer.forward/backward` |
| Dropout | Plumbing retained (softmax, attn/FFN residual, embedding) but defaults to `0.0`. | `src/minigpt/model.py` |
| Init | `N(0, 0.02)` for weight matrices; output projections (`W_down`, `W_o`) scaled by `0.02 / √(2·n_layers)` (residual-stream init); RMSNorm γ = ones | `src/minigpt/model.py` |
| KV cache | Inference only. `training=True` passes `cache=None` to attention, ~10× forward speedup preserved. | `src/minigpt/optimized_kv_cache.py` |

### Gradient tuple layout (what the trainer threads)

- `MiniTransformer.backward()` → `(dW_emb_out, layer_grads, dX_emb, d_gamma_final)`.
- Per-block (`TransformerBlock.backward`) → `(swiglu_grads, attn_grads, rms1_dgamma, rms2_dgamma)` where `swiglu_grads = (dW_gate, dW_up, dW_down)` and `attn_grads = (dW_qkv, dW_o)`.
- This shape is consumed by `scripts/train.py` (`global_grad_norm`, `apply_grads`, `consolidate_tied_embedding_grad`) and `scripts/validate_gradients.py`. Change all four together.

### Optimizer (GPT-1 faithful)

| Component | Implementation | Location |
|---|---|---|
| Algorithm | **Adam with coupled L2 weight decay**: `g ← g + wd·p` folded into the gradient, then standard Adam moments. This is the classic 2018 formulation, not decoupled AdamW. | `src/minigpt/optimizer.py::Adam` |
| Hyperparameters | `lr=2.5e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01` | config defaults |
| Parameter groups | Two groups: **decay** (2-D weight matrices: `W_emb, W_qkv, W_o, W_gate, W_up, W_down`) and **no-decay** (RMSNorm γ only — there are no biases or `W_pos`). Built via `build_param_groups(model)`. | `src/minigpt/optimizer.py::build_param_groups` |
| LR schedule | **Linear warmup** `0 → 2.5e-4` over 2000 steps, then **cosine decay** to `1e-5` over the remaining steps. Exposed as `LRSchedule(peak, min, warmup, total)`. | `src/minigpt/optimizer.py::LRSchedule` |
| Per-step call | `optimizer.step_grouped(param_groups, grad_groups, lr=lr_now)` | `src/minigpt/optimizer.py::Adam.step_grouped` |
| Gradient clipping | Global-norm clip at `1.0`, applied after accumulation, before optimizer step. All 1D params (γ, β, biases) are now in the clip set. | `src/minigpt/trainer.py::clip_grads` |
| Back-compat | `AdamW = Adam` alias kept; `Adam.step(params, grads)` single-group path kept for scripts that have not yet migrated to `step_grouped`. | `src/minigpt/optimizer.py` |

### Target Config (default in `config.py`)

```python
ModelConfig(
    vocab_size = 16_384,
    d_model    = 512,
    n_layers   = 8,
    n_heads    = 8,
    d_ff       = 1024,        # SwiGLU hidden width
    max_len    = 512,
    dropout    = 0.0,
    rope_theta = 10_000.0,
)

TrainConfig(
    learning_rate = 3e-4,
    min_lr        = 3e-5,
    betas         = (0.9, 0.95),
    eps           = 1e-8,
    weight_decay  = 0.1,
    warmup_steps  = 2000,
    max_steps     = 100_000,
    batch_size    = 32,       # micro-batch (safe 3060 default)
    accum_steps   = 4,        # effective batch = 128
    grad_clip     = 1.0,
    seq_len       = 512,
)
```

Instantiating `MiniTransformer(ModelConfig())` produces **29,368,832 parameters (~30M)** — verified by `scripts/smoke_test_gpt1.py`.

### Preserved from M0 (do not touch)

- `src/minigpt/backend.py` — `xp` CuPy/NumPy abstraction, `fp16_matmul` (Ampere SM86 Tensor-Core FP16 GEMM), `fuse` decorator, `scatter_add`, VRAM monitoring (`vram_stats`, `log_vram`, `estimate_model_vram` — now SwiGLU/bias-free aware).
- `src/minigpt/inference.py` — generation, sampling, repetition penalty, logit bias, query-classification RAG pipeline.
- `src/minigpt/optimized_kv_cache.py` — ring-buffer KV cache.
- `scripts/validate_gradients.py`, `scripts/benchmark_gpu.py`, `scripts/prepare_fineweb.py`.
- Checkpoint save/load logic in `scripts/train.py`.

---

## Key Files

| File | Purpose |
|---|---|
| `src/minigpt/model.py` | Modern Transformer: `RMSNorm`, `FeedForward` (SwiGLU: gate/up/down, no bias), `MultiHeadAttention` (full MHA + RoPE, no bias), `TransformerBlock` (pre-norm), `EmbeddingLayer` (token only — RoPE handles position), `MiniTransformer` (tied output, final RMSNorm). RoPE helpers `build_rope_tables/apply_rope`; fused `_fused_silu/_fused_swiglu/_fused_rmsnorm_apply/_fused_rope_combine`. |
| `src/minigpt/optimizer.py` | `Adam` (coupled L2 WD), `LRSchedule` (linear warmup + cosine), `build_param_groups` (no-decay = RMSNorm γ only). `AdamW = Adam` alias. |
| `src/minigpt/config.py` | Dataclasses: `TokenizerConfig` (16K vocab), `ModelConfig` (~30M defaults + `rope_theta`), `TrainConfig`. Helpers `estimate_training_memory_mb`, `solve_batch_plan`. |
| `src/minigpt/dataloader.py` | `BinDataLoader` — memmap a flat uint16 `.bin`, sample micro-batches to the active device (RAM-safe). `open_splits()` for train/val. |
| `src/minigpt/tokenizer.py` | Two classes: legacy `BPETokenizer` (pure Python, byte-level, `<EOS>` only), and `HFBPETokenizer` (HuggingFace `tokenizers` backend, 40K vocab, `<pad>` / `<eos>` / `<unk>` atomic specials, `train_on_fineweb()` classmethod). |
| `src/minigpt/trainer.py` | Rewritten for new 4-tuple backward API `(dW_emb_out, dW_pos, layer_grads, dX_emb)` and per-layer grad shapes including biases and LN β. Uses `step_grouped` with `LRSchedule`. |
| `src/minigpt/backend.py` | GPU/CPU abstraction, FP16 matmul, fused kernels, VRAM utilities. Preserved. |
| `src/minigpt/inference.py` | Generation engine. Preserved. |
| `src/minigpt/optimized_kv_cache.py` | Ring-buffer KV cache. Preserved. |
| `scripts/train.py` | Main training loop (FineWeb shards, gradient accumulation). Needs a touch-up to match the new backward tuple — mirror `trainer.py`. |
| `scripts/smoke_test_gpt1.py` | CPU-only instantiation + forward/backward + param-group verification at `d=384/L=6`. Run this first on any new machine. |
| `scripts/prepare_fineweb.py` | Stream FineWeb-Edu sample-10BT → 16K HFBPE → flat uint16 `train.bin`/`val.bin` with bounded-RAM flushes. |
| `scripts/calculate_vram_budget.py` | Recommend `batch_size`/`accum_steps` for the ~30M model on the 3060. |
| `scripts/validate_gradients.py` | Numerical gradient checker. Preserved. |
| `scripts/benchmark_gpu.py` | GPU throughput + VRAM stress tests + FP16 vs FP32. Preserved. |
| `docs/GPT1_IMPLEMENTATION_SPEC.md` | File-by-file diff-level spec that produced M1+M2. Reference for what changed and why. |
| `context.md` / `agents.md` | Milestone log + agent responsibilities. Update at milestone boundaries. |
| `MiniGPT_Progress_Report.pdf` | Original M0 progress report (pp. 1-5) + GPT-1 Roadmap supplement (pp. 6-9, added 2026-04-17). |

---

## Smoke Test

Run this before any new training effort:

```bash
MINIGPT_BACKEND=numpy python3 scripts/smoke_test_gpt1.py
```

Expected output:
```
[OK] Tied embedding, no W_pos, no biases/beta.
[OK] Parameter count: 12,194,688  (small-scale d=384/L=6/V=4096)
[OK] Forward+backward. step-0 loss ≈ 8.3  (≈ ln(V))
[OK] Only RMSNorm gammas excluded from weight decay.
[OK] Config defaults: d=512 L=8 H=8 d_ff=1024 V=16384 theta=10000 -> 29,368,832 params
========= ALL SMOKE TESTS PASSED =========
```

Also run the numerical gradient check (the real safety gate for the hand-written RoPE/SwiGLU/RMSNorm backward):
```bash
MINIGPT_BACKEND=numpy python3 scripts/validate_gradients.py   # -> ALL CHECKS PASSED
```

---

## Training Command (modern ~30M, RTX 3060)

```bash
# Pre-flight (CPU is fine for these)
MINIGPT_BACKEND=numpy python3 scripts/smoke_test_gpt1.py
MINIGPT_BACKEND=numpy python3 scripts/validate_gradients.py   # ALL CHECKS PASSED
MINIGPT_BACKEND=cupy  python3 scripts/benchmark_gpu.py        # confirm FP16 path on the card

# Data: stream FineWeb-Edu sample-10BT -> 16K BPE -> flat uint16 .bin (RAM-safe)
python3 scripts/prepare_fineweb.py --max_tokens 1_000_000_000

# Pick batch/accum for the 3060's VRAM
python3 scripts/calculate_vram_budget.py

# Full training (defaults are the modern ~30M config)
tmux new -s training
MINIGPT_BACKEND=cupy python3 scripts/train.py \
    --batch_size 32 --accum_steps 4 \
    --data_path data/train.bin \
    --output_dir outputs/modern_30m --checkpoint_interval 1000
```

> Dims/hparams (`d=512, L=8, H=8, d_ff=1024, V=16384, T=512`, RoPE `theta=10000`, lr `3e-4`, betas `(0.9,0.95)`, wd `0.1`) come from `config.py`. The `train.py` CLI exposes `--batch_size/--accum_steps/--steps/...`; override model dims via a `--config` JSON/YAML file.

**Notes:**
- Step-0 loss ≈ ln(16384) ≈ 9.70.
- `batch_size 32 / accum 4` (eff 128) is the safe default (~5GB est, ~8-9GB with FP32 backward recompute). The calculator suggests pushing micro-batch higher (≈48–64); confirm on the card with `benchmark_gpu.py` before maxing.
- No `nvcc`/`cupy.RawKernel` required: the RoPE/SwiGLU/RMSNorm kernels use `cupy.fuse`, which JIT-compiles element-wise kernels and is a no-op on the NumPy backend.

---

## GPT-1 Target Config Rationale

| Choice | Why |
|---|---|
| `d_model = 768`, `n_layers = 12`, `n_heads = 12` | Matches GPT-1 paper §3.1 exactly. |
| `d_ff = 3072` (4 × d_model) | GPT-1 paper §3.1. (The legacy auto-formula `2·4·d/3` was the SwiGLU ratio — removed.) |
| `max_len = 512` | GPT-1 paper §3.1. |
| `vocab_size = 40000` | GPT-1 paper §4.1 (BPE on BooksCorpus; we use FineWeb-Edu). |
| `lr = 2.5e-4`, `betas = (0.9, 0.98)`, `eps = 1e-8` | GPT-1 paper §4.1. Note: `beta2 = 0.98` is higher than the modern LM default of 0.95. |
| Coupled L2 (not decoupled AdamW) | GPT-1 (2018) predates decoupled weight decay (Loshchilov & Hutter 2019). Historical fidelity. |
| `warmup_steps = 2000` linear + cosine to `min_lr = 1e-5` | GPT-1 paper §4.1. |
| `dropout = 0.1` everywhere | GPT-1 paper §4.1 (residual, attention, embedding). |
| WD-exclude `W_pos` | GPT-1 convention: positional embeddings are not regularized via weight decay. The `ndim >= 2` gate alone fails here because `W_pos` is 2-D; `build_param_groups` adds an explicit name-based skip. |
| Tied input/output embeddings | GPT-1 paper §3.1. Saves ~30M parameters at 40K × 768. Gradient from `logits = h @ W_emb.T` accumulates into `dW_emb_out`; the input-lookup contribution is `scatter_add(dW_emb_out, token_ids, dX_emb)` in the trainer. |

---

## Design Choices Preserved from M0

| Decision | Rationale | Revisit if... |
|---|---|---|
| FP16 forward / FP32 backward | Ampere Tensor Cores (RTX 3060, SM86) without loss-scaling overhead. Manual backward needs FP32. | Moving to A100/H100 with BF16. |
| `cupy.fuse` element-wise kernels (SiLU, RoPE combine, RMS apply) | Decorator-based, automatic CPU no-op, correct scope (not matmul); compiles without `nvcc`, testable on CPU. | We need warp-level reductions or Flash Attention (then `cupy.RawKernel`). |
| FineWeb-Edu corpus | Large, quality-filtered, streamable. | Strict historical replication requires BooksCorpus. |
| Pure NumPy/CuPy (no PyTorch) | From-scratch pedagogical objective. | We hit the compiled-graph autograd ceiling. |
| KV cache bypass during training | Per-token Python loop was the #1 bottleneck; `start_pos=0` makes the cache unnecessary. | Streaming or prefix-tuning training. |

---

## VRAM Budget — ~30M on RTX 3060 12 GB

Run `python3 scripts/calculate_vram_budget.py` for the live table. The 30M model is tiny — params + optimizer + grads total only ~280 MB, so **activations dominate** (SwiGLU footprint ≈ `3·B·T·d_ff`). FP16-mixed estimates at T=512:

| micro-batch | est. peak | note |
|---|---|---|
| 32 | ~5.3 GB | safe default (eff batch 128 via accum 4) |
| 48 | ~7.8 GB | comfortable |
| 64 | ~10.1 GB | near the limit (raw estimate) |
| ≥96 | OOM | exceeds 12 GB |

`estimate_model_vram()` (in `backend.py`) and `estimate_training_memory_mb()` (in `config.py`) now model SwiGLU (`3·B·T·d_ff`) and the bias-free parameter count. The estimator does **not** model the FP32 backward recompute, so `calculate_vram_budget.py` derates the budget for its recommendation; `benchmark_gpu.py` on the real card is authoritative before maxing the micro-batch.

---

## Monitoring During a Training Run

```bash
tmux attach -t training                                    # Watch loss
watch -n 1 nvidia-smi                                      # GPU utilization
nvidia-smi -l 60 --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
tail -f checkpoints_gpt1/training_log.txt | grep "Step \|Train=\|Val="
```

**Key metrics:**
- Loss decreases smoothly; step-0 ≈ 10.60 for V=40K.
- VRAM < 8 GB on FP16 mixed path; if creeping toward 11 GB, reduce micro-batch.
- GPU utilization > 90%; temperature < 85 °C.
- Gradient norm in `[0.1, 5.0]` post-warmup.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| OOM during training | Reduce `--batch_size`; raise `--accum_steps` to preserve effective batch. |
| Loss spikes / NaN | Halve `--lr`, restart from last checkpoint. Confirm `grad_clip=1.0` is active. |
| Slow throughput | `nvidia-smi` → confirm GPU use; check FP16 mixed precision is on; re-run `benchmark_gpu.py`. |
| FP16 instability | `set_mixed_precision(False)` in `backend.py` to force FP32; debug in FP32 and re-enable. |
| Tokenizer not found | `HFBPETokenizer.train_on_fineweb(...)` or point `--tokenizer` at an existing `assets/tokenizer_gpt1_40k.json`. |
| `ValueError: Checkpoint vocab_size=4096 != model vocab_size=40000` | Expected — 4K/8K checkpoints are not forward-compatible with the 40K GPT-1 model. Retrain from scratch. |
| Dropout does nothing in generation | `model.eval()` (or not calling `model.train()`) disables dropout; confirm you're not inspecting a `training=True` forward pass. |

---

## Remaining Work (on the GPU box)

The architecture, optimizer, data pipeline, and VRAM tooling are done and CPU-verified
(`validate_gradients.py` + `smoke_test_gpt1.py` pass). What remains needs the RTX 3060
(this dev box has no cupy/nvcc):

- Install GPU deps: `pip install cupy-cuda12x datasets` (plus `requirements.txt`).
- `MINIGPT_BACKEND=cupy python3 scripts/benchmark_gpu.py` — confirm the FP16 Tensor-Core path and that the `cupy.fuse` RoPE/SwiGLU/RMSNorm kernels JIT-compile.
- `python3 scripts/prepare_fineweb.py --max_tokens 1_000_000_000` — build the 16K tokenizer + `data/train.bin`/`val.bin` (one-time, RAM-safe).
- Re-run `calculate_vram_budget.py`, then push micro-batch as high as `benchmark_gpu.py` confirms safe.
- 10-step GPU smoke run; confirm no OOM and step-0 loss ≈ ln(16384) ≈ 9.70.
- Launch the 100K-step run.

---

## Context for Future Sessions

This codebase implements a modern Llama-style Transformer (RoPE + SwiGLU + RMSNorm, ~30M) in hand-written NumPy/CuPy, with every backward pass derived by hand and verified numerically on the NumPy backend (`validate_gradients.py` → ALL CHECKS PASSED). The FP16 Tensor-Core matmul, `cupy.fuse` element-wise kernels, KV cache, and VRAM monitoring carry over. Tuned for an RTX 3060 12GB / i5-12400 / 16GB-RAM rig. Remaining work is GPU-side only (install cupy/datasets, build `data/train.bin`, benchmark, launch).

**Gradient tuple invariant:** keep `model.backward` (4-tuple), `TransformerBlock.backward` (per-block 4-tuple), `scripts/train.py`, and `scripts/validate_gradients.py` in lockstep.

**Last update:** 2026-06-23 (modern RoPE/SwiGLU/RMSNorm rebuild; CPU gradient-checked)
**Reference:** git history for the prior GPT-1 design; `docs/GPT1_IMPLEMENTATION_SPEC.md` (historical).

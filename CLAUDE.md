# MiniGPT — Claude Code Context

**Status:** GPT-1 (2018) architectural replica implemented. M1 (architecture) and M2 (optimizer) complete per `docs/GPT1_IMPLEMENTATION_SPEC.md`. M3 (40K BPE tokenizer with HuggingFace `tokenizers`) and M4 (scale to 117M on RTX 3060) remain.

---

## Current State

### What MiniGPT is now

A decoder-only Transformer implemented with a thin `xp` abstraction over NumPy (CPU) and CuPy (GPU). All modern-LLM components from the Llama-3-style predecessor have been replaced with GPT-1-faithful equivalents. The hand-written backward pass remains — no PyTorch, no autograd.

### Architecture (GPT-1 faithful)

| Component | Implementation | Location |
|---|---|---|
| Normalization | **Standard LayerNorm** (γ and β, `eps=1e-5`), compact Jacobian-vector-product backward | `src/minigpt/model.py::LayerNorm` |
| Block order | **Post-norm**: `a = x + Attention(x); b = LN1(a); c = b + FFN(b); y = LN2(c)` | `src/minigpt/model.py::TransformerBlock` |
| Activation | **GELU tanh approximation**: `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`; fused forward + backward via `cupy.fuse` | `src/minigpt/model.py::_fused_gelu, _fused_gelu_backward` |
| Positional encoding | **Learned absolute** `W_pos` shape `(max_len, d_model)`, init N(0, 0.01), added at input layer only (no RoPE anywhere) | `src/minigpt/model.py::EmbeddingLayer` |
| Attention | **Full MHA**: `n_kv_heads == n_heads` enforced in config post-init; QKV and output projections carry biases | `src/minigpt/model.py::MultiHeadAttention` |
| Output projection | **Tied to token embedding**: `logits = x_final @ W_emb.T`; gradient accumulates from both the output-projection path (`dW_emb_out`) and the input-lookup path (`dX_emb` scatter-added by the trainer). No final LayerNorm. | `src/minigpt/model.py::MiniTransformer.forward/backward` |
| Dropout (4 sites) | (1) softmax weights inside attention, (2) attention output residual branch, (3) after GELU in FFN, (4) post-embedding sum. Masks cached for backward. Toggled by `model.train()` / `model.eval()`. | `src/minigpt/model.py` |
| Init | `N(0, 0.02)` for weight matrices; output projections (`W_proj`, `W_o`) scaled by `0.02 / √(2·n_layers)` (GPT-2 residual-stream init); biases zero-init; `W_pos` init `N(0, 0.01)` | `src/minigpt/model.py` |
| KV cache | Inference only. `training=True` passes `cache=None` to attention, ~10× forward speedup preserved. | `src/minigpt/optimized_kv_cache.py` |

### Optimizer (GPT-1 faithful)

| Component | Implementation | Location |
|---|---|---|
| Algorithm | **Adam with coupled L2 weight decay**: `g ← g + wd·p` folded into the gradient, then standard Adam moments. This is the classic 2018 formulation, not decoupled AdamW. | `src/minigpt/optimizer.py::Adam` |
| Hyperparameters | `lr=2.5e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01` | config defaults |
| Parameter groups | Two groups: **decay** (2-D weight matrices: `W_emb, W_qkv, W_o, W_fc, W_proj`) and **no-decay** (LN γ, LN β, every bias, and `W_pos`). Built via `build_param_groups(model)`. | `src/minigpt/optimizer.py::build_param_groups` |
| LR schedule | **Linear warmup** `0 → 2.5e-4` over 2000 steps, then **cosine decay** to `1e-5` over the remaining steps. Exposed as `LRSchedule(peak, min, warmup, total)`. | `src/minigpt/optimizer.py::LRSchedule` |
| Per-step call | `optimizer.step_grouped(param_groups, grad_groups, lr=lr_now)` | `src/minigpt/optimizer.py::Adam.step_grouped` |
| Gradient clipping | Global-norm clip at `1.0`, applied after accumulation, before optimizer step. All 1D params (γ, β, biases) are now in the clip set. | `src/minigpt/trainer.py::clip_grads` |
| Back-compat | `AdamW = Adam` alias kept; `Adam.step(params, grads)` single-group path kept for scripts that have not yet migrated to `step_grouped`. | `src/minigpt/optimizer.py` |

### GPT-1 Target Config (default in `config.py`)

```python
ModelConfig(
    vocab_size = 40_000,
    d_model    = 768,
    n_layers   = 12,
    n_heads    = 12,
    d_ff       = 3072,        # 4 · d_model
    max_len    = 512,
    dropout    = 0.1,
)

TrainConfig(
    learning_rate = 2.5e-4,
    min_lr        = 1e-5,
    betas         = (0.9, 0.98),
    eps           = 1e-8,
    weight_decay  = 0.01,
    warmup_steps  = 2000,
    max_steps     = 100_000,
    batch_size    = 8,        # micro-batch
    accum_steps   = 8,        # effective batch = 64
    grad_clip     = 1.0,
    seq_len       = 512,
)
```

Instantiating `MiniTransformer(ModelConfig())` produces **~117M parameters**.

### Preserved from M0 (do not touch)

- `src/minigpt/backend.py` — `xp` CuPy/NumPy abstraction, `fp16_matmul`, `fuse` decorator, `scatter_add`, VRAM monitoring (`vram_stats`, `log_vram`, `estimate_model_vram`). The FFN-ratio constant inside `estimate_model_vram` is still the SwiGLU `2·4d/3`; fix as part of M4 before the 117M training run.
- `src/minigpt/inference.py` — generation, sampling, repetition penalty, logit bias, query-classification RAG pipeline.
- `src/minigpt/optimized_kv_cache.py` — ring-buffer KV cache.
- `scripts/validate_gradients.py`, `scripts/benchmark_gpu.py`, `scripts/prepare_fineweb.py`.
- Checkpoint save/load logic in `scripts/train.py`.

---

## Key Files

| File | Purpose |
|---|---|
| `src/minigpt/model.py` | GPT-1 Transformer: `LayerNorm`, `FeedForward` (Linear→GELU→Dropout→Linear), `MultiHeadAttention` (full MHA with softmax dropout), `TransformerBlock` (post-norm), `EmbeddingLayer` (token + learned pos), `MiniTransformer` (tied output, no final LN, `train()`/`eval()` toggle). |
| `src/minigpt/optimizer.py` | `Adam` (coupled L2 WD), `LRSchedule` (linear warmup + cosine), `build_param_groups`. `AdamW = Adam` alias. |
| `src/minigpt/config.py` | Dataclasses: `TokenizerConfig` (40K vocab + specials), `ModelConfig` (GPT-1 117M defaults), `TrainConfig` (Adam hparams + warmup 2000). Legacy `n_kv_heads`/`rope_theta` fields retained as no-op defaults for back-compat. |
| `src/minigpt/tokenizer.py` | Two classes: legacy `BPETokenizer` (pure Python, byte-level, `<EOS>` only), and `HFBPETokenizer` (HuggingFace `tokenizers` backend, 40K vocab, `<pad>` / `<eos>` / `<unk>` atomic specials, `train_on_fineweb()` classmethod). |
| `src/minigpt/trainer.py` | Rewritten for new 4-tuple backward API `(dW_emb_out, dW_pos, layer_grads, dX_emb)` and per-layer grad shapes including biases and LN β. Uses `step_grouped` with `LRSchedule`. |
| `src/minigpt/backend.py` | GPU/CPU abstraction, FP16 matmul, fused kernels, VRAM utilities. Preserved. |
| `src/minigpt/inference.py` | Generation engine. Preserved. |
| `src/minigpt/optimized_kv_cache.py` | Ring-buffer KV cache. Preserved. |
| `scripts/train.py` | Main training loop (FineWeb shards, gradient accumulation). Needs a touch-up to match the new backward tuple — mirror `trainer.py`. |
| `scripts/smoke_test_gpt1.py` | CPU-only instantiation + forward/backward + param-group verification at `d=384/L=6`. Run this first on any new machine. |
| `scripts/prepare_fineweb.py` | Stream FineWeb-Edu → tokenize → shard. Preserved. |
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
[OK] Tied embedding: W_emb appears exactly once, no W_out.
[OK] Parameter count: ~14,000,000  (at d=384/L=6/V=4096)
[OK] Forward+backward. step-0 loss ≈ 8.3
[OK] W_pos, LN gamma/beta, all biases correctly excluded from WD.
[OK] LR schedule: step0=1.25e-07 warmup_end=2.50e-04 final=1.00e-05
========= ALL SMOKE TESTS PASSED =========
```

---

## Training Command (117M, 72-Hour Budget)

```bash
# Pre-flight
MINIGPT_BACKEND=numpy python3 scripts/smoke_test_gpt1.py
MINIGPT_BACKEND=cupy  python3 scripts/validate_gradients.py
MINIGPT_BACKEND=cupy  python3 scripts/benchmark_gpu.py

# Tokenizer (one-time, if tokenizer_gpt1_40k.json not present)
python3 -c "from minigpt.tokenizer import HFBPETokenizer; \
            HFBPETokenizer.train_on_fineweb(n_docs=100_000, \
            vocab_size=40_000, save_path='assets/tokenizer_gpt1_40k.json')"

# Full training (defaults are GPT-1 117M)
tmux new -s training
MINIGPT_BACKEND=cupy python3 scripts/train.py \
    --dim 768 --n_layers 12 --n_heads 12 \
    --vocab_size 40000 --max_len 512 \
    --batch_size 8 --accum_steps 8 \
    --steps 100000 --lr 2.5e-4 --warmup_steps 2000 --min_lr 1e-5 \
    --weight_decay 0.01 --grad_clip 1.0 \
    --dropout 0.1 \
    --save_dir checkpoints_gpt1 --save_interval 1000
```

**Expected milestones** (projections from `MiniGPT_Progress_Report.pdf` §4):
- FP16 mixed, micro-batch 8, accum 8 → effective batch 64 → ~0.35 s/step → 50K steps in ~4.9 h, 100K in ~9.7 h.
- Peak VRAM ~7.2 GB / 12 GB on RTX 3060.
- Step-0 loss ≈ ln(40000) ≈ 10.60.
- Target final val loss: ~3.0 (PPL ~20) on FineWeb-Edu held-out split.

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
| FP16 forward / FP32 backward | Tensor Cores on RTX 3060 without loss-scaling overhead. Manual backward needs FP32. | Moving to A100/H100 with BF16. |
| `cupy.fuse` element-wise kernels | Decorator-based, automatic CPU no-op, correct scope (not matmul). | We need warp-level reductions or Flash Attention. |
| FineWeb-Edu corpus | Large, quality-filtered, streamable. | Strict historical replication requires BooksCorpus. |
| Pure NumPy/CuPy (no PyTorch) | From-scratch pedagogical objective. | We hit the compiled-graph autograd ceiling. |
| KV cache bypass during training | Per-token Python loop was the #1 bottleneck; `start_pos=0` makes the cache unnecessary. | Streaming or prefix-tuning training. |

---

## VRAM Budget — 117M on RTX 3060 12 GB

See `MiniGPT_Progress_Report.pdf` §3 for the full breakdown. Summary:

| Component | FP32 | FP16 mixed |
|---|---|---|
| Model params (FP32 master) | 468 MB | 468 MB |
| Optimizer (m + v, FP32) | 936 MB | 936 MB |
| Gradients (FP32) | 468 MB | 468 MB |
| Activations (12 layers, T=512, B=8) | ~9,100 MB | ~4,550 MB |
| FP16 temporaries | — | ~300 MB |
| Overhead | ~500 MB | ~500 MB |
| **Total** | **~11,500 MB** | **~7,200 MB** |
| **Headroom** | **~780 MB** | **~5,080 MB** |

**Verdict:** FP32 fits only at micro-batch 4; FP16 mixed is the default path. Update `estimate_model_vram()` to use GPT-1's `d_ff = 4·d` ratio before relying on its output for the 117M run (currently uses SwiGLU's `2·4d/3`).

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

## Remaining Work (M3, M4)

### M3: Tokenizer rebuild
- `HFBPETokenizer.train_on_fineweb(n_docs=100_000, vocab_size=40_000)` — run once, ~20 min.
- Update `scripts/prepare_fineweb.py` to emit `<eos>` at every document boundary.
- Re-encode training shards under the new tokenizer; version the output path as `data/fineweb_gpt1_40k/`.
- Update checkpoint save path to bundle `tokenizer_gpt1_40k.json` inline.

### M4: Scale to 117M
- Update `estimate_model_vram()` FFN ratio: `d_ff = 4 * d_model` (not `2*4*d/3`).
- Add biases and LN β to the parameter-count accounting in the estimator.
- Run `scripts/benchmark_gpu.py` at `d=768, L=12, H=12, V=40000, T=512, B=8, accum=8`; confirm `fits_12gb`.
- 10-step smoke run at full scale; confirm no OOM, step-0 loss ≈ 10.60, grad norm in target range.
- Launch 100K-step training run. Project ~10 h on RTX 3060 at FP16 mixed.

---

## Context for Future Sessions

This codebase now implements GPT-1 (Radford et al., 2018) faithfully in hand-written NumPy/CuPy at the architecture and optimizer level. The pre-existing M0 infrastructure (FP16, fused kernels, FineWeb pipeline, gradient validation, VRAM monitoring) carries over unchanged. The only pieces between here and a full 117M training run are: (1) train the 40K tokenizer on FineWeb-Edu, (2) fix the VRAM estimator's FFN ratio, (3) run the benchmark at full scale to pick the final batch config.

**Last update:** 2026-04-17 (M1 + M2 applied per `docs/GPT1_IMPLEMENTATION_SPEC.md`)
**Branch:** main (M1+M2 changes pending commit)
**Reference:** `docs/GPT1_IMPLEMENTATION_SPEC.md`, `MiniGPT_Progress_Report.pdf` (pp. 6-9)

# MiniGPT: A From-Scratch Modern Transformer in Pure NumPy / CuPy

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Backend](https://img.shields.io/badge/backend-NumPy%20%7C%20CuPy-orange.svg)
![Architecture](https://img.shields.io/badge/arch-RoPE%20%7C%20SwiGLU%20%7C%20RMSNorm-purple.svg)
![Params](https://img.shields.io/badge/params-~30M-green.svg)

**A dependency-light, Llama-style decoder-only Transformer — every layer, the optimizer, and the entire backward pass written by hand in NumPy / CuPy. No PyTorch, no TensorFlow, no autograd.**

MiniGPT demystifies the "black box" of Transformers by building one end-to-end with nothing but array math. A thin `xp` abstraction runs the *same* code on CPU (NumPy) or NVIDIA GPU (CuPy), so you can study the model on a laptop and train it on a GPU — tuned here for an **RTX 3060 12GB** rig.

> **History:** This repo previously shipped a GPT-1 (2018) faithful variant (LayerNorm/GELU/learned-positions). It has been rebuilt as a modern Llama-style model (RoPE + SwiGLU + RMSNorm). Old GPT-1 checkpoints are not compatible.

---

## What This Is

This codebase implements a **modern decoder-only Transformer** (Llama-style), with a hand-derived backward pass for every component — including RoPE, SwiGLU, and RMSNorm.

| Property | Value |
|---|---|
| Architecture | Decoder-only Transformer (pre-norm) |
| Target size | **~30M params** (`d=512, L=8, H=8, d_ff=1024`) |
| Normalization | **RMSNorm** (γ only), pre-norm + final norm |
| Activation | **SwiGLU** FFN (`down(silu(gate)·up)`), fused SiLU kernel |
| Positional encoding | **RoPE** (rotary), `theta=10000`, applied to Q/K |
| Attention | **Full multi-head** (8 heads), **no biases** |
| Output head | **Tied** to token embedding; final RMSNorm before projection |
| Optimizer | **Adam + coupled L2 weight decay** |
| Backend | `xp` abstraction over **NumPy (CPU)** and **CuPy (GPU)** |
| Precision | FP16 forward / FP32 backward (mixed) on GPU |
| Autograd | **None** — backward pass is hand-written |

---

## Architecture (modern, Llama-style)

- **RMSNorm** (`eps=1e-5`, scale γ only) with a compact Jacobian-vector-product backward; pre-norm blocks plus a final RMSNorm before the tied head.
- **Pre-norm block:** `h = x + Attn(RMSNorm1(x)); y = h + SwiGLU(RMSNorm2(h))`.
- **RoPE:** rotary embedding applied to Q and K per head (`theta=10000`); the backward is the inverse rotation. No learned positional table.
- **SwiGLU FFN:** `down( silu(gate(x)) · up(x) )` — three weight matrices, no biases; `silu` and the gate are fused via `cupy.fuse`.
- **Full MHA**, no biases anywhere (Llama convention).
- **Tied input/output embeddings** — `logits = RMSNorm(x_final) @ W_emb.Tᵀ`.
- **Dropout** plumbing retained (softmax, attention/FFN residual, embedding) but defaults to `0.0`.
- **KV cache** for inference only (bypassed during training for ~10× forward speedup).

## Optimizer (GPT-1 Faithful)

- **Adam with coupled L2 weight decay** — `g ← g + wd·p` folded into the gradient (the classic 2018 formulation, *not* decoupled AdamW).
- Hyperparameters: `lr=2.5e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01`.
- **Two parameter groups:** decay (2-D weight matrices) vs. no-decay (LayerNorm γ/β, all biases, and `W_pos`).
- **LR schedule:** linear warmup `0 → 2.5e-4` over 2000 steps, then cosine decay to `1e-5`.
- **Global-norm gradient clipping** at `1.0`, applied after accumulation.

---

## Live Training Run — Jetson Orin

A 98M-parameter variant was trained on an **NVIDIA Jetson Orin** (CuPy GPU backend, 7.4 GB VRAM) to validate the pipeline on edge hardware:

| Setting | Value |
|---|---|
| Model | `d=768, L=12, H=12, V=16384, T=512` → 98M params |
| Batch | micro-batch 1 × accum 32 → effective batch 32 |
| Throughput | ~790 tokens/sec |
| Loss | 12.63 (init) → 5.07 (step 100) → 4.29 (step 200) |
| Val loss | 4.33 @ step 200 (no overfitting vs. train) |

The full GPT-1 target (117M, 40K vocab) is configured for an RTX 3060 12 GB run (~10 h at FP16 mixed). See [`CLAUDE.md`](CLAUDE.md) for the VRAM budget and launch command.

---

## Quick Start

```bash
git clone https://github.com/X-Hell/MiniGPT.git
cd MiniGPT
pip install -r requirements.txt

# 1) Sanity-check the implementation (CPU, fast)
MINIGPT_BACKEND=numpy python3 scripts/smoke_test_gpt1.py

# 2) Numerically verify the hand-written gradients (GPU)
MINIGPT_BACKEND=cupy python3 scripts/validate_gradients.py

# 3) Generate text from a checkpoint
python3 scripts/generate.py --prompt "Once upon a time"
```

### Train (modern ~30M target)

```bash
# 0) Prepare data: stream FineWeb-Edu -> 16K BPE -> flat uint16 data/train.bin
python3 scripts/prepare_fineweb.py --max_tokens 1_000_000_000

# 1) Pick the batch/accum that maxes the 3060's VRAM
python3 scripts/calculate_vram_budget.py

# 2) Train (defaults are the modern ~30M config)
MINIGPT_BACKEND=cupy python3 scripts/train.py \
    --batch_size 32 --accum_steps 4 \
    --data_path data/train.bin \
    --output_dir outputs/modern_30m
```

> Model/optimizer defaults (`d=512, L=8, H=8, d_ff=1024, V=16384, T=512`, RoPE `theta=10000`, lr `3e-4`, betas `(0.9, 0.95)`) live in `src/minigpt/config.py`. Override dims via a `--config` JSON/YAML file.

> Select the compute backend with the `MINIGPT_BACKEND` env var: `numpy` (CPU) or `cupy` (GPU). The same code path runs on both.

---

## Project Structure

```
MiniGPT/
├── src/minigpt/
│   ├── model.py              # Modern Transformer (RMSNorm, RoPE, SwiGLU, tied head) + hand-written backward
│   ├── dataloader.py         # BinDataLoader: memmap a flat uint16 .bin, stream micro-batches to GPU
│   ├── optimizer.py          # Adam (coupled L2 WD), LRSchedule (warmup + cosine), param groups
│   ├── config.py             # ModelConfig / TrainConfig / Jetson memory planner
│   ├── tokenizer.py          # Legacy byte-level BPE + HuggingFace 40K BPE (HFBPETokenizer)
│   ├── backend.py            # xp abstraction: NumPy/CuPy, FP16 matmul, fused kernels, VRAM stats
│   ├── inference.py          # Generation, sampling, repetition penalty, RAG pipeline
│   ├── optimized_kv_cache.py # Ring-buffer KV cache (inference)
│   └── rag.py                # Retrieval-augmented generation
├── scripts/
│   ├── train.py              # Main training loop (grad accumulation, checkpointing)
│   ├── smoke_test_gpt1.py    # CPU instantiation + fwd/bwd + param-group checks
│   ├── validate_gradients.py # Numerical gradient checker
│   ├── benchmark_gpu.py      # GPU throughput + VRAM stress + FP16 vs FP32
│   ├── generate.py           # Text generation CLI
│   ├── rag_chat.py           # RAG-powered chat
│   ├── inference_jetson.py   # Jetson inference helper
│   ├── calculate_jetson_budget.py / calculate_vram_budget.py
│   └── launch_10h_training.sh / launch_72h_training.sh
├── configs/                  # YAML run configs (jetson_gpt1_10h, gpt1_72h, smoke_test, …)
├── tests/test_all.py         # Test suite
├── docs/GPT1_IMPLEMENTATION_SPEC.md  # File-by-file spec behind the GPT-1 retrofit
└── CLAUDE.md                 # Engineering context: full config, VRAM budget, troubleshooting
```

> Large artifacts — `outputs/` (checkpoints + logs), tokenized `data/*.npy`, and process files — are intentionally git-ignored; they are regenerable from the scripts above.

---

## Roadmap & Status

### Completed

- [x] **M0 — Engine core & GPU infra:** `xp` NumPy/CuPy abstraction, FP16 mixed precision, fused CUDA kernels, KV cache, VRAM monitoring, FineWeb data pipeline, gradient validation.
- [x] **M1 — GPT-1 architecture retrofit:** LayerNorm (post-norm), GELU tanh, learned absolute positions, full MHA, tied output head — with hand-derived backward for each.
- [x] **M2 — GPT-1 optimizer:** Adam + coupled L2 weight decay, two-group decay logic, linear-warmup + cosine schedule, global-norm clipping.
- [x] **Edge validation:** 98M-param run on Jetson Orin (CuPy), loss 12.6 → 4.3.

### Remaining

- [ ] **M3 — Tokenizer rebuild:** train the 40K BPE tokenizer (`HFBPETokenizer.train_on_fineweb`), re-encode shards, bundle the tokenizer into checkpoints.
- [ ] **M4 — Scale to 117M:** fix the VRAM estimator FFN ratio, benchmark the final batch config, launch the full 100K-step run on RTX 3060.

---

## Design Choices

| Decision | Rationale |
|---|---|
| Pure NumPy / CuPy, no PyTorch | From-scratch pedagogical objective — the backward pass is the point. |
| RoPE + SwiGLU + RMSNorm | Modern Llama-style components, each with a hand-derived backward pass. |
| FP16 forward / FP32 backward | Ampere Tensor Cores (RTX 3060, SM86) without loss-scaling overhead; manual backward needs FP32. |
| `cupy.fuse` element-wise kernels (SiLU, RoPE combine, RMS apply) | Automatic CPU no-op, correct scope (element-wise, not matmul); portable and testable on CPU. |
| KV cache bypassed in training | `start_pos=0` makes it unnecessary; removes the per-token Python loop bottleneck. |
| Flat `uint16` `.bin` + `np.memmap` loader | RAM-safe over FineWeb-Edu sample-10BT on a 16GB-RAM rig — corpus stays on the NVMe SSD. |

---

## License

MIT

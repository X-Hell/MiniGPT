# MiniGPT: A From-Scratch GPT-1 in Pure NumPy / CuPy

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Backend](https://img.shields.io/badge/backend-NumPy%20%7C%20CuPy-orange.svg)
![Architecture](https://img.shields.io/badge/arch-GPT--1%20(2018)-purple.svg)
![Params](https://img.shields.io/badge/params-117M%20target-green.svg)

**A faithful, dependency-light reimplementation of GPT-1 (Radford et al., 2018) — every layer, the optimizer, and the entire backward pass written by hand in NumPy / CuPy. No PyTorch, no TensorFlow, no autograd.**

MiniGPT demystifies the "black box" of Transformers by building one end-to-end with nothing but array math. A thin `xp` abstraction runs the *same* code on CPU (NumPy) or NVIDIA GPU (CuPy), so you can study the model on a laptop and train it on a GPU — including edge devices like the Jetson Orin.

---

## What This Is

This codebase implements the **GPT-1 architecture exactly as described in the 2018 paper**, with a hand-derived backward pass for every component. It began as a Llama-3-style educational model and was retrofitted to be historically faithful to GPT-1 (see the milestone history below).

| Property | Value |
|---|---|
| Architecture | Decoder-only Transformer (GPT-1, post-norm) |
| Target size | **~117M params** (`d=768, L=12, H=12`) |
| Normalization | **LayerNorm** (γ, β), post-norm block order |
| Activation | **GELU** (tanh approximation), fused fwd/bwd kernel |
| Positional encoding | **Learned absolute** embeddings (no RoPE) |
| Attention | **Full multi-head** (12 heads), QKV + output biases |
| Output head | **Tied** to token embedding; no final LayerNorm |
| Optimizer | **Adam + coupled L2 weight decay** (classic 2018 form) |
| Backend | `xp` abstraction over **NumPy (CPU)** and **CuPy (GPU)** |
| Precision | FP16 forward / FP32 backward (mixed) on GPU |
| Autograd | **None** — backward pass is hand-written |

---

## Architecture (GPT-1 Faithful)

- **Standard LayerNorm** (`eps=1e-5`) with a compact Jacobian-vector-product backward.
- **Post-norm block:** `a = x + Attn(x); b = LN1(a); c = b + FFN(b); y = LN2(c)`.
- **GELU (tanh approx):** `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`, fused via `cupy.fuse`.
- **Learned absolute positions** `W_pos` added at the input layer only.
- **Full MHA** with biases on QKV and output projections.
- **Tied input/output embeddings** — `logits = x_final @ W_emb.Tᵀ`; saves ~30M params at 40K×768.
- **Dropout at 4 sites** (softmax weights, attention residual, post-GELU, post-embedding), toggled by `model.train()` / `model.eval()`.
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

### Train (GPT-1 117M target)

```bash
MINIGPT_BACKEND=cupy python3 scripts/train.py \
    --dim 768 --n_layers 12 --n_heads 12 \
    --vocab_size 40000 --max_len 512 \
    --batch_size 8 --accum_steps 8 \
    --steps 100000 --lr 2.5e-4 --warmup_steps 2000 --min_lr 1e-5 \
    --weight_decay 0.01 --grad_clip 1.0 --dropout 0.1 \
    --save_dir checkpoints_gpt1 --save_interval 1000
```

> Select the compute backend with the `MINIGPT_BACKEND` env var: `numpy` (CPU) or `cupy` (GPU). The same code path runs on both.

---

## Project Structure

```
MiniGPT/
├── src/minigpt/
│   ├── model.py              # GPT-1 Transformer (LayerNorm, GELU, MHA, tied head) + hand-written backward
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
| Coupled L2 (not decoupled AdamW) | GPT-1 (2018) predates decoupled weight decay; historical fidelity. |
| FP16 forward / FP32 backward | Tensor Cores without loss-scaling overhead; manual backward needs FP32. |
| `cupy.fuse` element-wise kernels | Automatic CPU no-op, correct scope (element-wise, not matmul). |
| KV cache bypassed in training | `start_pos=0` makes it unnecessary; removes the per-token Python loop bottleneck. |

---

## License

MIT

# MiniGPT: The NumPy-Only LLM Inference Engine

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Params](https://img.shields.io/badge/params-1.25M-orange.svg)
![Perplexity](https://img.shields.io/badge/perplexity-7.35-green.svg)

**A dependency-free implementation of modern LLM architectures (Llama 3, Mistral) built from scratch in NumPy.**

MiniGPT is designed to demystify the "black box" of Transformers. By implementing everything—from the attention mechanism to the tokenizer—using only `numpy`, this project provides a clean, educational look at how Large Language Models actually "think," without the abstraction layers of PyTorch or TensorFlow.

---

## Current Model

| Metric | Value |
|--------|-------|
| Parameters | **1.25M** |
| Architecture | 4-layer Transformer (Llama 3-style) |
| Embedding Dim | 128 |
| Attention | 4 heads, GQA (2 KV heads) |
| Vocabulary | 2,048 tokens (BPE, GPT-4 regex) |
| Context Window | 256 tokens |
| Perplexity | **7.35** (down from 181.3 baseline) |
| Latency | **1.2 ms/token** (down from 55 ms) |
| Memory | 4.9 MB |

> See [`data/model_card.md`](data/model_card.md) for full architecture details, weight health reports, and improvement roadmap.

---

## Architecture Features

- **RMSNorm** — Pre-normalization for training stability
- **SwiGLU** — Gated activation function (Llama 3 standard)
- **RoPE** — Rotary Positional Embeddings (θ=500K)
- **GQA** — Grouped Query Attention (4 Q heads → 2 KV heads)
- **BPE Tokenizer** — GPT-4-style regex splitting
- **OptimizedKVCache** — Ring buffer with dynamic batch resizing
- **Weight Tying** — Shared embedding/output projection
- **AdamW** — With cosine LR decay and gradient clipping

---

## Roadmap & Status

### Completed Milestones

- [x] **Milestone 1: Engine Core** — Base Transformer with character-level tokenization.
  > Perplexity: 181.3 → established baseline.

- [x] **Milestone 2: Inference Optimization** — KV Caching (ring buffer), static memory allocation.
  > Latency: 55ms → 42ms/token.

- [x] **Milestone 3: Parameter Efficiency** — Weight Tying, INT8 quantization support.
  > -19% parameters, significant memory reduction.

- [x] **Milestone 4: Modern Architecture** — SwiGLU, RMSNorm, GQA, RoPE.
  > Perplexity: 67.4, basic grammar acquisition.

- [x] **Milestone 5: Semantic Coherence** — BPE tokenizer (GPT-4 regex), scaled architecture to 4 layers/128 dim.
  > Perplexity: 7.35, overfitted on clean dataset.

- [x] **Milestone 6: Training Stability** — Cosine LR decay, gradient clipping, AdamW optimizer.
  > Stable training with no NaN/explosion issues.

- [x] **Milestone 8: Packaging** — Refactored into `src/minigpt` package with `dataclass` configs.
  > Clean pip-installable structure.

### In Progress

- [~] **Milestone 7: Scaling Laws** — Expand to 384 dim / 6 layers (~15M params), increase context to 512+.
  > **Current blocker:** Model is overfitted on small dataset (75 KB). Need larger, diverse training data before scaling.

### Future

- [ ] **Milestone 9: Speed** — Custom C++/CUDA kernels for critical MatMul ops.
- [ ] **Milestone 10: Instruction Tuning** — Fine-tune on Alpaca-style dataset.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/elsoro/MiniGPT.git
cd MiniGPT

# Install dependencies (only numpy, tqdm, regex, requests)
pip install -r requirements.txt

# Run generation
python scripts/generate.py --prompt "Once upon a time"

# Train the model
python scripts/train.py

# Start the web interface
python website_api/app.py          # Backend on :8000
cd website && npm run dev          # Frontend on :5173
```

---

## Project Structure

```
MiniGPT/
├── src/minigpt/              # Core library
│   ├── model.py              # Transformer (RMSNorm, SwiGLU, GQA, RoPE)
│   ├── inference.py          # Generation engine (sampling, RAG, safety)
│   ├── tokenizer.py          # BPE tokenizer (GPT-4 regex)
│   ├── optimized_kv_cache.py # Ring buffer KV cache
│   ├── optimizer.py          # AdamW optimizer
│   ├── trainer.py            # Training loop utilities
│   ├── rag.py                # Retrieval-Augmented Generation
│   └── config.py             # Dataclass configs
├── scripts/                  # CLI tools
│   ├── train.py              # Training script (cosine LR, grad clip)
│   ├── generate.py           # Text generation CLI
│   ├── rag_chat.py           # RAG-powered chat
│   ├── build_corpus.py       # Data pipeline
│   └── generate_curated_data.py
├── data/                     # Training data & metrics
│   ├── train_clean.txt       # Active training dataset
│   ├── phase_metrics.csv     # Performance tracking (16 phases)
│   └── model_card.md         # Detailed model diagnostics
├── checkpoints/              # Trained model weights
│   └── model_latest.pkl      # Current best model (20 MB)
├── website/                  # React/Vite frontend
├── website_api/              # Flask inference API
└── assets/
    └── tokenizer.model       # Trained BPE tokenizer
```

---

## Performance Tracking

All 16 optimization phases are tracked in [`data/phase_metrics.csv`](data/phase_metrics.csv):

| Phase | Perplexity | Latency (ms) | Key Change |
|-------|-----------|-------------|------------|
| 1 | 181.3 | 55.0 | FP32 baseline |
| 7 | 67.4 | 18.1 | SwiGLU + RMSNorm + RoPE + GQA |
| 11 | 45.6 | 12.8 | BPE tokenizer + fused QKV |
| 13 | 17.3 | 9.8 | 4-layer scale-up |
| **16** | **7.35** | **1.2** | **Clean dataset retrain + OptimizedKVCache** |

---

## Known Issues

| Priority | Issue | Details |
|----------|-------|---------|
| 🔴 High | **Overfitting** | Low perplexity (7.35) but incoherent generation on novel prompts. Training set is only 75 KB. |
| 🟡 Medium | Small context | 256-token limit restricts multi-turn conversation. |
| 🟡 Medium | Small vocab | 2,048 BPE tokens causes subword fragmentation. |
| 🟢 Low | Scale | 1.25M params limits representational capacity. |

> See [`data/model_card.md`](data/model_card.md) for detailed diagnostics and fix strategies.

---

## License

MIT

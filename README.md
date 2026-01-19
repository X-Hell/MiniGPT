<div align="center">

# MiniGPT: The Educational Inference Engine
### From Naive FP32 to Research-Grade Optimization in 10 Phases

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Numpy-Only](https://img.shields.io/badge/dependency-numpy_only-red)](https://numpy.org)
[![Status: Optimized](https://img.shields.io/badge/Status-Optimized-success)]()

**MiniGPT** is a journey through the mechanics of Large Language Models. Built entirely in `numpy` without autograd libraries, it demonstrates how raw arithmetic transforms into intelligence, optimizing a Transformer from a memory-hogging baseline to a lean, quantized inference engine.

[ **[Explore Codebase](mini_transformer/)** ] • [ **[Evolution](phase_metrics.png)** ] • [ **[Visualization](attn_interactive.html)** ]

---

</div>

## 📈 Project Evolution (Phases 1-10)

This project was built in 10 distinct phases, mirroring the real-world optimization pipeline of modern LLMs. Below is the exact data on how each architectural decision impacted Model Size and Memory Footprint.

![Evolution Graph](phase_metrics.png)

### Evolution Markers

| **Phase** | **Major Change** | **Params** | **Peak Memory** | **Notes** |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `FP32 Transformer` | ~4.8M | ~6.2 MB | Baseline implementation. Functional but inefficient. |
| **2** | `Explicit Logger` | ~4.8M | ~6.2 MB | Added visibility into MatMul operations (Flops tracking). |
| **3** | `KV Cache` | ~4.8M | ~6.9 MB | **Key Feature:** Caching keys/values to speed up autoregressive decoding. |
| **4** | `Weight Tying` | **~3.9M** | **~5.3 MB** | 📉 **-19% Params**. Shared embeddings for input and output projection. |
| **5** | `INT8 Attention` | ~3.9M | **~3.9 MB** | Quantized Q/K/V projections for lower memory bandwidth. |
| **6** | `INT8 FFN` | ~3.9M | **~3.1 MB** | **Stable**. Post-training quantization of the Feed-Forward Network. |
| **7** | `Low-Rank FFN` | **~1.6M** | **~2.2 MB** | 📉 **Huge Drop**. Replaced dense FFN with Low-Rank approximation. |
| **8** | `QKV Fusion` | ~1.6M | ~2.0 MB | Merged matrices to reduce kernel launch overhead (simulated). |
| **9** | `Multi-Head Viz` | ~1.6M | ~2.0 MB | Added `matplotlib` visualizations for attention heads. |
| **10** | `Entropy Analysis` | ~1.6M | ~2.0 MB | **Research-Grade**. Entropy heatmaps & head similarity metrics. |

---

## 🧠 Technical Highlights

### 1. Manual Autograd Engine
Every gradient in this project is calculated **by hand**.
- `loss.backward()`? No. We calculate $\frac{\partial L}{\partial W}$ manually for Attention, LayerNorm, and Softmax.
- This unveils the "black box" of backpropagation.

### 2. INT8 Quantization & Low-Rank Adaptation
We don't just compress; we redesign.
- **Quantization**: Symmetric per-channel scaling for weights, reducing memory by 4x.
- **Low-Rank**: Factorizing large weight matrices $W \approx A \times B$ where $rank(A,B) \ll rank(W)$.

### 3. Nucleus Sampling (Top-p)
Instead of greedy decoding, we implement **Nucleus Sampling** to dynamically cut off the tail of the probability distribution, balancing creativity with coherence.

---

## 🚀 Quick Start

### Installation
Clone the repository and install the single dependency: `numpy`.

```bash
git clone https://github.com/your-username/minigpt.git
cd minigpt
pip install numpy matplotlib seaborn
```

### Training
Train the model on the Shakespeare dataset (auto-downloaded).

```bash
python mini_transformer/train.py
```
*Outputs: `mini_transformer_model.pkl`*

### Inference & Visualization
Run the inference engine. This will generate the attention maps and memory logs.

```bash
python -m mini_transformer.run_inference
```

**Generated Artifacts**:
- `attn_prefill.png`: Attention patterns during prompt processing.
- `head_similarity.png`: Cosine similarity between attention heads.
- `entropy_heatmap.png`: Uncertainty quantification per head.

---

## 🎨 Visualization Gallery

| Attention Pattern | Head Similarity | Entropy Heatmap |
| :---: | :---: | :---: |
| ![Attn](attn_prefill.png) | ![Sim](head_similarity.png) | ![Ent](entropy_heatmap.png) |

---

<div align="center">

**Educational Use Only**
Designed for understanding, not production.
*Built with ❤️ in Python*

</div>

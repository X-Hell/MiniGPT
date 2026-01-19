# Mini Transformer: An Educational Inference Engine with INT8 Quantization

## Abstract
**Mini Transformer** is a lightweight, widely accessible implementation of the Transformer architecture, designed to demonstrate the mechanics of Large Language Models (LLMs) at a microscopic scale. Unlike opaque deep learning frameworks, this project implements **manual backpropagation**, **post-training INT8 quantization**, and **advanced sampling strategies** (Nucleus Sampling) using only `numpy`. It serves as a transparent testbed for understanding the low-level arithmetic of modern AI.

## Technical Specifications

### Architecture
- **Type**: Decoder-only Transformer (GPT-style).
- **Depth**: 2 Layers (Stackable `TransformerBlock` units).
- **Dimensions**: $d_{model}=240$, $h=4$ heads, $d_{head}=60$.
- **Context Window**: 128 tokens.
- **Parameters**: ~1.5M parameters (uncompressed).

### Key Features
1.  **Manual Auto-Differentiation**:
    - Complete implementation of the backward pass for Attention, LayerNorm, and FeedForward layers without automatic differentiation libraries.
2.  **Weight Tying**:
    - The embedding matrix $W_{emb}$ is transposed and reused as the output projection head, reducing memory usage by ~25% and improving regularization.
3.  **Per-Channel Quantization**:
    - Post-training quantization of FFN weights ($W_1, W_2$) to `int8`.
    - Uses per-channel scaling (Axis 0/1) to preserve activation dominance, reducing FFN memory footprint by **4x** (450 KB vs 1.8 MB) with minimal degradation.
4.  **Nucleus Sampling (Top-$p$)**:
    - Implements Top-$p=0.9$ decoding to dynamically truncate the vocabulary distribution, balancing creativity distribution.
5.  **Optimized Training**:
    - Gradient Clipping (Norm $\le 1.0$).
    - Cosine Learning Rate Decay.

## Performance Metrics

| Metric | Value |
| :--- | :--- |
| **Training Data** | TinyShakespeare (1MB) |
| **Training Loss (2k steps)** | ~5.84 nats (Cross-Entropy) |
| **Perplexity** | ~345.0 |
| **Quantization Compression** | 4.0x (FFN Weights) |
| **Inference Latency (CPU)** | < 50ms / token |

*Note: Training was conducted on a minimal regime (2000 steps) for demonstration. Loss values are expected to converge further with extended training.*

## Installation & Usage

### Prerequisites
- Python 3.9+
- `numpy`, `matplotlib`

```bash
git clone https://github.com/your-username/mini-transformer.git
cd mini-transformer
pip install numpy matplotlib
```

### Training
Train the model from scratch using the Shakespeare dataset. The script automatically manages data downloading and preprocessing.

```bash
python mini_transformer/train.py
```
*Outputs: `mini_transformer_model.pkl`*

### Quantization
Compress the trained model using per-channel INT8 quantization.

```bash
python mini_transformer/quantize.py
```
*Outputs: `mini_transformer_model_quantized.pkl`*

### Inference
Run the inference engine with visualization support.

```bash
python -m mini_transformer.run_inference
```

## Visualization
The engine generates real-time visualization artifacts:
- `attn_prefill.png`: Multi-head attention patterns.
- `memory_log.png`: Dynamic memory usage tracking per operation.

## Citation
If you find this codebase useful for educational purposes, please cite:
```bibtex
@misc{minitransformer2025,
  author = {Project Contributor},
  title = {Mini Transformer: A Numpy-only LLM Inference Engine},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
}
```

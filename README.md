# MiniGPT: The NumPy-Only LLM Inference Engine

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

**A dependency-free implementation of modern LLM architectures (Llama 3, Mistral) built from scratch in NumPy.**

MiniGPT is designed to demystify the "black box" of Transformers. By implementing everything—from the attention mechanism to the tokenizer—using only `numpy`, this project provides a clean, educational look at how Large Language Models actually "think," without the abstraction layers of PyTorch or TensorFlow.

---

## Roadmap & Status

The project follows a strategic 10-milestone roadmap to evolve from a basic script to a production-grade inference engine.

### Completed Milestones
- [x] **Milestone 1: The Engine Core.** *Achieved.*
  Implemented the base Transformer architecture with character-level tokenization.
  > *Data:* Established baseline perplexity of ~181.3.

- [x] **Milestone 2: Inference Optimization.** *Achieved.*
  Implemented KV Caching (Ring Buffer) and Static Memory Allocation.
  > *Data:* Reduced inference latency from ~55ms to ~42ms per token.

- [x] **Milestone 3: Parameter Efficiency.** *Achieved.*
  Implemented Weight Tying and INT8 Quantization support.
  > *Data:* Reduced parameter count by 19% and memory footprint significantly.

- [x] **Milestone 4: Modern Architecture.** *Achieved.*
  Upgraded to Llama 3 standards: SwiGLU activations, RMSNorm, Grouped Query Attention (GQA), and Rotary Embeddings (RoPE).
  > *Data:* Lowered perplexity to 67.4, enabling basic grammar acquisition.

### In-Progress
- [~] **Milestone 5: Semantic Coherence.** *In Progress.*
  Transitioning from character-level to Sub-word BPE Tokenization (GPT-4 regex style) and scaling training on the "TinyStories" dataset to fix word-gluing and context shock.

### Future Milestones
- [ ] **Milestone 6: Training Stability.**
  Implement Cosine Learning Rate Decay and Gradient Clipping to support deeper networks.

- [ ] **Milestone 7: Scaling Laws.**
  Expand architecture to 6 Layers / 384 Dim (approx. 15M params) and increase context window to 256+.

- [ ] **Milestone 8: Packaging.**
  Refactor into a pip-installable `src/minigpt` package with formal configuration management (`dataclasses`).

- [ ] **Milestone 9: Speed.**
  Implement custom C++/CUDA kernels for the critical Matrix Multiplication operations (optional `ctypes` bridge).

- [ ] **Milestone 10: Instruction Tuning.**
  Fine-tune on an Alpaca-style dataset to transition the model from "Storyteller" to "Assistant".

---

## Features & Usage

MiniGPT is built for transparency. It includes built-in tools to visualize the model's internal state during inference.

### Quick Start

Running inference is as simple as executing the script. The model will auto-download dependencies and pre-trained weights if strictly necessary (or use random weights for demo).

```bash
# Clone the repository
git clone https://github.com/elsoro/MiniGPT.git
cd MiniGPT

# Install dependencies (only numpy and tqdm)
pip install -r requirements.txt

# Run generation
python scripts/generate.py --prompt "Once upon a time"
```

### Visualization

MiniGPT provides rich visual feedback to understand token generation:

*   **Log Probabilities:** View the confidence distribution for each generated token.
    *(See `logprobs.png`)*
*   **Attention Maps:** Interactive HTML visualization of attention weights across heads.
    *(See `attn_interactive.html`)*
*   **Inference Timeline:** Profiling data showing time spent in Compute vs. Memory operations.
    *(See `inference_timeline.png`)*

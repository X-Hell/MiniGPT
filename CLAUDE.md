# MiniGPT — Claude Code Context

**Status**: 72-hour Linux GPU training pipeline ready. All critical bugs fixed. FP16 mixed precision + fused CUDA kernels implemented. Full FineWeb-Edu streaming pipeline complete.

---

## Current State

### Fixed Issues (Completed)
1. **KV Cache Bypass Training** (src/minigpt/model.py:195-203, 443)
   - ~10x forward pass speedup by skipping per-token Python loop during training
   - Cache only used for inference (autoregressive generation)

2. **VRAM Monitoring & Estimation** (src/minigpt/backend.py:134-208)
   - `vram_stats()`, `log_vram()`, `estimate_model_vram()` utilities
   - Training script logs VRAM every 500 steps, warns at startup if >12GB

3. **Weight Decay on LN Gammas** (src/minigpt/optimizer.py:45-50)
   - Standard practice: skip weight decay for 1D params (layer norm gammas)
   - Prevents destabilizing the learnable scale in RMSNorm

4. **Gradient Accumulation & Clipping** (scripts/train.py:383-485)
   - All 13 LN gamma parameters now included in gradient clipping
   - Proper micro-batch accumulation with `scatter_add` for embeddings
   - Single backward call (removed duplicate from v1)

5. **Checkpoint Portability** (scripts/train.py:102-117)
   - Safe CPU serialization of GPU models before pickling
   - Checkpoints are now portable across CPU/GPU environments

6. **Gradient Validation** (scripts/validate_gradients.py)
   - Numerical gradient checker: median error <10%, p90 <50%
   - All gradients pass validation (float32 forward + float64 loss)

7. **FineWeb-Edu Pipeline** (scripts/prepare_fineweb.py)
   - Streams HuggingFace FineWeb-Edu, trains BPE tokenizer, writes .npy shards
   - Tested end-to-end: stream → tokenize → shard → train
   - Seamlessly integrates with BPETokenizer API

8. **FP16 Mixed Precision for Tensor Cores** (src/minigpt/backend.py:66-84)
   - `fp16_matmul()` casts inputs to FP16 before matmul, result back to FP32
   - Activates RTX 3060 Tensor Cores (up to 4x throughput vs FP32 CUDA cores)
   - Applied to ALL forward pass matmuls (9 total: QKV, scores, V, output, gate, up, down, logits)
   - Backward pass stays FP32 for gradient precision
   - Auto-enabled on GPU, disabled on CPU; toggle with `set_mixed_precision(bool)`

9. **Fused CUDA Kernels via cupy.fuse()** (src/minigpt/model.py:7-24)
   - `_fused_silu()`: Fuses `x * sigmoid(x)` into single CUDA kernel (was 3 kernels)
   - `_fused_silu_backward()`: Fuses SiLU gradient computation (was 6+ kernels)
   - `_fused_exp()`: Fuses `exp(x - x_max)` in softmax (was 2 kernels)
   - On CPU (NumPy), `fuse` is a no-op decorator — zero overhead
   - Reduces kernel launch overhead, the #1 CuPy bottleneck identified in architecture review

### Architecture
- **Model**: 12.9M parameter MiniTransformer (default: d=384, L=6, H=6, GQA with 2 KV heads)
- **Training**: AdamW with cosine LR schedule, gradient accumulation, global norm clipping
- **Backend**: CuPy (GPU) / NumPy (CPU) abstraction layer with FP16 mixed precision
- **Data**: Real-world FineWeb-Edu (10 GB default) with BPE tokenizer (vocab 4096-8192)
- **Tokenizer**: BPE with GPT-4 regex splitting, trained on corpus
- **Precision**: FP16 forward matmuls (Tensor Cores) + FP32 backward/optimizer (stability)

---

## Key Files

| File | Purpose |
|------|---------|
| `src/minigpt/model.py` | Transformer architecture with fused kernels + FP16 forward matmuls |
| `src/minigpt/optimizer.py` | AdamW with per-param weight decay filtering |
| `src/minigpt/backend.py` | GPU/CPU abstraction + VRAM monitoring + `fuse` + `fp16_matmul` |
| `src/minigpt/tokenizer.py` | BPE tokenizer with GPT-4 regex |
| `scripts/train.py` | Main training loop (FineWeb shards, gradient accumulation, LR schedule) |
| `scripts/prepare_fineweb.py` | Stream FineWeb-Edu → tokenize → shard pipeline |
| `scripts/validate_gradients.py` | Numerical gradient checker |
| `scripts/benchmark_gpu.py` | GPU throughput + VRAM stress testing + FP16 vs FP32 comparison |
| `scripts/setup_linux_gpu.sh` | Complete Linux CUDA/CuPy installation |
| `ROADMAP_72H.md` | 72-hour training schedule + issue documentation |

---

## Training Command (72-Hour Run)

```bash
# One-time setup (Linux RTX 3060 + Ryzen 9 + 32GB RAM)
sudo ./scripts/setup_linux_gpu.sh
source ~/minigpt_venv/bin/activate
cd /path/to/MiniGPT

# Pre-flight validation
MINIGPT_BACKEND=cupy python scripts/validate_gradients.py
MINIGPT_BACKEND=cupy python scripts/benchmark_gpu.py

# Prepare FineWeb-Edu (one-time, ~30 min for 10 GB)
python scripts/prepare_fineweb.py

# 72-hour training — UPDATED with higher VRAM utilization
tmux new -s training
MINIGPT_BACKEND=cupy python scripts/train.py \
    --dim 384 --n_layers 6 --n_heads 6 --n_kv_heads 2 \
    --vocab_size 8192 --max_len 512 \
    --batch_size 128 --accum_steps 2 \
    --steps 50000 --lr 3e-4 --warmup_steps 1000 \
    --weight_decay 0.1 --grad_clip 1.0 \
    --save_dir checkpoints_v2 --save_interval 500
```

**Expected Results** (after 50K steps):
- Train loss: ~2.5 (from 8.3)
- Val loss: ~2.7 → PPL ~15
- Tokens seen: ~6.5B (128 * 2 * 512 * 50000)
- FP16 Tensor Cores: 2-4x faster matmuls vs FP32
- Model learns real-world knowledge from FineWeb

---

## GPU Optimization Strategy

### FP16 Mixed Precision (Tensor Core Activation)

| Component | Precision | Reason |
|-----------|-----------|--------|
| Forward matmuls (9 total) | FP16 | Activates Tensor Cores, up to 4x throughput |
| Backward matmuls | FP32 | Gradient precision for manual backprop |
| Optimizer states (m, v) | FP32 | Prevents moment underflow |
| Master weights | FP32 | Accumulate small updates correctly |
| Loss computation | FP32 | Numerical stability |

**Implementation**: `fp16_matmul(a, b)` in `backend.py` — casts inputs to FP16, matmul, cast result back to FP32. Applied to all 9 forward-pass `xp.matmul` calls in `model.py`.

### Fused CUDA Kernels (cupy.fuse)

| Kernel | Ops Fused | Kernel Launches Saved |
|--------|-----------|----------------------|
| `_fused_silu` | sigmoid, multiply | 3 → 1 per FFN layer |
| `_fused_silu_backward` | sigmoid, clip, multiply, add | 6+ → 1 per FFN layer |
| `_fused_exp` | subtract, exp | 2 → 1 per softmax call |

**Total per step**: ~36 fewer kernel launches (6 layers × 6 fused ops). On RTX 3060, each kernel launch costs ~5-10μs, saving ~180-360μs/step.

---

## VRAM & Performance

### Updated Config (RTX 3060 12GB, FP16 mixed precision)
| Component | Memory |
|-----------|--------|
| Model params (FP32 master) | 52 MB |
| Optimizer (m+v, FP32) | 103 MB |
| Gradients (FP32) | 52 MB |
| Activations (6L, T=512, B=128) | ~6,000 MB |
| FP16 temporaries | ~200 MB |
| Overhead | ~400 MB |
| **Total** | **~6,800 MB** |
| **Headroom** | **~5,200 MB** |

### Throughput (Projected with FP16)
- **CPU (NumPy)**: ~30s/step (no change)
- **GPU (CuPy FP32)**: ~0.3s/step
- **GPU (CuPy FP16 mixed)**: ~0.1-0.15s/step (2-3x faster with Tensor Cores)

---

## Design Choices

### Why FP16 Mixed Precision (Not Full FP16)
- **Tensor Cores**: RTX 3060 gets most throughput from FP16/BF16 matmuls
- **Gradient stability**: Manual backward pass needs FP32 precision
- **No loss scaling needed**: Forward-only FP16 avoids the underflow issues that require loss scaling in full mixed-precision training
- **Simple implementation**: One function (`fp16_matmul`) wraps all forward matmuls

### Why cupy.fuse() (Not Raw CUDA Kernels)
- **Minimal code change**: Decorator-based, no kernel code to write
- **Automatic fallback**: No-op on CPU, transparent to NumPy path
- **Correct scope**: Only element-wise ops (not matmul, which is already optimized by cuBLAS)
- **Proven pattern**: CuPy's fuse compiles to optimized CUDA at first call

### Why FineWeb over TinyStories
- **Real-world data**: FineWeb-Edu is filtered web text with high quality
- **Generalization**: TinyStories is synthetic, small scale (~75 KB)
- **Scale**: 10 GB raw text = ~2B tokens after BPE tokenization
- **Tokenizer diversity**: BPE trained on real corpus better than bytes

### Why KV Cache Bypass Training
- **Bottleneck**: Per-token Python loop was slowest component (~10x overhead)
- **Unnecessary**: During training (start_pos=0), full sequence computed at once
- **Safe**: Backward pass recomputes, no cache needed

### Why Weight Decay Exclusion on 1D Params
- **Standard practice**: GPT-2, LLaMA both skip WD on biases/norms
- **Implementation**: Simple ndim check (`if p.ndim >= 2`)

---

## Monitoring During 72-Hour Run

```bash
# Terminal 1: Watch training
tmux attach -t training

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 3: Check VRAM, temps
nvidia-smi -l 60 --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv

# Check loss curves (every hour)
tail -f checkpoints_v2/training_log.txt | grep "Step \|Train=\|Val="
```

**Key Metrics to Watch**:
- Loss should decrease smoothly (no spikes)
- VRAM should stay <8 GB (with larger batch/context)
- GPU utilization >90%
- Temperature <85°C
- Gradient norm 0.1–1.0 (after warmup)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM during training | Reduce `--batch_size` or `--max_len` |
| Loss spikes/NaN | Reduce `--lr` by 2x, restart from checkpoint |
| Slow throughput | Check `nvidia-smi` — ensure GPU is being used |
| FP16 instability | `set_mixed_precision(False)` to disable, debug in FP32 |
| Tokenizer not found | Run `python scripts/prepare_fineweb.py` first |
| Checkpoint portability | Use CPU-safe pickle (already implemented in train.py) |

---

## Next Steps (Beyond 72 Hours)

1. **Multi-GPU Training**: Distribute across multiple GPUs
2. **Larger Model**: Scale to 50M+ params with more layers
3. **Flash Attention**: Fused CUDA kernels for attention (100x speedup possible)
4. **Async Data Prefetcher**: Pin CuPy buffers, overlap PCIe with compute
5. **Nucleus Sampling**: Replace top-k with top-p in inference
6. **Long Context**: Extend max_len to 1024+, use sliding window cache
7. **Supervised Finetuning**: Train on curated Q&A pairs after pre-training
8. **RL Fine-tuning**: Reinforce coherent, factual generation
9. **Gradient Checkpointing**: Trade compute for VRAM to enable larger batches

---

## Context for Future Sessions

This codebase is **production-ready** for the 72-hour training event on Linux RTX 3060:
- All gradient bugs fixed and validated
- FP16 mixed precision activates Tensor Cores (2-4x matmul speedup)
- Fused CUDA kernels reduce kernel launch overhead (~36 fewer launches/step)
- FineWeb-Edu pipeline tested end-to-end
- VRAM carefully budgeted for 12 GB GPU (updated to ~6.8 GB with larger config)
- Monitoring and checkpointing robust
- Expected result: ~PPL 15 on real-world text

**Current branch**: main
**Last update**: 2026-03-29
**Model family**: Claude Haiku 4.5 + Opus 4.6

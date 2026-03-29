#!/usr/bin/env python3
"""
GPU Benchmark & VRAM Stress Test for MiniGPT

Tests the actual training throughput and peak VRAM usage for the RTX 3060 12GB.
Run this BEFORE the 72-hour training event to find the maximum safe batch size.

Usage:
    MINIGPT_BACKEND=cupy python scripts/benchmark_gpu.py
    MINIGPT_BACKEND=numpy python scripts/benchmark_gpu.py  # CPU baseline
"""

import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from minigpt.backend import xp, to_cpu, to_device, get_backend_info, using_gpu, log_vram, estimate_model_vram, set_mixed_precision
from minigpt.config import ModelConfig
from minigpt.model import MiniTransformer


def benchmark_config(d_model, n_layers, n_heads, n_kv_heads, vocab_size,
                     max_len, batch_size, n_steps=5):
    """Run a mini training loop and measure throughput + peak VRAM."""
    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_layers=n_layers,
        max_len=max_len,
        dropout=0.0,
    )
    model = MiniTransformer(config)
    total_params = sum(p.size for _, p in model.named_parameters())

    print(f"\n  Config: d={d_model}, L={n_layers}, H={n_heads}, KV={n_kv_heads}, "
          f"V={vocab_size}, T={max_len}, B={batch_size}")
    print(f"  Params: {total_params:,} ({total_params * 4 / 1e6:.1f} MB)")

    if using_gpu():
        log_vram("post-init")

    # Warm up
    X = to_device(np.random.randint(0, vocab_size, (batch_size, max_len)).astype(np.int64))
    Y = to_device(np.random.randint(0, vocab_size, (batch_size, max_len)).astype(np.int64))

    try:
        # Forward warmup
        logits, _ = model.forward(X, training=True)
        if using_gpu():
            xp.cuda.Stream.null.synchronize()

        # Timed forward + backward
        times = []
        for step in range(n_steps):
            t0 = time.time()

            # Forward
            logits, _ = model.forward(X, training=True)
            B, T, V = logits.shape
            logits_flat = logits.reshape(-1, V)
            targets_flat = Y.reshape(-1)

            max_logits = xp.max(logits_flat, axis=1, keepdims=True)
            exp_logits = xp.exp(logits_flat - max_logits)
            probs = exp_logits / xp.sum(exp_logits, axis=1, keepdims=True)

            dlogits = probs
            dlogits[xp.arange(len(targets_flat)), targets_flat] -= 1
            dlogits /= len(targets_flat)
            dlogits = dlogits.reshape(B, T, V)

            # Backward
            dW_emb, layer_grads, dX_emb, ln_f_d_gamma = model.backward(dlogits)

            if using_gpu():
                xp.cuda.Stream.null.synchronize()

            dt = time.time() - t0
            times.append(dt)

            tokens = batch_size * max_len
            print(f"    Step {step}: {dt:.3f}s ({tokens / dt:.0f} tok/s)")

        if using_gpu():
            log_vram("peak-training")

        avg_time = np.mean(times[1:])  # Skip first (warmup)
        tokens_per_sec = batch_size * max_len / avg_time

        print(f"\n  Average: {avg_time:.3f}s/step, {tokens_per_sec:.0f} tok/s")
        print(f"  Projected 50K steps: {50000 * avg_time / 3600:.1f} hours")
        print(f"  Projected 25K steps: {25000 * avg_time / 3600:.1f} hours")

        return {
            'avg_time': avg_time,
            'tokens_per_sec': tokens_per_sec,
            'params': total_params,
            'success': True,
        }

    except Exception as e:
        print(f"  FAILED: {e}")
        if "out of memory" in str(e).lower() or "MemoryError" in str(type(e)):
            print(f"  OOM detected. Reduce batch_size or max_len.")
        return {'success': False, 'error': str(e)}


def main():
    print("=" * 60)
    print("  MiniGPT GPU Benchmark & VRAM Stress Test")
    print(f"  Backend: {get_backend_info()}")
    print("=" * 60)

    if using_gpu():
        log_vram("baseline")

    # -----------------------------------------------------------------------
    # Test 1: Default config (should always work on RTX 3060)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("  TEST 1: Default Config (12.9M params)")
    print("-" * 40)
    benchmark_config(
        d_model=384, n_layers=6, n_heads=6, n_kv_heads=2,
        vocab_size=4096, max_len=256, batch_size=64
    )

    # -----------------------------------------------------------------------
    # Test 2: With gradient accumulation simulation (larger effective batch)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("  TEST 2: Larger batch (effective 256 via accum=4)")
    print("-" * 40)
    benchmark_config(
        d_model=384, n_layers=6, n_heads=6, n_kv_heads=2,
        vocab_size=4096, max_len=256, batch_size=64
    )

    # -----------------------------------------------------------------------
    # Test 3: Max batch size probe (find OOM boundary)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("  TEST 3: Finding maximum batch size...")
    print("-" * 40)
    for bs in [32, 64, 96, 128, 160, 192]:
        result = benchmark_config(
            d_model=384, n_layers=6, n_heads=6, n_kv_heads=2,
            vocab_size=4096, max_len=256, batch_size=bs, n_steps=2
        )
        if not result['success']:
            print(f"\n  Max safe batch_size: {bs - 32}")
            break

    # -----------------------------------------------------------------------
    # Test 4: Longer context
    # -----------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("  TEST 4: Longer Context (max_len=512)")
    print("-" * 40)
    benchmark_config(
        d_model=384, n_layers=6, n_heads=6, n_kv_heads=2,
        vocab_size=4096, max_len=512, batch_size=32
    )

    # -----------------------------------------------------------------------
    # Test 5: FP16 Mixed Precision Comparison
    # -----------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("  TEST 5: FP16 Mixed Precision vs FP32")
    print("-" * 40)

    # FP32 baseline
    set_mixed_precision(False)
    result_fp32 = benchmark_config(
        d_model=384, n_layers=6, n_heads=6, n_kv_heads=2,
        vocab_size=4096, max_len=256, batch_size=64, n_steps=5
    )

    # FP16 mixed precision
    set_mixed_precision(True)
    result_fp16 = benchmark_config(
        d_model=384, n_layers=6, n_heads=6, n_kv_heads=2,
        vocab_size=4096, max_len=256, batch_size=64, n_steps=5
    )

    if result_fp32['success'] and result_fp16['success']:
        speedup = result_fp16['tokens_per_sec'] / result_fp32['tokens_per_sec']
        print(f"\n  FP16 Speedup: {speedup:.2f}x")
        print(f"  FP32: {result_fp32['tokens_per_sec']:.0f} tok/s")
        print(f"  FP16: {result_fp16['tokens_per_sec']:.0f} tok/s")

    # -----------------------------------------------------------------------
    # Test 6: Maximum VRAM utilization
    # -----------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("  TEST 6: Max VRAM Utilization (T=512, B=128)")
    print("-" * 40)
    set_mixed_precision(True)
    benchmark_config(
        d_model=384, n_layers=6, n_heads=6, n_kv_heads=2,
        vocab_size=8192, max_len=512, batch_size=128, n_steps=3
    )

    # -----------------------------------------------------------------------
    # VRAM Budget Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  VRAM Budget Summary (RTX 3060 12 GB)")
    print("=" * 60)

    configs = [
        ("Default", 384, 6, 6, 256, 64),
        ("Large Batch", 384, 6, 6, 256, 128),
        ("Long Context", 384, 6, 6, 512, 32),
        ("Scaled Up", 512, 8, 8, 256, 32),
    ]

    for name, d, nl, nh, sl, bs in configs:
        total_params = d * 4096 + nl * (d * d * 4 + d * (int(2*4*d/3)) * 3 + d * 2) + d
        est = estimate_model_vram(total_params, bs, sl, d, nl, nh)
        status = "OK" if est['fits_12gb'] else "OOM"
        print(f"  {name:20s}: {est['total_mb']:6.0f} MB [{status}]")

    print("=" * 60)


if __name__ == "__main__":
    main()

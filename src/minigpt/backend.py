"""
GPU/CPU Backend Abstraction Layer for MiniGPT.

Provides a unified array library (`xp`) that is either CuPy (GPU) or NumPy (CPU),
controlled by the MINIGPT_BACKEND environment variable.

Usage:
    from minigpt.backend import xp, scatter_add, to_cpu

    # All array operations use xp instead of np:
    a = xp.zeros((3, 4), dtype=xp.float32)

    # Transfer data to GPU:
    gpu_array = xp.asarray(cpu_numpy_array)

    # Transfer data back to CPU for logging/saving:
    cpu_array = to_cpu(gpu_array)

Environment:
    MINIGPT_BACKEND="cupy"  -> Use CuPy (CUDA GPU acceleration)
    MINIGPT_BACKEND="numpy" -> Use NumPy (CPU fallback)
    Default: Tries CuPy first, falls back to NumPy if unavailable.
"""

import os
import warnings
from typing import Any, Dict, Optional

_BACKEND = os.environ.get("MINIGPT_BACKEND", "auto").lower()

if _BACKEND == "auto":
    try:
        import cupy as xp
        import cupyx
        _USING_GPU = True
    except ImportError:
        import numpy as xp
        _USING_GPU = False
elif _BACKEND == "cupy":
    import cupy as xp
    import cupyx
    _USING_GPU = True
elif _BACKEND == "numpy":
    import numpy as xp
    _USING_GPU = False
else:
    warnings.warn(f"Unknown MINIGPT_BACKEND='{_BACKEND}', falling back to numpy.")
    import numpy as xp
    _USING_GPU = False


# ---------------------------------------------------------------------------
# Fused kernel decorator (cupy.fuse on GPU, no-op on CPU)
# ---------------------------------------------------------------------------
if _USING_GPU:
    fuse = xp.fuse
else:
    def fuse(func=None, *, kernel_name=None):
        """No-op fuse decorator for NumPy backend."""
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper


# ---------------------------------------------------------------------------
# FP16 Mixed Precision matmul for Tensor Core acceleration
# ---------------------------------------------------------------------------
# Default OFF (pure FP32). On this ~30M model the matmuls are too small for
# Ampere Tensor Cores to beat the Python/orchestration overhead, so FP16 gave
# ~no speedup while costing ~1GB extra VRAM (FP16 temporaries on top of the
# FP32 master weights). Re-enable explicitly via set_mixed_precision(True).
_MIXED_PRECISION = False


def set_mixed_precision(enabled: bool) -> None:
    """Enable or disable FP16 mixed precision matmul."""
    global _MIXED_PRECISION
    _MIXED_PRECISION = enabled


def fp16_matmul(a: Any, b: Any) -> Any:
    """Matrix multiply using FP16 for Tensor Core acceleration.

    On the RTX 3060 (Ampere, SM86) FP16 GEMMs dispatch to the hardware Tensor
    Cores, roughly doubling matmul throughput vs. FP32. Inputs are cast to FP16,
    the result is accumulated and returned in FP32 (so the hand-written FP32
    backward stays numerically stable — see README "FP16 fwd / FP32 bwd").
    No-op precision change on CPU (NumPy has no Tensor Cores)."""
    if _MIXED_PRECISION:
        return xp.matmul(a.astype(xp.float16), b.astype(xp.float16)).astype(xp.float32)
    return xp.matmul(a, b)


def scatter_add(target: Any, indices: Any, source: Any) -> None:
    """
    Atomic scatter-add: target[indices] += source.
    Uses cupyx.scatter_add on GPU, np.add.at on CPU.
    """
    if _USING_GPU:
        cupyx.scatter_add(target, indices, source)
    else:
        xp.add.at(target, indices, source)


def to_cpu(array: Any) -> Any:
    """
    Transfer array to CPU (NumPy). No-op if already NumPy.
    Useful for logging, saving checkpoints, and plotting.
    """
    if _USING_GPU:
        return xp.asnumpy(array)
    return array


def to_device(array: Any) -> Any:
    """
    Transfer a NumPy array to the active device (GPU if available).
    No-op if already on the correct device or if using NumPy backend.
    """
    if _USING_GPU:
        return xp.asarray(array)
    return array


def get_backend_info() -> str:
    """Return a human-readable string describing the active backend."""
    if _USING_GPU:
        device = xp.cuda.runtime.getDeviceProperties(0)
        name = device['name'].decode() if isinstance(device['name'], bytes) else device['name']
        mem_gb = device['totalGlobalMem'] / (1024**3)
        return f"CuPy (GPU: {name}, {mem_gb:.1f} GB VRAM)"
    else:
        return "NumPy (CPU)"


def get_device_info() -> str:
    """Backward-compatible alias used by older diagnostics/scripts."""
    return get_backend_info()


def using_gpu() -> bool:
    """Return True if CuPy/GPU backend is active."""
    return _USING_GPU


def vram_stats() -> Optional[Dict[str, float]]:
    """
    Return VRAM usage stats for the active GPU.
    Returns dict with keys: total_mb, used_mb, free_mb, utilization_pct.
    Returns None if not using GPU.
    """
    if not _USING_GPU:
        return None
    pool = xp.cuda.runtime.memGetInfo()
    free_bytes, total_bytes = pool
    used_bytes = total_bytes - free_bytes
    return {
        'total_mb': total_bytes / (1024 ** 2),
        'used_mb': used_bytes / (1024 ** 2),
        'free_mb': free_bytes / (1024 ** 2),
        'utilization_pct': 100.0 * used_bytes / total_bytes
    }


def log_vram(label: str = "") -> None:
    """Print current VRAM usage. No-op on CPU."""
    stats = vram_stats()
    if stats is None:
        return
    prefix = f"[VRAM {label}] " if label else "[VRAM] "
    print(f"{prefix}{stats['used_mb']:.0f} / {stats['total_mb']:.0f} MB "
          f"({stats['utilization_pct']:.1f}% used, {stats['free_mb']:.0f} MB free)")


def estimate_model_vram(n_params: int, batch_size: int, seq_len: int,
                        d_model: int, n_layers: int, n_heads: int,
                        d_ff: Optional[int] = None,
                        vocab_size: Optional[int] = None,
                        mixed_precision: bool = False,
                        recompute_factor: float = 1.7) -> Dict[str, float]:
    """
    Estimate peak VRAM usage for training the modern (SwiGLU) model.
    Returns dict with component-wise breakdown in MB.

    Calibrated against the measured Run-1 peak of 11,294 MB at batch_size=32
    (FP32) on the RTX 3060 (I-7). The original estimate (5,453 MB) under-predicted
    by ~2.07x because it omitted two large FP32 tensors:

      * logits + dlogits, each B*T*vocab*4 bytes (~1.07 GB *each* at V=16384,
        B=32, T=512) -- only added when `vocab_size` is provided.
      * the hand-written backward recomputes attention/SwiGLU forwards in FP32
        (model.py), inflating peak activations by ~`recompute_factor` (1.7x,
        empirically matched to the 11,294 MB anchor).
    """
    bytes_per_param = 2 if mixed_precision else 4  # FP16 vs FP32

    # Model parameters
    params_mb = n_params * bytes_per_param / (1024 ** 2)

    # Optimizer states (always FP32): m + v = 2x params
    optim_mb = n_params * 4 * 2 / (1024 ** 2)

    # Gradients (FP32)
    grads_mb = n_params * 4 / (1024 ** 2)

    # Activations per layer (approximate):
    # Attention scores: B * H * T * T * 4 bytes
    attn_scores_mb = (batch_size * n_heads * seq_len * seq_len * 4) / (1024 ** 2)
    # SwiGLU intermediates: gate, up, and (silu(gate)*up) are each (B, T, d_ff),
    # so ~3 * B * T * d_ff * 4 bytes. Default d_ff = round(8/3 * d) to a mult of 64.
    if d_ff is None:
        d_ff = int(round((8 * d_model / 3) / 64)) * 64
    ffn_mb = (batch_size * seq_len * d_ff * 3 * 4) / (1024 ** 2)
    # Layer input/output: B * T * D * 4 bytes
    layer_io_mb = (batch_size * seq_len * d_model * 4 * 2) / (1024 ** 2)

    activations_per_layer = attn_scores_mb + ffn_mb + layer_io_mb
    # FP32 backward recompute inflates the activation peak (I-7).
    total_activations = activations_per_layer * n_layers * recompute_factor

    # Output logits + their gradient: each B * T * vocab * 4 bytes (FP32).
    logits_mb = 0.0
    if vocab_size:
        logits_mb = 2.0 * (batch_size * seq_len * vocab_size * 4) / (1024 ** 2)

    # Working memory overhead (~20%)
    overhead_mb = (params_mb + optim_mb + grads_mb + total_activations + logits_mb) * 0.20

    total_mb = params_mb + optim_mb + grads_mb + total_activations + logits_mb + overhead_mb

    return {
        'params_mb': params_mb,
        'optimizer_mb': optim_mb,
        'gradients_mb': grads_mb,
        'activations_mb': total_activations,
        'activations_per_layer_mb': activations_per_layer,
        'logits_mb': logits_mb,
        'overhead_mb': overhead_mb,
        'total_mb': total_mb,
        'fits_12gb': total_mb < 11500,  # Leave 500 MB headroom
    }

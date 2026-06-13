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
_MIXED_PRECISION = _USING_GPU  # Auto-enable on GPU


def set_mixed_precision(enabled: bool) -> None:
    """Enable or disable FP16 mixed precision matmul."""
    global _MIXED_PRECISION
    _MIXED_PRECISION = enabled


def fp16_matmul(a: Any, b: Any) -> Any:
    """Matrix multiply using FP16 for Tensor Core acceleration.
    Inputs are cast to FP16, result is cast back to FP32.
    No-op precision change on CPU (NumPy doesn't have Tensor Cores)."""
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
                        mixed_precision: bool = False) -> Dict[str, float]:
    """
    Estimate peak VRAM usage for training.
    Returns dict with component-wise breakdown in MB.
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
    # FFN intermediates: B * T * d_ff * 4 bytes (GPT-1 uses d_ff = 4 * d_model)
    d_ff = 4 * d_model
    ffn_mb = (batch_size * seq_len * d_ff * 4) / (1024 ** 2)
    # Layer input/output: B * T * D * 4 bytes
    layer_io_mb = (batch_size * seq_len * d_model * 4 * 2) / (1024 ** 2)

    activations_per_layer = attn_scores_mb + ffn_mb + layer_io_mb
    total_activations = activations_per_layer * n_layers

    # Working memory overhead (~20%)
    overhead_mb = (params_mb + optim_mb + grads_mb + total_activations) * 0.20

    total_mb = params_mb + optim_mb + grads_mb + total_activations + overhead_mb

    return {
        'params_mb': params_mb,
        'optimizer_mb': optim_mb,
        'gradients_mb': grads_mb,
        'activations_mb': total_activations,
        'activations_per_layer_mb': activations_per_layer,
        'overhead_mb': overhead_mb,
        'total_mb': total_mb,
        'fits_12gb': total_mb < 11500,  # Leave 500 MB headroom
    }

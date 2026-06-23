from dataclasses import dataclass
from math import ceil
from typing import Dict, Optional, Tuple


@dataclass
class TokenizerConfig:
    """Modern BPE tokenizer, 16K vocab + 3 special tokens."""
    vocab_size: int = 16384
    min_frequency: int = 2
    # GPT-4 split pattern
    pattern: str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    special_tokens: Tuple[str, ...] = ("<pad>", "<eos>", "<unk>")


@dataclass
class ModelConfig:
    """Modern Llama-style target: ~30M parameters (d=512, L=8, H=8).

    Architecture: RoPE positions, SwiGLU FFN, RMSNorm (pre-norm), no biases,
    tied input/output embeddings. See src/minigpt/model.py.
    """
    vocab_size: int = 16384
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024                      # SwiGLU hidden width (gate/up/down)
    max_len: int = 512
    dropout: float = 0.0                  # modern small LMs train dropout-free

    # Rotary positional embedding base frequency (now USED — applied to Q/K).
    rope_theta: float = 10000.0

    # Full multi-head attention: n_kv_heads always == n_heads (no GQA here).
    n_kv_heads: int = 8

    def __post_init__(self):
        if self.d_ff == 0:
            # Llama SwiGLU convention: ~8/3 * d, rounded to a multiple of 64.
            self.d_ff = int(round((8 * self.d_model / 3) / 64)) * 64
        if self.n_kv_heads != self.n_heads:
            self.n_kv_heads = self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert (self.d_model // self.n_heads) % 2 == 0, "d_head must be even for RoPE"


@dataclass
class TrainConfig:
    """Training: Adam + L2, linear warmup + cosine decay.

    Defaults tuned for the ~30M model on an RTX 3060 12GB (see
    scripts/calculate_vram_budget.py): micro-batch 32 x accum 4 -> eff batch 128.
    """
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    batch_size: int = 32                  # micro-batch (30M model, T=512, FP32)
    accum_steps: int = 4                  # effective batch = 128
    max_steps: int = 36000                # 36K * 128 * 512 ≈ 2.36B tokens (~80 tok/param); ~42-45h on a 3060
    warmup_steps: int = 2000
    weight_decay: float = 0.1             # modern decoupled-ish value (coupled L2 here)
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95                   # modern LM default
    eps: float = 1e-8

    # Checkpointing & Logging
    eval_interval: int = 500
    log_interval: int = 10
    save_interval: int = 1000
    save_dir: str = "checkpoints_modern"

    # Dataloader
    seq_len: int = 512


@dataclass
class JetsonOrinNanoMemoryConfig:
    """Memory budget assumptions for Jetson Orin Nano (8GB unified memory)."""

    total_system_memory_gb: float = 8.0
    os_reserved_memory_gb: float = 1.5
    runtime_safety_margin_gb: float = 0.5
    target_effective_batch: int = 128
    seq_len: int = 512


def estimate_training_memory_mb(
    n_params: int,
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    vocab_size: Optional[int] = None,
    mixed_precision: bool = True,
    recompute_factor: float = 1.7,
) -> float:
    """Estimate peak training memory (MB), matching backend.estimate_model_vram.

    SwiGLU activations dominate: gate, up, and (silu(gate)*up) are each
    (B, T, d_ff), so the FFN intermediate footprint is ~3 * B * T * d_ff.

    Calibrated to the measured Run-1 peak (11,294 MB @ batch 32) by adding the
    two terms the original estimate omitted (I-7): the FP32 logits + dlogits
    (each B*T*vocab*4; added when `vocab_size` is given) and the FP32 backward
    recompute (`recompute_factor` on activations).
    """
    bytes_per_param = 2 if mixed_precision else 4
    params_mb = n_params * bytes_per_param / (1024 ** 2)
    optimizer_mb = n_params * 4 * 2 / (1024 ** 2)
    gradients_mb = n_params * 4 / (1024 ** 2)

    attn_scores_mb = (batch_size * n_heads * seq_len * seq_len * 4) / (1024 ** 2)
    ffn_mb = (batch_size * seq_len * d_ff * 3 * 4) / (1024 ** 2)   # SwiGLU: gate+up+prod
    layer_io_mb = (batch_size * seq_len * d_model * 4 * 2) / (1024 ** 2)
    activations_mb = (attn_scores_mb + ffn_mb + layer_io_mb) * n_layers * recompute_factor

    logits_mb = 0.0
    if vocab_size:
        logits_mb = 2.0 * (batch_size * seq_len * vocab_size * 4) / (1024 ** 2)

    base_mb = params_mb + optimizer_mb + gradients_mb + activations_mb + logits_mb
    return base_mb * 1.20


def solve_batch_plan(
    *,
    n_params: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    seq_len: int = 512,
    vocab_size: Optional[int] = None,
    vram_budget_mb: float = 11_500.0,
    target_effective_batch: int = 128,
    max_micro_batch: int = 256,
    mixed_precision: bool = True,
) -> Tuple[int, int, Dict[str, float]]:
    """Return (micro_batch, grad_accum_steps, diagnostics) maxing a VRAM budget.

    Picks the largest micro-batch whose estimated peak fits vram_budget_mb, then
    sets grad_accum to reach at least target_effective_batch.
    """
    micro = 1
    for b in range(1, max_micro_batch + 1):
        peak = estimate_training_memory_mb(
            n_params=n_params, batch_size=b, seq_len=seq_len,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_ff, vocab_size=vocab_size, mixed_precision=mixed_precision,
        )
        if peak > vram_budget_mb:
            break
        micro = b

    grad_accum = max(1, ceil(target_effective_batch / micro))
    peak = estimate_training_memory_mb(
        n_params=n_params, batch_size=micro, seq_len=seq_len,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        d_ff=d_ff, vocab_size=vocab_size, mixed_precision=mixed_precision,
    )
    diagnostics = {
        "vram_budget_mb": vram_budget_mb,
        "estimated_peak_mb": peak,
        "headroom_mb": vram_budget_mb - peak,
        "effective_batch": float(micro * grad_accum),
    }
    return micro, grad_accum, diagnostics


def calculate_jetson_batch_plan(
    *,
    n_params: int = 29_368_832,
    d_model: int = 512,
    n_layers: int = 8,
    n_heads: int = 8,
    d_ff: int = 1024,
    seq_len: int = 512,
    memory: Optional[JetsonOrinNanoMemoryConfig] = None,
) -> Tuple[int, int, Dict[str, float]]:
    """Return (max_batch_size, grad_accum_steps, diagnostics) for Jetson 8GB.

    Defaults correspond to the modern ~30M layout in this codebase.
    """
    mem_cfg = memory or JetsonOrinNanoMemoryConfig()

    budget_mb = (
        mem_cfg.total_system_memory_gb
        - mem_cfg.os_reserved_memory_gb
        - mem_cfg.runtime_safety_margin_gb
    ) * 1024.0
    if budget_mb <= 0.0:
        raise ValueError("Invalid Jetson memory budget: non-positive usable memory.")

    return solve_batch_plan(
        n_params=n_params, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        d_ff=d_ff, seq_len=seq_len, vocab_size=16384, vram_budget_mb=budget_mb,
        target_effective_batch=mem_cfg.target_effective_batch,
    )


@dataclass
class Config:
    """Legacy flat config kept for script compatibility."""

    # Model
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    max_len: int = 512
    vocab_size: int = 16384
    dropout: float = 0.0
    rope_theta: float = 10000.0

    # Optimizer / schedule
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    total_steps: int = 36000
    log_interval: int = 10
    checkpoint_interval: int = 1000
    eval_interval: int = 500
    fp16: bool = False

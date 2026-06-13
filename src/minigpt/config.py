from dataclasses import dataclass
from math import ceil
from typing import Dict, Optional, Tuple


@dataclass
class TokenizerConfig:
    """GPT-1 BPE tokenizer, 40K vocab + 3 special tokens."""
    vocab_size: int = 40000
    min_frequency: int = 2
    # GPT-4 split pattern
    pattern: str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    special_tokens: Tuple[str, ...] = ("<pad>", "<eos>", "<unk>")


@dataclass
class ModelConfig:
    """GPT-1 target: 117M parameters (d=768, L=12, H=12)."""
    vocab_size: int = 40000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 0                         # auto = 4 * d_model
    max_len: int = 512
    dropout: float = 0.1

    # --- Legacy fields kept at no-op defaults for back-compat with old
    #     checkpoints and scripts/train.py CLI flags. Do NOT use them in new
    #     code; model.py ignores rope_theta entirely and forces n_kv_heads =
    #     n_heads.
    n_kv_heads: int = 12                  # always == n_heads for GPT-1
    rope_theta: float = 0.0               # unused (no RoPE in GPT-1)

    def __post_init__(self):
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model          # GPT-1 ratio
        if self.n_kv_heads != self.n_heads:
            # GPT-1 has full MHA; silently normalize
            self.n_kv_heads = self.n_heads


@dataclass
class TrainConfig:
    """GPT-1 training: Adam + L2, linear warmup + cosine decay."""
    learning_rate: float = 2.5e-4
    min_lr: float = 1e-5
    batch_size: int = 8                   # micro-batch for 117M at T=512 FP16
    accum_steps: int = 8                  # effective batch = 64
    max_steps: int = 100000
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98                   # GPT-1 paper value
    eps: float = 1e-8

    # Checkpointing & Logging
    eval_interval: int = 500
    log_interval: int = 10
    save_interval: int = 1000
    save_dir: str = "checkpoints_gpt1"

    # Dataloader
    seq_len: int = 512


@dataclass
class JetsonOrinNanoMemoryConfig:
    """Memory budget assumptions for Jetson Orin Nano (8GB unified memory)."""

    total_system_memory_gb: float = 8.0
    os_reserved_memory_gb: float = 1.5
    runtime_safety_margin_gb: float = 0.5
    target_effective_batch: int = 64
    seq_len: int = 512


def estimate_training_memory_mb(
    n_params: int,
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    mixed_precision: bool = True,
) -> float:
    """Estimate peak training memory (MB) using the same model as backend.py."""
    bytes_per_param = 2 if mixed_precision else 4
    params_mb = n_params * bytes_per_param / (1024 ** 2)
    optimizer_mb = n_params * 4 * 2 / (1024 ** 2)
    gradients_mb = n_params * 4 / (1024 ** 2)

    attn_scores_mb = (batch_size * n_heads * seq_len * seq_len * 4) / (1024 ** 2)
    d_ff = 4 * d_model
    ffn_mb = (batch_size * seq_len * d_ff * 4) / (1024 ** 2)
    layer_io_mb = (batch_size * seq_len * d_model * 4 * 2) / (1024 ** 2)
    activations_mb = (attn_scores_mb + ffn_mb + layer_io_mb) * n_layers

    base_mb = params_mb + optimizer_mb + gradients_mb + activations_mb
    return base_mb * 1.20


def calculate_jetson_batch_plan(
    *,
    n_params: int = 116_167_680,
    d_model: int = 768,
    n_layers: int = 12,
    n_heads: int = 12,
    seq_len: int = 512,
    memory: Optional[JetsonOrinNanoMemoryConfig] = None,
) -> Tuple[int, int, Dict[str, float]]:
    """Return (max_batch_size, grad_accum_steps, diagnostics) for Jetson 8GB.

    Budget:
      available_for_training = total - os_reserved - runtime_safety_margin

    Defaults correspond to the GPT-1 117M layout in this codebase.
    """
    mem_cfg = memory or JetsonOrinNanoMemoryConfig()

    budget_mb = (
        mem_cfg.total_system_memory_gb
        - mem_cfg.os_reserved_memory_gb
        - mem_cfg.runtime_safety_margin_gb
    ) * 1024.0
    if budget_mb <= 0.0:
        raise ValueError("Invalid Jetson memory budget: non-positive usable memory.")

    max_batch_size = 1
    for batch_size in range(1, 129):
        total_mb = estimate_training_memory_mb(
            n_params=n_params,
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            mixed_precision=True,
        )
        if total_mb > budget_mb:
            break
        max_batch_size = batch_size

    grad_accum_steps = max(1, ceil(mem_cfg.target_effective_batch / max_batch_size))
    estimated_peak_mb = estimate_training_memory_mb(
        n_params=n_params,
        batch_size=max_batch_size,
        seq_len=seq_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        mixed_precision=True,
    )

    diagnostics = {
        "budget_mb": budget_mb,
        "estimated_peak_mb": estimated_peak_mb,
        "headroom_mb": budget_mb - estimated_peak_mb,
        "effective_batch": float(max_batch_size * grad_accum_steps),
    }
    return max_batch_size, grad_accum_steps, diagnostics


@dataclass
class Config:
    """Legacy flat config kept for script compatibility.

    Some older scripts use `Config()` and expect a JSON-serializable mapping via
    `vars(Config())`. This wrapper mirrors the main defaults from ModelConfig
    and TrainConfig in a single dataclass.
    """

    # Model
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    max_len: int = 512
    vocab_size: int = 40000
    dropout: float = 0.1

    # Optimizer / schedule
    lr: float = 2.5e-4
    min_lr: float = 1e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 8
    total_steps: int = 100000
    log_interval: int = 10
    checkpoint_interval: int = 1000
    eval_interval: int = 500
    fp16: bool = True

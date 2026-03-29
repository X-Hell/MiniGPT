from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class TokenizerConfig:
    """Configuration for the BPETokenizer."""
    vocab_size: int = 4096
    min_frequency: int = 3
    # GPT-4 split pattern
    pattern: str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

@dataclass
class ModelConfig:
    """Configuration for the MiniTransformer model."""
    vocab_size: int = 4096
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2  # GQA: Grouped Query Attention
    d_ff: int = 0  # 0 means auto-calculated (SwiGLU: 2/3 * 4 * d_model)
    max_len: int = 512
    dropout: float = 0.0  # Large dataset vs small model -- regularization unnecessary
    rope_theta: float = 500000.0  # RoPE base period

    def __post_init__(self):
        # Auto-calculate d_ff for SwiGLU if not set
        if self.d_ff == 0:
            # Standard SwiGLU hidden dim: 2/3 * 4 * d_model, rounded to multiple of 256 for efficiency
            hidden = int(2 * 4 * self.d_model / 3)
            self.d_ff = 256 * ((hidden + 255) // 256)

@dataclass
class TrainConfig:
    """Configuration for the training loop."""
    # Optimization
    learning_rate: float = 6e-4     # Chinchilla-scaled for 12.6M params
    min_lr: float = 6e-5            # 10% of peak
    batch_size: int = 64            # Micro-batch size
    accum_steps: int = 2            # Gradient accumulation -> effective batch = 128
    max_steps: int = 50000
    warmup_steps: int = 500         # ~1% of total steps
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95             # More stable for LMs than 0.999

    # Checkpointing & Logging
    eval_interval: int = 250
    log_interval: int = 10
    save_interval: int = 1000
    save_dir: str = "checkpoints_v2"

    # Dataloader
    seq_len: int = 256  # Phase 1: 256, Phase 2: 512

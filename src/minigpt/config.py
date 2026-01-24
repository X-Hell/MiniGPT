from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class TokenizerConfig:
    """Configuration for the BPETokenizer."""
    vocab_size: int = 4096
    min_frequency: int = 2
    # GPT-4 split pattern
    pattern: str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

@dataclass
class ModelConfig:
    """Configuration for the MiniTransformer model."""
    vocab_size: int = 4096
    d_model: int = 384
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2  # GQA: Grouped Query Attention
    d_ff: int = 0  # 0 means auto-calculated (SwiGLU: 2/3 * 4 * d_model)
    max_len: int = 512
    dropout: float = 0.1
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
    learning_rate: float = 3e-4
    batch_size: int = 32
    max_steps: int = 1000
    weight_decay: float = 1e-1
    grad_clip: float = 1.0
    
    # Checkpointing & Logging
    eval_interval: int = 100
    eval_steps: int = 20
    log_interval: int = 10
    save_dir: str = "checkpoints"
    
    # Device (Mental model, since we are NumPy only)
    device: str = "cpu" 
    
    # Dataloader
    seq_len: int = 128  # Context window implementation during training

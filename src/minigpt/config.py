from dataclasses import dataclass, field
from typing import Optional, Tuple


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

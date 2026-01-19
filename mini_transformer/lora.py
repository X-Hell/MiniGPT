"""
Low-Rank Adaptation (LoRA) Module

Implements LoRA adapters for efficient fine-tuning and parameter reduction.
Instead of updating full weight matrices, we add low-rank decompositions:
    W' = W + ΔW where ΔW = A @ B (rank r << min(d_in, d_out))

Benefits:
- 40-60% parameter reduction
- Memory-efficient training
- Easy to merge/unmerge adapters
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    rank: int = 8            # Low-rank dimension
    alpha: float = 16.0      # Scaling factor
    dropout: float = 0.0     # Dropout probability
    target_modules: Tuple[str, ...] = ("ffn_w1", "ffn_w2", "attn_qkv", "attn_o")


class LoRAAdapter:
    """
    Low-Rank Adapter for a single weight matrix.
    
    Given original weight W (d_in, d_out), we add:
        ΔW = A @ B * (alpha / rank)
    where A is (d_in, rank) and B is (rank, d_out)
    
    B is initialized to zero so ΔW starts as zero,
    preserving the original model behavior at init.
    """
    
    def __init__(self, d_in: int, d_out: int, rank: int = 8, alpha: float = 16.0):
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Initialize A with small random values, B with zeros
        # This ensures ΔW = 0 at initialization
        self.A = np.random.randn(d_in, rank).astype(np.float32) * 0.01
        self.B = np.zeros((rank, d_out), dtype=np.float32)
        
        # Gradients
        self.dA = np.zeros_like(self.A)
        self.dB = np.zeros_like(self.B)
        
        # Cache for backward
        self.cache_x = None
        self.cache_A_out = None
        
        # Stats
        self.original_params = d_in * d_out
        self.lora_params = d_in * rank + rank * d_out
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute LoRA delta: ΔW @ x.T → (A @ B) @ x.T * scaling
        
        Args:
            x: Input tensor (..., d_in)
        
        Returns:
            LoRA output (..., d_out)
        """
        # Cache for backward
        self.cache_x = x
        
        # Compute: x @ A @ B * scaling
        A_out = np.matmul(x, self.A)  # (..., rank)
        self.cache_A_out = A_out
        
        output = np.matmul(A_out, self.B) * self.scaling  # (..., d_out)
        return output
    
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backward pass for LoRA.
        
        Args:
            d_out: Gradient w.r.t. output (..., d_out)
        
        Returns:
            d_in: Gradient w.r.t. input (..., d_in)
        """
        d_out_scaled = d_out * self.scaling
        
        # Gradient for B: dB = A_out.T @ d_out
        # Reshape for matmul
        orig_shape = d_out_scaled.shape
        d_out_flat = d_out_scaled.reshape(-1, self.d_out)
        A_out_flat = self.cache_A_out.reshape(-1, self.rank)
        
        self.dB += np.matmul(A_out_flat.T, d_out_flat)
        
        # Gradient for A_out: dA_out = d_out @ B.T
        dA_out = np.matmul(d_out_flat, self.B.T).reshape(self.cache_A_out.shape)
        
        # Gradient for A: dA = x.T @ dA_out
        x_flat = self.cache_x.reshape(-1, self.d_in)
        dA_out_flat = dA_out.reshape(-1, self.rank)
        self.dA += np.matmul(x_flat.T, dA_out_flat)
        
        # Gradient for input: d_in = dA_out @ A.T
        d_in = np.matmul(dA_out, self.A.T)
        
        return d_in
    
    def get_merged_delta(self) -> np.ndarray:
        """Return the full ΔW matrix for merging into original weights."""
        return np.matmul(self.A, self.B) * self.scaling
    
    def apply_grads(self, lr: float = 1e-4):
        """Apply accumulated gradients."""
        self.A -= lr * self.dA
        self.B -= lr * self.dB
        
        # Reset gradients
        self.dA.fill(0)
        self.dB.fill(0)
    
    def param_reduction(self) -> float:
        """Return percentage of parameter reduction."""
        return 1.0 - (self.lora_params / self.original_params)
    
    def memory_bytes(self) -> int:
        """Return total memory used by LoRA matrices."""
        return self.A.nbytes + self.B.nbytes


class LoRALinear:
    """
    Linear layer with LoRA adapter.
    
    Forward: y = x @ W + lora(x)
    
    Can operate in three modes:
    - disabled: y = x @ W (original)
    - enabled: y = x @ W + lora(x) 
    - merged: y = x @ (W + ΔW) (for inference)
    """
    
    def __init__(self, W: np.ndarray, rank: int = 8, alpha: float = 16.0):
        self.W = W
        self.d_in, self.d_out = W.shape
        self.lora = LoRAAdapter(self.d_in, self.d_out, rank, alpha)
        self.enabled = True
        self.merged = False
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        base_out = np.matmul(x, self.W)
        
        if self.enabled and not self.merged:
            lora_out = self.lora.forward(x)
            return base_out + lora_out
        
        return base_out
    
    def backward(self, d_out: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass.
        
        Returns:
            d_in: Gradient w.r.t. input
            dW: Gradient w.r.t. base weight (None if using LoRA frozen)
        """
        # Base gradient
        d_in_base = np.matmul(d_out, self.W.T)
        
        if self.enabled and not self.merged:
            # LoRA gradient
            d_in_lora = self.lora.backward(d_out)
            d_in = d_in_base + d_in_lora
        else:
            d_in = d_in_base
        
        # We don't compute dW for base weights when using LoRA
        # (that's the point - freeze base, train adapters)
        return d_in, None
    
    def merge_lora(self):
        """Merge LoRA weights into base weights for inference."""
        if not self.merged:
            delta_W = self.lora.get_merged_delta()
            self.W = self.W + delta_W
            self.merged = True
            print(f"[LoRA] Merged adapter into base weights")
    
    def unmerge_lora(self, original_W: np.ndarray):
        """Restore original weights (requires keeping original)."""
        self.W = original_W.copy()
        self.merged = False


class LoRAFFN:
    """
    Feed-Forward Network with LoRA adapters on W1 and W2.
    
    Original: y = GELU(x @ W1) @ W2
    With LoRA: y = GELU(x @ W1 + lora1(x)) @ W2 + lora2(...)
    """
    
    def __init__(self, W1: np.ndarray, W2: np.ndarray, rank: int = 8, alpha: float = 16.0):
        self.lora_w1 = LoRALinear(W1, rank, alpha)
        self.lora_w2 = LoRALinear(W2, rank, alpha)
        
        # Stats
        original_params = W1.size + W2.size
        lora_params = (self.lora_w1.lora.lora_params + 
                       self.lora_w2.lora.lora_params)
        
        print(f"[LoRA FFN] Original: {original_params:,} params")
        print(f"[LoRA FFN] LoRA: {lora_params:,} params ({100*lora_params/original_params:.1f}%)")
        print(f"[LoRA FFN] Reduction: {100*(1-lora_params/original_params):.1f}%")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward with LoRA."""
        # First linear + activation
        h = self.lora_w1.forward(x)
        h = np.maximum(0, h)  # ReLU (or use GELU)
        
        # Second linear
        out = self.lora_w2.forward(h)
        return out
    
    def apply_grads(self, lr: float = 1e-4):
        """Apply gradients to LoRA adapters only."""
        self.lora_w1.lora.apply_grads(lr)
        self.lora_w2.lora.apply_grads(lr)
    
    def merge(self):
        """Merge LoRA into base for inference."""
        self.lora_w1.merge_lora()
        self.lora_w2.merge_lora()


def demo():
    print("=== LoRA Module Demo ===\n")
    
    # Simulate FFN dimensions
    d_model = 240
    d_ff = 600
    rank = 8
    
    # Create weight matrices
    W1 = np.random.randn(d_model, d_ff).astype(np.float32) * 0.02
    W2 = np.random.randn(d_ff, d_model).astype(np.float32) * 0.02
    
    print(f"Original FFN params: {W1.size + W2.size:,}")
    print(f"Original FFN memory: {(W1.nbytes + W2.nbytes) / 1024:.1f} KB\n")
    
    # Create LoRA FFN
    lora_ffn = LoRAFFN(W1, W2, rank=rank, alpha=16.0)
    
    # Forward pass
    x = np.random.randn(1, 32, d_model).astype(np.float32)
    y = lora_ffn.forward(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Adapter memory
    lora_mem = (lora_ffn.lora_w1.lora.memory_bytes() + 
                lora_ffn.lora_w2.lora.memory_bytes())
    print(f"\nLoRA adapter memory: {lora_mem / 1024:.1f} KB")


if __name__ == "__main__":
    demo()

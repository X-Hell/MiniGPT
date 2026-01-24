import numpy as np
from typing import Tuple, Dict, Optional

class KVCache:
    """
    KV Cache with tiered storage and recency-based eviction.
    """
    def __init__(self, max_len: int, n_heads: int, d_head: int, 
                 n_layers: int = 1, window_size: int = 64, n_kv_heads: int = None):
        self.max_len = max_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_layers = n_layers
        self.window_size = window_size
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        
        # Ring Buffer: Lazy allocation
        # Shape: (n_layers, B, n_kv_heads, max_len, d_head)
        self.k_cache = None
        self.v_cache = None
        
        self.current_len = 0
        
    def update(self, new_k: np.ndarray, new_v: np.ndarray, 
               start_pos: int, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Write new tokens to Ring Buffer and return the attention view.
        
        Args:
            new_k, new_v: (1, H_kv, T, D)
            start_pos: current sequence position
            layer_idx: current layer index
            
        Returns:
        Returns:
            (k_out, v_out): k_out is (1, H_kv, D, window_size), v_out is (1, H_kv, window_size, D)
        """
        B, H, T, D = new_k.shape
        
        # Initialize or Resize if batch size changed
        if self.k_cache is None or self.k_cache.shape[1] != B:
            print(f"[KVCache] Allocating buffer for B={B} (Layers={self.n_layers}, Len={self.max_len})")
            
            # Pickle Compat: Ensure n_kv_heads exists
            if not hasattr(self, 'n_kv_heads'):
                self.n_kv_heads = self.n_heads
                
            # Optimization: Store K as (B, H, D, T) and V as (B, H, T, D) (V is usually multiplied as T x D)
            # Optimization: Store K as (B, H, D, T) and V as (B, H, T, D) (V is usually multiplied as T x D)
            # Standard Attention: softmax(Q @ K.T) @ V
            # Q: (B, H, 1, D)
            # K.T: (B, H, D, T) -> So we want K to be (B, H, D, T) natively?
            # Yes. This avoids transposing K every step.
            
            # V: (B, H, T, D) is correct for the second matmul (Attn @ V).
            
            self.k_cache = np.zeros((self.n_layers, B, self.n_kv_heads, self.d_head, self.max_len), dtype=np.float16)
            self.v_cache = np.zeros((self.n_layers, B, self.n_kv_heads, self.max_len, self.d_head), dtype=np.float16)
        
        # Ring Buffer Write
        for t in range(T):
            pos = (start_pos + t) % self.max_len
            # K is stored as (B, H, D, T)
            # Input new_k is (B, H, T, D)
            # We want vector at time t: new_k[:, :, t:t+1, :] -> (B, H, 1, D)
            # Transpose to (B, H, D, 1) and insert at pos
            
            k_vec = new_k[:, :, t:t+1, :].transpose(0, 1, 3, 2).astype(np.float16)
            self.k_cache[layer_idx, :, :, :, pos:pos+1] = k_vec
            
            # V remains (B, H, T, D) -> Time axis is dim 2
            self.v_cache[layer_idx, :, :, pos:pos+1, :] = new_v[:, :, t:t+1, :].astype(np.float16)
            # V logic: Original was (B, H, T, D).
            # My comment said: "V as (B, H, T, D)". 
            # So V layout is UNCHANGED?
            # Let's check my init code: 
            # self.v_cache = np.zeros((..., self.max_len, self.d_head))
            # Wait, `v_cache` shape in init was `(..., self.max_len, self.d_head)`.
            # That is (T, D). So V is Time-Major last.
            # So V write should be `pos:pos+1` on dim -2? 
            # Let's check Line 43 in previous tool call.
            # `self.v_cache = np.zeros((..., self.max_len, self.d_head))`
            # That corresponds to (..., T, D).
            # So V write logic is correct as is?
            # Yes. V is (T, D). K is (D, T).
            
            # Re-verify K write:
            # new_k is (B, H, D, T)? No, `new_k` passed to `update` usually comes from `model.forward`.
            # In `model.py` (Line 183): `k = k...transpose(0, 2, 1, 3)` -> (B, H, T, D).
            # So `new_k` input is (B, H, T, D).
            # If we want to store it as (D, T), we must transpose `new_k` before writing? 
            # OR transpose the slice.
            # `new_k` is (B, H, T, D). `new_k[:, :, t:t+1, :]` is (B, H, 1, D).
            # We want to write to `k_cache[..., :, pos:pos+1]`. That slot is (B, H, D, 1).
            # So we need to transpose the slice.
            
            k_slice = new_k[:, :, t:t+1, :].transpose(0, 1, 3, 2).astype(np.float16)
            self.k_cache[layer_idx, :, :, :, pos:pos+1] = k_slice
            self.v_cache[layer_idx, :, :, pos:pos+1, :] = new_v[:, :, t:t+1, :].astype(np.float16)
            
        if layer_idx == 0:
            self.current_len = max(self.current_len, start_pos + T)
        
        # Construct Contiguous View
        end_idx = start_pos + T
        
        if end_idx <= self.window_size:
             # Early sequence, no wrap
             view_start = 0
             view_end = end_idx
             # K: (B, H, D, T) -> Slice last dim (Time)
             k_out = self.k_cache[layer_idx, :, :, :, :end_idx].astype(np.float32)
             v_out = self.v_cache[layer_idx, :, :, :end_idx, :].astype(np.float32)
        else:
             # Window mode (possibly wrapping)
             # k_out matches internal layout (D, Window)
             k_out = np.zeros((B, self.n_kv_heads, self.d_head, self.window_size), dtype=np.float32)
             v_out = np.zeros((B, self.n_kv_heads, self.window_size, self.d_head), dtype=np.float32)
             
             # Optimization: If no wrap logic needed for indices?
             # Virtual indices: [end_idx - window_size, end_idx)
             for i in range(self.window_size):
                 virtual_pos = end_idx - self.window_size + i
                 buffer_pos = virtual_pos % self.max_len
                 # buffer_pos indexes Time (last dim for K, 2nd-to-last for V)
                 k_out[:, :, :, i] = self.k_cache[layer_idx, :, :, :, buffer_pos].astype(np.float32)
                 v_out[:, :, i, :] = self.v_cache[layer_idx, :, :, buffer_pos, :].astype(np.float32)
                 
        return (k_out, v_out)

    def reset(self) -> None:
        """Reset cache for new sequence."""
        if self.k_cache is not None:
            self.k_cache.fill(0)
            self.v_cache.fill(0)
        self.current_len = 0
    
    def utilization(self) -> float:
        return self.current_len / self.max_len


import numpy as np
from typing import Tuple, Optional

class OptimizedKVCache:
    """
    High-performance KV Cache with pre-allocation and zero-copy views.
    Layout: (n_layers, batch_size, n_kv_heads, d_head, max_len)
    """
    def __init__(self, max_len: int, n_heads: int, d_head: int, 
                 n_layers: int = 1, window_size: int = 64, n_kv_heads: int = None, batch_size: int = 1):
        
        self.max_len = max_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_layers = n_layers
        self.window_size = window_size
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.batch_size = batch_size
        
        # Pre-allocate full buffer efficiently
        # K stored as (D, T) for direct matmul usage
        # V stored as (T, D) usually, but check compute pattern.
        # Attention: softmax(Q @ K) @ V
        # Q: (B, H, 1, D)
        # K (transposed layout): (B, H, D, T) -> Q@K -> (B, H, 1, T) weights
        # V: (B, H, T, D) -> weights @ V -> (B, H, 1, D)
        # So K should be (..., D, T) and V should be (..., T, D).
        
        # K Layout: (n_layers, B, n_kv_heads, d_head, max_len)
        self.k_buffer = np.zeros((n_layers, batch_size, self.n_kv_heads, d_head, max_len), dtype=np.float16)
        
        # V Layout: (n_layers, B, n_kv_heads, max_len, d_head)
        self.v_buffer = np.zeros((n_layers, batch_size, self.n_kv_heads, max_len, d_head), dtype=np.float16)
        
        self.current_len = 0
        self.roll_count = 0 
        
    def reset(self, batch_size: Optional[int] = None):
        """Reset cache pointers and optionally resize batch dimension."""
        self.current_len = 0
        self.roll_count = 0
        self.k_buffer.fill(0)
        self.v_buffer.fill(0)
        
        if batch_size is not None and batch_size != self.batch_size:
            print(f"[KVCache] Resizing batch: {self.batch_size} -> {batch_size}")
            self.batch_size = batch_size
            self.k_buffer = np.zeros((self.n_layers, batch_size, self.n_kv_heads, self.d_head, self.max_len), dtype=np.float16)
            self.v_buffer = np.zeros((self.n_layers, batch_size, self.n_kv_heads, self.max_len, self.d_head), dtype=np.float16)

    def update(self, new_k: np.ndarray, new_v: np.ndarray, 
               start_pos: int, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update cache with new tokens and return the valid view for attention.
        
        Args:
            new_k: (B, H, T_new, D) 
            new_v: (B, H, T_new, D)
        """
        # Validate batch size dimension (dim 0)
        B, H, T_new, D = new_k.shape
        
        if B != self.batch_size:
            # Auto-resize if mismatch on the fly (safer for dynamic usage)
            self.reset(batch_size=B)
        
        # Ring Buffer Logic
        # We perform circular writes.
        # But for 'view' we prefer contiguous memory if possible.
        # To get contiguous view from circular buffer in numpy requires roll or double-allocation.
        # Optimization: Just use roll? Or return slice if not wrapped?
        
        # Assume start_pos is authoritative.
        
        for t in range(T_new):
            pos = (start_pos + t) % self.max_len
            
            # K Update: Input (B, H, 1, D) -> Store (B, H, D, 1) at pos
            # new_k slice: (B, H, 1, D) -> transpose -> (B, H, D, 1)
            k_slice = new_k[:, :, t:t+1, :].transpose(0, 1, 3, 2).astype(np.float16)
            self.k_buffer[layer_idx, :, :, :, pos:pos+1] = k_slice
            
            # V Update: Input (B, H, 1, D) -> Store (B, H, 1, D) at pos
            self.v_buffer[layer_idx, :, :, pos:pos+1, :] = new_v[:, :, t:t+1, :].astype(np.float16)
            
        if layer_idx == 0:
            self.current_len = max(self.current_len, start_pos + T_new)
            
        # Construct output view
        # We need the full valid context up to 'current_len' (or limited by window)
        # If wrapped, we must roll to make it contiguous for MatMul.
        # MatMul with non-contiguous slices is okay in Numpy, but 'roll' creates copy.
        
        # Simpler approach: If not wrapped, return slice.
        # If wrapped, we must construct.
        
        end_idx = start_pos + T_new
        
        if end_idx <= self.max_len:
             # Contiguous Linear Case: 0 to end_idx
             # Return valid prefix.
             # Windowing: if end_idx > window_size, take suffix?
             
             if end_idx <= self.window_size:
                 # Short sequence
                 k_view = self.k_buffer[layer_idx, :, :, :, :end_idx]
                 v_view = self.v_buffer[layer_idx, :, :, :end_idx, :]
             else:
                 # Windowed, but contiguous in buffer
                 start_v = end_idx - self.window_size
                 k_view = self.k_buffer[layer_idx, :, :, :, start_v:end_idx]
                 v_view = self.v_buffer[layer_idx, :, :, start_v:end_idx, :]
        else:
             # Wrapped Case.
             # We need 'window_size' tokens ending at end_idx.
             # Indices: [end_idx-window, ... end_idx] (modulo max_len)
             # Construct contiguous view.
             
             k_view = np.zeros((B, self.n_kv_heads, self.d_head, self.window_size), dtype=np.float32)
             v_view = np.zeros((B, self.n_kv_heads, self.window_size, self.d_head), dtype=np.float32)
             
             # This loop is avoiding 'np.roll' overhead for full buffer? 
             # Or just manual copy.
             for i in range(self.window_size):
                 virtual = end_idx - self.window_size + i
                 p = virtual % self.max_len
                 k_view[:, :, :, i] = self.k_buffer[layer_idx, :, :, :, p]
                 v_view[:, :, i, :] = self.v_buffer[layer_idx, :, :, p, :]
                 
        return k_view.astype(np.float32), v_view.astype(np.float32)

    def utilization(self) -> float:
        return self.current_len / self.max_len

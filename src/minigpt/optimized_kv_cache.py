from minigpt.backend import xp
from typing import Any, Optional, Tuple

class OptimizedKVCache:
    """
    High-performance KV Cache with pre-allocation and zero-copy views.
    Layout: K (n_layers, batch_size, n_kv_heads, d_head, max_len)
            V (n_layers, batch_size, n_kv_heads, max_len, d_head)

    Uses FP32 for training stability (no gradient corruption from FP16 truncation).
    """
    def __init__(self, max_len: int, n_heads: int, d_head: int,
                 n_layers: int = 1, window_size: int = None, n_kv_heads: int = None, batch_size: int = 1):

        self.max_len = max_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_layers = n_layers
        self.window_size = window_size if window_size is not None else max_len
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.batch_size = batch_size

        # FP32 for training stability (was FP16 -- caused precision loss in long generation)
        self.k_buffer = xp.zeros((n_layers, batch_size, self.n_kv_heads, d_head, max_len), dtype=xp.float32)
        self.v_buffer = xp.zeros((n_layers, batch_size, self.n_kv_heads, max_len, d_head), dtype=xp.float32)

        self.current_len = 0
        self.roll_count = 0

    def reset(self, batch_size: Optional[int] = None) -> None:
        """Reset cache pointers and optionally resize batch dimension."""
        self.current_len = 0
        self.roll_count = 0
        self.k_buffer.fill(0)
        self.v_buffer.fill(0)

        if batch_size is not None and batch_size != self.batch_size:
            print(f"[KVCache] Resizing batch: {self.batch_size} -> {batch_size}")
            self.batch_size = batch_size
            self.k_buffer = xp.zeros((self.n_layers, batch_size, self.n_kv_heads, self.d_head, self.max_len), dtype=xp.float32)
            self.v_buffer = xp.zeros((self.n_layers, batch_size, self.n_kv_heads, self.max_len, self.d_head), dtype=xp.float32)

    def update(self, new_k: Any, new_v: Any, start_pos: int, layer_idx: int) -> Tuple[Any, Any]:
        """
        Update cache with new tokens and return the valid view for attention.

        Args:
            new_k: (B, H, T_new, D)
            new_v: (B, H, T_new, D)
        """
        B, H, T_new, D = new_k.shape

        if B != self.batch_size:
            self.reset(batch_size=B)

        for t in range(T_new):
            pos = (start_pos + t) % self.max_len

            # K Update: Input (B, H, 1, D) -> Store (B, H, D, 1) at pos
            k_slice = new_k[:, :, t:t+1, :].transpose(0, 1, 3, 2).astype(xp.float32)
            self.k_buffer[layer_idx, :, :, :, pos:pos+1] = k_slice

            # V Update: Input (B, H, 1, D) -> Store (B, H, 1, D) at pos
            self.v_buffer[layer_idx, :, :, pos:pos+1, :] = new_v[:, :, t:t+1, :].astype(xp.float32)

        if layer_idx == 0:
            self.current_len = max(self.current_len, start_pos + T_new)

        end_idx = start_pos + T_new

        if end_idx <= self.max_len:
             if end_idx <= self.window_size:
                 k_view = self.k_buffer[layer_idx, :, :, :, :end_idx]
                 v_view = self.v_buffer[layer_idx, :, :, :end_idx, :]
             else:
                 start_v = end_idx - self.window_size
                 k_view = self.k_buffer[layer_idx, :, :, :, start_v:end_idx]
                 v_view = self.v_buffer[layer_idx, :, :, start_v:end_idx, :]
        else:
             k_view = xp.zeros((B, self.n_kv_heads, self.d_head, self.window_size), dtype=xp.float32)
             v_view = xp.zeros((B, self.n_kv_heads, self.window_size, self.d_head), dtype=xp.float32)

             for i in range(self.window_size):
                 virtual = end_idx - self.window_size + i
                 p = virtual % self.max_len
                 k_view[:, :, :, i] = self.k_buffer[layer_idx, :, :, :, p]
                 v_view[:, :, i, :] = self.v_buffer[layer_idx, :, :, p, :]

        return k_view.astype(xp.float32), v_view.astype(xp.float32)

    def utilization(self) -> float:
        """Return used cache fraction in [0, 1]."""
        return self.current_len / self.max_len

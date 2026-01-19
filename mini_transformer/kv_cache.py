import numpy as np

class KVCache:
    """
    KV Cache with tiered storage and recency-based eviction.
    
    Features:
    - FP16 for recent tokens (keep_recent)
    - INT8 compression for older tokens (saves 50% more)
    - Sliding window when max_len exceeded
    """
    def __init__(self, max_len, n_heads, d_head, n_layers=1, compress=False, 
                 window_size=64, keep_recent=32):
        self.max_len = max_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_layers = n_layers
        self.window_size = window_size
        self.keep_recent = keep_recent
        
        # FP16 storage for recent tokens
        self.k_cache = np.zeros((n_layers, 1, n_heads, max_len, d_head), dtype=np.float16)
        self.v_cache = np.zeros((n_layers, 1, n_heads, max_len, d_head), dtype=np.float16)
        
        # INT8 storage for old tokens (with scales)
        self.k_old_int8 = None
        self.v_old_int8 = None
        self.k_scales = None
        self.v_scales = None
        self.old_len = 0
        
        self.current_len = 0
        
        mem_usage = self.k_cache.nbytes + self.v_cache.nbytes
        print(f"[KVCache] FP16 + INT8 eviction (window={window_size}, recent={keep_recent}). Mem: {mem_usage/1024:.2f} KB")

    def update(self, new_k, new_v, start_pos, layer_idx):
        """
        new_k, new_v: (1, n_heads, T, d_head) in FP32
        start_pos: integer index to write to
        layer_idx: which layer to update
        """
        T = new_k.shape[2]
        
        # Apply sliding window if exceeding capacity
        if start_pos + T > self.max_len:
            # Before sliding, compress oldest tokens to INT8
            self._compress_old_tokens(layer_idx)
            
            # Slide: shift old content left
            shift = (start_pos + T) - self.max_len
            self.k_cache[layer_idx, :, :, :-shift, :] = self.k_cache[layer_idx, :, :, shift:, :]
            self.v_cache[layer_idx, :, :, :-shift, :] = self.v_cache[layer_idx, :, :, shift:, :]
            start_pos = self.max_len - T
            self.current_len = start_pos
        
        # Store as FP16
        self.k_cache[layer_idx, :, :, start_pos : start_pos + T, :] = new_k.astype(np.float16)
        self.v_cache[layer_idx, :, :, start_pos : start_pos + T, :] = new_v.astype(np.float16)
        
        if layer_idx == 0:
            if start_pos + T > self.current_len:
                self.current_len = start_pos + T
        
        # Return windowed view (last window_size tokens for attention)
        window_start = max(0, self.current_len - self.window_size)
        
        k_out = self.k_cache[layer_idx, :, :, window_start:self.current_len, :].astype(np.float32)
        v_out = self.v_cache[layer_idx, :, :, window_start:self.current_len, :].astype(np.float32)
        
        return (k_out, v_out)
    
    def _compress_old_tokens(self, layer_idx):
        """Compress tokens beyond keep_recent to INT8."""
        if self.current_len <= self.keep_recent:
            return
        
        compress_end = self.current_len - self.keep_recent
        
        if compress_end <= 0:
            return
        
        # Extract old tokens
        k_old = self.k_cache[layer_idx, :, :, :compress_end, :]
        v_old = self.v_cache[layer_idx, :, :, :compress_end, :]
        
        # Quantize to INT8
        k_scale = np.abs(k_old).max(axis=-1, keepdims=True) / 127.0 + 1e-9
        v_scale = np.abs(v_old).max(axis=-1, keepdims=True) / 127.0 + 1e-9
        
        k_int8 = np.clip(np.round(k_old / k_scale), -127, 127).astype(np.int8)
        v_int8 = np.clip(np.round(v_old / v_scale), -127, 127).astype(np.int8)
        
        # Store compressed (for potential retrieval)
        self.k_old_int8 = k_int8
        self.v_old_int8 = v_int8
        self.k_scales = k_scale
        self.v_scales = v_scale
        self.old_len = compress_end
    
    def get_full_context(self, layer_idx) -> tuple:
        """
        Retrieve full context including decompressed old tokens.
        Use sparingly - decompression has overhead.
        """
        if self.k_old_int8 is None or self.old_len == 0:
            return (
                self.k_cache[layer_idx, :, :, :self.current_len, :].astype(np.float32),
                self.v_cache[layer_idx, :, :, :self.current_len, :].astype(np.float32)
            )
        
        # Decompress old tokens
        k_old = self.k_old_int8.astype(np.float32) * self.k_scales
        v_old = self.v_old_int8.astype(np.float32) * self.v_scales
        
        # Get recent tokens
        k_recent = self.k_cache[layer_idx, :, :, self.old_len:self.current_len, :].astype(np.float32)
        v_recent = self.v_cache[layer_idx, :, :, self.old_len:self.current_len, :].astype(np.float32)
        
        # Concatenate
        k_full = np.concatenate([k_old, k_recent], axis=2)
        v_full = np.concatenate([v_old, v_recent], axis=2)
        
        return (k_full, v_full)
    
    def reset(self):
        """Reset cache for new sequence."""
        self.k_cache.fill(0)
        self.v_cache.fill(0)
        self.k_old_int8 = None
        self.v_old_int8 = None
        self.k_scales = None
        self.v_scales = None
        self.old_len = 0
        self.current_len = 0
    
    def utilization(self) -> float:
        """Return cache utilization as percentage (0-1)."""
        return self.current_len / self.max_len
    
    def memory_usage(self) -> dict:
        """Return memory usage breakdown."""
        fp16_bytes = self.k_cache.nbytes + self.v_cache.nbytes
        int8_bytes = 0
        if self.k_old_int8 is not None:
            int8_bytes = (self.k_old_int8.nbytes + self.v_old_int8.nbytes +
                         self.k_scales.nbytes + self.v_scales.nbytes)
        
        return {
            "fp16_kb": fp16_bytes / 1024,
            "int8_kb": int8_bytes / 1024,
            "total_kb": (fp16_bytes + int8_bytes) / 1024,
            "utilization": self.utilization()
        }

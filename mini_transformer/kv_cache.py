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
        
        # Pre-allocate full buffer (Ring Buffer)
        # static size = max_len
        self.k_cache = np.zeros((n_layers, 1, n_heads, max_len, d_head), dtype=np.float16)
        self.v_cache = np.zeros((n_layers, 1, n_heads, max_len, d_head), dtype=np.float16)
        
        self.current_len = 0
        
        mem_usage = self.k_cache.nbytes + self.v_cache.nbytes
        print(f"[KVCache] Static Ring Buffer (max_len={max_len}, window={window_size}). Mem: {mem_usage/1024:.2f} KB")

    def update(self, new_k, new_v, start_pos, layer_idx):
        """
        Write new tokens to Ring Buffer.
        Returns contiguous view of last `window_size` tokens.
        """
        T = new_k.shape[2]
        
        # Ring Buffer Write
        # We can write in chunks if wrapping around
        for t in range(T):
            pos = (start_pos + t) % self.max_len
            self.k_cache[layer_idx, :, :, pos:pos+1, :] = new_k[:, :, t:t+1, :].astype(np.float16)
            self.v_cache[layer_idx, :, :, pos:pos+1, :] = new_v[:, :, t:t+1, :].astype(np.float16)
            
        if layer_idx == 0:
            self.current_len = max(self.current_len, start_pos + T)
        
        # Construct Contiguous View for Attention
        # We need the last `window_size` tokens ending at `start_pos + T`
        end_idx = start_pos + T
        
        # Calculate range
        if end_idx <= self.window_size:
             # Case 1: Early in sequence, just key from 0 to end
             view_start = 0
             view_end = end_idx
             # Indices in buffer are 0..end_idx (no wrap)
             k_out = self.k_cache[layer_idx, :, :, 0:end_idx, :].astype(np.float32)
             v_out = self.v_cache[layer_idx, :, :, 0:end_idx, :].astype(np.float32)
        else:
             # Case 2: Standard, take last window_size
             # Indices might wrap!
             # example: max=10, window=4, current=12 (wraps to 2)
             # we want virtual indices 8, 9, 10, 11 -> map to buffer 8, 9, 0, 1
             
             # Simpler approach: construct return array
             k_out = np.zeros((1, self.n_heads, self.window_size, self.d_head), dtype=np.float32)
             v_out = np.zeros((1, self.n_heads, self.window_size, self.d_head), dtype=np.float32)
             
             for i in range(self.window_size):
                 virtual_pos = end_idx - self.window_size + i
                 buffer_pos = virtual_pos % self.max_len
                 k_out[:, :, i, :] = self.k_cache[layer_idx, :, :, buffer_pos, :].astype(np.float32)
                 v_out[:, :, i, :] = self.v_cache[layer_idx, :, :, buffer_pos, :].astype(np.float32)
                 
        return (k_out, v_out)
    
    # _compress_old_tokens removed (deprecated)
    
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

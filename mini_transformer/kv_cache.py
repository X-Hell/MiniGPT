import numpy as np

class KVCache:
    """
    Explicit KV Cache for autoregressive generation.
    Pre-allocates memory to avoid re-allocations.
    """
    def __init__(self, max_len, n_heads, d_head, n_layers=1):
        self.max_len = max_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_layers = n_layers
        
        # Pre-allocate: (Layers, Batch=1, N_heads, Max_len, D_head)
        self.k_cache = np.zeros((n_layers, 1, n_heads, max_len, d_head), dtype=np.float32)
        self.v_cache = np.zeros((n_layers, 1, n_heads, max_len, d_head), dtype=np.float32)
        
        self.current_len = 0
        
        mem_usage = self.k_cache.nbytes + self.v_cache.nbytes
        print(f"[KVCache] Initialized. Capacity: {max_len} tokens. Mem: {mem_usage/1024:.2f} KB")

    def update(self, new_k, new_v, start_pos, layer_idx):
        """
        new_k, new_v: (1, n_heads, T, d_head)
        start_pos: integer index to write to
        layer_idx: which layer to update
        """
        T = new_k.shape[2]
        
        if start_pos + T > self.max_len:
             # Check if we should circle buffer or error. Error for now.
             # Actually, if we are training with T=64 < max_len=128, it's fine.
             # If inference runs long, we crash or window.
             # For MiniGPT demo, simple crash or clamp is fine.
             raise ValueError("KV Cache overflow")
             
        # Write into cache (using layer index)
        self.k_cache[layer_idx, :, :, start_pos : start_pos + T, :] = new_k
        self.v_cache[layer_idx, :, :, start_pos : start_pos + T, :] = new_v
        
        # current_len global or per layer?
        # Usually synced across layers.
        if layer_idx == 0:
            if start_pos + T > self.current_len:
                self.current_len = start_pos + T
            
        # Return valid view for THIS layer
        return (
            self.k_cache[layer_idx, :, :, :self.current_len, :],
            self.v_cache[layer_idx, :, :, :self.current_len, :]
        )

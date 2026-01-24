
import sys
import os
import time
import numpy as np
import timeit

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from minigpt.config import ModelConfig
from minigpt.model import FeedForward, MultiHeadAttention, KVCache

def benchmark_layers():
    config = ModelConfig(d_model=128, n_heads=4, n_layers=1, max_len=2048)
    
    # Random Inputs
    B, T, D = 1, 1, 128
    x = np.random.randn(B, T, D).astype(np.float32)
    
    # 1. FFN Benchmark (Generation step T=1)
    ffn = FeedForward(config)
    print("=== FFN Benchmark (1 Step) ===")
    
    def run_ffn():
        return ffn.forward(x)
        
    t_ffn = timeit.timeit(run_ffn, number=1000)
    print(f"FFN (1000 iter): {t_ffn:.4f}s")
    
    # 2. Attention Benchmark (Generation step T=1, Context=500)
    attn = MultiHeadAttention(config)
    kv_cache = KVCache(max_len=2048, n_heads=4, d_head=32, n_kv_heads=2)
    
    # k_past: (B, H_kv, T, D) - API expects standard layout, update() handles transpose
    # Config: n_heads=4 -> n_kv_heads=2 (GQA)
    n_kv_heads = 2
    k_past = np.random.randn(1, n_kv_heads, 500, 32).astype(np.float32)
    # V remains (B, H_kv, T, D)
    v_past = np.random.randn(1, n_kv_heads, 500, 32).astype(np.float32)
    kv_cache.update(k_past, v_past, 0, 0)
    
    print("\n=== Attention Benchmark (Ctx=500) ===")
    def run_attn():
        attn.forward(x, kv_cache, start_pos=500, layer_idx=0)
        
    t_attn = timeit.timeit(run_attn, number=1000)
    print(f"Attn (1000 iter): {t_attn:.4f}s")

    # 3. SiLU Profiling
    print("\n=== SiLU Profile ===")
    large_x = np.random.randn(100, 1000).astype(np.float32) # 100k elements
    def run_silu_numpy():
        return large_x * (1.0 / (1.0 + np.exp(-large_x)))
        
    t_silu = timeit.timeit(run_silu_numpy, number=1000)
    print(f"Standard SiLU (100k elems): {t_silu:.4f}s")

if __name__ == "__main__":
    benchmark_layers()

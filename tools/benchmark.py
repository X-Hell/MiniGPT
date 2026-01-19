"""
Benchmarking Suite for MiniGPT

Measures:
- Inference latency (ms per token)
- Peak memory usage (MB)
- Throughput (tokens/sec)
- GFLOP/s estimation

Outputs results to phase_metrics.csv for README graphs.
"""

import numpy as np
import time
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional
import csv

sys.path.append(os.getcwd())


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    name: str
    latency_ms: float
    memory_peak_mb: float
    tokens_per_sec: float
    gflops: float = 0.0
    extra: dict = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "name": self.name,
            "latency_ms": round(self.latency_ms, 2),
            "memory_peak_mb": round(self.memory_peak_mb, 2),
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "gflops": round(self.gflops, 3),
            **self.extra
        }


class MemoryTracker:
    """
    Simple memory tracking using numpy array allocations.
    For more accurate tracking, use tracemalloc or scalene.
    """
    
    def __init__(self):
        self.allocations = []
        self.peak = 0
    
    def track(self, array: np.ndarray, name: str = ""):
        """Track an array allocation."""
        size_mb = array.nbytes / (1024 * 1024)
        self.allocations.append((name, size_mb))
        total = sum(a[1] for a in self.allocations)
        self.peak = max(self.peak, total)
    
    def reset(self):
        self.allocations = []
        self.peak = 0
    
    def get_peak_mb(self) -> float:
        return self.peak


def estimate_gflops(model, seq_len: int, batch_size: int = 1) -> float:
    """
    Estimate GFLOP for one forward pass.
    
    For transformer: 
    - Embedding lookup: O(V * d)
    - Attention: O(n * d^2 + n^2 * d) per layer
    - FFN: O(n * d * d_ff) per layer
    """
    d = model.d_model
    d_ff = model.layers[0].ffn.d_ff if hasattr(model.layers[0].ffn, 'd_ff') else d * 4
    n_layers = model.n_layers
    n_heads = model.n_heads
    
    n = seq_len
    B = batch_size
    
    # Attention per layer: QKV proj + attention + out proj
    attn_flops = (
        3 * B * n * d * d +  # QKV projection
        B * n_heads * n * n * (d // n_heads) +  # Attention scores
        B * n_heads * n * n * (d // n_heads) +  # Softmax @ V
        B * n * d * d  # Output projection
    )
    
    # FFN per layer: two linear layers
    ffn_flops = (
        B * n * d * d_ff +  # W1
        B * n * d_ff * d    # W2
    )
    
    # Total
    total_flops = n_layers * (attn_flops + ffn_flops)
    
    return total_flops / 1e9  # GFLOP


def benchmark_inference(model, tokenizer, prompt: str, 
                        n_tokens: int = 50, 
                        n_warmup: int = 3,
                        n_runs: int = 5) -> BenchmarkResult:
    """
    Benchmark inference performance.
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer
        prompt: Input prompt
        n_tokens: Number of tokens to generate
        n_warmup: Warmup runs (not counted)
        n_runs: Number of benchmark runs
    
    Returns:
        BenchmarkResult with averaged metrics
    """
    from mini_transformer.kv_cache import KVCache
    
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    
    latencies = []
    
    for run in range(n_warmup + n_runs):
        # Reset cache
        model.kv_cache = KVCache(
            128, 
            model.n_kv_heads, 
            model.d_model // model.n_heads,
            model.n_layers
        )
        
        # Start timing
        start = time.perf_counter()
        
        # Prefill
        logits, _ = model.forward(np.array(prompt_tokens), start_pos=0)
        
        # Generate tokens
        current_ids = [np.argmax(logits[0, -1])]
        for _ in range(n_tokens - 1):
            pos = len(prompt_tokens) + len(current_ids) - 1
            logits, _ = model.forward(np.array(current_ids[-1:]), start_pos=pos)
            current_ids.append(np.argmax(logits[0, -1]))
        
        end = time.perf_counter()
        
        if run >= n_warmup:
            latencies.append((end - start) * 1000)  # ms
    
    avg_latency = np.mean(latencies)
    tokens_per_sec = n_tokens / (avg_latency / 1000)
    
    # Estimate memory (rough)
    memory_peak = sum(
        layer.attn.W_qkv.nbytes + layer.attn.W_o.nbytes +
        layer.ffn.W1.nbytes + layer.ffn.W2.nbytes
        for layer in model.layers
    ) / (1024 * 1024)
    memory_peak += model.embeddings.W_emb.nbytes / (1024 * 1024)
    
    # GFLOP estimation
    gflops = estimate_gflops(model, len(prompt_tokens) + n_tokens)
    gflops_per_sec = gflops / (avg_latency / 1000)
    
    return BenchmarkResult(
        name="inference",
        latency_ms=avg_latency,
        memory_peak_mb=memory_peak,
        tokens_per_sec=tokens_per_sec,
        gflops=gflops_per_sec,
        extra={
            "n_tokens": n_tokens,
            "prompt_len": len(prompt_tokens),
            "latency_std": float(np.std(latencies))
        }
    )


def benchmark_forward(model, batch_size: int = 1, seq_len: int = 64, n_runs: int = 10) -> BenchmarkResult:
    """Benchmark raw forward pass (no generation loop)."""
    from mini_transformer.kv_cache import KVCache
    
    latencies = []
    
    for _ in range(n_runs):
        # Reset
        model.kv_cache = KVCache(
            128,
            model.n_kv_heads,
            model.d_model // model.n_heads,
            model.n_layers
        )
        
        # Random input
        x = np.random.randint(0, model.embeddings.W_emb.shape[0], size=(batch_size, seq_len))
        
        start = time.perf_counter()
        model.forward(x, start_pos=0)
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)
    
    avg_latency = np.mean(latencies)
    gflops = estimate_gflops(model, seq_len, batch_size)
    
    return BenchmarkResult(
        name="forward_pass",
        latency_ms=avg_latency,
        memory_peak_mb=0,  # Not tracked
        tokens_per_sec=seq_len / (avg_latency / 1000),
        gflops=gflops / (avg_latency / 1000),
        extra={
            "batch_size": batch_size,
            "seq_len": seq_len
        }
    )


def save_results(results: List[BenchmarkResult], path: str = "phase_metrics.csv"):
    """Save benchmark results to CSV."""
    if not results:
        return
    
    # Get all keys
    all_keys = set()
    for r in results:
        all_keys.update(r.to_dict().keys())
    
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())
    
    print(f"[Benchmark] Saved results to {path}")


def print_results(results: List[BenchmarkResult]):
    """Pretty print benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    for r in results:
        print(f"\n{r.name.upper()}")
        print("-" * 40)
        print(f"  Latency:     {r.latency_ms:.2f} ms")
        print(f"  Memory:      {r.memory_peak_mb:.2f} MB")
        print(f"  Throughput:  {r.tokens_per_sec:.1f} tokens/sec")
        print(f"  GFLOP/s:     {r.gflops:.3f}")
        
        for k, v in r.extra.items():
            print(f"  {k}: {v}")


def run_all_benchmarks():
    """Run complete benchmark suite."""
    import pickle
    from mini_transformer.tokenizer import BPETokenizer, TokenizerConfig
    from mini_transformer.kv_cache import KVCache
    
    print("=== MiniGPT Benchmark Suite ===\n")
    
    # Load tokenizer
    if not os.path.exists("tokenizer.model"):
        print("[Error] Tokenizer not found. Run train.py first.")
        return
    
    config = TokenizerConfig(vocab_size=300)
    tokenizer = BPETokenizer(config)
    tokenizer.load("tokenizer.model")
    
    # Load model  
    model_path = "mini_transformer_model.pkl"
    if not os.path.exists(model_path):
        print("[Error] Model not found. Run train.py first.")
        return
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Reset cache
    model.kv_cache = KVCache(
        128, 
        model.n_kv_heads,
        model.d_model // model.n_heads,
        model.n_layers
    )
    
    results = []
    
    # Benchmark 1: Forward pass
    print("[1/3] Benchmarking forward pass...")
    results.append(benchmark_forward(model))
    
    # Benchmark 2: Inference (generation)
    print("[2/3] Benchmarking inference...")
    results.append(benchmark_inference(model, tokenizer, "Hello AI", n_tokens=50))
    
    # Benchmark 3: Long context
    print("[3/3] Benchmarking long context...")
    results.append(benchmark_forward(model, seq_len=100))
    
    # Print and save
    print_results(results)
    save_results(results)
    
    return results


if __name__ == "__main__":
    run_all_benchmarks()

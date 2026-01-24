
import sys
import os
import time
import pickle
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from minigpt.inference import InferenceEngine
from minigpt.config import TokenizerConfig
from minigpt.tokenizer import BPETokenizer

def benchmark():
    print("=== Inference Optimization Benchmark ===")
    
    # Load Model
    ckpt_path = "checkpoints/model_latest.pkl"
    if not os.path.exists(ckpt_path):
        print("Error: No checkpoint found.")
        return
        
    with open(ckpt_path, "rb") as f:
        model = pickle.load(f)
        
    # Load Tokenizer
    tok_asset = "assets/tokenizer.model"
    config = TokenizerConfig(vocab_size=4096)
    tokenizer = BPETokenizer(config)
    if os.path.exists(tok_asset):
        tokenizer.load(tok_asset)
    
    engine = InferenceEngine(model, tokenizer)
    
    prompt = "Q: What is the meaning of life?\nA:"
    
    # 1. Test Single Generation (Batched API)
    print("\n[Test 1] Single Generation (num_return_sequences=1)")
    start = time.time()
    texts, stats = engine.generate(prompt, max_tokens=20, num_return_sequences=1)
    dt = time.time() - start
    print(f"Time: {dt*1000:.2f} ms")
    print(f"Output: {texts[0].strip()}")
    
    # 2. Test Batched Generation (Ensemble)
    print("\n[Test 2] Self-Consistency Ensemble (n=3, Batched)")
    start = time.time()
    candidates = engine.self_consistency_ensemble("What is the meaning of life?", "Context: Life is good.", n=3)
    dt = time.time() - start
    print(f"Time: {dt*1000:.2f} ms")
    print(f"Candidates generated: {len(candidates)}")
    for i, (txt, _) in enumerate(candidates):
        print(f"  {i}: {txt.strip().replace('\n', ' ')}")
        
    # 3. Baseline Comparison Estimation
    # Serial would be roughly 3 * single generation time?
    # Not exact because prompt encoding is cached/fast.
    # But generation loop is dominant.
    print(f"\n[Analysis]")
    print(f"Single Gen: ~{dt:.2f}ms (if we assume linear scaling, 3x would be {dt*3:.2f}ms)")
    # Wait, the logic above is comparing Test 2 time vs single. 
    # Let's run a fake serial benchmark?
    
    print("\n[Test 3] Serial Execution (Simulated 3 calls)")
    start = time.time()
    for _ in range(3):
         engine.generate(prompt, max_tokens=20, num_return_sequences=1)
    dt_serial = time.time() - start
    print(f"Serial Time: {dt_serial*1000:.2f} ms")
    
    print(f"\nSpeedup: {dt_serial / max(dt, 1e-9):.2f}x")

if __name__ == "__main__":
    benchmark()

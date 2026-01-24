
import sys
import os
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from minigpt.config import ModelConfig, TokenizerConfig
from minigpt.model import MiniTransformer
from minigpt.inference import InferenceEngine
from minigpt.tokenizer import BPETokenizer

def test_optimized_integration():
    print("=== OptimizedKVCache Integration Test ===")
    
    # 1. Initialize Fresh Model (Uses OptimizedKVCache via model.py import)
    config = ModelConfig(
        d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, max_len=128
    )
    model = MiniTransformer(config)
    
    # Check if proper class is used
    print(f"KV Cache Type: {type(model.kv_cache).__name__}")
    
    tok_config = TokenizerConfig(vocab_size=config.vocab_size)
    tokenizer = BPETokenizer(tok_config)
    
    engine = InferenceEngine(model, tokenizer)
    
    # 2. Test Generation (Batch=1)
    print("\n[Test 1] Single Generation")
    engine.generate("Test prompt", max_tokens=10, num_return_sequences=1)
    print("Success.")
    
    # 3. Test Batched Generation (Batch=3)
    # This triggers reset() inside OptimizedKVCache to resize buffer
    print("\n[Test 2] Batched Generation (B=3)")
    engine.generate("Test prompt", max_tokens=10, num_return_sequences=3)
    print("Success.")

if __name__ == "__main__":
    test_optimized_integration()

import numpy as np
import pickle
import sys
import os

# Ensure modules in path
sys.path.append(os.getcwd())
from mini_transformer.transformer import MiniTransformer
from mini_transformer.kv_cache import KVCache

def main():
    if len(sys.argv) < 2:
        model_path = "mini_transformer_model.pkl"
    else:
        model_path = sys.argv[1]
        
    print(f"Loading {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    print("Quantizing FFN weights...")
    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            print(f"[Layer {i}] Quantizing FFN...")
            layer.quantize()
    else:
        # Fallback for old models (though we shouldn't use them)
        if hasattr(model, 'ffn'):
            model.ffn.quantize()
    
    # We must reset KV cache before saving to avoid large state or dirty state
    n_layers = getattr(model, 'n_layers', 1)
    # Get heads
    if hasattr(model, 'layers'):
        n_heads = model.layers[0].attn.n_heads
        d_head = model.layers[0].attn.d_head
    else:
        n_heads = model.attn.n_heads
        d_head = model.attn.d_head
        
    model.kv_cache = KVCache(model.kv_cache.max_len, n_heads, d_head, n_layers=n_layers)
    # Remove training caches if present
    if hasattr(model, 'cache_x_emb'): del model.cache_x_emb
    if hasattr(model, 'cache_x2'): del model.cache_x2
    if hasattr(model, 'cache_x3'): del model.cache_x3
    if hasattr(model, 'cache_x_final'): del model.cache_x_final
    
    save_path = model_path.replace(".pkl", "_quantized.pkl")
    print(f"Saving quantized model to {save_path}...")
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
        
    print("Done.")

if __name__ == "__main__":
    main()

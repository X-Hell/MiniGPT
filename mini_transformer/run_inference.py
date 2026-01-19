import numpy as np
import sys
import os

# Ensure we can import modules
sys.path.append(os.getcwd())

from mini_transformer.tokenizer import MiniTokenizer, TokenizerConfig
from mini_transformer.transformer import MiniTransformer
from mini_transformer.matmul import get_stats
from mini_transformer.visualize import plot_attention, plot_memory_log

def main():
    print("=== Mini Transformer Inference Engine ===")
    
    # Configuration
    config = TokenizerConfig(vocab_size=1024)
    tokenizer = MiniTokenizer(config)
    
    # Try to load trained model
    # Prefer quantized
    if os.path.exists("mini_transformer_model_quantized.pkl"):
        model_path = "mini_transformer_model_quantized.pkl"
    else:
        model_path = "mini_transformer_model.pkl"
        
    if os.path.exists(model_path):
        print(f"[Init] Loading trained model from {model_path}...")
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Reset KV Cache state from training
        from mini_transformer.kv_cache import KVCache
        # We need to re-init it to clear history
        # model.n_layers might not exist on old models, but we assume re-training
        # model.attn no longer exists on MiniTransformer, use layers[0].attn
        if hasattr(model, 'layers'):
             n_heads = model.layers[0].attn.n_heads
             d_head = model.layers[0].attn.d_head
             # model should have n_layers
             n_layers = getattr(model, 'n_layers', 2)
        else:
             n_heads = 4
             d_head = 60
             n_layers = 1 # Fallback for old models
             
        model.kv_cache = KVCache(model.kv_cache.max_len, n_heads, d_head, n_layers=n_layers)
    else:
        print("[Init] No trained model found. Initializing random weights.")
        model = MiniTransformer(
            vocab_size=1024,
            d_model=240,
            n_heads=4,
            max_len=128
        )
    
    # Input
    text = "Hello AI"
    print(f"\n[Input] '{text}'")
    
    # Encoding
    tokens = tokenizer.encode(text)
    print(f"[Tokens] {tokens}")
    
    # Inference "prefill" (processing the prompt)
    print("\n--- Phase 1: Prefill (Prompt Processing) ---")
    logits, attn_weights = model.forward(np.array(tokens), start_pos=0)
    
    # Decode last token prediction
    # next_token_id = np.argmax(logits[0, -1])
    next_token_id = sample_next_token(logits[0, -1], temperature=0.8, top_k=5)
    
    print(f"[Prediction] Next token ID: {next_token_id}")
    print(f"[Prediction] Next char: '{tokenizer.decode([next_token_id])}'")
    
    # Visualization items
    token_strs = [tokenizer.decode([t]) for t in tokens]
    plot_attention(attn_weights, token_strs, save_path="attn_prefill.png")
    
    # Generation Step (Autoregressive)
    print("\n--- Phase 2: Generation (Step-by-Step) ---")
    
    generated_ids = list(tokens)
    current_ids = [next_token_id]
    
    # Let's generate 20 more tokens
    for i in range(20):
        pos = len(generated_ids)
        # print(f"Gen Step {i+1} at pos {pos}")
        
        # Forward pass for ONE token
        logits, attn_weights = model.forward(np.array(current_ids), start_pos=pos)
        
        # next_id = sample_next_token(logits[0, -1], temperature=0.8, top_k=5)
        next_id = sample_next_token(logits[0, -1], temperature=0.85, top_p=0.9)
        
        char = tokenizer.decode([next_id])
        # print(f" -> Predicted: '{char}' (ID {next_id})")
        sys.stdout.write(char)
        sys.stdout.flush()
        
        generated_ids.append(current_ids[0])
        current_ids = [next_id]

    print(f"\n\n[Final Output] '{tokenizer.decode(generated_ids)}'")
    
    # Stats
    stats = get_stats()
    # ...

def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    """
    logits: (Vocab_Size,)
    """
    # 1. Temperature
    if temperature > 0:
        logits = logits / temperature
    else:
        # Greedy
        return np.argmax(logits)
    
    # Calculate probabilities once (stabilized)
    max_val = np.max(logits)
    exp_vals = np.exp(logits - max_val)
    probs = exp_vals / np.sum(exp_vals)
    
    # 2. Top-K
    if top_k is not None:
        # Keep only top k
        indices = np.argsort(probs)[-top_k:]
        probs_k = probs[indices]
        probs_k /= np.sum(probs_k)
        
        # We need to map back to original indices
        choice_idx = np.random.choice(len(indices), p=probs_k)
        return indices[choice_idx]
        
    # 3. Top-P (Nucleus)
    if top_p is not None and top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Remove tokens with cumulative probability above the threshold
        # We include the first token that crosses the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
        sorted_indices_to_remove[0] = False
        
        # Set removed probs to 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0
        probs /= np.sum(probs)
        
        return np.random.choice(len(probs), p=probs)
        
    return np.random.choice(len(logits), p=probs)
    print("\n=== Performance Metrics ===")
    print(f"Total MatMul Ops Logged: {len(stats)}")
    
    # Plot memory usage
    plot_memory_log(stats, save_path="memory_log.png")

if __name__ == "__main__":
    main()

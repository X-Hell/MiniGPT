import numpy as np
import sys
import os

# Ensure we can import modules
sys.path.append(os.getcwd())

from mini_transformer.tokenizer import MiniTokenizer, TokenizerConfig
from mini_transformer.transformer import MiniTransformer
from mini_transformer.matmul import get_stats
from mini_transformer.visualize import plot_attention, plot_memory_log, plot_entropy, plot_head_similarity, plot_entropy_heatmap

def sample_next_token(logits, temperature=1.0, top_k=40, top_p=0.9, repetition_penalty=1.0, context_tokens=None):
    """
    Robust sampling: Temp -> RepPenalty -> TopK -> TopP -> Softmax -> Sample
    logits: (Vocab_Size,)
    """
    # 1. Temperature
    logits = logits / (temperature + 1e-9)

    # 2. Repetition Penalty
    if repetition_penalty != 1.0 and context_tokens is not None:
        # Use a set for O(1)
        for token in set(context_tokens):
            if logits[token] < 0:
                logits[token] *= repetition_penalty
            else:
                logits[token] /= repetition_penalty

    # 3. Top-K
    if top_k is not None and top_k > 0:
        kdict = min(top_k, len(logits))
        kth_val = np.partition(logits, -kdict)[-kdict]
        indices_to_remove = logits < kth_val
        logits[indices_to_remove] = -float('Inf')

    # 4. Top-P (Nucleus)
    if top_p is not None and top_p < 1.0:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        
        # Softmax on sorted logits
        max_proto = np.max(sorted_logits)
        if max_proto == -float('Inf'):
            probs = np.ones_like(logits) / len(logits)
        else:
            exp_proto = np.exp(sorted_logits - max_proto)
            sorted_probs = exp_proto / np.sum(exp_proto)
            
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Remove tokens with cumulative probability above the threshold
        # Shift rights to keep first
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
        sorted_indices_to_remove[0] = False
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')

    # 5. Final Softmax & Sample
    max_val = np.max(logits)
    if max_val == -float('Inf'):
         probs = np.ones_like(logits) / len(logits)
    else:
         exp_vals = np.exp(logits - max_val)
         probs = exp_vals / np.sum(exp_vals)
         
    return np.random.choice(len(probs), p=probs)

def main():
    print("=== Mini Transformer Inference Engine ===")
    
    # Configuration
    config = TokenizerConfig(vocab_size=256)
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
        # model.n_layers check
        if hasattr(model, 'layers'):
             n_heads = model.layers[0].attn.n_heads
             d_head = model.layers[0].attn.d_head
             n_layers = getattr(model, 'n_layers', 2)
        else:
             n_heads = 4
             d_head = 60
             n_layers = 2
             
        model.kv_cache = KVCache(model.kv_cache.max_len, n_heads, d_head, n_layers=n_layers)
    else:
        print("[Init] No trained model found. Initializing random weights.")
        model = MiniTransformer(
            vocab_size=256,
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
    next_token_id = sample_next_token(logits[0, -1], temperature=0.8, top_k=5)
    
    print(f"[Prediction] Next token ID: {next_token_id}")
    print(f"[Prediction] Next char: '{tokenizer.decode([next_token_id])}'")
    
    # Visualization items
    token_strs = [tokenizer.decode([t]) for t in tokens]
    from mini_transformer.visualize import plot_attention, plot_memory_log, plot_entropy, plot_head_similarity, plot_entropy_heatmap, plot_interactive_attention
    plot_attention(attn_weights, token_strs, save_path="attn_prefill.png")
    plot_interactive_attention(attn_weights, token_strs, save_path="attn_interactive.html")
    plot_entropy(attn_weights, save_path="entropy.png")
    plot_head_similarity(attn_weights, save_path="head_similarity.png")
    plot_entropy_heatmap(attn_weights, save_path="entropy_heatmap.png")
    # plot_interactive_attention will be added in visualize.py update
    
    # Generation Step (Autoregressive)
    print("\n--- Phase 2: Generation (Step-by-Step) ---")
    
    generated_ids = list(tokens)
    current_ids = [next_token_id]
    
    # Let's generate 50 tokens (longer)
    for i in range(50):
        pos = len(generated_ids)
        
        # Forward pass for ONE token
        logits, attn_weights = model.forward(np.array(current_ids), start_pos=pos)
        
        # Sampling with context for repetition penalty
        next_id = sample_next_token(logits[0, -1], 
                                    temperature=0.85, 
                                    top_p=0.9, 
                                    repetition_penalty=1.2, 
                                    context_tokens=generated_ids)
        
        char = tokenizer.decode([next_id])
        sys.stdout.write(char)
        sys.stdout.flush()
        
        generated_ids.append(current_ids[0])
        current_ids = [next_id]

    print(f"\n\n[Final Output] '{tokenizer.decode(generated_ids)}'")
    
    # Stats
    stats = get_stats()
    print("\n=== Performance Metrics ===")
    print(f"Total MatMul Ops Logged: {len(stats)}")
    
    # Plot memory usage
    plot_memory_log(stats, save_path="memory_log.png")
    
    # Timeline
    from mini_transformer.visualize import plot_inference_timeline
    plot_inference_timeline(stats, save_path="inference_timeline.png")

if __name__ == "__main__":
    main()

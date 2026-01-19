import matplotlib.pyplot as plt
import numpy as np

def plot_attention(attn_weights, tokens, save_path="attention_map.png"):
    """
    attn_weights: (1, n_heads, T_q, T_k)
    tokens: list of strings (length T)
    Plots a grid of attention maps for all heads.
    """
    B, n_heads, T_q, T_k = attn_weights.shape
    
    # We assume square plot mostly, or single step
    is_generation = (T_q == 1)
    
    # Create grid
    cols = 2
    rows = (n_heads + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
    axes = axes.flatten()
    
    for h in range(n_heads):
        ax = axes[h]
        w = attn_weights[0, h] # (T_q, T_k)
        
        if is_generation:
            # Bar chart
             w = w[0] # (T_k,)
             ax.bar(range(len(w)), w, color='skyblue')
             # Only show some ticks if too long
             ax.set_title(f"Head {h} (Generation)")
        else:
            # Heatmap
            im = ax.imshow(w, cmap='Blues')
            # Labels
            if len(tokens) <= 30: # Only label if short
                ax.set_xticks(range(T_k))
                ax.set_xticklabels(tokens, rotation=90, fontsize=8)
                ax.set_yticks(range(T_q))
                ax.set_yticklabels(tokens, fontsize=8)
            ax.set_title(f"Head {h}")
            
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved multi-head attention to {save_path}")

def plot_memory_log(log_data, save_path="memory_usage.png"):
    names = [entry['name'] for entry in log_data]
    mems = [entry['mem_bytes'] / 1024 for entry in log_data] # KB
    
    plt.figure(figsize=(10, 6))
    plt.barh(names, mems, color='salmon')
    plt.xlabel("Instantaneous Memory (KB)")
    plt.title("Memory Footprint per Matrix Mult Operation")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved memory log to {save_path}")

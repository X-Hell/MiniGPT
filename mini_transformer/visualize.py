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

def plot_entropy(attn_weights, save_path="entropy.png"):
    """
    Plots the entropy of the attention distribution per head over tokens.
    attn_weights: (1, n_heads, T, T)
    """
    B, n_heads, T_q, T_k = attn_weights.shape
    # attn_weights sum to 1 over axis -1 (T_k)
    # H = -sum(p * log(p))
    
    # Avoid log(0)
    p = attn_weights[0] # (H, T, T)
    log_p = np.log(p + 1e-9)
    entropy_per_token = -np.sum(p * log_p, axis=-1) # (H, T)
    
    plt.figure(figsize=(10, 6))
    for h in range(n_heads):
        plt.plot(range(T_q), entropy_per_token[h], label=f"Head {h}")
        
    plt.xlabel("Token Position")
    plt.ylabel("Entropy (nats)")
    plt.title("Attention Entropy per Head (Focus Metric)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved entropy plot to {save_path}")

def plot_head_similarity(attn_weights, save_path="head_similarity.png"):
    """
    Plots cosine similarity between attention maps of different heads.
    attn_weights: (1, n_heads, T, T)
    """
    n_heads = attn_weights.shape[1]
    
    # Flatten maps to vectors
    # maps: (H, T*T)
    maps = attn_weights[0].reshape(n_heads, -1)
    
    # Normalize
    norms = np.linalg.norm(maps, axis=1, keepdims=True)
    maps_norm = maps / (norms + 1e-9)
    
    # Cosine Sim
    sim_matrix = np.matmul(maps_norm, maps_norm.T)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(sim_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    plt.xticks(range(n_heads), [f"H{i}" for i in range(n_heads)])
    plt.yticks(range(n_heads), [f"H{i}" for i in range(n_heads)])
    plt.title("Head Specialization (Similarity)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved head similarity to {save_path}")

def plot_entropy_heatmap(attn_weights, save_path="entropy_heatmap.png"):
    """
    2D heatmap of Entropy: Heads x Tokens
    """
    B, n_heads, T_q, T_k = attn_weights.shape
    p = attn_weights[0] # (H, T, T)
    log_p = np.log(p + 1e-9)
    entropy = -np.sum(p * log_p, axis=-1) # (H, T_q)
    
    plt.figure(figsize=(8, 4))
    plt.imshow(entropy, cmap='magma', aspect='auto')
    plt.colorbar(label="Entropy (nats)")
    plt.xlabel("Token Position")
    plt.ylabel("Head Index")
    plt.title("Attention Entropy Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved entropy heatmap to {save_path}")

def plot_interactive_attention(attn_weights, tokens, save_path="attn_interactive.html"):
    """
    Saves an interactive HTML heatmap using Plotly.
    attn_weights: (1, n_heads, T, T)
    tokens: list of str
    """
    try:
        import plotly.express as px
    except ImportError:
        print("[Viz] Plotly not installed. Skipping interactive plot.")
        return
        
    B, n_heads, T_q, T_k = attn_weights.shape
    # Plot Head 0 for now, or make a dropdown? 
    # For now, let's plot Head 0 as an example, or average?
    # User asked for "Head {head_idx} attention".
    # Let's verify if we can plot all heads. No, 3D is hard.
    # Let's plot Head 0 for simplicity or loop.
    # We'll stick to Head 0 as "Default detailed view".
    
    # Actually, let's plot Head 0.
    w = attn_weights[0, 0] # (T, T)
    
    fig = px.imshow(w, 
                    x=tokens, 
                    y=tokens, 
                    color_continuous_scale='Blues',
                    labels=dict(x="Key", y="Query", color="Attention"),
                    title="Attention Map (Head 0) - Interactive")
                    
    fig.write_html(save_path)
    print(f"[Viz] Saved interactive attention to {save_path}")

def plot_inference_timeline(stats, save_path="inference_timeline.png"):
    """
    Plots a timeline of matrix multiplications and memory usage.
    stats: list of dicts from MatMulLogger
    """
    if not stats:
        return
        
    ops = [s['name'] for s in stats]
    mems = [s['mem_bytes'] / 1024 / 1024 for s in stats] # MB
    indices = range(len(stats))
    
    plt.figure(figsize=(10, 6))
    plt.plot(indices, mems, marker='o', linestyle='-', color='purple')
    plt.title("Inference Memory Timeline")
    plt.xlabel("Operation Step")
    plt.ylabel("Memory (MB)")
    plt.grid(True)
    
    # Annotate peaks
    max_mem = max(mems)
    for i, m in enumerate(mems):
        if m == max_mem:
            plt.annotate(f"{ops[i]}\n{m:.2f} MB", (i, m), textcoords="offset points", xytext=(0,10), ha='center')
            
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved timeline to {save_path}")

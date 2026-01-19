import matplotlib.pyplot as plt
import numpy as np

# Set style for aesthetic plots - Dark Theme
plt.style.use('dark_background')

def generate_phase_plots():
    phases = [
        "1. Baseline",
        "2. Logger",
        "3. KV Cache",
        "4. W-Tying",
        "5. Int8 Attn",
        "6. Int8 FFN",
        "7. LowRank",
        "8. Fusion",
        "9. Visualize",
        "10. Research"
    ]
    
    # User Provided Data
    # Params in Millions
    params = [4.8, 4.8, 4.8, 3.9, 3.9, 3.9, 1.6, 1.6, 1.6, 1.6]
    # Memory in MB
    memory = [6.2, 6.2, 6.9, 5.3, 3.9, 3.1, 2.2, 2.0, 2.0, 2.0]
    
    x = np.arange(len(phases))
    
    # Create the figure
    fig, ax1 = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#0d1117') # Github Dark Dimmed
    ax1.set_facecolor('#0d1117')

    # --- Plot Params (Bar Chart) ---
    color_bar = '#58a6ff' # Github Blue
    ax1.set_xlabel('Project Phase', fontsize=12, labelpad=15, color='white', weight='bold')
    ax1.set_ylabel('Parameters (Millions)', color=color_bar, fontsize=12, weight='bold')
    
    # Gradient bars effect (simulated with alpha)
    bars = ax1.bar(x, params, width=0.6, color=color_bar, alpha=0.3, label='Model Size (Params)')
    # Add border to bars
    for bar in bars:
        bar.set_edgecolor(color_bar)
        bar.set_linewidth(1.5)
        
    ax1.tick_params(axis='y', labelcolor=color_bar, colors='white')
    ax1.tick_params(axis='x', labelcolor='white')
    
    # --- Plot Memory (Line Chart) ---
    ax2 = ax1.twinx()
    color_line = '#ff7b72' # Github Red
    ax2.set_ylabel('Peak Memory Usage (MB)', color=color_line, fontsize=12, rotation=270, labelpad=20, weight='bold')
    
    line, = ax2.plot(x, memory, marker='o', markersize=8, linestyle='-', linewidth=3, color=color_line, label='Peak Memory (MB)')
    
    # Fill under line
    ax2.fill_between(x, memory, alpha=0.1, color=color_line)
    
    ax2.tick_params(axis='y', labelcolor=color_line, colors='white')
    
    # Set X Ticks
    ax1.set_xticks(x)
    ax1.set_xticklabels(phases, rotation=45, ha='right', fontsize=10)
    
    # Title
    plt.title('MiniGPT Evolution: Optimization & Efficiency (Phase 1-10)', fontsize=18, pad=20, weight='bold', color='white')
    
    # Grid
    ax1.grid(True, linestyle='--', alpha=0.1, color='white')
    
    # Annotations for major drops
    # Phase 4 Drop
    ax1.annotate('−19% Params\n(Weight Tying)', xy=(3, 3.9), xytext=(3, 4.5),
                 arrowprops=dict(facecolor='white', arrowstyle='->', connectionstyle="arc3,rad=.2", color='white'),
                 fontsize=9, ha='center', color='white', bbox=dict(boxstyle="round,pad=0.3", fc="#161b22", ec=color_bar, alpha=0.9))

    # Phase 7 Drop
    ax1.annotate('−60% Params\n(Low-Rank FFN)', xy=(6, 1.6), xytext=(6, 3.0),
                 arrowprops=dict(facecolor='white', arrowstyle='->', connectionstyle="arc3,rad=.2", color='white'),
                 fontsize=9, ha='center', color='white', bbox=dict(boxstyle="round,pad=0.3", fc="#161b22", ec=color_bar, alpha=0.9))

    # Phase 6 Drop (Memory)
    ax2.annotate('Quantized FFN\n(Stable)', xy=(5, 3.1), xytext=(4, 2.5),
                 arrowprops=dict(facecolor='white', arrowstyle='->', connectionstyle="arc3,rad=-.2", color='white'),
                 fontsize=9, ha='center', color='white', bbox=dict(boxstyle="round,pad=0.3", fc="#161b22", ec=color_line, alpha=0.9))

    # Combine legends manually
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    # Custom legend
    fig.legend(handles1 + handles2, labels1 + labels2, loc='upper right', 
               bbox_to_anchor=(0.90, 0.88), 
               facecolor='#161b22', edgecolor='white', labelcolor='white')

    plt.tight_layout()
    
    # Save
    output_path = "phase_metrics.png"
    plt.savefig(output_path, dpi=300, facecolor='#0d1117', bbox_inches='tight')
    print(f"Generated {output_path}")

if __name__ == "__main__":
    generate_phase_plots()

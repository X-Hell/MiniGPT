#!/usr/bin/env python3
"""
Phase Metrics Visualization Tool
================================
Generates publication-quality plots from phase_metrics.csv for README embedding.

Usage:
    python tools/plot_phase_metrics.py [--csv path/to/metrics.csv]

Outputs:
    plots/quality_across_phases.png    - LogProb, Entropy, Perplexity trends
    plots/latency_memory_across_phases.png - Dual-axis latency/memory
    plots/matmul_ops_per_phase.png     - Bar chart of MatMul operations
    plots/quality_vs_memory.png        - Pareto frontier visualization
"""

import os
import sys
import argparse
import numpy as np

# Graceful import handling
try:
    import pandas as pd
except ImportError:
    print("pandas not installed. Run: pip install pandas")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'logprob': '#2E86AB',
    'entropy': '#A23B72',
    'perplexity': '#F18F01',
    'latency': '#C73E1D',
    'memory': '#3A7D44',
    'matmul': '#6C5B7B'
}

def ensure_output_dir(path: str = "plots"):
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path

def plot_quality_metrics(df: pd.DataFrame, output_dir: str):
    """
    Line plot: avg_logprob, avg_entropy, perplexity vs phase.
    Uses twin y-axis if perplexity scale differs significantly.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Primary axis: logprob and entropy
    ax1.plot(df['phase'], df['avg_logprob'], 
             marker='o', linewidth=2, markersize=8,
             color=COLORS['logprob'], label='Avg LogProb')
    ax1.plot(df['phase'], df['avg_entropy'], 
             marker='s', linewidth=2, markersize=8,
             color=COLORS['entropy'], label='Avg Entropy (nats)')
    
    ax1.set_xlabel('Phase', fontsize=12, fontweight='bold')
    ax1.set_ylabel('LogProb / Entropy', fontsize=11)
    ax1.set_xticks(df['phase'])
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Secondary axis: perplexity (if present and has variance)
    if 'perplexity' in df.columns and df['perplexity'].notna().any():
        ax2 = ax1.twinx()
        ax2.plot(df['phase'], df['perplexity'], 
                 marker='^', linewidth=2, markersize=8,
                 color=COLORS['perplexity'], label='Perplexity', linestyle='--')
        ax2.set_ylabel('Perplexity', fontsize=11, color=COLORS['perplexity'])
        ax2.tick_params(axis='y', labelcolor=COLORS['perplexity'])
        ax2.spines['right'].set_color(COLORS['perplexity'])
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    else:
        ax1.legend(loc='upper right', fontsize=10)
    
    ax1.set_title('Quality Metrics Across Phases', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Annotate key phases
    for _, row in df.iterrows():
        if row['phase'] in [1, 7, 13, 15]:  # Milestone phases
            ax1.annotate(f"P{int(row['phase'])}", 
                        (row['phase'], row['avg_logprob']),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'quality_across_phases.png')
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✓ Saved: {outpath}")

def plot_latency_memory(df: pd.DataFrame, output_dir: str):
    """
    Dual-axis plot: latency_ms (left) and memory_kb (right) vs phase.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Left axis: Latency
    ax1.plot(df['phase'], df['latency_ms'], 
             marker='o', linewidth=2.5, markersize=9,
             color=COLORS['latency'], label='Latency (ms)')
    ax1.set_xlabel('Phase', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Latency (ms)', fontsize=11, color=COLORS['latency'])
    ax1.tick_params(axis='y', labelcolor=COLORS['latency'])
    ax1.set_xticks(df['phase'])
    ax1.spines['left'].set_color(COLORS['latency'])
    
    # Right axis: Memory
    ax2 = ax1.twinx()
    ax2.plot(df['phase'], df['memory_kb'], 
             marker='s', linewidth=2.5, markersize=9,
             color=COLORS['memory'], label='Memory (KB)')
    ax2.set_ylabel('Peak Memory (KB)', fontsize=11, color=COLORS['memory'])
    ax2.tick_params(axis='y', labelcolor=COLORS['memory'])
    ax2.spines['right'].set_color(COLORS['memory'])
    
    # Grid and title
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_title('Latency & Memory Footprint Across Phases', fontsize=14, fontweight='bold', pad=15)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'latency_memory_across_phases.png')
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✓ Saved: {outpath}")

def plot_matmul_ops(df: pd.DataFrame, output_dir: str):
    """
    Bar chart: MatMul operations per phase.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    bars = ax.bar(df['phase'], df['matmul_ops'], 
                  color=COLORS['matmul'], edgecolor='black', linewidth=0.8, alpha=0.85)
    
    # Add value labels on bars
    for bar, ops in zip(bars, df['matmul_ops']):
        height = bar.get_height()
        ax.annotate(f'{int(ops)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Phase', fontsize=12, fontweight='bold')
    ax.set_ylabel('MatMul Operations', fontsize=11)
    ax.set_title('MatMul Operations Per Phase', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(df['phase'])
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    # Highlight reduction
    if len(df) > 1:
        reduction = (1 - df['matmul_ops'].iloc[-1] / df['matmul_ops'].iloc[0]) * 100
        ax.text(0.98, 0.95, f'Total Reduction: {reduction:.1f}%', 
                transform=ax.transAxes, ha='right', va='top',
                fontsize=11, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'matmul_ops_per_phase.png')
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✓ Saved: {outpath}")

def plot_quality_vs_memory(df: pd.DataFrame, output_dir: str):
    """
    Scatter plot: Quality (avg_logprob) vs Cost (memory_kb), phase-labeled.
    Shows Pareto frontier for optimization decisions.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter with phase as color
    scatter = ax.scatter(df['memory_kb'], df['avg_logprob'], 
                         c=df['phase'], cmap='viridis', 
                         s=150, edgecolors='black', linewidth=1.2, alpha=0.9)
    
    # Phase labels
    for _, row in df.iterrows():
        ax.annotate(f"P{int(row['phase'])}", 
                    (row['memory_kb'], row['avg_logprob']),
                    textcoords="offset points", xytext=(8, 3),
                    ha='left', fontsize=10, fontweight='bold')
    
    # Draw Pareto frontier (connect points from high memory/quality to low)
    df_sorted = df.sort_values('memory_kb', ascending=False)
    pareto_mem, pareto_log = [df_sorted['memory_kb'].iloc[0]], [df_sorted['avg_logprob'].iloc[0]]
    for _, row in df_sorted.iterrows():
        if row['avg_logprob'] >= pareto_log[-1]:
            pareto_mem.append(row['memory_kb'])
            pareto_log.append(row['avg_logprob'])
    ax.plot(pareto_mem, pareto_log, 'r--', linewidth=1.5, alpha=0.6, label='Pareto Frontier')
    
    ax.set_xlabel('Peak Memory (KB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg LogProb (higher = better)', fontsize=12, fontweight='bold')
    ax.set_title('Quality vs Memory Tradeoff (Phase Labeled)', fontsize=14, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(scatter, ax=ax, label='Phase')
    cbar.ax.tick_params(labelsize=10)
    
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'quality_vs_memory.png')
    fig.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✓ Saved: {outpath}")

def generate_summary_table(df: pd.DataFrame) -> str:
    """Generate markdown table for README embedding."""
    headers = ['Phase', 'Avg LogProb', 'Avg Entropy', 'Latency (ms)', 'Memory (KB)', 'MatMul Ops', 'Notes']
    
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join([':---:'] * len(headers)) + ' |'
    ]
    
    for _, row in df.iterrows():
        notes = row.get('notes', '')
        lines.append(
            f"| {int(row['phase'])} | {row['avg_logprob']:.2f} | {row['avg_entropy']:.2f} | "
            f"{row['latency_ms']:.2f} | {row['memory_kb']:.0f} | {int(row['matmul_ops'])} | {notes} |"
        )
    
    return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description='Generate phase metric visualizations')
    parser.add_argument('--csv', default='phase_metrics.csv', help='Path to metrics CSV')
    parser.add_argument('--output', default='plots', help='Output directory for plots')
    args = parser.parse_args()
    
    # Load data
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        print("Expected columns: phase, avg_logprob, min_logprob, avg_entropy, perplexity, latency_ms, memory_kb, matmul_ops, notes")
        sys.exit(1)
    
    df = pd.read_csv(args.csv)
    
    # Validate required columns
    required = ['phase', 'avg_logprob', 'avg_entropy', 'latency_ms', 'memory_kb', 'matmul_ops']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        sys.exit(1)
    
    print(f"\n📊 Generating Phase Metrics Visualizations")
    print(f"   Source: {args.csv} ({len(df)} phases)")
    print(f"   Output: {args.output}/\n")
    
    output_dir = ensure_output_dir(args.output)
    
    # Generate all plots
    plot_quality_metrics(df, output_dir)
    plot_latency_memory(df, output_dir)
    plot_matmul_ops(df, output_dir)
    plot_quality_vs_memory(df, output_dir)
    
    # Print summary table
    print("\n📋 Markdown Table for README:\n")
    print(generate_summary_table(df))
    
    print(f"\n✅ All plots saved to {output_dir}/")

if __name__ == '__main__':
    main()

#!/bin/bash
set -euo pipefail

# Ampere-oriented runtime flags (RTX 30xx / A-series).
export MINIGPT_BACKEND=cupy
export CUPY_TF32=1
export CUPY_ACCELERATORS="cub,cutensor"

# Pre-flight checks
echo "=== Pre-flight Checks ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found (non-GPU environment)."
fi
python3 -c "import os,sys; sys.path.insert(0, os.path.abspath('src')); from minigpt.backend import xp; print(f'Backend module: {xp.__name__}')"
python3 -c "import os,sys; sys.path.insert(0, os.path.abspath('src')); from minigpt.config import ModelConfig; c=ModelConfig(); print(f'Config loaded: d_model={c.d_model}, n_layers={c.n_layers}, n_heads={c.n_heads}')"

# Create output directory
OUTPUT_DIR="outputs/gpt1_72h_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Copy config to output dir for reproducibility
cp configs/gpt1_72h.yaml "$OUTPUT_DIR/config.yaml"

# Launch training with nohup (survives terminal disconnect)
nohup python3 -u scripts/train.py \
    --config configs/gpt1_72h.yaml \
    --output_dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/training.log" 2>&1 &

# Save PID
echo $! > "$OUTPUT_DIR/training.pid"
echo "Training PID: $(cat "$OUTPUT_DIR/training.pid")"

echo "=== Training Launched ==="
echo "Monitor with: tail -f $OUTPUT_DIR/training.log"
echo "Stop with: kill $(cat "$OUTPUT_DIR/training.pid")"

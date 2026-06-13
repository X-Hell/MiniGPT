#!/usr/bin/env bash
# Launch the 10-hour GPT-1 GPU training run on the Jetson, detached.
#
#   ./scripts/launch_10h_training.sh [CONFIG] [STEPS]
#
# - GPU is hard-gated: jetson_gpu.sh sets MINIGPT_BACKEND=cupy, which makes the
#   backend raise ImportError (no silent CPU fallback) if CuPy is unavailable.
# - timeout gives a hard 10.5h backstop; train.py checkpoints every
#   checkpoint_interval steps (latest.pkl) and is resumable via
#   --resume_from_checkpoint outputs/gpt1_jetson_10h/checkpoints/latest.pkl
set -euo pipefail
cd "$(dirname "$0")/.."

CFG="${1:-configs/jetson_gpt1_10h.yaml}"
STEPS="${2:-1700}"
OUT="outputs/gpt1_jetson_10h"
mkdir -p "$OUT"

nohup timeout 10.5h ./jetson_gpu.sh python3 scripts/train.py \
    --config "$CFG" \
    --steps "$STEPS" \
    --output_dir "$OUT" \
    > "$OUT/training_console.log" 2>&1 &

echo $! > "$OUT/train.pid"
echo "launched PID $(cat "$OUT/train.pid")"
echo "console log: $OUT/training_console.log"
echo "checkpoints: $OUT/checkpoints/  (latest.pkl, resumable)"

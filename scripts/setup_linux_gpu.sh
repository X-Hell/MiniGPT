#!/bin/bash
# =============================================================================
# MiniGPT Linux GPU Server Setup Script
# Target: Ubuntu 22.04/24.04, NVIDIA RTX 3060 12GB, Ryzen 9, 32GB RAM
#
# This script installs everything needed for the 72-hour training event:
#   1. NVIDIA drivers + CUDA toolkit 12.x
#   2. Python 3.11 virtual environment
#   3. CuPy (CUDA-accelerated NumPy drop-in)
#   4. All MiniGPT dependencies
#   5. Monitoring tools (nvtop, tmux)
#
# Usage:
#   chmod +x scripts/setup_linux_gpu.sh
#   sudo ./scripts/setup_linux_gpu.sh
#
# After running, activate the venv:
#   source ~/minigpt_venv/bin/activate
#   MINIGPT_BACKEND=cupy python scripts/validate_gradients.py
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  MiniGPT Linux GPU Server Setup${NC}"
echo -e "${GREEN}  Target: RTX 3060 12GB + Ryzen 9 + 32GB RAM${NC}"
echo -e "${GREEN}================================================${NC}"

# -------------------------------------------------------------------------
# 1. System Update & Build Essentials
# -------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/7] Updating system packages...${NC}"
apt-get update -qq
apt-get install -y -qq \
    build-essential \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    tmux \
    htop \
    nvtop \
    unzip \
    pkg-config

# -------------------------------------------------------------------------
# 2. NVIDIA Driver (if not already installed)
# -------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/7] Checking NVIDIA driver...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo "  Installing NVIDIA driver 550..."
    apt-get install -y -qq nvidia-driver-550
    echo -e "  ${RED}REBOOT REQUIRED after driver install.${NC}"
    echo "  Run 'sudo reboot' then re-run this script."
    echo "  The script will skip driver install on second run."
else
    echo "  NVIDIA driver already installed:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

# -------------------------------------------------------------------------
# 3. CUDA Toolkit 12.4 (Runtime + Development)
# -------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/7] Installing CUDA Toolkit 12.4...${NC}"
if ! command -v nvcc &> /dev/null; then
    # Add NVIDIA CUDA repository
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-4

    # Add to PATH for current session
    export PATH=/usr/local/cuda-12.4/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}

    # Persist in profile
    echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}' >> /etc/profile.d/cuda.sh
    echo "  CUDA 12.4 installed."
else
    echo "  CUDA already installed: $(nvcc --version | grep release)"
fi

# -------------------------------------------------------------------------
# 4. Python Virtual Environment
# -------------------------------------------------------------------------
VENV_DIR="$HOME/minigpt_venv"
echo -e "\n${YELLOW}[4/7] Creating Python virtual environment at ${VENV_DIR}...${NC}"
if [ ! -d "$VENV_DIR" ]; then
    python3.11 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q

# -------------------------------------------------------------------------
# 5. Install CuPy (CUDA 12.x wheel — no compilation needed)
# -------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/7] Installing CuPy for CUDA 12.x...${NC}"
pip install cupy-cuda12x -q
echo "  CuPy installed. Testing GPU access..."
python3 -c "
import cupy as cp
print(f'  CuPy version: {cp.__version__}')
print(f'  GPU: {cp.cuda.runtime.getDeviceProperties(0)[\"name\"].decode()}')
free, total = cp.cuda.runtime.memGetInfo()
print(f'  VRAM: {total/1024**3:.1f} GB total, {free/1024**3:.1f} GB free')
# Quick matmul benchmark
import time
a = cp.random.randn(4096, 4096, dtype=cp.float32)
b = cp.random.randn(4096, 4096, dtype=cp.float32)
cp.cuda.Stream.null.synchronize()
t0 = time.time()
for _ in range(10):
    c = cp.matmul(a, b)
cp.cuda.Stream.null.synchronize()
tflops = 10 * 2 * 4096**3 / (time.time() - t0) / 1e12
print(f'  FP32 matmul: {tflops:.2f} TFLOPS')
"

# -------------------------------------------------------------------------
# 6. Install MiniGPT Dependencies
# -------------------------------------------------------------------------
echo -e "\n${YELLOW}[6/7] Installing MiniGPT dependencies...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -r "$PROJECT_DIR/requirements.txt" -q
fi

# Install MiniGPT package in editable mode
if [ -f "$PROJECT_DIR/setup.py" ]; then
    pip install -e "$PROJECT_DIR" -q
fi

# Optional: monitoring tools
pip install wandb gpustat -q 2>/dev/null || true

# -------------------------------------------------------------------------
# 7. System Tuning for 72-Hour Training
# -------------------------------------------------------------------------
echo -e "\n${YELLOW}[7/7] Applying system tuning...${NC}"

# Disable GPU power management (keep GPU at max clocks)
nvidia-smi -pm 1 2>/dev/null || true
# Set power limit to maximum for RTX 3060
nvidia-smi -pl 170 2>/dev/null || true

# Increase file descriptor limits
if ! grep -q "minigpt_training" /etc/security/limits.conf 2>/dev/null; then
    echo "# minigpt_training" >> /etc/security/limits.conf
    echo "* soft nofile 65536" >> /etc/security/limits.conf
    echo "* hard nofile 65536" >> /etc/security/limits.conf
fi

# Disable swap to prevent OOM thrashing (optional, 32GB should be enough)
# swapoff -a  # Uncomment if you want to disable swap

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "  Next steps:"
echo "    1. source ~/minigpt_venv/bin/activate"
echo "    2. cd $PROJECT_DIR"
echo "    3. MINIGPT_BACKEND=cupy python scripts/validate_gradients.py"
echo "    4. MINIGPT_BACKEND=cupy python scripts/benchmark_gpu.py"
echo ""
echo "  To start the 72-hour training run:"
echo "    tmux new -s training"
echo "    MINIGPT_BACKEND=cupy python scripts/train.py \\"
echo "      --dim 384 --n_layers 6 --n_heads 6 --n_kv_heads 2 \\"
echo "      --vocab_size 4096 --max_len 256 \\"
echo "      --batch_size 64 --accum_steps 4 \\"
echo "      --steps 50000 --lr 3e-4 \\"
echo "      --save_dir checkpoints_v2 --save_interval 500"
echo ""

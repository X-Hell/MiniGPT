#!/usr/bin/env bash
# Jetson GPU launcher for MiniGPT.
#
# Why this exists: this box has mismatched pip `nvidia-*-cu13` (CUDA 13) wheels
# installed alongside the real system CUDA 12.6. cuda-pathfinder (used by CuPy)
# scans site-packages first and greedily picks the NEWEST version, so it loads
# NVRTC 13 / cuRAND 13 against a 12.6 driver -> CUDA_ERROR_INVALID_IMAGE on every
# JIT kernel and CURAND_STATUS_INITIALIZATION_FAILED. Pre-loading the system
# CUDA-12.6 libs makes pathfinder's "already loaded" short-circuit pick the
# correct versions. Non-destructive: nothing is uninstalled or modified.
#
# Usage: ./jetson_gpu.sh python3 scripts/train.py ...
L=/usr/local/cuda-12.6/targets/aarch64-linux/lib
export LD_PRELOAD="$L/libnvrtc.so.12 $L/libnvrtc-builtins.so.12.6 $L/libcurand.so.10${LD_PRELOAD:+ $LD_PRELOAD}"
export MINIGPT_BACKEND=cupy
exec "$@"

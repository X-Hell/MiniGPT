#!/usr/bin/env python3
"""Real-time training monitor for MiniGPT runs."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def monitor(output_dir: str) -> None:
    """Display recent logs, GPU stats, and latest checkpoints."""

    run_dir = Path(output_dir)
    log_file = run_dir / "training.log"
    ckpt_dir = run_dir / "checkpoints"

    while True:
        os.system("clear")
        print("=" * 60)
        print(f"MiniGPT Training Monitor - {run_dir}")
        print("=" * 60)

        if log_file.exists():
            print("\nRecent log lines:\n")
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-15:]:
                print(line)
        else:
            print("No training.log found yet.")

        print("\n" + "=" * 60)
        print("GPU stats:")
        if os.system("command -v nvidia-smi >/dev/null 2>&1") == 0:
            os.system(
                "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu "
                "--format=csv,noheader,nounits"
            )
        else:
            print("nvidia-smi not available on this machine")

        print("\n" + "=" * 60)
        print("Recent checkpoints:")
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("step_*.pkl"), key=os.path.getmtime, reverse=True)[:5]
            if ckpts:
                for ckpt in ckpts:
                    size_mb = ckpt.stat().st_size / (1024 * 1024)
                    mtime = time.ctime(ckpt.stat().st_mtime)
                    print(f"  {ckpt.name:24s} {size_mb:7.1f} MB  {mtime}")
            else:
                print("  No step checkpoints yet.")
        else:
            print("  No checkpoints directory yet.")

        time.sleep(10)


def main() -> int:
    """CLI entrypoint."""

    if len(sys.argv) < 2:
        print("Usage: python scripts/monitor_training.py <output_dir>")
        return 1
    monitor(sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify a token .bin meets the single-epoch budget and is uncorrupted (I-4).

Run 1 trained over 0.99B tokens for a 2.36B-token schedule -> a 2.38x looping
factor (every token seen ~2.4x). This script gates the Run-2 launch on:

    looping factor <= 1.0   (data >= single-epoch requirement)
    max token id  <  vocab_size
    no out-of-range token ids (corruption)

Usage:
    .venv/bin/python scripts/verify_data.py \
        --bin data/train.bin --vocab 16384 \
        --steps 36000 --batch 32 --accum 4 --seq_len 512
"""
from __future__ import annotations

import argparse

import numpy as np


def verify(bin_path: str, vocab_size: int, total_steps: int, batch: int,
           accum: int, seq_len: int, val_frac: float = 0.01,
           scan_tokens: int = 50_000_000) -> bool:
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    n = int(data.shape[0])
    val_size = max(seq_len + 2, int(n * val_frac))
    train_n = n - val_size
    need = total_steps * batch * accum * seq_len
    loop = need / train_n if train_n > 0 else float("inf")

    scan = np.asarray(data[: min(n, scan_tokens)])
    mx = int(scan.max()) if scan.size else 0
    oob = int((scan >= vocab_size).sum()) if scan.size else 0

    print(f"file                 : {bin_path}")
    print(f"tokens (total)       : {n:,}")
    print(f"tokens (train, 1-{val_frac:.0%} split): {train_n:,}")
    print(f"single-epoch need    : {need:,}  ({batch}x{accum}x{seq_len}x{total_steps})")
    print(f"looping factor       : {loop:.3f}x   "
          f"{'OK (<=1.0)' if loop <= 1.0 else 'FAIL (>1.0 -- rebuild larger)'}")
    print(f"max token id (scan {scan.size:,}): {mx}   "
          f"{'OK' if mx < vocab_size else f'FAIL (>= vocab {vocab_size})'}")
    print(f"out-of-range ids     : {oob}   {'OK' if oob == 0 else 'FAIL -- corruption'}")

    ok = (loop <= 1.0) and (mx < vocab_size) and (oob == 0)
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def verify_non_repeating(bin_path: str, seq_len: int, batch: int,
                         n_batches: int = 64, seed: int = 0) -> None:
    """Sample n_batches; confirm random start-index collisions are ~0 (I-4 sanity)."""
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    hi = int(data.shape[0]) - seq_len - 1
    rng = np.random.RandomState(seed)
    seen: set[int] = set()
    coll = 0
    draws = 0
    for _ in range(n_batches):
        for s in rng.randint(0, hi, size=batch):
            s = int(s)
            coll += s in seen
            seen.add(s)
            draws += 1
    print(f"start-index collisions over {draws:,} draws: {coll} "
          f"({100.0 * coll / max(1, draws):.3f}%)")


def main() -> int:
    p = argparse.ArgumentParser(description="Verify token .bin against the step budget (I-4).")
    p.add_argument("--bin", default="data/train.bin")
    p.add_argument("--vocab", type=int, default=16384)
    p.add_argument("--steps", type=int, default=36000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--val_frac", type=float, default=0.01)
    a = p.parse_args()

    ok = verify(a.bin, a.vocab, a.steps, a.batch, a.accum, a.seq_len, a.val_frac)
    verify_non_repeating(a.bin, a.seq_len, a.batch)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

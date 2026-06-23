"""
RAM-safe data loading for MiniGPT.

`BinDataLoader` reads a flat `uint16` token stream (a `.bin` file produced by
`scripts/prepare_fineweb.py`) via `numpy.memmap`. The full corpus stays
file-backed on the NVMe SSD; only the small per-batch windows are ever copied
into host RAM (and immediately pushed to the GPU). This keeps system-RAM usage
near-zero regardless of corpus size — the key constraint on a 16GB-RAM rig
training over FineWeb-Edu sample-10BT (~20GB of text → ~5-10GB of tokens).

Usage:
    from minigpt.dataloader import BinDataLoader
    train = BinDataLoader("data/train.bin", seq_len=512, batch_size=32)
    x, y = train.next_batch()          # already on the active device (GPU/CPU)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from minigpt.backend import to_device


class BinDataLoader:
    """Sample contiguous (x, y) micro-batches from a flat uint16 `.bin` stream.

    The file is opened with `np.memmap` (mode="r"), so it is never fully read
    into RAM. Each batch slices `batch_size` random windows of `seq_len + 1`
    tokens, upcasts to int64, and transfers to the active backend device.
    """

    def __init__(
        self,
        bin_path: str,
        seq_len: int,
        batch_size: int,
        dtype=np.uint16,
        seed: Optional[int] = None,
    ) -> None:
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Token .bin not found: {bin_path}")
        self.bin_path = bin_path
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        self.n_tokens = int(self.data.shape[0])
        if self.n_tokens < self.seq_len + 1:
            raise ValueError(
                f"{bin_path} has {self.n_tokens} tokens, need at least "
                f"seq_len+1={self.seq_len + 1}."
            )
        self._rng = np.random.RandomState(seed) if seed is not None else np.random

    def __len__(self) -> int:
        """Number of non-overlapping sequences in the stream (rough epoch size)."""
        return self.n_tokens // (self.seq_len + 1)

    def sample_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return one (x, y) batch as host int64 arrays (no device transfer)."""
        hi = self.n_tokens - self.seq_len - 1
        starts = self._rng.randint(0, hi, size=self.batch_size)
        x = np.stack([self.data[i : i + self.seq_len] for i in starts], axis=0).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + self.seq_len + 1] for i in starts], axis=0).astype(np.int64)
        return x, y

    def next_batch(self) -> Tuple[object, object]:
        """Return one (x, y) batch already on the active device (GPU/CPU)."""
        x, y = self.sample_numpy()
        return to_device(x), to_device(y)


def open_splits(
    data_dir: str,
    seq_len: int,
    batch_size: int,
    seed: Optional[int] = None,
) -> Tuple[BinDataLoader, Optional[BinDataLoader]]:
    """Open `data_dir/train.bin` (+ optional `val.bin`) as BinDataLoaders."""
    train = BinDataLoader(os.path.join(data_dir, "train.bin"), seq_len, batch_size, seed=seed)
    val_path = os.path.join(data_dir, "val.bin")
    val = (
        BinDataLoader(val_path, seq_len, batch_size, seed=seed)
        if os.path.exists(val_path)
        else None
    )
    return train, val

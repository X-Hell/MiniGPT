#!/usr/bin/env python3
"""
prepare_fineweb.py — Stream FineWeb-Edu from Hugging Face, tokenize with the
modern 16K BPE, and write a single flat uint16 `.bin` token stream per split.

RAM-safe by design (the 16GB-RAM constraint):
  - `streaming=True` never downloads the whole dataset; documents arrive one at
    a time over HTTP.
  - Tokens accumulate in a small Python buffer that is flushed to disk (binary
    append + os.fsync) every `--flush_tokens` tokens, so resident RAM stays
    near `flush_tokens * 2 bytes` (default ~20 MB) regardless of corpus size.
  - Output is a flat `uint16` `.bin`, read back at train time via `np.memmap`
    (see src/minigpt/dataloader.py::BinDataLoader) — micro-batches stream from
    the NVMe SSD straight to the GPU, bypassing host RAM.

Usage
-----
    # Train the 16K tokenizer (once) and write ~1B tokens to data/train.bin:
    python scripts/prepare_fineweb.py --max_tokens 1_000_000_000

    # Reuse an existing tokenizer, custom budget:
    python scripts/prepare_fineweb.py \
        --out_dir data --vocab_size 16384 \
        --tokenizer_path assets/tokenizer_modern_16k.json \
        --max_tokens 2_000_000_000 --val_tokens 5_000_000 --flush_tokens 10_000_000

Output
------
  data/train.bin   # flat uint16 token stream (documents joined by <eos>)
  data/val.bin     # first --val_tokens tokens held out for validation
  assets/tokenizer_modern_16k.json
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from minigpt.config import TokenizerConfig
from minigpt.tokenizer import HFBPETokenizer


def get_fineweb_stream(split: str = "train", name: str = "sample-10BT"):
    """Return a streaming HuggingFace dataset iterator for FineWeb-Edu."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[prepare_fineweb] ERROR: `datasets` not found. pip install datasets")
        sys.exit(1)
    print(f"[prepare_fineweb] Streaming FineWeb-Edu ({name}, split={split})...")
    return load_dataset("HuggingFaceFW/fineweb-edu", name=name, split=split, streaming=True)


def train_tokenizer(name: str, vocab_size: int, n_docs: int, save_path: str) -> HFBPETokenizer:
    """Train a 16K HFBPE tokenizer on the first n_docs of FineWeb-Edu and save it."""
    print(f"[prepare_fineweb] Training {vocab_size}-vocab BPE on {n_docs:,} docs ...")
    ds = get_fineweb_stream("train", name)

    def text_iter():
        for i, ex in enumerate(ds):
            if i >= n_docs:
                break
            yield ex["text"]

    cfg = TokenizerConfig(vocab_size=vocab_size, special_tokens=("<pad>", "<eos>", "<unk>"))
    tok = HFBPETokenizer(cfg)
    tok.train_from_iterator(text_iter())
    tok.save(save_path)
    print(f"[prepare_fineweb] Saved tokenizer → {save_path} "
          f"(pad={tok.pad_id}, eos={tok.eos_id}, unk={tok.unk_id})")
    return tok


class BinWriter:
    """Append-only uint16 writer with a bounded in-RAM buffer + fsync flushes."""

    def __init__(self, path: str, flush_tokens: int):
        self.path = path
        self.flush_tokens = flush_tokens
        self._buf: list = []
        self.total = 0
        # Truncate/create the file.
        open(path, "wb").close()

    def add(self, ids) -> None:
        self._buf.extend(ids)
        if len(self._buf) >= self.flush_tokens:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        arr = np.asarray(self._buf, dtype=np.uint16)
        with open(self.path, "ab") as fh:
            fh.write(arr.tobytes())
            fh.flush()
            os.fsync(fh.fileno())
        self.total += arr.size
        self._buf = []


def main():
    parser = argparse.ArgumentParser(
        description="Stream FineWeb-Edu and tokenize into flat uint16 .bin files."
    )
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--fineweb_name", default="sample-10BT")
    parser.add_argument("--max_tokens", type=int, default=1_000_000_000,
                        help="Total training tokens to write (default: 1B)")
    parser.add_argument("--val_tokens", type=int, default=5_000_000,
                        help="Tokens held out for validation (written to val.bin)")
    parser.add_argument("--flush_tokens", type=int, default=10_000_000,
                        help="Buffer size before flushing to disk (bounds RAM)")
    parser.add_argument("--vocab_size", type=int, default=16384)
    parser.add_argument("--tokenizer_path", default="assets/tokenizer_modern_16k.json")
    parser.add_argument("--tokenizer_train_docs", type=int, default=100_000,
                        help="Docs used to train the tokenizer if it must be built")
    parser.add_argument("--retrain_tokenizer", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.tokenizer_path) or "assets", exist_ok=True)

    print("=" * 60)
    print("  MiniGPT — FineWeb-Edu Data Preparation (flat .bin, 16K BPE)")
    print("=" * 60)
    print(f"  Subset    : {args.fineweb_name}")
    print(f"  Max toks  : {args.max_tokens:,}  (+{args.val_tokens:,} val)")
    print(f"  Flush at  : {args.flush_tokens:,} tokens "
          f"(~{args.flush_tokens * 2 / 1e6:.0f} MB RAM buffer)")
    print(f"  Vocab     : {args.vocab_size}")
    print()

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = HFBPETokenizer(TokenizerConfig(vocab_size=args.vocab_size))
    if os.path.exists(args.tokenizer_path) and not args.retrain_tokenizer:
        tokenizer.load(args.tokenizer_path)
        print(f"  Loaded tokenizer from {args.tokenizer_path}")
    else:
        tokenizer = train_tokenizer(
            args.fineweb_name, args.vocab_size,
            args.tokenizer_train_docs, args.tokenizer_path,
        )
    eos_id = tokenizer.eos_id
    print(f"  EOS id    : {eos_id}\n")

    # ── 2. Stream → tokenize → flat .bin ─────────────────────────────────────
    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")
    val_writer = BinWriter(val_path, args.flush_tokens)
    train_writer = BinWriter(train_path, args.flush_tokens)

    stream = get_fineweb_stream("train", args.fineweb_name)
    total_docs = 0
    pbar = tqdm(total=args.max_tokens, unit="tok", unit_scale=True, desc="Tokenizing")

    for record in stream:
        text = record.get("text", "")
        if not text.strip():
            continue
        ids = tokenizer.encode(text)
        ids.append(eos_id)                       # document boundary

        if val_writer.total + len(val_writer._buf) < args.val_tokens:
            val_writer.add(ids)
        else:
            train_writer.add(ids)
            pbar.update(len(ids))

        total_docs += 1
        if train_writer.total + len(train_writer._buf) >= args.max_tokens:
            break

    pbar.close()
    val_writer.flush()
    train_writer.flush()

    print("\n" + "=" * 60)
    print("  Preparation Complete!")
    print(f"  Documents processed : {total_docs:,}")
    print(f"  Train tokens        : {train_writer.total:,} → {train_path}")
    print(f"  Val tokens          : {val_writer.total:,} → {val_path}")
    print()
    print("  Train with:")
    print(f"    MINIGPT_BACKEND=cupy python scripts/train.py --data_path {train_path}")
    print("=" * 60)

    # The HuggingFace streaming iterator keeps a background reconnect thread that
    # can race the interpreter finalizer at shutdown (SIGABRT / "no thread-state"
    # after we are already done). All token data is flushed+fsync'd above, so
    # exit hard and clean to avoid that benign-but-alarming crash on big runs.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

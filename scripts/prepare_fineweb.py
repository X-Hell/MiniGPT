#!/usr/bin/env python3
"""
prepare_fineweb.py — Stream FineWeb-Edu from Hugging Face and tokenize into
.npy shards ready for MiniGPT training.

Usage
-----
# Stream ~10 GB (≈ 5M documents) of FineWeb-Edu and write shards to data/:
    python scripts/prepare_fineweb.py

# Custom options:
    python scripts/prepare_fineweb.py \\
        --out_dir data \\
        --max_tokens 2_000_000_000 \\
        --shard_size 100_000_000 \\
        --vocab_size 8192 \\
        --tokenizer_chars 20_000_000 \\
        --retrain_tokenizer

Output
------
  data/
    fineweb_train_00000.npy   # uint16 token arrays, one per shard
    fineweb_train_00001.npy
    ...
    fineweb_val_00000.npy
  assets/tokenizer_fineweb.model   # BPE tokenizer trained on FineWeb text

The .npy shards are directly compatible with the BPETokenizer encode/decode
API and the MiniGPT train.py data loader.
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

# ── Make sure src/ is importable ──────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from minigpt.config import TokenizerConfig
from minigpt.tokenizer import BPETokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_fineweb_stream(split: str = "train", name: str = "sample-10BT"):
    """
    Return a streaming HuggingFace dataset iterator for FineWeb-Edu.

    `name` options (approximate raw-text sizes):
        "sample-10BT"   ~10 GB   ← default (real-world scale, tractable)
        "sample-100BT"  ~100 GB  (large-scale pre-training)
        "CC-MAIN-2024-10" etc.   (specific CommonCrawl dumps)

    Each record has keys: id, text, token_count, dump, url, date, score.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[prepare_fineweb] ERROR: `datasets` library not found.")
        print("  Install it with:  pip install datasets")
        sys.exit(1)

    print(f"[prepare_fineweb] Loading FineWeb-Edu stream ({name}, split={split})...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=name,
        split=split,
        streaming=True,
    )
    return ds


def collect_tokenizer_text(stream, n_chars: int) -> str:
    """
    Walk the stream and collect `n_chars` worth of text for BPE training.
    Returns the concatenated text.
    """
    chunks = []
    collected = 0
    with tqdm(total=n_chars, unit="chars", desc="[Tokenizer corpus]") as bar:
        for record in stream:
            txt = record["text"]
            remaining = n_chars - collected
            chunks.append(txt[:remaining])
            added = min(len(txt), remaining)
            collected += added
            bar.update(added)
            if collected >= n_chars:
                break
    return "\n\n".join(chunks)


def write_shard(tokens: list, out_dir: str, split: str, shard_idx: int):
    """Flush a list of int token IDs to a uint16 .npy shard file."""
    arr = np.array(tokens, dtype=np.uint16)
    fname = f"fineweb_{split}_{shard_idx:05d}.npy"
    path = os.path.join(out_dir, fname)
    np.save(path, arr)
    return path, len(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stream FineWeb-Edu and tokenize into .npy shards for MiniGPT"
    )
    parser.add_argument("--out_dir", default="data",
                        help="Output directory for .npy shards (default: data/)")
    parser.add_argument("--fineweb_name", default="sample-10BT",
                        help="FineWeb-Edu subset name (default: sample-10BT ≈ 10 GB)")
    parser.add_argument("--max_tokens", type=int, default=2_000_000_000,
                        help="Max tokens to write in total (default: 2B ≈ ~10 GB text)")
    parser.add_argument("--shard_size", type=int, default=100_000_000,
                        help="Tokens per shard .npy file (default: 100M → 200 MB uint16)")
    parser.add_argument("--val_docs", type=int, default=5_000,
                        help="First N documents held out as validation (default: 5000)")
    # Tokenizer
    parser.add_argument("--vocab_size", type=int, default=8192,
                        help="BPE vocabulary size (default: 8192)")
    parser.add_argument("--tokenizer_chars", type=int, default=20_000_000,
                        help="Characters of FineWeb text used to train BPE (default: 20M)")
    parser.add_argument("--tokenizer_path", default="assets/tokenizer_fineweb.model",
                        help="Path to save/load the BPE tokenizer model")
    parser.add_argument("--retrain_tokenizer", action="store_true",
                        help="Force retrain tokenizer even if saved model exists")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.tokenizer_path) or "assets", exist_ok=True)

    print("=" * 60)
    print("  MiniGPT — FineWeb-Edu Data Preparation")
    print("=" * 60)
    print(f"  Subset   : {args.fineweb_name}")
    print(f"  Max toks : {args.max_tokens:,}")
    print(f"  Shard sz : {args.shard_size:,} tokens → "
          f"{args.shard_size * 2 / 1e6:.0f} MB per shard (uint16)")
    print(f"  Vocab    : {args.vocab_size}")
    print()

    # ── 1. Train / Load BPE tokenizer ─────────────────────────────────────────
    tok_cfg = TokenizerConfig(vocab_size=args.vocab_size)
    tokenizer = BPETokenizer(tok_cfg)

    if os.path.exists(args.tokenizer_path) and not args.retrain_tokenizer:
        tokenizer.load(args.tokenizer_path)
        print(f"  Loaded tokenizer from {args.tokenizer_path} "
              f"(merges: {len(tokenizer.merges)})")
    else:
        print(f"  Training BPE on {args.tokenizer_chars:,} chars from FineWeb-Edu ...")
        # Stream once for tokenizer corpus
        stream_tok = get_fineweb_stream("train", args.fineweb_name)
        tok_text = collect_tokenizer_text(stream_tok, args.tokenizer_chars)
        tokenizer.train(tok_text)
        tokenizer.save(args.tokenizer_path)
        del tok_text  # Free memory

    eos_id = tokenizer.eos_id
    print(f"  EOS id   : {eos_id}")
    print()

    # ── 2. Stream & tokenize into shards ──────────────────────────────────────
    print("  Streaming FineWeb-Edu and writing tokenized shards ...")

    stream = get_fineweb_stream("train", args.fineweb_name)

    current_split = "val"      # First val_docs docs → val shard(s)
    val_doc_count = 0
    shard_idx = {"train": 0, "val": 0}
    token_counts = {"train": 0, "val": 0}
    shard_buf = []             # Accumulates token IDs for current shard
    total_tokens_written = 0
    total_docs = 0

    pbar = tqdm(desc="Tokenizing", unit="tokens", unit_scale=True,
                total=args.max_tokens)

    for record in stream:
        if total_tokens_written >= args.max_tokens:
            break

        # Switch from val to train after val_docs documents
        if current_split == "val" and val_doc_count >= args.val_docs:
            if shard_buf:
                path, n = write_shard(shard_buf, args.out_dir, "val",
                                      shard_idx["val"])
                token_counts["val"] += n
                shard_idx["val"] += 1
                shard_buf = []
                print(f"\n  [Val shard {shard_idx['val']-1:05d}] {n:,} tokens → {path}")
            current_split = "train"

        text = record.get("text", "")
        if not text.strip():
            continue

        ids = tokenizer.encode(text)
        ids.append(eos_id)  # Document boundary

        shard_buf.extend(ids)
        n_new = len(ids)
        total_tokens_written += n_new
        total_docs += 1
        pbar.update(n_new)

        if current_split == "val":
            val_doc_count += 1

        # Flush shard when buffer is large enough
        while len(shard_buf) >= args.shard_size:
            to_write = shard_buf[:args.shard_size]
            shard_buf = shard_buf[args.shard_size:]
            path, n = write_shard(to_write, args.out_dir, current_split,
                                  shard_idx[current_split])
            token_counts[current_split] += n
            shard_idx[current_split] += 1
            print(f"\n  [{current_split} shard {shard_idx[current_split]-1:05d}] "
                  f"{n:,} tokens → {path}")

    pbar.close()

    # Flush remaining buffer
    if shard_buf:
        path, n = write_shard(shard_buf, args.out_dir, current_split,
                              shard_idx[current_split])
        token_counts[current_split] += n
        shard_idx[current_split] += 1
        print(f"\n  [{current_split} shard {shard_idx[current_split]-1:05d}] "
              f"{n:,} tokens → {path}")

    print()
    print("=" * 60)
    print("  Preparation Complete!")
    print(f"  Documents processed : {total_docs:,}")
    print(f"  Train tokens        : {token_counts['train']:,} "
          f"in {shard_idx['train']} shard(s)")
    print(f"  Val tokens          : {token_counts['val']:,} "
          f"in {shard_idx['val']} shard(s)")
    print(f"  Total tokens        : {total_tokens_written:,}")
    print()
    print("  Next step — train with:")
    print(f"    python scripts/train.py --fineweb --data_dir {args.out_dir} "
          f"--vocab_size {args.vocab_size}")
    print("=" * 60)


if __name__ == "__main__":
    main()

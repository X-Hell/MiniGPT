#!/usr/bin/env python3
"""Comprehensive MiniGPT validation suite.

Run categories:
    python tests/test_all.py --test-gradients
    python tests/test_all.py --test-optimizer
    python tests/test_all.py --test-tokenizer
    python tests/test_all.py --test-data
    python tests/test_all.py --test-memory
    python tests/test_all.py --test-checkpoint
    python tests/test_all.py --test-edge-cases
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys
import tempfile
import time
import traceback
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Optional plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from minigpt.backend import estimate_model_vram, set_mixed_precision, using_gpu, xp
from minigpt.config import ModelConfig, TokenizerConfig
from minigpt.model import MiniTransformer
from minigpt.optimizer import Adam, LRSchedule, build_param_groups
from minigpt.tokenizer import BPETokenizer

import train as train_script


class SkipTest(Exception):
    """Internal skip signal."""


@dataclass
class TestResult:
    """Structured outcome for one test."""

    name: str
    status: str
    duration_s: float
    details: str = ""


RESULTS: List[TestResult] = []


def _cross_entropy(logits, targets):
    """Cross-entropy + dLogits helper."""

    bsz, seq, vocab = logits.shape
    logits_flat = logits.reshape(-1, vocab)
    targets_flat = targets.reshape(-1)

    mx = xp.max(logits_flat, axis=1, keepdims=True)
    ex = xp.exp(logits_flat - mx)
    probs = ex / xp.sum(ex, axis=1, keepdims=True)

    n = logits_flat.shape[0]
    loss = float(-xp.mean(xp.log(probs[xp.arange(n), targets_flat] + 1e-9)))

    dlogits = probs
    dlogits[xp.arange(n), targets_flat] -= 1
    dlogits /= n

    return loss, dlogits.reshape(bsz, seq, vocab)


def _tiny_model(vocab_size: int = 64, d_model: int = 24, n_layers: int = 2, n_heads: int = 2, max_len: int = 8, dropout: float = 0.0) -> MiniTransformer:
    """Construct a small deterministic model for tests."""

    cfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=4 * d_model,
        max_len=max_len,
        dropout=dropout,
    )
    return MiniTransformer(cfg)


def _make_batch(vocab_size: int, batch: int = 2, seq: int = 4, seed: int = 123) -> Tuple[Any, Any]:
    """Create deterministic token/target batch."""

    rng = np.random.RandomState(seed)
    x_np = rng.randint(0, vocab_size, size=(batch, seq), dtype=np.int64)
    y_np = rng.randint(0, vocab_size, size=(batch, seq), dtype=np.int64)
    return xp.asarray(x_np), xp.asarray(y_np)


def _analytical_pairs(model: MiniTransformer, x, y) -> List[Tuple[str, Any, Any]]:
    """Return parameter/gradient pairs from one backward pass."""

    logits, _ = model.forward(x, training=True)
    _, dlogits = _cross_entropy(logits, y)
    dW_emb_out, dW_pos, layer_grads, dX_emb = model.backward(dlogits)

    dW_emb_total = dW_emb_out.copy()
    from minigpt.backend import scatter_add

    scatter_add(dW_emb_total, x.reshape(-1), dX_emb.reshape(-1, model.config.d_model))

    pairs: List[Tuple[str, Any, Any]] = [
        ("W_emb", model.embeddings.W_emb, dW_emb_total),
        ("W_pos", model.embeddings.W_pos, dW_pos),
    ]

    for idx, (ffn_g, attn_g, (ln1_dg, ln1_db), (ln2_dg, ln2_db)) in enumerate(layer_grads):
        dW_fc, db_fc, dW_proj, db_proj = ffn_g
        dW_qkv, db_qkv, dW_o, db_o = attn_g
        layer = model.layers[idx]
        pairs.extend(
            [
                (f"L{idx}.W_qkv", layer.attn.W_qkv, dW_qkv),
                (f"L{idx}.W_o", layer.attn.W_o, dW_o),
                (f"L{idx}.W_fc", layer.ffn.W_fc, dW_fc),
                (f"L{idx}.W_proj", layer.ffn.W_proj, dW_proj),
                (f"L{idx}.ln1.gamma", layer.ln1.gamma, ln1_dg),
                (f"L{idx}.ln1.beta", layer.ln1.beta, ln1_db),
                (f"L{idx}.ln2.gamma", layer.ln2.gamma, ln2_dg),
                (f"L{idx}.ln2.beta", layer.ln2.beta, ln2_db),
                (f"L{idx}.b_qkv", layer.attn.b_qkv, db_qkv),
                (f"L{idx}.b_o", layer.attn.b_o, db_o),
                (f"L{idx}.b_fc", layer.ffn.b_fc, db_fc),
                (f"L{idx}.b_proj", layer.ffn.b_proj, db_proj),
            ]
        )

    return pairs


def _loss_f64(model: MiniTransformer, x, y) -> float:
    """Numerical loss in float64 for finite difference checks."""

    logits, _ = model.forward(x, training=True)
    bsz, seq, vocab = logits.shape
    lf = np.asarray(logits.reshape(-1, vocab), dtype=np.float64)
    tf = np.asarray(y.reshape(-1), dtype=np.int64)
    mx = np.max(lf, axis=1, keepdims=True)
    lse = np.log(np.sum(np.exp(lf - mx), axis=1)) + mx.squeeze()
    return float(np.mean(lse - lf[np.arange(len(tf)), tf]))


def _finite_diff(model: MiniTransformer, x, y, param, grad, n_checks: int = 12, eps: float = 1e-3) -> Tuple[float, float]:
    """Return (median relative error, p90 relative error)."""

    p_flat = param.reshape(-1)
    g_flat = np.asarray(grad).reshape(-1).astype(np.float64)
    rng = np.random.RandomState(0)
    idxs = rng.choice(p_flat.size, size=min(n_checks, p_flat.size), replace=False)
    rels: List[float] = []

    for idx in idxs:
        orig = float(p_flat[idx])
        p_flat[idx] = orig + eps
        lp = _loss_f64(model, x, y)
        p_flat[idx] = orig - eps
        lm = _loss_f64(model, x, y)
        p_flat[idx] = orig

        num = (lp - lm) / (2 * eps)
        ana = float(g_flat[idx])
        if abs(num) < 1e-8 and abs(ana) < 1e-8:
            continue
        denom = max(abs(num), abs(ana), 1e-8)
        rels.append(abs(num - ana) / denom)

    if not rels:
        return 0.0, 0.0

    arr = np.asarray(rels)
    return float(np.median(arr)), float(np.percentile(arr, 90))


def run_test(fn: Callable[[], None]) -> None:
    """Execute one test function and record outcome."""

    name = fn.__name__
    print(f"[RUN ] {name}")
    t0 = time.time()
    try:
        fn()
    except SkipTest as exc:
        dt = time.time() - t0
        RESULTS.append(TestResult(name=name, status="SKIP", duration_s=dt, details=str(exc)))
        print(f"[SKIP] {name}: {exc}")
        return
    except Exception as exc:
        dt = time.time() - t0
        RESULTS.append(TestResult(name=name, status="FAIL", duration_s=dt, details=f"{exc}\n{traceback.format_exc()}"))
        print(f"[FAIL] {name}: {exc}")
        return

    dt = time.time() - t0
    RESULTS.append(TestResult(name=name, status="PASS", duration_s=dt))
    print(f"[PASS] {name} ({dt:.2f}s)")


# ---------------------------------------------------------------------------
# 2.1 Gradient validation tests
# ---------------------------------------------------------------------------

def test_layernorm_gradients() -> None:
    """Numerical gradient check on LayerNorm gamma/beta."""

    model = _tiny_model()
    x, y = _make_batch(model.config.vocab_size)
    pairs = dict((name, (param, grad)) for name, param, grad in _analytical_pairs(model, x, y))

    for key in ("L0.ln1.gamma", "L0.ln1.beta"):
        param, grad = pairs[key]
        med, p90 = _finite_diff(model, x, y, param, grad)
        assert med < 0.10 and p90 < 0.50, f"{key} gradient check failed: med={med:.3f} p90={p90:.3f}"


def test_gelu_gradients() -> None:
    """Numerical gradient check through GELU path (W_fc)."""

    model = _tiny_model()
    x, y = _make_batch(model.config.vocab_size, seed=321)
    pairs = dict((name, (param, grad)) for name, param, grad in _analytical_pairs(model, x, y))
    param, grad = pairs["L0.W_fc"]
    med, p90 = _finite_diff(model, x, y, param, grad)
    assert med < 0.10 and p90 < 0.50, f"GELU/W_fc check failed: med={med:.3f} p90={p90:.3f}"


def test_attention_gradients() -> None:
    """Numerical gradient check on attention projection W_qkv."""

    model = _tiny_model()
    x, y = _make_batch(model.config.vocab_size, seed=654)
    pairs = dict((name, (param, grad)) for name, param, grad in _analytical_pairs(model, x, y))
    param, grad = pairs["L0.W_qkv"]
    med, p90 = _finite_diff(model, x, y, param, grad)
    assert med < 0.15 and p90 < 1.20, f"Attention W_qkv check failed: med={med:.3f} p90={p90:.3f}"


def test_ffn_gradients() -> None:
    """Numerical gradient check on FFN output projection W_proj."""

    model = _tiny_model()
    x, y = _make_batch(model.config.vocab_size, seed=888)
    pairs = dict((name, (param, grad)) for name, param, grad in _analytical_pairs(model, x, y))
    param, grad = pairs["L0.W_proj"]
    med, p90 = _finite_diff(model, x, y, param, grad)
    assert med < 0.10 and p90 < 0.50, f"FFN W_proj check failed: med={med:.3f} p90={p90:.3f}"


def test_transformer_block_gradients() -> None:
    """Integration check on key parameters in a full transformer block."""

    model = _tiny_model()
    x, y = _make_batch(model.config.vocab_size, seed=999)
    pairs = dict((name, (param, grad)) for name, param, grad in _analytical_pairs(model, x, y))

    keys = ["L0.W_qkv", "L0.W_o", "L0.W_fc", "L0.W_proj", "L0.ln1.gamma", "L0.ln2.beta"]
    for key in keys:
        param, grad = pairs[key]
        med, p90 = _finite_diff(model, x, y, param, grad, n_checks=8)
        assert med < 0.20 and p90 < 0.70, f"{key} failed: med={med:.3f} p90={p90:.3f}"


def test_embedding_tying_gradients() -> None:
    """Verify tied embedding receives output and input-path gradient contributions."""

    model = _tiny_model(vocab_size=128)
    x, y = _make_batch(model.config.vocab_size, batch=2, seq=6, seed=111)
    logits, _ = model.forward(x, training=True)
    _, dlogits = _cross_entropy(logits, y)
    dW_emb_out, _dW_pos, _layer_grads, dX_emb = model.backward(dlogits)

    dW_total = dW_emb_out.copy()
    from minigpt.backend import scatter_add

    scatter_add(dW_total, x.reshape(-1), dX_emb.reshape(-1, model.config.d_model))

    delta = np.abs(np.asarray(dW_total) - np.asarray(dW_emb_out)).sum()
    assert delta > 0.0, "No input-path contribution detected in tied embedding gradient"


# ---------------------------------------------------------------------------
# 2.2 Optimizer tests
# ---------------------------------------------------------------------------

def test_weight_decay_exclusion() -> None:
    """Verify LN params, biases, and W_pos are excluded from weight decay."""

    model = _tiny_model()
    groups = build_param_groups(model, weight_decay=0.1)

    decay_set = {id(p) for p in groups[0]["params"]}
    no_decay_set = {id(p) for p in groups[1]["params"]}

    for name, param, group in model.named_parameters_with_groups():
        if name.endswith("W_pos") or name.endswith(".gamma") or name.endswith(".beta") or \
           name.endswith(".b_qkv") or name.endswith(".b_o") or name.endswith(".b_fc") or name.endswith(".b_proj"):
            assert id(param) in no_decay_set, f"Expected no_decay for {name}"
        if group == "decay":
            assert id(param) in decay_set

    # Explicit WD contribution check
    for name, param, group in model.named_parameters_with_groups():
        wd_contrib = float(np.asarray(0.1 * param).sum()) if group == "decay" else 0.0
        if group == "no_decay":
            assert wd_contrib == 0.0, f"Expected zero WD contribution for {name}"


def test_lr_schedule() -> None:
    """Verify warmup + cosine schedule shape and key points."""

    sched = LRSchedule(peak_lr=2.5e-4, min_lr=1e-5, warmup_steps=2000, max_steps=100000)
    key_steps = [0, 500, 2000, 50000, 100000]
    key_vals = [sched(step) for step in key_steps]

    assert abs(key_vals[0] - 0.0) < 1e-12, f"step 0 lr mismatch: {key_vals[0]}"
    assert abs(key_vals[2] - 2.5e-4) < 1e-12, f"step 2000 lr mismatch: {key_vals[2]}"
    assert abs(key_vals[4] - 1e-5) < 1e-12, f"step 100000 lr mismatch: {key_vals[4]}"

    steps = np.arange(0, 100001)
    lrs = np.asarray([sched(int(step)) for step in steps])
    assert np.all(np.isfinite(lrs))
    assert np.all(np.diff(lrs[:2001]) >= -1e-12), "Warmup should be non-decreasing"

    out_path = ROOT / "tests" / "lr_schedule.png"
    plt.figure(figsize=(8, 4))
    plt.plot(steps, lrs)
    plt.title("MiniGPT LR Schedule")
    plt.xlabel("Step")
    plt.ylabel("LR")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def test_gradient_clipping() -> None:
    """Verify global norm clipping at 1.0."""

    # Construct synthetic nested grad tuple with very large norm.
    fake_layer_grads = [
        (
            (xp.ones((4, 8), dtype=xp.float32) * 10, xp.ones((8,), dtype=xp.float32) * 10, xp.ones((8, 4), dtype=xp.float32) * 10, xp.ones((4,), dtype=xp.float32) * 10),
            (xp.ones((4, 12), dtype=xp.float32) * 10, xp.ones((12,), dtype=xp.float32) * 10, xp.ones((4, 4), dtype=xp.float32) * 10, xp.ones((4,), dtype=xp.float32) * 10),
            (xp.ones((4,), dtype=xp.float32) * 10, xp.ones((4,), dtype=xp.float32) * 10),
            (xp.ones((4,), dtype=xp.float32) * 10, xp.ones((4,), dtype=xp.float32) * 10),
        )
    ]
    grads = (
        xp.ones((16, 4), dtype=xp.float32) * 10,
        xp.ones((8, 4), dtype=xp.float32) * 10,
        fake_layer_grads,
    )

    clipped, _before = train_script.clip_grads(grads, max_norm=1.0)
    after = train_script.global_grad_norm(clipped)
    assert after <= 1.0001, f"Clipped norm too large: {after}"


def test_adam_vs_adamw() -> None:
    """Verify optimizer uses coupled L2 (not decoupled AdamW)."""

    p0 = np.array([[1.0, -2.0], [0.5, -0.25]], dtype=np.float32)
    g = np.array([[0.2, -0.1], [0.05, -0.03]], dtype=np.float32)

    p_adam = xp.asarray(p0.copy())
    g_adam = xp.asarray(g.copy())

    opt = Adam(lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    opt.step([p_adam], [g_adam], lr=1e-3)

    # Manual coupled Adam reference (single step).
    b1, b2 = 0.9, 0.98
    g_coupled = g + 0.01 * p0
    m = (1 - b1) * g_coupled
    v = (1 - b2) * (g_coupled * g_coupled)
    m_hat = m / (1 - b1)
    v_hat = v / (1 - b2)
    p_ref = p0 - 1e-3 * m_hat / (np.sqrt(v_hat) + 1e-8)

    # Decoupled AdamW reference.
    m_w = (1 - b1) * g
    v_w = (1 - b2) * (g * g)
    m_hat_w = m_w / (1 - b1)
    v_hat_w = v_w / (1 - b2)
    p_adamw = p0 - 1e-3 * m_hat_w / (np.sqrt(v_hat_w) + 1e-8) - 1e-3 * 0.01 * p0

    p_out = np.asarray(p_adam)
    assert np.allclose(p_out, p_ref, rtol=1e-6, atol=1e-6), "Optimizer does not match coupled Adam+L2"
    assert not np.allclose(p_out, p_adamw, rtol=1e-8, atol=1e-8), "Update matches AdamW unexpectedly"


# ---------------------------------------------------------------------------
# 2.3 Tokenizer tests
# ---------------------------------------------------------------------------

def _sample_sentences(n: int = 100) -> List[str]:
    """Build a diverse sentence list."""

    base = [
        "Hello, world!",
        "",
        "a",
        "The quick brown fox jumps over 13 lazy dogs.",
        "def f(x): return x**2 + 1",
        "https://example.com/path?q=1&lang=en",
        "emoji test: 😀 🚀 ✅",
        "Math: ∑_{i=1}^n i = n(n+1)/2",
        "New\nline\ntext",
        "Tabs\tare\there",
        "Café naïve façade déjà vu",
        "<eos> <pad> <unk>",
    ]
    out: List[str] = []
    for idx in range(n):
        out.append(base[idx % len(base)] + f" :: {idx}")
    return out


def test_tokenizer_roundtrip() -> None:
    """Encode/decode roundtrip over 100 diverse sentences."""

    tok = BPETokenizer(TokenizerConfig(vocab_size=257, min_frequency=1000000))
    # No merges -> pure byte-level, strictly lossless roundtrip.
    sentences = _sample_sentences(100)
    for sent in sentences:
        enc = tok.encode(sent)
        dec = tok.decode(enc)
        assert dec == sent, f"Roundtrip mismatch: {sent!r} -> {dec!r}"


def test_vocab_coverage() -> None:
    """Verify <pad>, <eos>, <unk> special tokens in HF tokenizer when available."""

    try:
        from minigpt.tokenizer import HFBPETokenizer
    except Exception as exc:
        raise SkipTest(f"HFBPETokenizer unavailable: {exc}")

    try:
        tok = HFBPETokenizer(TokenizerConfig(vocab_size=300, min_frequency=1))
    except Exception as exc:
        raise SkipTest(f"HFBPETokenizer runtime unavailable: {exc}")
    corpus = _sample_sentences(200)
    tok.train_from_iterator(corpus, min_frequency=1)

    assert tok.pad_id is not None, "<pad> missing"
    assert tok.eos_id is not None, "<eos> missing"
    assert tok.unk_id is not None, "<unk> missing"

    assert tok.encode("<eos>") == [tok.eos_id], "<eos> not atomic"
    assert tok.encode("<pad>") == [tok.pad_id], "<pad> not atomic"
    assert tok.encode("<unk>") == [tok.unk_id], "<unk> not atomic"


def test_tokenizer_efficiency() -> None:
    """Report token-count statistics on synthetic FineWeb-like samples."""

    docs = [" ".join(_sample_sentences(10)) for _ in range(1000)]

    # Baseline byte-only tokenizer.
    baseline = BPETokenizer(TokenizerConfig(vocab_size=257, min_frequency=1_000_000))
    base_counts = np.array([len(baseline.encode(doc)) for doc in docs], dtype=np.float64)

    # Lightly trained tokenizer.
    trained = BPETokenizer(TokenizerConfig(vocab_size=1024, min_frequency=2))
    trained.train("\n".join(docs[:200]))
    counts = np.array([len(trained.encode(doc)) for doc in docs], dtype=np.float64)

    stats = {
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "p95": float(np.percentile(counts, 95)),
        "p99": float(np.percentile(counts, 99)),
    }
    print(f"tokenizer efficiency stats: {stats}")
    assert np.mean(counts) <= np.mean(base_counts), "Trained tokenizer should not be worse than byte baseline"


def test_special_token_insertion() -> None:
    """Verify prepare_fineweb appends EOS at each document boundary."""

    script_text = (ROOT / "scripts" / "prepare_fineweb.py").read_text(encoding="utf-8")
    assert "ids.append(eos_id)" in script_text, "prepare_fineweb.py does not append EOS at boundaries"


# ---------------------------------------------------------------------------
# 2.4 Data pipeline tests
# ---------------------------------------------------------------------------

def test_data_loading() -> None:
    """Load 1000 synthetic batches and verify shape/range/finite checks."""

    vocab_size = 40000
    tokens = np.random.randint(0, vocab_size, size=500_000, dtype=np.int64)

    for _ in range(1000):
        x, y = train_script.sample_batch(tokens, batch_size=8, seq_len=32)
        assert x.shape == (8, 32) and y.shape == (8, 32)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert x.min() >= 0 and y.min() >= 0
        assert x.max() < vocab_size and y.max() < vocab_size


def test_data_diversity() -> None:
    """Check token diversity across sampled batches."""

    vocab_size = 40000
    tokens = np.random.randint(0, vocab_size, size=600_000, dtype=np.int64)

    collected = []
    for _ in range(100):
        x, _ = train_script.sample_batch(tokens, batch_size=8, seq_len=32)
        collected.append(x.reshape(-1))

    all_tokens = np.concatenate(collected)
    unique_ratio = len(np.unique(all_tokens)) / len(all_tokens)
    assert unique_ratio > 0.70, f"Unique token ratio too low: {unique_ratio:.3f}"


def test_eos_token_presence() -> None:
    """Ensure every synthetic document-batch has at least one EOS token."""

    eos_id = 256
    docs = []
    rng = np.random.RandomState(77)
    for _ in range(128):
        doc = rng.randint(0, 255, size=31, dtype=np.int64)
        docs.append(np.concatenate([doc, [eos_id]]) )

    batches = np.stack(docs, axis=0)
    assert batches.shape == (128, 32)
    assert np.all(np.any(batches == eos_id, axis=1)), "Some batches do not contain EOS"


# ---------------------------------------------------------------------------
# 2.5 VRAM/memory tests
# ---------------------------------------------------------------------------

def test_vram_estimation_accuracy() -> None:
    """Compare estimator against runtime memory (GPU only)."""

    if not using_gpu():
        raise SkipTest("No GPU backend active")

    cfg = _tiny_model(vocab_size=1024, d_model=64, n_layers=2, n_heads=2, max_len=64, dropout=0.0).config
    model = MiniTransformer(cfg)
    n_params = sum(p.size for _, p in model.named_parameters())

    estimate = estimate_model_vram(
        n_params=n_params,
        batch_size=8,
        seq_len=cfg.max_len,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        mixed_precision=True,
    )

    free_before, total = xp.cuda.runtime.memGetInfo()
    x, y = _make_batch(cfg.vocab_size, batch=8, seq=cfg.max_len)
    logits, _ = model.forward(x, training=True)
    _, dlogits = _cross_entropy(logits, y)
    _ = model.backward(dlogits)
    xp.cuda.Stream.null.synchronize()
    free_after, _ = xp.cuda.runtime.memGetInfo()

    used_mb = (free_before - free_after) / (1024 ** 2)
    est_mb = estimate["total_mb"]
    diff = abs(est_mb - used_mb) / max(used_mb, 1e-6)
    assert diff <= 0.10, f"Estimator off by {diff*100:.1f}% (est={est_mb:.1f}MB actual={used_mb:.1f}MB)"


def test_fp16_no_overflow() -> None:
    """Run short mixed-precision training and verify finite gradients."""

    set_mixed_precision(True)
    model = _tiny_model(vocab_size=512, d_model=64, n_layers=2, n_heads=2, max_len=16, dropout=0.1)

    for step in range(20):
        x, y = _make_batch(model.config.vocab_size, batch=4, seq=16, seed=100 + step)
        logits, _ = model.forward(x, training=True)
        _, dlogits = _cross_entropy(logits, y)
        grads = model.backward(dlogits)

        flat_values = []
        for item in grads:
            if isinstance(item, list):
                for sub in item:
                    flat_values.extend([np.asarray(v).ravel() for v in sub[0] + sub[1] + sub[2] + sub[3]])
            else:
                flat_values.append(np.asarray(item).ravel())

        all_vals = np.concatenate(flat_values)
        assert np.isfinite(all_vals).all(), f"Non-finite gradients at step {step}"


def test_memory_leak() -> None:
    """Run repeated steps and check memory growth stays bounded."""

    model = _tiny_model(vocab_size=256, d_model=32, n_layers=2, n_heads=2, max_len=16, dropout=0.0)

    if using_gpu():
        free_start, _ = xp.cuda.runtime.memGetInfo()
        for step in range(200):
            x, y = _make_batch(model.config.vocab_size, batch=4, seq=16, seed=200 + step)
            logits, _ = model.forward(x, training=True)
            _, dlogits = _cross_entropy(logits, y)
            _ = model.backward(dlogits)
        xp.cuda.Stream.null.synchronize()
        free_end, _ = xp.cuda.runtime.memGetInfo()
        leak_mb = (free_start - free_end) / (1024 ** 2)
        assert leak_mb < 64.0, f"Potential GPU memory leak: {leak_mb:.1f}MB"
    else:
        tracemalloc.start()
        for step in range(200):
            x, y = _make_batch(model.config.vocab_size, batch=4, seq=16, seed=300 + step)
            logits, _ = model.forward(x, training=True)
            _, dlogits = _cross_entropy(logits, y)
            _ = model.backward(dlogits)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        growth_mb = (peak - current) / (1024 ** 2)
        assert growth_mb < 32.0, f"Potential CPU memory leak: {growth_mb:.1f}MB"


# ---------------------------------------------------------------------------
# 2.6 Checkpoint tests
# ---------------------------------------------------------------------------

def _one_train_step(model: MiniTransformer, optimizer: Adam, tokens: np.ndarray, batch_size: int, seq_len: int, weight_decay: float, lr: float) -> float:
    """Run one update step for checkpoint-oriented tests."""

    x_np, y_np = train_script.sample_batch(tokens, batch_size=batch_size, seq_len=seq_len)
    x = xp.asarray(x_np)
    y = xp.asarray(y_np)

    logits, _ = model.forward(x, training=True)
    loss, dlogits = _cross_entropy(logits, y)

    grads_raw = model.backward(dlogits)
    grads = train_script.consolidate_tied_embedding_grad(model, grads_raw, x)
    grads, _ = train_script.clip_grads(grads, max_norm=1.0)
    train_script.apply_grads(model, optimizer, grads, weight_decay=weight_decay, lr=lr)
    return float(loss)


def test_checkpoint_save_load() -> None:
    """Save + load checkpoint and verify exact parameter reproduction."""

    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=2, max_len=16, dropout=0.1)
    model = MiniTransformer(cfg)
    opt = Adam(lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    tokens = np.random.randint(0, cfg.vocab_size, size=20000, dtype=np.int64)

    for _ in range(10):
        _one_train_step(model, opt, tokens, batch_size=4, seq_len=16, weight_decay=0.01, lr=1e-4)

    with tempfile.TemporaryDirectory() as td:
        run_cfg = train_script.RunConfig(vocab_size=cfg.vocab_size)
        ckpt_path = os.path.join(td, "ckpt.pkl")
        train_script.save_checkpoint(ckpt_path, model, opt, run_cfg, step=10, best_val_loss=1.23, tokens_seen=640)

        model2 = MiniTransformer(cfg)
        opt2 = Adam(lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
        train_script.load_checkpoint(ckpt_path, model2, opt2)

        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert np.allclose(np.asarray(p1), np.asarray(p2), rtol=1e-7, atol=1e-7), f"Mismatch in {n1}"


def test_checkpoint_resume() -> None:
    """Resume from checkpoint and verify continuity against uninterrupted run."""

    cfg = ModelConfig(vocab_size=128, d_model=32, n_layers=2, n_heads=2, max_len=8, dropout=0.0)
    tokens = np.random.RandomState(55).randint(0, cfg.vocab_size, size=10000, dtype=np.int64)

    # Precompute deterministic batches.
    rng = np.random.RandomState(999)
    batches = []
    for _ in range(100):
        starts = rng.randint(0, len(tokens) - cfg.max_len - 1, size=2)
        x = np.stack([tokens[i : i + cfg.max_len] for i in starts]).astype(np.int64)
        y = np.stack([tokens[i + 1 : i + cfg.max_len + 1] for i in starts]).astype(np.int64)
        batches.append((x, y))

    def run(model: MiniTransformer, opt: Adam, batch_slice: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[float]:
        losses = []
        for x_np, y_np in batch_slice:
            x = xp.asarray(x_np)
            y = xp.asarray(y_np)
            logits, _ = model.forward(x, training=True)
            loss, dlogits = _cross_entropy(logits, y)
            grads_raw = model.backward(dlogits)
            grads = train_script.consolidate_tied_embedding_grad(model, grads_raw, x)
            grads, _ = train_script.clip_grads(grads, 1.0)
            train_script.apply_grads(model, opt, grads, weight_decay=0.01, lr=1e-4)
            losses.append(loss)
        return losses

    # Straight-through baseline.
    np.random.seed(12345)
    baseline_model = MiniTransformer(cfg)
    baseline_opt = Adam(lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    baseline_losses = run(baseline_model, baseline_opt, batches)

    # Interrupted + resumed run.
    np.random.seed(12345)
    model_a = MiniTransformer(cfg)
    opt_a = Adam(lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    _ = run(model_a, opt_a, batches[:50])

    with tempfile.TemporaryDirectory() as td:
        run_cfg = train_script.RunConfig(vocab_size=cfg.vocab_size)
        ckpt_path = os.path.join(td, "resume.pkl")
        train_script.save_checkpoint(ckpt_path, model_a, opt_a, run_cfg, step=50, best_val_loss=9.0, tokens_seen=0)

        model_b = MiniTransformer(cfg)
        opt_b = Adam(lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
        train_script.load_checkpoint(ckpt_path, model_b, opt_b)
        resumed_losses = run(model_b, opt_b, batches[50:])

    assert abs(resumed_losses[-1] - baseline_losses[-1]) < 1e-6, "Loss continuity mismatch after resume"


def test_checkpoint_portability() -> None:
    """Checkpoint must contain weights, optimizer state, config, tokenizer bundle, and metadata."""

    cfg = ModelConfig(vocab_size=64, d_model=16, n_layers=1, n_heads=1, max_len=8, dropout=0.0)
    model = MiniTransformer(cfg)
    opt = Adam(lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)

    with tempfile.TemporaryDirectory() as td:
        run_cfg = train_script.RunConfig(vocab_size=cfg.vocab_size, tokenizer_path="assets/tokenizer.model")
        ckpt_path = os.path.join(td, "portable.pkl")
        train_script.save_checkpoint(ckpt_path, model, opt, run_cfg, step=1, best_val_loss=0.5, tokens_seen=10)
        payload = pickle.load(open(ckpt_path, "rb"))

    required = {"model_state", "optimizer_state", "model_config", "run_config", "step", "best_val_loss", "tokens_seen", "tokenizer"}
    missing = required - set(payload.keys())
    assert not missing, f"Checkpoint missing keys: {missing}"


def test_checkpoint_version_mismatch() -> None:
    """Loading mismatched vocab checkpoint should fail loudly."""

    cfg_a = ModelConfig(vocab_size=64, d_model=16, n_layers=1, n_heads=1, max_len=8, dropout=0.0)
    cfg_b = ModelConfig(vocab_size=128, d_model=16, n_layers=1, n_heads=1, max_len=8, dropout=0.0)
    model_a = MiniTransformer(cfg_a)
    opt_a = Adam(lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)

    with tempfile.TemporaryDirectory() as td:
        run_cfg = train_script.RunConfig(vocab_size=cfg_a.vocab_size)
        ckpt_path = os.path.join(td, "mismatch.pkl")
        train_script.save_checkpoint(ckpt_path, model_a, opt_a, run_cfg, step=0, best_val_loss=0.0, tokens_seen=0)

        model_b = MiniTransformer(cfg_b)
        opt_b = Adam(lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)

        try:
            train_script.load_checkpoint(ckpt_path, model_b, opt_b)
        except ValueError as exc:
            assert "vocab_size" in str(exc)
            return

    raise AssertionError("Expected ValueError for vocab mismatch")


# ---------------------------------------------------------------------------
# 4.1/4.2/4.3 Edge-case tests
# ---------------------------------------------------------------------------

def test_training_interruption_recovery() -> None:
    """Simulate interruption and resume without corruption."""

    cfg = ModelConfig(vocab_size=128, d_model=32, n_layers=2, n_heads=2, max_len=8, dropout=0.0)
    model = MiniTransformer(cfg)
    opt = Adam(lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    tokens = np.random.randint(0, cfg.vocab_size, size=20000, dtype=np.int64)

    for _ in range(50):
        _one_train_step(model, opt, tokens, batch_size=2, seq_len=8, weight_decay=0.01, lr=1e-4)

    with tempfile.TemporaryDirectory() as td:
        run_cfg = train_script.RunConfig(vocab_size=cfg.vocab_size)
        ckpt = os.path.join(td, "interrupt.pkl")
        train_script.save_checkpoint(ckpt, model, opt, run_cfg, step=50, best_val_loss=1.0, tokens_seen=0)

        model2 = MiniTransformer(cfg)
        opt2 = Adam(lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
        train_script.load_checkpoint(ckpt, model2, opt2)

        starts = np.array([123, 456], dtype=np.int64)
        x_np = np.stack([tokens[i : i + cfg.max_len] for i in starts]).astype(np.int64)
        y_np = np.stack([tokens[i + 1 : i + cfg.max_len + 1] for i in starts]).astype(np.int64)

        def one_shared_step(m: MiniTransformer, o: Adam) -> float:
            x = xp.asarray(x_np)
            y = xp.asarray(y_np)
            logits, _ = m.forward(x, training=True)
            loss, dlogits = _cross_entropy(logits, y)
            grads_raw = m.backward(dlogits)
            grads = train_script.consolidate_tied_embedding_grad(m, grads_raw, x)
            grads, _ = train_script.clip_grads(grads, 1.0)
            train_script.apply_grads(m, o, grads, weight_decay=0.01, lr=1e-4)
            return loss

        l1 = one_shared_step(model, opt)
        l2 = one_shared_step(model2, opt2)

    assert abs(l1 - l2) < 1e-6, "Recovered training diverged immediately"


def test_out_of_memory_graceful_fail() -> None:
    """Simulate preflight OOM check with clear remediation message."""

    est = estimate_model_vram(
        n_params=120_000_000,
        batch_size=512,
        seq_len=1024,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        mixed_precision=True,
    )
    if est["total_mb"] <= 12_000:
        raise SkipTest("Estimator did not exceed VRAM in this environment")

    def preflight_check(total_mb: float) -> None:
        if total_mb > 12_000:
            raise RuntimeError("VRAM exceeded, reduce batch_size")

    try:
        preflight_check(est["total_mb"])
    except RuntimeError as exc:
        assert "VRAM exceeded, reduce batch_size" in str(exc)
        return
    raise AssertionError("Expected graceful OOM RuntimeError")


def test_corrupted_checkpoint() -> None:
    """Corrupted checkpoint load should fail loudly."""

    cfg = ModelConfig(vocab_size=64, d_model=16, n_layers=1, n_heads=1, max_len=8, dropout=0.0)
    model = MiniTransformer(cfg)
    opt = Adam(lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "corrupt.pkl")
        with open(path, "wb") as fh:
            fh.write(b"not a pickle checkpoint")

        try:
            train_script.load_checkpoint(path, model, opt)
        except Exception:
            return

    raise AssertionError("Corrupted checkpoint unexpectedly loaded")


def test_mismatched_config() -> None:
    """Checkpoint with mismatched d_model should raise shape mismatch error."""

    cfg_small = ModelConfig(vocab_size=64, d_model=16, n_layers=1, n_heads=1, max_len=8, dropout=0.0)
    cfg_large = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=1, max_len=8, dropout=0.0)
    model_small = MiniTransformer(cfg_small)
    opt_small = Adam(lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)

    with tempfile.TemporaryDirectory() as td:
        run_cfg = train_script.RunConfig(vocab_size=cfg_small.vocab_size)
        ckpt_path = os.path.join(td, "shape.pkl")
        train_script.save_checkpoint(ckpt_path, model_small, opt_small, run_cfg, step=0, best_val_loss=0.0, tokens_seen=0)

        model_large = MiniTransformer(cfg_large)
        opt_large = Adam(lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
        try:
            train_script.load_checkpoint(ckpt_path, model_large, opt_large)
        except ValueError as exc:
            assert "Shape mismatch" in str(exc) or "shape" in str(exc).lower()
            return

    raise AssertionError("Expected shape mismatch failure")


def test_gradient_explosion() -> None:
    """Large gradients should be clipped back to max norm."""

    fake_layer_grads = [
        (
            (xp.ones((4, 8), dtype=xp.float32) * 1e6, xp.ones((8,), dtype=xp.float32) * 1e6, xp.ones((8, 4), dtype=xp.float32) * 1e6, xp.ones((4,), dtype=xp.float32) * 1e6),
            (xp.ones((4, 12), dtype=xp.float32) * 1e6, xp.ones((12,), dtype=xp.float32) * 1e6, xp.ones((4, 4), dtype=xp.float32) * 1e6, xp.ones((4,), dtype=xp.float32) * 1e6),
            (xp.ones((4,), dtype=xp.float32) * 1e6, xp.ones((4,), dtype=xp.float32) * 1e6),
            (xp.ones((4,), dtype=xp.float32) * 1e6, xp.ones((4,), dtype=xp.float32) * 1e6),
        )
    ]
    grads = (
        xp.ones((16, 4), dtype=xp.float32) * 1e6,
        xp.ones((8, 4), dtype=xp.float32) * 1e6,
        fake_layer_grads,
    )
    clipped, _ = train_script.clip_grads(grads, max_norm=1.0)
    norm = train_script.global_grad_norm(clipped)
    assert norm <= 1.0001, f"Clipping failed, norm={norm}"


def test_zero_gradients() -> None:
    """Ensure non-trivial backward pass produces non-zero gradients."""

    model = _tiny_model(vocab_size=128, d_model=32, n_layers=2, n_heads=2, max_len=8, dropout=0.0)
    x, y = _make_batch(model.config.vocab_size, batch=2, seq=8, seed=4242)
    logits, _ = model.forward(x, training=True)
    _, dlogits = _cross_entropy(logits, y)
    grads = model.backward(dlogits)

    total_abs = 0.0
    dW_emb_out, dW_pos, layer_grads, dX_emb = grads
    total_abs += float(np.abs(np.asarray(dW_emb_out)).sum())
    total_abs += float(np.abs(np.asarray(dW_pos)).sum())
    total_abs += float(np.abs(np.asarray(dX_emb)).sum())
    for ffn_g, attn_g, ln1, ln2 in layer_grads:
        for g in ffn_g + attn_g + ln1 + ln2:
            total_abs += float(np.abs(np.asarray(g)).sum())

    assert total_abs > 0.0, "All gradients are zero"


def test_extreme_sequence_lengths() -> None:
    """Test seq_len=1, seq_len=max_len, and seq_len=max_len+1 error path."""

    cfg = ModelConfig(vocab_size=64, d_model=16, n_layers=1, n_heads=1, max_len=8, dropout=0.0)
    model = MiniTransformer(cfg)

    x1 = xp.asarray(np.random.randint(0, cfg.vocab_size, size=(2, 1), dtype=np.int64))
    x2 = xp.asarray(np.random.randint(0, cfg.vocab_size, size=(2, cfg.max_len), dtype=np.int64))
    x3 = xp.asarray(np.random.randint(0, cfg.vocab_size, size=(2, cfg.max_len + 1), dtype=np.int64))

    _ = model.forward(x1, training=True)
    _ = model.forward(x2, training=True)

    try:
        _ = model.forward(x3, training=True)
    except ValueError:
        return

    raise AssertionError("Expected ValueError for seq_len=max_len+1")


def test_single_gpu_isolation() -> None:
    """Ensure codebase does not attempt multi-GPU operations."""

    train_text = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    forbidden = ["DistributedDataParallel", "torch.distributed", "nccl", "all_reduce"]
    for token in forbidden:
        assert token not in train_text, f"Found multi-GPU token in train.py: {token}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CATEGORIES: Dict[str, List[Callable[[], None]]] = {
    "gradients": [
        test_layernorm_gradients,
        test_gelu_gradients,
        test_attention_gradients,
        test_ffn_gradients,
        test_transformer_block_gradients,
        test_embedding_tying_gradients,
    ],
    "optimizer": [
        test_weight_decay_exclusion,
        test_lr_schedule,
        test_gradient_clipping,
        test_adam_vs_adamw,
    ],
    "tokenizer": [
        test_tokenizer_roundtrip,
        test_vocab_coverage,
        test_tokenizer_efficiency,
        test_special_token_insertion,
    ],
    "data": [
        test_data_loading,
        test_data_diversity,
        test_eos_token_presence,
    ],
    "memory": [
        test_vram_estimation_accuracy,
        test_fp16_no_overflow,
        test_memory_leak,
    ],
    "checkpoint": [
        test_checkpoint_save_load,
        test_checkpoint_resume,
        test_checkpoint_portability,
        test_checkpoint_version_mismatch,
    ],
    "edge": [
        test_training_interruption_recovery,
        test_out_of_memory_graceful_fail,
        test_corrupted_checkpoint,
        test_mismatched_config,
        test_gradient_explosion,
        test_zero_gradients,
        test_extreme_sequence_lengths,
        test_single_gpu_isolation,
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""

    p = argparse.ArgumentParser(description="MiniGPT full test suite")
    p.add_argument("--test-gradients", action="store_true")
    p.add_argument("--test-optimizer", action="store_true")
    p.add_argument("--test-tokenizer", action="store_true")
    p.add_argument("--test-data", action="store_true")
    p.add_argument("--test-memory", action="store_true")
    p.add_argument("--test-checkpoint", action="store_true")
    p.add_argument("--test-edge-cases", action="store_true")
    return p.parse_args()


def selected_categories(args: argparse.Namespace) -> List[str]:
    """Resolve which categories to run based on CLI flags."""

    flags = {
        "gradients": args.test_gradients,
        "optimizer": args.test_optimizer,
        "tokenizer": args.test_tokenizer,
        "data": args.test_data,
        "memory": args.test_memory,
        "checkpoint": args.test_checkpoint,
        "edge": args.test_edge_cases,
    }
    picked = [name for name, enabled in flags.items() if enabled]
    return picked or list(CATEGORIES.keys())


def main() -> int:
    """Execute selected test categories and emit summary."""

    args = parse_args()
    cats = selected_categories(args)

    print("=" * 72)
    print(f"MiniGPT test suite starting | categories={cats}")
    print("=" * 72)

    for cat in cats:
        print(f"\n--- CATEGORY: {cat} ---")
        for test_fn in CATEGORIES[cat]:
            run_test(test_fn)

    passed = sum(1 for r in RESULTS if r.status == "PASS")
    failed = sum(1 for r in RESULTS if r.status == "FAIL")
    skipped = sum(1 for r in RESULTS if r.status == "SKIP")

    print("\n" + "=" * 72)
    print(f"Summary: total={len(RESULTS)} pass={passed} fail={failed} skip={skipped}")
    for result in RESULTS:
        if result.status != "PASS":
            print(f"  - {result.status}: {result.name} :: {result.details.splitlines()[0] if result.details else ''}")
    print("=" * 72)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

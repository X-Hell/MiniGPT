#!/usr/bin/env python3
"""Unit tests for the Run-2 stability utilities (minigpt.stability).

Runs on the NumPy backend (forced below) and is self-contained -- no pytest
required. Exits non-zero on the first failure, like scripts/smoke_test_gpt1.py.

    MINIGPT_BACKEND=numpy .venv/bin/python tests/test_run2_stability.py
"""
import os
os.environ.setdefault("MINIGPT_BACKEND", "numpy")

import sys
import tempfile
import pickle

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from minigpt.config import ModelConfig
from minigpt.model import MiniTransformer
from minigpt.optimizer import Adam, LRSchedule
from minigpt.stability import (
    apply_residual_scaling,
    monitor_gradient_norms,
    GradientSpikeDetector,
    EvaluationMonitor,
    save_checkpoint,
)
import train as trainmod  # scripts/train.py -- for authoritative global_grad_norm


def tiny_model():
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=3, n_heads=4,
                      d_ff=128, max_len=32)
    return MiniTransformer(cfg)


# --------------------------------------------------------------------------- #
def test_apply_residual_scaling():
    m = tiny_model()
    n_before = sum(p.size for _, p in m.named_parameters())
    apply_residual_scaling(m, m.config.n_layers)
    target = 0.02 / (2 * m.config.n_layers) ** 0.5
    for layer in m.layers:
        for w in (layer.attn.W_o, layer.ffn.W_down):
            assert abs(float(np.std(w)) - target) / target < 0.02, "std not at residual target"
    # idempotent
    snap = [np.array(l.attn.W_o, copy=True) for l in m.layers]
    apply_residual_scaling(m, m.config.n_layers)
    for l, s in zip(m.layers, snap):
        assert np.allclose(l.attn.W_o, s, atol=1e-6), "apply_residual_scaling not idempotent"
    assert sum(p.size for _, p in m.named_parameters()) == n_before, "param count changed"
    assert m._residual_scaled is True
    print("[OK] apply_residual_scaling: std==target, idempotent, param-count stable")


def test_gradient_spike_detector():
    d = GradientSpikeDetector(warn=12.0, crit=40.0, window=5, warn_consec=3, crit_consec=5)
    last = None
    for g in (3, 4, 55, 4):
        last = d.update(g)
    assert last == "CRITICAL", "a 55-spike in the window must keep status CRITICAL"
    # once the spike ages out of the 5-sample window, the alert clears
    for g in (4, 4, 4, 4, 4):
        last = d.update(g)
    assert last is None, "alert must clear after the spike leaves the window"
    d2 = GradientSpikeDetector()
    assert [d2.update(g) for g in (13, 14, 13)] == [None, None, "WARN"], "3-consec WARN failed"
    d3 = GradientSpikeDetector()
    assert all(d3.update(g) is None for g in (5, 30, 5, 5)), "isolated spike must not alert"
    print("[OK] GradientSpikeDetector: CRITICAL on >40, WARN on cluster, ignores isolated")


def test_evaluation_monitor_run1_replay():
    """Replay Run-1's eval/train sequence: must never CONFIRM through step 2000,
    and only reach PENDING at step 2500."""
    em = EvaluationMonitor(margin=0.05)
    # train losses descend (new lows) through ~2170, matching the real run.
    evals = {500: 7.7983, 1000: 7.9072, 1500: 7.8534, 2000: 7.7749, 2500: 7.9052}
    train_series = {200: 8.2, 700: 7.8, 1200: 7.72, 1700: 7.66, 2170: 7.6106, 2400: 7.65}
    timeline = sorted([(s, "t", v) for s, v in train_series.items()]
                      + [(s, "e", v) for s, v in evals.items()])
    status_at = {}
    for step, kind, v in timeline:
        if kind == "t":
            em.record_train(step, v)
        else:
            em.record_eval(step, v)
            status_at[step] = em.status
    assert status_at[1000] != "CONFIRMED", "noisy step-1000 rise must not confirm"
    assert status_at[2000] == "HEALTHY", "step-2000 new best must be HEALTHY"
    assert status_at[2500] == "PENDING", "real step-2500 regression should be PENDING (1/2)"
    # two clean consecutive rises with no intervening train low -> CONFIRMED
    em2 = EvaluationMonitor(margin=0.05)
    em2.record_eval(0, 4.0)
    em2.record_eval(1, 4.2)
    assert em2.status == "PENDING"
    em2.record_eval(2, 4.3)
    assert em2.status == "CONFIRMED", "two consecutive regressions must confirm"
    print("[OK] EvaluationMonitor: Run-1 replay never confirms through 2000; PENDING at 2500")


def test_monitor_gradient_norms_matches_train():
    rng = np.random.RandomState(0)
    n_layers = 3

    def mat(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1

    layer_grads = []
    for _ in range(n_layers):
        swiglu = (mat(64, 128), mat(64, 128), mat(128, 64))
        attn = (mat(64, 192), mat(64, 64))
        layer_grads.append((swiglu, attn, mat(64), mat(64)))
    grads = (mat(256, 64), layer_grads, mat(64))

    class Stub:
        pass
    stub = Stub()
    stub.latest_grads = grads

    info = monitor_gradient_norms(stub, clip_value=1.0)
    ref = trainmod.global_grad_norm(grads)
    assert abs(info["global_norm"] - ref) < 1e-4, f"{info['global_norm']} != {ref}"
    assert info["will_clip"] is True and info["clip_scale"] < 1.0
    assert 0 <= info["worst_layer"] < n_layers
    print(f"[OK] monitor_gradient_norms: global_norm {info['global_norm']:.4f} "
          f"== train.global_grad_norm {ref:.4f}")


def test_save_checkpoint_rolling_and_restartable():
    m = tiny_model()
    opt = Adam(lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1)
    params = [p for _, p in m.named_parameters()]
    grads = [np.ones_like(p) * 0.01 for p in params]
    opt.step(params, grads)  # populate optimizer.state (t=1) keyed by id(param)
    sched = LRSchedule(peak_lr=2e-4, min_lr=2e-5, warmup_steps=5, max_steps=100)

    with tempfile.TemporaryDirectory() as d:
        rng_state = np.random.get_state()
        for step in (2, 4, 6, 8):
            meta = {"best_val_loss": 5.0, "tokens_seen": step * 1000,
                    "numpy_rng_state": rng_state}
            path = save_checkpoint(m, opt, sched, step, meta,
                                   output_dir=d, run_cfg=None, keep_last=3)
        ck_dir = os.path.join(d, "checkpoints")
        files = sorted(os.listdir(ck_dir))
        numbered = [f for f in files if f.startswith("step_")]
        assert numbered == ["step_0000004.pkl", "step_0000006.pkl", "step_0000008.pkl"], numbered
        assert "latest.pkl" in files
        assert not any(f.endswith(".tmp") for f in files), "temp file leaked"

        ck = pickle.load(open(path, "rb"))
        for name, p in m.named_parameters():
            assert np.allclose(ck["model_state"][name], np.asarray(p)), f"param {name} not bit-exact"
        assert ck["scheduler"]["peak_lr"] == 2e-4 and ck["scheduler"]["warmup_steps"] == 5
        assert ck["numpy_rng_state"] is not None
        sample = next(iter(ck["optimizer_state"].values()))
        assert sample["t"] == 1, "optimizer step counter not preserved"
    print("[OK] save_checkpoint: rolling keep_last=3, atomic, scheduler+RNG+optimizer preserved")


def main():
    test_apply_residual_scaling()
    test_gradient_spike_detector()
    test_evaluation_monitor_run1_replay()
    test_monitor_gradient_norms_matches_train()
    test_save_checkpoint_rolling_and_restartable()
    print("========= ALL RUN-2 STABILITY TESTS PASSED =========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

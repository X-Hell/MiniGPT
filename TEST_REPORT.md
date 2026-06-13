# MiniGPT Test Report
**Date:** 2026-04-17
**Commit:** fb87226

## Summary
- Total tests run: 32
- Passed: 30
- Failed: 0
- Skipped: 2
- Smoke test: PARTIAL PASS (10-step validated; 500-step blocked by CPU-only runtime)
- Production config draft: `batch_size=16`, `gradient_accumulation_steps=4`, `vocab_size=40000` (`configs/gpt1_72h.yaml`)

## Test Results by Category

### Gradient Validation
- `python3 tests/test_all.py --test-gradients` -> PASS (6/6)
- `python3 scripts/validate_gradients.py` -> PASS (all monitored params under configured thresholds)
- Key observation: deeper `W_qkv`/`W_proj` parameters show higher finite-difference p90 but remain within relaxed known-noise bounds.

### Optimizer
- `python3 tests/test_all.py --test-optimizer` -> PASS (4/4)
- Verified:
  - Coupled Adam+L2 behavior (not decoupled AdamW)
  - Weight-decay exclusion for LN gamma/beta, all biases, and `W_pos`
  - Global-norm clipping at 1.0
  - LR schedule key points: step0=0, step2000=peak, step100000=min
- LR schedule plot generated at `tests/lr_schedule.png`.

### Tokenizer
- `python3 tests/test_all.py --test-tokenizer` -> PASS with 1 skip
- Roundtrip and efficiency checks pass for byte-level BPE path.
- Skip reason: `HFBPETokenizer` runtime dependency (`tokenizers`) not installed in this environment.

### Data Pipeline
- `python3 tests/test_all.py --test-data` -> PASS (3/3)
- 1000-batch load test passes shape/range checks.
- Diversity and EOS-presence checks pass on synthetic pipeline fixtures.

### VRAM & Memory
- `python3 tests/test_all.py --test-memory` -> PASS with 1 skip
- FP16 overflow guard and CPU memory leak checks pass.
- Skip reason: no active CUDA backend, so measured-vs-estimated VRAM accuracy could not be validated on GPU.

### Checkpoints
- `python3 tests/test_all.py --test-checkpoint` -> PASS (4/4)
- Save/load equivalence, resume continuity, portability payload keys, and vocab mismatch fail-loud behavior verified.

### Edge Cases
- `python3 tests/test_all.py --test-edge-cases` -> PASS (8/8)
- Includes interruption recovery, corrupted checkpoint handling, config mismatch, clipping under gradient explosion, zero-gradient sanity, sequence length boundaries, and single-GPU isolation checks.

## Smoke Test Results
Command executed:
```bash
python3 scripts/train.py \
  --config configs/smoke_test.yaml \
  --steps 10 \
  --batch_size 4 \
  --log_interval 1 \
  --checkpoint_interval 5 \
  --output_dir outputs/smoke_test/
```

- Initial loss: 9.5834
- Final loss (step 10): 7.6746
- Tokens/sec: ~1.5K–2.0K tok/s on CPU (NumPy backend)
- Checkpoints: `step_0000005.pkl`, `step_0000010.pkl`, `final.pkl`
- Log file: `outputs/smoke_test/training.log`

## Known Issues
- Full 500-step smoke run at `d=384, L=6, H=6, T=512, batch=32, accum=2` is not practical on CPU-only environment; throughput observed ~396 tok/s at that scale.
- `tokenizers` package is missing; HF tokenizer special-token coverage test is skipped.
- GPU-specific VRAM accuracy check is skipped because CUDA/NVIDIA runtime is unavailable in this workspace.

## Recommendations
1. Install `tokenizers` and rerun tokenizer coverage test.
2. Run memory/VRAM validation on RTX 3060 with `MINIGPT_BACKEND=cupy`.
3. Execute full 500-step smoke test on GPU using `configs/smoke_test.yaml` before launch.
4. Confirm 72h launch with `scripts/launch_72h_training.sh` after GPU preflight checks pass.

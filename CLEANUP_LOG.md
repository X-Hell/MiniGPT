# Cleanup Log

## Date
- 2026-04-17

## Removed / Refactored Dead or Stale Code
- Replaced `scripts/train.py` (legacy RMSNorm/SwiGLU-era trainer references) with GPT-1-compatible training pipeline.
- Removed stale imports in training path (`requests`, `precompute_freqs_cis`) and obsolete parameter references (`W_gate`, `W_up`, `W_down`, `ln_f`).
- Replaced `scripts/benchmark_gpu.py` with GPT-1-compatible benchmark logic that matches current `model.backward()` outputs.
- Replaced `scripts/validate_gradients.py` with GPT-1-compatible gradient checks.
- Replaced `scripts/rag_chat.py` to eliminate broken RoPE-context patch path and missing `precompute_freqs_cis` dependency.
- Added regex fallback in `src/minigpt/tokenizer.py` for environments without the `regex` package to remove broken `\p{...}` runtime path.

## Compatibility Cleanups
- Added `Config` dataclass in `src/minigpt/config.py` for legacy scripts expecting `Config()`.
- Added `get_device_info()` alias in `src/minigpt/backend.py` for legacy diagnostics.
- Added missing public-function type hints/docstrings across `src/minigpt/` modules (model/backend/tokenizer/trainer/rag/inference compatibility surfaces).

## Behavior Fixes Included in Cleanup
- Updated LR schedule to start at step-0 LR = 0 and hit peak at warmup boundary.
- Updated `estimate_model_vram()` FFN ratio from SwiGLU-era approximation to GPT-1 `d_ff = 4 * d_model`.
- Added explicit sequence overflow check in `MiniTransformer.forward()` for `seq_len > max_len` safety.

## Notes
- Legacy scripts not on the 72-hour critical path (`generate_curated_data.py`, `generate_synthetic.py`) were left intact to preserve dataset-generation workflows.

---

## Date
- 2026-05-25

## Additional Cleanup (Ampere Prep)
- Removed unreferenced legacy module `src/minigpt/trainer.py` (not used by active `scripts/train.py` pipeline).
- Removed unused imports:
  - `src/minigpt/model.py` (`Union`)
  - `scripts/train.py` (`Iterable`)
  - `scripts/generate.py` (`Optional`)
  - `scripts/generate_curated_data.py` (`random`)
  - `scripts/smoke_test_gpt1.py` (`TrainConfig`, `Adam`)
- Purged stale runtime artifacts:
  - all files under `checkpoints/`
  - all files under `outputs/`
  - `scripts/__pycache__/`, `src/minigpt/__pycache__/`, and all `*.pyc`
  - macOS metadata file `.DS_Store`
- Updated `README.md` project tree to remove deleted `trainer.py`.

## Ampere Runtime Optimization
- Updated `scripts/launch_72h_training.sh` to export Ampere/CuPy-friendly runtime flags before launch:
  - `MINIGPT_BACKEND=cupy`
  - `CUPY_TF32=1`
  - `CUPY_ACCELERATORS=\"cub,cutensor\"`

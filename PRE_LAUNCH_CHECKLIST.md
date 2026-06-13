# 72-Hour Training Pre-Launch Checklist

Run through this checklist immediately before launching. Every box must be checked.

## Code Quality
- [ ] All tests in `tests/test_all.py` pass
- [ ] Smoke test (500 steps) completed successfully
- [ ] No debug print statements in production code
- [ ] No dead code remains in `src/minigpt/`
- [ ] All public functions have type hints and docstrings

## Configuration
- [ ] `configs/gpt1_72h.yaml` exists and is complete
- [ ] `batch_size` and `gradient_accumulation_steps` calculated for 12GB VRAM
- [ ] Effective batch size = 64 (or documented replacement)
- [ ] Total steps = 100,000
- [ ] Checkpoint interval = 5,000
- [ ] FP16 enabled

## Data Pipeline
- [ ] FineWeb-Edu dataset accessible and streaming works
- [ ] Tokenizer trained with vocab_size=40K (or fallback documented)
- [ ] `<eos>` token correctly inserted at document boundaries
- [ ] Data loader tested for 1000 batches without errors

## Hardware
- [ ] `nvidia-smi` shows RTX 3060 with 12GB VRAM
- [ ] GPU not running other processes (check `nvidia-smi`)
- [ ] At least 200GB free disk space for checkpoints
- [ ] At least 32GB system RAM
- [ ] CUDA drivers up to date

## Monitoring
- [ ] `scripts/monitor_training.py` tested and working
- [ ] WandB or TensorBoard configured (optional)
- [ ] Log directory has write permissions

## Backup & Recovery
- [ ] Checkpoint save/load tested
- [ ] Resumption from checkpoint tested
- [ ] Output directory is on a disk with backups
- [ ] Training PID will be saved for emergency stop

## Validation Metrics
- [ ] Gradient checks pass on all modules
- [ ] Initial loss ≈ ln(vocab_size)
- [ ] VRAM estimate matches actual usage within 10%
- [ ] Tokens/sec measured for target hardware

## Emergency Contacts
- [ ] You know how to kill training: `kill $(cat outputs/*/training.pid)`
- [ ] You know how to resume: rerun launch script with `--resume_from_checkpoint`

## Final Actions Before Launch
- [ ] Commit all code changes
- [ ] Create git tag: `v1.0-gpt1-training-start`
- [ ] Close all other GPU applications
- [ ] Disable system sleep/hibernate

**All boxes checked? Ready to launch:** `./scripts/launch_72h_training.sh`

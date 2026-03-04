# Phase 1 (Math Foundations) — 7-Day Build Checklist

Goal: finish a small neural-net foundation using only NumPy and your current project structure.

## Day 1 — Linear Algebra Core
- Implement and test matrix multiplication, transpose, and shape checks in [src/math/linear_algebra.py](src/math/linear_algebra.py).
- Add quick sanity script in [scripts/run_phase1_demo.py](scripts/run_phase1_demo.py).
- Compare outputs with NumPy reference values.

**Done when**
- `matmul` works for 2D matrices and basic batched cases.
- You can run `python scripts/run_phase1_demo.py` without errors.

## Day 2 — Numerically Stable Activations
- Implement `stable_softmax` thoroughly (already stubbed) and add `log_softmax` helper in [src/math/linear_algebra.py](src/math/linear_algebra.py).
- Validate behavior for large positive/negative values.
- Add a tiny test module in [tests](tests) for overflow/underflow checks.

**Done when**
- Softmax sums to 1 across the last axis.
- Extreme values do not produce `nan` or `inf`.

## Day 3 — Loss Functions
- Add cross-entropy and MSE loss utilities (NumPy only) in [src/math](src/math).
- Create toy classification data inside [experiments](experiments) and run a forward loss computation.
- Log loss values for 10 random batches.

**Done when**
- Loss decreases on a trivial synthetic dataset after manual parameter nudges.
- You can explain cross-entropy vs MSE usage in one short note.

## Day 4 — Backprop from Scratch (Core)
- Expand [src/math/autodiff.py](src/math/autodiff.py) from `Value` placeholder into minimal scalar autodiff (`+`, `*`, optional `tanh`/`relu`).
- Implement topological backward pass.
- Reproduce a tiny computational graph gradient check by finite differences.

**Done when**
- Analytic gradients match numerical gradients (small tolerance, e.g., < 1e-4).
- At least 3 graph examples pass gradient checks.

## Day 5 — Mini Neural Net (NumPy)
- Build a 2-layer MLP (forward + backward) in a new file under [src/model](src/model).
- Add SGD step and track train loss per epoch.
- Use tiny synthetic data first (e.g., XOR or 2-class blobs).

**Done when**
- Loss consistently decreases over epochs.
- Model achieves strong accuracy on toy data (target: >90% on simple set).

## Day 6 — Training Loop Hygiene
- Add gradient clipping helper and learning-rate schedule function in [src/training/optim.py](src/training/optim.py).
- Improve [src/training/trainer.py](src/training/trainer.py) to report metrics each step/epoch.
- Save run artifacts to [experiments](experiments) (loss curves as CSV is enough).

**Done when**
- Training script runs end-to-end for at least 100+ steps.
- Metrics are reproducible with fixed random seed.

## Day 7 — Consolidation + Validation
- Refactor duplicated math code, clean naming, and add docstrings.
- Add/expand tests in [tests](tests) for math ops, losses, and gradient checks.
- Update [docs/phases.md](docs/phases.md) with what worked, what failed, and next steps for Phase 2.

**Done when**
- `python scripts/run_phase1_demo.py` succeeds.
- Core tests pass for your Phase 1 modules.
- You have a short retrospective and a clear handoff into tokenizer work.

---

## Suggested Daily Time Box
- 60–90 min implementation
- 20–30 min debugging
- 15 min notes + commit-quality cleanup

## Output of Week 1
- Reliable NumPy math primitives
- Basic autodiff understanding (and implementation)
- First working neural network training loop
- Ready to start Phase 2 (BPE tokenizer)

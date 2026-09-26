# Release check — 2026-09-27

This maintenance update does not change the trained weights, architecture,
loss formulas, training schedule, or reported benchmark values.

## Checks performed

- All 13 unittest cases passed with PyTorch 2.5.1+cu124 on CPU.
- All three checkpoint SHA-256 hashes match the existing Table 1 verification record.
- All three checkpoints load with `strict=True` and restricted
  `torch.load(..., weights_only=True)`.
- Precision and correction parameter counts are 18,768 and 28,066.
- Every model output tensor is bitwise equal to the pre-update model on a
  seeded synthetic complex input for each of the three checkpoints.
- Silent-input inference produces finite outputs; the Cartesian correction
  bounds hold on the tested synthetic inputs.
- Existing loss-value and staged-training regression tests still pass.
- A synthetic-data CLI smoke test passed: one update in each training stage,
  best-checkpoint saving, restricted checkpoint reload, and finite mono WAV
  inference with the expected length and sampling rate. This checks workflow
  execution only, not enhancement quality or convergence.

## Corrections

- The architecture illustration now labels precision convolution width as
  **32**, while its temporal GRU remains **48**.
- Documented frequency BiGRU hidden size **24 per direction**, 48-to-32
  projection, residual addition and the final two-coordinate linear head.
- Added explicit invalid-RTF, audio and frequency-shape checks.
- Rejected `--batch-size 0` instead of silently selecting the default.
- Removed obsolete submission-status wording and added manuscript BibTeX.

This is a software regression check, not a new complete training run or a
remeasurement of Table 1 quality/runtime. The original restricted benchmark
is still not independently reconstructible from the public release.

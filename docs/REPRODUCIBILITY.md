# Model and training

## Architecture

Four channels at 16 kHz; 512-point FFT, 400-sample square-root Hann window,
160-sample hop, centered STFT. Input spectrum: `[B,4,257,T]`;
enhanced spectrum: `[B,257,T]`.

| Module | Layers | Parameters |
| --- | --- | ---: |
| Precision | 16 features → 1×1 convolution (32) → four convolutional blocks → temporal GRU (48) → linear Cholesky head (16) | 18,768 |
| Correction | 18 features → 1×1 convolution (32) → four convolutional blocks → frequency BiGRU (24 per direction) → projection (48→32) + residual addition → temporal GRU (48) → linear head (2) | 28,066 |

Convolutional blocks use 3×3 depthwise-separable convolutions, temporal dilations
1, 2, 4, 8, and left-only time padding. The BiGRU runs along frequency.
The two output coordinates define
`delta = 4 * max(abs(z), 1e-8) * (tanh(h_real) + j*tanh(h_imag))`;
the enhanced spectrum is `z + delta`. The residual-derived scale `v` is
an input feature, not the output multiplier. Its internal key
`likelihood_variance` does not imply an exact posterior variance.

The analytic estimate preserves the modeled target under matching calibration;
this constraint does not guarantee distortion-free neural correction.
The released audio path processes whole files.

## Training settings

| Stage | Updates | Batch size | Learning rate | Run-1 seed |
| --- | ---: | ---: | ---: | ---: |
| Spatial estimator | 600 | 3 | 0.002 | 20260816 |
| Corrector; spatial estimator frozen | 800 | 4 | 0.002 | 20260813 |
| Joint fine-tuning | 600 | 3 | 0.0005 | 20260817 |

Runs 2 and 3 add 10,000 and 20,000 to each seed. All stages use AdamW,
weight decay 0.0001 and gradient-norm clipping at 5.
Validation uses ten batches every 100 updates: fresh batches in stages 1–2,
fixed batches in stage 3, with provider seed equal to the stage seed + 10,000.
The best validation SI-SDR-improvement checkpoint initializes the next stage;
the best joint checkpoint is the final output.

Numerical floors and feature clipping are retained in code. In particular,
the correction loss floors `v` at `1e-8`; joint scale calibration floors
`v` at `1e-10` and target power at `1e-8`.

## Online data boundary

For a custom online STFT generator:

```bash
python scripts/train.py --batch-factory my_batches:make_batchers --rtf data/target_rtf.npy --seed-index 1 --output outputs/sarc.pt
```

Supply `make_batchers(stage, train_seed, validation_seed, config, target_transfer)`
in your own module. It returns separate training and validation providers,
each exposing `segment_samples` and `sample_batch(batch_size)`.
A batch contains complex tensors `mixture [B,4,257,T]` and `target [B,257,T]`.

The NPZ adapter and this interface do not include the restricted original
generator. Training with substitute data does not guarantee the reported scores.

## Checkpoints and verification

Three frozen checkpoints are included. Their hashes and original-benchmark
rescoring results are in [the verification record](../results/table1_sarc_verification.json).
Those scores match the SARC row of Table 1 at the reported precision using
the original shared metric-scaling protocol. This is not independent public
benchmark reconstruction or a new from-scratch training result.

Tests cover architecture dimensions, checkpoint integrity, analytic constraints,
finite inference, and staged-training/loss regressions.

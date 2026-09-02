# Reproducibility notes

## Frozen setup

- Four microphones, fixed channel order.
- Sampling rate: 16 kHz.
- STFT: 512-point FFT, 400-sample square-root Hann window, 160-sample hop.
- Model size: 46,834 trainable parameters during joint training.
- Three optimization stages: analytic estimator, correction network, joint fine-tuning.
- Main results use three independent frozen checkpoints.

## Exact correction used by the released checkpoints

The correction network predicts two Cartesian coordinates and applies

```text
delta = 4 * abs(z) * (tanh(h_real) + j * tanh(h_imag))
S_hat = z + delta
```

The variance proxy `v` is included among the correction-network features, but
the main-result checkpoints do not use `sqrt(v)` as the output scale.

## Reported results

Machine-readable copies of the main comparison and ablation tables are in
`results/main_comparison.csv` and `results/ablation.csv`. The private evaluation
audio is not included.


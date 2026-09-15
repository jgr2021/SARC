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

## Table 1 verification (2026-09-15)

The three public checkpoints were rerun through the public model on all 80
original evaluation mixtures. SI-SDR improvement, STOI, and wideband PESQ
match the SARC row of Table 1 at the reported three-decimal precision:

| Metric | Recomputed mean | Sample SD | Table 1 |
| --- | ---: | ---: | --- |
| SI-SDR improvement (dB) | 17.527482 | 0.053008 | 17.527 +/- 0.053 |
| STOI | 0.961038 | 0.001436 | 0.961 +/- 0.001 |
| PESQ | 2.729519 | 0.057336 | 2.730 +/- 0.057 |

Checkpoint hashes and per-seed results are recorded in
[`results/table1_sarc_verification.json`](../results/table1_sarc_verification.json).
The check uses the original scoring protocol, including its shared metric
scaling across method outputs. Small floating-point waveform differences exist
relative to archived outputs; the reported table precision is preserved.
This verification covers the SARC quality scores, not new baseline training
or runtime measurements. The original restricted test data remains unavailable
for independent public reconstruction.

## Training recipe from the successful experiments

`scripts/train.py` follows the experiment's three stages rather than treating
the paper's abbreviated equations as an implementation specification.

| Stage | Updates | Batch size | Learning rate | Seed for run 1 |
| --- | ---: | ---: | ---: | ---: |
| Spatial estimator only | 600 | 3 | 0.002 | 20260816 |
| Corrector; spatial estimator frozen | 800 | 4 | 0.002 | 20260813 |
| Both networks jointly | 600 | 3 | 0.0005 | 20260817 |

Run 2 and run 3 add 10,000 and 20,000, respectively, to each stage's seed.
Each stage uses AdamW with weight decay 0.0001 and gradient-norm clipping at 5.
Validation occurs every 100 updates using ten batches. The first two stages
draw fresh validation batches per check; the joint stage reuses ten fixed
batches. The validation provider seed is the stage seed plus 10,000.
The best validation SI-SDR-improvement checkpoint initializes the next stage.
The final output is the best joint checkpoint, not automatically the last one.
Per-stage checkpoints and a validation history are saved beside the output.

Numerical operations from the experiment are retained. In particular:

- RMS, magnitude and denominator floors and log-feature clipping remain in
  the model even where omitted from the manuscript for readability.
- The correction loss uses `v.clamp_min(1e-8)`; joint scale calibration uses
  `v.clamp_min(1e-10)` and a target-power floor of `1e-8`.
- Cholesky precision remains `L @ L.conj().transpose(-1, -2)`, and correction
  remains scaled by `4 * abs(z)`, as used by the released weights.

Regression tests include loss values computed by the archived experimental
functions, including low-amplitude inputs that exercise the numerical floors.
An additional local comparison of initialization, losses, gradients and one
AdamW update is recorded in
[`results/table1_training_parity.json`](../results/table1_training_parity.json).
These checks establish agreement of the tested training computations, not a
new complete training run reproducing the Table 1 scores.

### Online data boundary

The successful experiments consumed complex STFT batches from an online
generator. To connect a compatible generator without converting those batches
to WAV and back, use:

```bash
python scripts/train.py --batch-factory my_batches:make_batchers \
  --rtf data/target_rtf.npy --seed-index 1 --output outputs/sarc.pt
```

`my_batches` is a user-supplied module, not included in this repository. Its
`make_batchers(stage, train_seed, validation_seed, config, target_transfer)`
function returns separate training and validation providers. Each provider
exposes `segment_samples` and `sample_batch(batch_size)`, returning a dictionary
with complex `mixture` of shape `[B,4,257,T]` and `target` of shape `[B,257,T]`.
The reference microphone and RTF channel order must agree.

The NPZ command in the README is a self-contained waveform-pair retraining
adapter. Its source draws and STFT construction do not recreate the restricted
online generator. Matching the optimizer recipe alone is insufficient to
promise the same trained checkpoint or Table 1 result on different data.


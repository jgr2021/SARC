# Architecture matched to the released checkpoints

Input complex STFT: `[B, 4, F, T]`; enhanced STFT: `[B, F, T]`.
At 16 kHz the 512-point FFT gives `F = 257`.

## Precision network: 18,768 parameters

- 16 real-valued feature maps from the RTF-aligned observation.
- 1×1 convolution: **16 → 32**, followed by leaky ReLU.
- Four residual depthwise-separable convolutional blocks, width **32**,
  3×3 kernels and temporal dilations 1, 2, 4, 8. Padding is left-only in time.
- Unidirectional temporal GRU: input **32**, hidden **48**, independently per frequency.
- Linear head: **48 → 16** real coordinates for a complex 4×4 Cholesky factor.
- Deterministic positive-definite weighting, analytic estimate, spatial residuals
  and residual-derived scale. These operations have no trainable parameters.

**The convolution width is 32, not 48.** The number 48 belongs to the temporal
GRU. The corrected illustration reflects the checkpoint, not a model change.

## Correction network: 28,066 parameters

- 18 real-valued feature maps: six scalar maps from `(z, v)` and 12 from `r`.
- 1×1 convolution: **18 → 32**, leaky ReLU, then four convolutional blocks as above.
- Frequency-axis BiGRU: input **32**, hidden **24 per direction**, concatenated output **48**.
- Linear frequency projection: **48 → 32**, added to the pre-BiGRU representation.
- Unidirectional temporal GRU: input **32**, hidden **48**.
- Linear head: **48 → 2**, producing the real coordinates `h_real, h_imag`.
- Cartesian bounded correction:
  `delta = 4 * max(abs(z), 1e-8) * (tanh(h_real) + j*tanh(h_imag))`.
  Each Cartesian component is bounded by `4 * max(abs(z), 1e-8)`;
  the complex magnitude bound is larger by a factor of `sqrt(2)`.

The diagram is a compact overview; the input projection and frequency
projection/residual connection in the correction branch are detailed here.
The two output coordinates are not magnitude and phase.

## Interpretation and implementation boundaries

- `z`, `r` and `delta` are complex. `v` is a positive conditioning scale.
- `likelihood_variance` is a checkpoint-compatible internal key for `v`,
  not a claim of exact posterior variance.
- The raw identity `y_m = z + r_m` preserves the aligned observation.
  This does not imply every normalized/clipped feature map is invertible.
- The analytic unit-response constraint preserves the modeled target under
  matching calibration. Neural correction is not itself distortionless.
- Neural operations do not access future time frames, but the public audio
  path uses centered STFT and whole-file inference. Frequency bidirectionality
  is not temporal lookahead; whole-file speed is not deployment latency.
- Numerical floors and bounded features remain part of the implementation,
  even where compact manuscript equations omit them.

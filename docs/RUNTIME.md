# Runtime comparison

These values reproduce the current manuscript's Table 1. Each input is a
four-channel, 4-second waveform at 16 kHz; batch size is 1. Values are median
milliseconds per complete utterance, including analysis and synthesis but
excluding disk I/O and host/device transfers. CPU execution uses one thread.

| Method | GPU (ms) | CPU (ms) |
| --- | ---: | ---: |
| Norm-constrained GSC | Not tested | 174.0 |
| Online SDW-MWF | Not tested | 348.9 |
| ML-AIC | Not tested | 209.6 |
| Online McNet | 170.0 | 2831.5 |
| AFnet | 74.6 | 1459.0 |
| EaBNet | 37.3 | 701.6 |
| SARC | 28.0 | 458.7 |

Hardware: Intel Core i7-13650HX and NVIDIA GeForce RTX 4060 Laptop GPU.
Neural methods use PyTorch 2.5.1, CUDA 12.4, FP32, and the first training seed.
Twenty timed repetitions pool two rounds of ten, with three warmups per round
and reversed method order. CUDA is synchronized around inference; TF32 and
cuDNN autotuning are disabled.

Classical methods use MATLAB R2024b and FP64, with three warmups and ten timed
repetitions in a separate session. Calibration is loaded before timing;
existing wrapper diagnostics are included. Cross-family comparisons therefore
include software and numerical-precision differences. These values do not
measure strict-frame streaming latency or establish deployment readiness.

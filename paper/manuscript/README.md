# SARC manuscript

Main document: `Template.tex`.

Use pdfLaTeX and BibTeX, or run `latexmk -pdf Template.tex`.

The package includes the compiled manuscript, the editable vector figure,
the bibliography files, and the original conference template dependencies.

The method assumes four input microphone channels and a fixed frontal RTF
available before enhancement. The RTF is shared across methods requiring
calibration. Array coordinates are not direct inputs to the networks.

Version 10 removes Section 4.5 and adds a four-panel STFT example directly
within Section 4.2: clean reference,
SARC (Proposed), ML-AIC, and EaBNet. The panels use the same paired test example,
time interval, STFT settings, and magnitude reference; no per-signal amplitude
normalization is applied. The example is selected near the median SARC
improvement within the 0-dB group, using the first training seed for neural
outputs. Figure 2 is illustrative, not a new aggregate evaluation.
The final paragraphs of Section 4.2 compare full-utterance runtime for SARC, McNet, AFnet, and EaBNet,
and adds CPU-only results for norm-constrained GSC, online SDW-MWF, and ML-AIC.
All runtime values are now columns in Table 1, replacing implementation
status; there is no separate runtime table or Section 4.4. Section 4.3 explains the design
choices supported by the user's revised ablation table, which is preserved.
The protocol uses the same four-channel 4-s input, batch size 1, the first
training seed and FP32. Neural times now pool twenty timed repetitions from
two rounds of ten, with three warmups per round and reversed method order.
They supersede the earlier neural timings; classical timings are unchanged
and originate from a separate session. Reported
times are medians, include analysis and synthesis, and exclude disk I/O and
host/device transfers. CPU execution uses one thread. GPU timings synchronize
CUDA before and after inference; TF32 and cuDNN autotuning are disabled.
Hardware: Intel Core i7-13650HX and NVIDIA GeForce RTX 4060 Laptop GPU.
Neural software: PyTorch 2.5.1 with CUDA 12.4. Classical methods use their
existing MATLAB R2024b implementations in FP64, with the same input, three
warmups, ten timed repetitions, and one computational thread. Their fixed
calibration is loaded before timing; waveform processing and existing wrapper
diagnostics are included. Classical GPU execution is not tested. Therefore
cross-family numbers compare implementations with different software and
precision, not hardware-independent algorithm complexity.
The timing comparison is not a
strict-frame streaming or deployment-latency benchmark.

The enhancement scores, user's ablation table, and algorithm equations are preserved;
the runtime comparison is a new timing experiment. The evaluation description does
not claim a fully public spatial dataset. Release of supporting data and the completeness of author
affiliations require separate confirmation before submission.

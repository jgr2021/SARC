# SARC: Structured Analytic-Residual Correction

This repository contains the PyTorch implementation and frozen checkpoints for
**Structured Analytic-Residual Correction (SARC)**, a lightweight calibrated
four-microphone speech-enhancement method.

SARC first divides out a fixed frontal relative transfer function (RTF), so
the modeled target is common across microphones. A precision network produces a
Hermitian positive-definite matrix and an analytic unit-response estimate `z`.
The spatial residual `r = y - 1z` exposes channel-to-channel nuisance differences
to guide correction of interference remaining in `z`. A second network uses `(z, r, v)` to
predict a bounded complex correction, and the enhanced STFT is `z + delta`.

![SARC architecture](paper/sarc_architecture.png)

## Main result

On the fixed four-microphone evaluation set, SARC achieves
**17.527 +/- 0.053 dB SI-SDR improvement**, **0.961 +/- 0.001 STOI**, and
**2.730 +/- 0.057 PESQ** over three runs. The complete model has 46,834
parameters. Machine-readable tables are available in [`results/`](results/).

## Architecture and runtime

The architecture uses single convolution blocks marked ×4, feature-map labels,
and the precision factorization `Psi = L L^H`.

The precision and correction networks contain 18,768 and 28,066 trainable
parameters, respectively. These are parameter counts, not storage units.
The bidirectional GRU operates along frequency, not future time frames.

Table 1 includes CPU and GPU full-utterance runtimes; see
[`docs/RUNTIME.md`](docs/RUNTIME.md) for the protocol and limitations.
These are implementation timings, not proof of real-time streaming deployment.

## Installation

```bash
git clone https://github.com/jgr2021/SARC.git
cd SARC
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e .
```

## Inference

Input must be a synchronized four-channel, 16-kHz WAV file in the same channel
order as the released model.

```bash
python scripts/infer.py input_4ch.wav enhanced.wav \
  --checkpoint checkpoints/sarc_seed1.pt
```

The fixed RTF is stored in the checkpoint. The command writes a mono
floating-point WAV file.

## Verification

```bash
python -m unittest discover -s tests -v
python scripts/inspect_checkpoint.py checkpoints/sarc_seed1.pt
```

The tests check the 46,834-parameter count, unit-response constraint, and the
lossless identity between the canonicalized observation and `(z, r)`.

## Repository structure

```text
SARC/
|-- sarc/          model, STFT, and checkpoint loading
|-- scripts/       training, inference, and checkpoint inspection
|-- checkpoints/   three frozen SARC runs
|-- results/       paper tables in CSV form
|-- docs/          data and reproducibility notes
|-- paper/         architecture figure
`-- tests/         structural tests
```

## Data availability

The evaluation mixtures combine public LibriSpeech speech and ESC-50 sounds
with confidential spatial transfer assets; they are not a public-only synthetic
dataset and the final audio is not distributed. See [`docs/DATA.md`](docs/DATA.md)
for source links, training/evaluation construction, and reproducibility limits.
The [80-trial public-source manifest](data/manifests/evaluation_public_sources.csv)
is available without restricted spatial assets or internal identifiers.

For retraining, place fixed-length `.npz` examples in one directory. Each file
must contain float32 arrays `mixture` with shape `[4, samples]` and `target` with
shape `[samples]`, together with a complex `[257, 4]` RTF saved as `.npy`:

```bash
python scripts/train.py --train data/train --rtf data/target_rtf.npy \
  --output outputs/sarc.pt
```

## Citation

If you use SARC in your research, please cite:

> G. Jin, X.-P. Zhang, L. Xiao, Y. Wang, and Z. Liu, “SARC: Structured Analytic-Residual Correction for Lightweight Multichannel Speech Enhancement,” manuscript submitted to the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2027), 2026.

```bibtex
@unpublished{jin2026sarc,
  author = {Jin, Gurui and Zhang, Xiao-Ping and Xiao, Le and Wang, Yue and Liu, Zhenyu},
  title  = {{SARC}: Structured Analytic-Residual Correction for Lightweight Multichannel Speech Enhancement},
  year   = {2026},
  note   = {Manuscript submitted to the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2027)},
  url    = {https://github.com/jgr2021/SARC}
}
```

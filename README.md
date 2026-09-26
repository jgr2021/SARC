# SARC: Structured Analytic-Residual Correction

PyTorch implementation of **SARC: Structured Analytic-Residual Correction for
Lightweight Multichannel Speech Enhancement**.

**Gurui Jin, Xiao-Ping Zhang, Le Xiao, Yue Wang, and Zhenyu Liu**

**Submitted to ICASSP 2027**

## Overview

SARC combines an analytic spatial estimate with neural correction. Given a
calibrated target relative transfer function (RTF), it aligns four microphone
signals, forms a target-preserving estimate, and uses target-canceling spatial
residuals to guide a compact correction network.

![SARC architecture](paper/sarc_architecture.png)

The model has **46,834 parameters**. On the fixed 80-mixture evaluation set,
three runs achieve **17.527 ± 0.053 dB SI-SDR improvement**, **0.961 ± 0.001
STOI**, and **2.730 ± 0.057 PESQ**.
See [result tables](results/), [implementation details](docs/REPRODUCIBILITY.md),
and [runtime measurements](docs/RUNTIME.md).

## Installation

Python 3.10+; original experiments used PyTorch 2.5.1.

```bash
git clone https://github.com/jgr2021/SARC.git
cd SARC
pip install -e .
```

## Inference

```bash
python scripts/infer.py input_4ch.wav enhanced.wav --checkpoint checkpoints/sarc_seed1.pt
```

Input: synchronized four-channel, 16-kHz WAV. Output: mono WAV.
The checkpoint includes the fixed RTF; channel order and calibration must match.
This is whole-file inference, not a stateful streaming implementation.

## Training

Prepare separate training and validation folders of fixed-length NPZ pairs:
float32 `mixture [4, samples]` and `target [samples]`, plus a complex
`[257, 4]` RTF in a NumPy file.

```bash
python scripts/train.py --train data/train --validation data/validation --rtf data/target_rtf.npy --seed-index 1 --output outputs/sarc.pt
```

See [training settings and online-batch interface](docs/REPRODUCIBILITY.md).

## Data

Speech comes from [LibriSpeech](https://www.openslr.org/12); evaluation
interferers come from [ESC-50](https://github.com/karolpiczak/ESC-50).
Confidential training noise and spatial synthesis assets are not distributed,
so the original benchmark cannot be fully reconstructed from this release.
See [data details](docs/DATA.md) and the
[public-source manifest](data/manifests/evaluation_public_sources.csv).

## Tests

```bash
python -m unittest discover -s tests -v
python scripts/inspect_checkpoint.py checkpoints/sarc_seed1.pt
```

## Citation

```bibtex
@unpublished{jin2026sarc,
  author = {Jin, Gurui and Zhang, Xiao-Ping and Xiao, Le and Wang, Yue and Liu, Zhenyu},
  title = {{SARC}: Structured Analytic-Residual Correction for Lightweight Multichannel Speech Enhancement},
  year = {2026},
  note = {Submitted to ICASSP 2027},
  url = {https://github.com/jgr2021/SARC}
}
```

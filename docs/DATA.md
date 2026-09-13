# Data sources, synthesis, and availability

**The reported evaluation set is custom-synthesized, but it is not a public-only dataset.**
It combines public speech and environmental sounds with confidential spatial
transfer assets. Public source audio does not make the resulting multichannel
mixtures automatically redistributable.

## 1. Where the data comes from

| Component | Source and role | Availability |
| --- | --- | --- |
| Clean speech | [LibriSpeech (OpenSLR 12)](https://www.openslr.org/12), by Panayotov et al. Used for training/validation speech and evaluation targets. The 80-trial evaluation uses selected files from `test-clean`. | Download from the original provider; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). |
| Evaluation interference | [ESC-50](https://github.com/karolpiczak/ESC-50), by Karol J. Piczak. One or two environmental sound clips per mixture. | Public downloads and [clip metadata](https://github.com/karolpiczak/ESC-50/blob/master/meta/esc50.csv). The dataset is CC BY-NC 3.0; see its [license and individual-clip attribution](https://github.com/karolpiczak/ESC-50/blob/master/LICENSE). |
| Training noise | A confidential multichannel noise collection; not an ESC-50 training split. | Not distributed under the applicable confidentiality restrictions. |
| Target and interference spatial mapping | Confidential calibration/transfer assets used to map sources into microphone channels. | The complete synthesis assets are not distributed. The released checkpoints already contain the fixed target RTF used for inference. |
| Sensor-noise component | Seeded Gaussian noise generated numerically during evaluation synthesis. | Its generation is not dependent on a public audio corpus; parameters and seeds are in the public-source manifest below. |
| Final evaluation mixtures | Custom mixtures combining the above components. | Not distributed because their spatial mapping depends on restricted assets. |

No company names, internal data labels, internal paths, or calibration-source
locators are included in the public manifest. This page documents provenance
without disclosing those identifiers.

## 2. Training and validation

The original experiments generate mixtures **online**, rather than training on
a fixed published mixture dataset:

- Select a LibriSpeech utterance and a random 2.56-second crop; pad shorter
  utterances and normalize the speech.
- Map the speech into the microphone channels using the fixed target RTF.
- Add a synchronized crop from the confidential multichannel noise collection.
- Sample the nominal SNR uniformly from -10 to +15 dB and a common gain from
  -8 to +3 dB.

The two validation speaker IDs are `61` and `121`. The 20 evaluation speakers
are excluded from training and validation; their IDs are available in the
evaluation manifest. The original strict-split training pipeline restricts
training and validation noise crops to separate time intervals.

This is a **custom speaker split**, not a claim that the experiments use
LibriSpeech's standard ASR training/evaluation protocol. The original sampler
selects eligible files from the locally available LibriSpeech pool. A complete
frozen training-file inventory and total training-corpus hours are not provided
in this release; downloading an arbitrary LibriSpeech subset will not recreate
the exact original training stream.

The released [training script](../scripts/train.py) accepts pre-rendered NPZ
pairs. It is a public retraining interface, not the original confidential
online data generator.

## 3. Fixed evaluation set

The reported main comparison uses **80 mixtures**, each **4 seconds** long,
from **20 LibriSpeech speakers**, evaluated with **four channels at 16 kHz**.
The nominal SNRs are **-10, -5, 0, +5, and +10 dB**. Each mixture contains
one or two ESC-50 interference sources. The design also spans four restricted
spatial conditions; their identifiers and transfer assets are not released.

In outline, synthesis performs the following steps:

1. Read the specified speech and interference crops, resample to 16 kHz,
   remove the mean, and normalize each source to RMS 0.05.
2. Apply the target transfer function to speech and the interference transfer
   functions to the environmental sounds in the STFT domain.
3. Sum the interference sources and scale them to the nominal SNR at the
   reference microphone.
4. Add independent Gaussian sensor noise at 30 dB relative to target power.
5. Apply a common peak-control gain to the mixture and target.

The nominal SNR is set **before adding sensor noise**; the final input SNR can
therefore differ slightly. The clean target is the target speech image at the
reference microphone, not the noisy input.

### Public source manifest

Download [`evaluation_public_sources.csv`](../data/manifests/evaluation_public_sources.csv).
It contains one row per trial:

- LibriSpeech file path, speaker ID, and speech crop offset;
- ESC-50 file names, fold, category, original source ID, and crop offsets;
- trial seed, duration, source count, nominal SNR, and sensor-noise SNR.

Paths are relative to the original corpus folders, not local machine paths.
Offsets and durations are in seconds. Empty second-source fields indicate a
single-interferer trial. `ESC-50-master` is just the corpus extraction-folder
name; adjust it to your local download location.

**This manifest identifies the public ingredients; it is not a complete
reproduction recipe.** Restricted spatial transfer values are intentionally
absent. Substituting simulated or independently obtained transfer functions
creates a different dataset and does not reproduce the reported scores.

## 4. What is and is not released

Released:

- Model code, three frozen checkpoints, and result tables.
- The sanitized 80-trial public-source manifest.
- Public-corpus links and the synthesis outline above.

Not released:

- Confidential training noise and complete spatial synthesis assets.
- Final evaluation mixtures and their spatialized components.
- An exact, end-to-end reproduction of the original training/evaluation data.

There is currently **no separate public-only synthetic audio benchmark in this
repository**. The existing 80 mixtures must not be described as such. A future
public-only set would need independently shareable spatial simulation, source
attribution, its own manifest, and separate evaluation results.

## 5. Input and retraining format

Inference requires a synchronized four-channel WAV at 16 kHz. Channel order
and acoustic calibration must be compatible with the released checkpoint.
The fixed target RTF is embedded in each checkpoint, so no separate RTF file
is needed for inference.

For retraining, each fixed-length NPZ example must contain:

| Key | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `mixture` | `[4, samples]` | `float32` | Synchronized noisy microphone signals at 16 kHz |
| `target` | `[samples]` | `float32` | Corresponding clean reference-channel target |

Supply a separate complex `[257, 4]` target RTF in a `.npy` file to
[`scripts/train.py`](../scripts/train.py). Keep training, validation, and test
sources separate. For a different array or channel ordering, obtain a matching
target RTF and retrain; the frozen model is not an array-independent baseline.

# Data sources and availability

The benchmark combines public audio with restricted spatial assets; it is
not a public-only synthetic dataset.

| Component | Source | Availability |
| --- | --- | --- |
| Speech | [LibriSpeech / OpenSLR 12](https://www.openslr.org/12) | Public; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Evaluation interference | [ESC-50](https://github.com/karolpiczak/ESC-50) | Public; [CC BY-NC 3.0 and clip attribution](https://github.com/karolpiczak/ESC-50/blob/master/LICENSE) |
| Training noise | Confidential multichannel noise | Not distributed |
| Spatial synthesis assets | Confidential target/interference transfers | Complete assets not distributed; fixed target RTF is included in checkpoints |
| Evaluation mixtures | Synthesis using the above components | Not distributed |

## Training and validation

Mixtures are generated online using 2.56-second LibriSpeech crops, the target
RTF, synchronized noise crops, nominal SNR uniformly from −10 to +15 dB,
and common gain from −8 to +3 dB. Short utterances are padded.

Validation speakers are `61` and `121`; the 20 evaluation speakers are
excluded from training and validation. Training and validation noise use
separate time intervals. This is a custom speaker split, not the standard
LibriSpeech ASR protocol. A complete training-file inventory and corpus-hour
count are not provided.

## Evaluation

80 four-second mixtures from 20 LibriSpeech `test-clean` speakers; four
channels at 16 kHz; four spatial conditions; nominal SNRs of −10, −5, 0,
+5 and +10 dB; one or two spatialized ESC-50 interferers per mixture.

Sources are resampled, mean-centered and normalized to RMS 0.05, then mapped
to microphone channels in the STFT domain. Interference is scaled to the
nominal reference-channel SNR, followed by independent sensor noise at 30 dB
relative to target power and common peak control. The clean target is the
reference-channel speech image. Nominal SNR precedes sensor-noise addition.

The [public-source manifest](../data/manifests/evaluation_public_sources.csv)
lists source files, speakers, crop offsets, interference metadata, trial seeds
and SNRs. Paths are relative to corpus folders; offsets are in seconds.
An empty second-source field denotes a single interferer.

Restricted spatial assets are omitted, so this manifest alone cannot reproduce
the original mixtures. Using simulated transfers creates a different benchmark.

## Your own data

Inference requires a synchronized four-channel WAV at 16 kHz with calibration
and channel order matching the checkpoint. For another array, obtain its target
RTF and retrain.

Training accepts fixed-length NPZ pairs with float32 `mixture [4, samples]`
and `target [samples]`, plus a complex `[257,4]` RTF in a NumPy file.
Keep training, validation and test sources separate. See
[training and online-batch settings](REPRODUCIBILITY.md).

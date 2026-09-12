# Data

The released checkpoints expect four synchronous channels sampled at 16 kHz.
The channel order must match the four channels used to estimate the RTF stored
inside each checkpoint.

Evaluation audio and example-generation assets are not included. Public clean
speech and benchmark corpora should be obtained from their original providers
and used under their respective licenses. This repository alone does not provide
all assets needed to reproduce the complete evaluation dataset.

For inference, no separate calibration file is required: the fixed frontal
RTF is stored in the checkpoint. For retraining on a different array or
channel order, estimate a new target RTF and train a new model.


# Data

The released checkpoints expect four synchronous channels sampled at 16 kHz.
The channel order must match the four channels used to estimate the RTF stored
inside each checkpoint.

The Shokz recordings are not redistributed in this repository. They contain
locally recorded material and must be supplied by the data owner. Public clean
speech and public benchmark corpora should be downloaded from their original
providers and used under their respective licenses.

For inference, no separate calibration file is required: the measured frontal
RTF is stored in the checkpoint. For retraining on a different headset or
channel order, estimate a new target RTF and train a new model.


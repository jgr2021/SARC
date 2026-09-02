"""Simple public data interface for synchronized mixture/target waveform pairs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NpzWaveDataset(Dataset):
    """Read fixed-length `.npz` files containing `mixture` and `target` arrays.

    `mixture` must have shape [4, samples], `target` must have shape [samples],
    and both arrays must be float32 at 16 kHz.
    """

    def __init__(self, root: str | Path) -> None:
        self.paths = sorted(Path(root).glob("*.npz"))
        if not self.paths:
            raise ValueError(f"no .npz examples found in {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with np.load(self.paths[index], allow_pickle=False) as example:
            mixture = np.asarray(example["mixture"], dtype=np.float32)
            target = np.asarray(example["target"], dtype=np.float32)
        if mixture.ndim != 2 or mixture.shape[0] != 4:
            raise ValueError(f"{self.paths[index]}: mixture must have shape [4,N]")
        if target.ndim != 1 or target.shape[0] != mixture.shape[1]:
            raise ValueError(f"{self.paths[index]}: target must have shape [N]")
        return torch.from_numpy(mixture), torch.from_numpy(target)


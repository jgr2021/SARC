"""Checkpoint loading with compatibility checks for the frozen SARC weights."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import SARC


def load_sarc(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[SARC, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "target_transfer" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("checkpoint must contain target_transfer and model_state")
    model = SARC(checkpoint["target_transfer"], freeze_spatial=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if model.parameter_count() != 46_834:
        raise RuntimeError(f"unexpected parameter count: {model.parameter_count()}")
    model.to(device).eval()
    return model, checkpoint


"""Remove local paths and historical prototype labels from release checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    old_manifest = source.get("manifest", {})
    manifest = {
        "model": "SARC",
        "model_class": "SARC",
        "parameters": 46_834,
        "microphones": 4,
        "seed": args.seed,
        "stage": "joint",
        "spatial_frozen": False,
        "correction_parameterization": "4*abs(z)*[tanh(h_real)+j*tanh(h_imag)]",
        "stft": old_manifest.get(
            "stft",
            {
                "sample_rate": 16_000,
                "n_fft": 512,
                "win_length": 400,
                "hop_length": 160,
                "center": True,
            },
        ),
    }
    public = {
        "model_state": source["model_state"],
        "target_transfer": source["target_transfer"],
        "step": source.get("step", 600),
        "best_validation_delta_si_sdr": source.get("best_validation_delta_si_sdr"),
        "manifest": manifest,
    }
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(public, args.destination)


if __name__ == "__main__":
    main()


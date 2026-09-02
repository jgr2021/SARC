"""Enhance a four-channel WAV file with a released SARC checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from sarc import STFTConfig, istft, load_sarc, make_window, stft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="four-channel WAV at 16 kHz")
    parser.add_argument("output", type=Path, help="enhanced mono WAV")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/sarc_seed1.pt")
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    audio, sample_rate = sf.read(args.input, dtype="float32", always_2d=True)
    config = STFTConfig()
    if sample_rate != config.sample_rate:
        raise ValueError(f"expected {config.sample_rate} Hz, received {sample_rate} Hz")
    if audio.shape[1] != 4:
        raise ValueError(f"expected four channels, received {audio.shape[1]}")

    device = torch.device(args.device)
    model, checkpoint = load_sarc(args.checkpoint, device)
    manifest = checkpoint.get("manifest", {})
    if "stft" in manifest:
        config = STFTConfig(**manifest["stft"])
    window = make_window(config, device)
    waveform = torch.from_numpy(audio.T).to(device)
    spectrum = stft(waveform, window, config)[None]
    enhanced = istft(model(spectrum)["mean"], window, config, len(audio))[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, enhanced.cpu().numpy(), sample_rate, subtype="FLOAT")
    print(args.output.resolve())


if __name__ == "__main__":
    main()


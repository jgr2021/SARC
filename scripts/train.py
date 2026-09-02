"""Three-stage SARC training from fixed-length waveform examples."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from sarc import SARC, STFTConfig, make_window, stft
from sarc.dataset import NpzWaveDataset
from sarc.losses import correction_objective, joint_objective, spatial_objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--rtf", type=Path, required=True, help="complex [257,4] .npy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--spatial-updates", type=int, default=600)
    parser.add_argument("--correction-updates", type=int, default=800)
    parser.add_argument("--joint-updates", type=int, default=600)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def batches(loader: DataLoader):
    while True:
        yield from loader


def optimize_stage(
    model: SARC,
    loader: DataLoader,
    updates: int,
    optimizer: torch.optim.Optimizer,
    objective,
    device: torch.device,
    window: torch.Tensor,
    config: STFTConfig,
) -> None:
    iterator = batches(loader)
    model.train()
    for update in range(1, updates + 1):
        mixture_wave, target_wave = next(iterator)
        mixture_wave = mixture_wave.to(device)
        target_wave = target_wave.to(device)
        length = target_wave.shape[-1]
        mixture = stft(mixture_wave, window, config)
        target = stft(target_wave[:, None], window, config)[:, 0]
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model(mixture), target, window, config, length)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]], 5.0
        )
        optimizer.step()
        if update == 1 or update % 20 == 0 or update == updates:
            print(f"update {update:4d}/{updates}: loss={float(loss.detach()):.5f}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    transfer = torch.from_numpy(np.load(args.rtf, allow_pickle=False)).to(torch.complex64)
    if transfer.shape != (257, 4):
        raise ValueError(f"RTF must have shape [257,4], received {tuple(transfer.shape)}")
    model = SARC(transfer).to(device)
    data = NpzWaveDataset(args.train)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    config = STFTConfig()
    window = make_window(config, device)

    print("stage 1/3: analytic estimator")
    optimize_stage(
        model,
        loader,
        args.spatial_updates,
        torch.optim.AdamW(model.spatial_model.parameters(), lr=2e-3, weight_decay=1e-4),
        spatial_objective,
        device,
        window,
        config,
    )
    print("stage 2/3: correction network")
    for parameter in model.spatial_model.parameters():
        parameter.requires_grad_(False)
    optimize_stage(
        model,
        loader,
        args.correction_updates,
        torch.optim.AdamW(model.score_model.parameters(), lr=2e-3, weight_decay=1e-4),
        correction_objective,
        device,
        window,
        config,
    )
    print("stage 3/3: joint fine-tuning")
    for parameter in model.spatial_model.parameters():
        parameter.requires_grad_(True)
    optimize_stage(
        model,
        loader,
        args.joint_updates,
        torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4),
        joint_objective,
        device,
        window,
        config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "target_transfer": transfer.cpu(),
            "step": args.joint_updates,
            "manifest": {
                "model": "SARC",
                "parameters": model.parameter_count(),
                "seed": args.seed,
                "stft": vars(config),
                "correction_parameterization": "4*abs(z)*[tanh(h_real)+j*tanh(h_imag)]",
            },
        },
        args.output,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()


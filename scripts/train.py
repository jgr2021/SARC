"""Train using the Table 1 recipe; data can come from NPZ pairs or an online factory."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from sarc import STFTConfig, make_window, stft
from sarc.dataset import NpzWaveDataset
from sarc.training import experiment_stages, initialize_stage, optimize_stage, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--batch-factory", help="module:function returning train and validation STFT batch providers")
    parser.add_argument("--rtf", type=Path, required=True, help="complex [257,4] .npy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, help="override the experimental stage sizes 3,4,3")
    parser.add_argument("--seed-index", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seed", type=int, help="custom joint seed; earlier stages use seed-1 and seed-4")
    parser.add_argument("--spatial-updates", type=int, default=600)
    parser.add_argument("--correction-updates", type=int, default=800)
    parser.add_argument("--joint-updates", type=int, default=600)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_factory:
        if args.train or args.validation:
            parser.error("choose --batch-factory or --train with --validation")
    elif args.train is None or args.validation is None:
        parser.error("provide both --train and --validation, or --batch-factory")
    elif args.train.resolve() == args.validation.resolve():
        parser.error("training and validation directories must differ")
    if min(args.spatial_updates, args.correction_updates, args.joint_updates,
           args.validation_every, args.validation_batches, args.batch_size or 1) < 1:
        parser.error("update counts and batch sizes must be positive")
    return args


class NpzBatcher:
    """Retraining adapter, not a replacement for the original online generator."""
    def __init__(self, root, config, seed):
        self.data = NpzWaveDataset(root)
        self.segment_samples = self.data[0][1].numel()
        self.config = config
        self.window = make_window(config, torch.device("cpu"))
        self.rng = np.random.default_rng(seed)

    def sample_batch(self, batch_size):
        examples = [self.data[int(i)] for i in self.rng.integers(0, len(self.data), size=batch_size)]
        mixture = torch.stack([x[0] for x in examples])
        target = torch.stack([x[1] for x in examples])
        return {"mixture": stft(mixture, self.window, self.config),
                "target": stft(target[:, None], self.window, self.config)[:, 0]}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    transfer = torch.from_numpy(np.load(args.rtf, allow_pickle=False)).to(torch.complex64)
    if transfer.shape != (257, 4):
        raise ValueError(f"RTF must have shape [257,4], received {tuple(transfer.shape)}")
    config = STFTConfig()
    window = make_window(config, device)

    stages = experiment_stages(args.seed_index)
    if args.seed is not None:
        offset = args.seed - stages[-1].seed
        stages = tuple(replace(s, seed=s.seed + offset) for s in stages)
    counts = (args.spatial_updates, args.correction_updates, args.joint_updates)
    stages = tuple(replace(s, updates=n, batch_size=args.batch_size or s.batch_size)
                   for s, n in zip(stages, counts))
    factory = None
    if args.batch_factory:
        module, name = args.batch_factory.split(":", 1)
        factory = getattr(importlib.import_module(module), name)
    stage_dir = args.output.parent / (args.output.stem + "_stages")
    if args.output.exists() or stage_dir.exists():
        raise FileExistsError("choose a new --output path to preserve prior checkpoints")
    stage_dir.mkdir(parents=True)
    previous = None
    history = []
    for index, stage in enumerate(stages, 1):
        print(f"stage {index}/3: {stage.name}", flush=True)
        set_seed(stage.seed)
        # In stage 1 the original code constructs batchers before the model;
        # later stages construct the model before batchers.
        def batchers():
            if factory:
                return factory(stage=stage.name, train_seed=stage.seed,
                               validation_seed=stage.seed + 10_000,
                               config=config, target_transfer=transfer)
            return (NpzBatcher(args.train, config, stage.seed),
                    NpzBatcher(args.validation, config, stage.seed + 10_000))
        if stage.name == "spatial":
            train, validation = batchers()
        model = initialize_stage(stage, transfer, previous).to(device)
        if stage.name != "spatial":
            train, validation = batchers()
        best = optimize_stage(model, stage, train, validation, device, window, config,
                              args.validation_every, args.validation_batches, history.append)
        checkpoint = {**best, "target_transfer": transfer.cpu(), "manifest": {
            "model": "SARCSpatial" if stage.name == "spatial" else "SARC",
            "parameters": model.parameter_count(), **asdict(stage), "stage": stage.name,
            "spatial_frozen": stage.name == "correction", "stft": asdict(config),
            "correction_parameterization": "4*abs(z)*[tanh(h_real)+j*tanh(h_imag)]",
            "optimizer": "AdamW", "weight_decay": 1e-4, "gradient_clip_l2": 5.0,
            "validation_seed": stage.seed + 10_000,
            "validation_every": args.validation_every,
            "validation_batches": args.validation_batches,
            "fixed_validation_batches": stage.name == "joint",
            "checkpoint_rule": "maximum mean validation SI-SDR improvement",
            "data_interface": "online_factory" if factory else "prerendered_npz",
            "all_stages": [asdict(s) for s in stages],
        }}
        torch.save(checkpoint, stage_dir / f"{stage.name}_best.pt")
        previous = best["model_state"]
        (stage_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save(checkpoint, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()


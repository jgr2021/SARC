"""Table 1 optimization recipe, with the original sample_batch data boundary.

Batch providers return complex mixture [B,4,F,T] and target [B,F,T] tensors
and expose segment_samples. Restricted data generators are not included.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import numpy as np
import torch

from .audio import istft
from .losses import correction_objective, joint_objective, si_sdr, spatial_objective
from .model import SARC, SARCSpatial


@dataclass(frozen=True)
class Stage:
    name: str
    seed: int
    updates: int
    batch_size: int
    learning_rate: float


def experiment_stages(seed_index=1):
    if seed_index not in (1, 2, 3):
        raise ValueError("seed_index must be 1, 2, or 3")
    offset = (seed_index - 1) * 10_000
    return (
        Stage("spatial", 20260816 + offset, 600, 3, 2e-3),
        Stage("correction", 20260813 + offset, 800, 4, 2e-3),
        Stage("joint", 20260817 + offset, 600, 3, 5e-4),
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def initialize_stage(stage, transfer, previous=None):
    """Call set_seed before constructing the stage and its batch providers."""
    if stage.name == "spatial":
        return SARCSpatial(transfer)
    if stage.name not in ("correction", "joint") or previous is None:
        raise ValueError("correction/joint stages need the preceding best state")
    model = SARC(transfer, freeze_spatial=stage.name == "correction")
    if stage.name == "correction":
        model.spatial_model.load_state_dict(previous, strict=True)
    else:
        model.load_state_dict(previous, strict=True)
    return model


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def validate(model, batches, device, window, config, length):
    model.eval()
    scores = []
    for batch in batches:
        mixture, target = batch["mixture"].to(device), batch["target"].to(device)
        estimate = istft(model(mixture)["mean"], window, config, length)
        clean = istft(target, window, config, length)
        reference = istft(mixture[:, 0], window, config, length)
        scores.append(float((si_sdr(estimate, clean) - si_sdr(reference, clean)).mean()))
    return float(np.mean(scores))


def optimize_stage(model, stage, train, validation, device, window, config,
                   validation_every=100, validation_batches=10, record=None):
    if stage.name == "spatial":
        if not isinstance(model, SARCSpatial):
            raise ValueError("stage 1 trains only SARCSpatial")
        objective = spatial_objective
    elif stage.name == "correction":
        if not model.freeze_spatial:
            raise ValueError("stage 2 requires the spatial model to be frozen")
        objective = correction_objective
    elif stage.name == "joint":
        if model.freeze_spatial:
            raise ValueError("joint training requires both networks")
        objective = joint_objective
    else:
        raise ValueError(f"unknown stage {stage.name}")
    parameters = list(model.parameters()) if stage.name != "correction" else list(model.score_model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=stage.learning_rate, weight_decay=1e-4)
    fixed = None
    if stage.name == "joint":
        fixed = [validation.sample_batch(stage.batch_size) for _ in range(validation_batches)]
    best = None
    for step in range(1, stage.updates + 1):
        model.train()
        batch = train.sample_batch(stage.batch_size)
        mixture, target = batch["mixture"].to(device), batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(mixture)
        extra = {}
        if stage.name != "correction":
            spatial = model if stage.name == "spatial" else model.spatial_model
            extra = {"mixture": mixture, "transfer": spatial.target_transfer}
        loss = objective(output, target, window, config, train.segment_samples, **extra)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        if step == 1 or step % 20 == 0 or step == stage.updates:
            print(f"{stage.name} {step}/{stage.updates}: loss={float(loss.detach()):.6f}", flush=True)
        if step % validation_every == 0 or step == stage.updates:
            batches = fixed if fixed is not None else [
                validation.sample_batch(stage.batch_size) for _ in range(validation_batches)
            ]
            score = validate(model, batches, device, window, config, validation.segment_samples)
            if not np.isfinite(score):
                raise FloatingPointError("nonfinite validation score; no checkpoint selected")
            if record is not None:
                record({"stage": stage.name, "step": step, "loss": float(loss.detach()),
                        "validation_delta_si_sdr": score})
            print(f"{stage.name} validation {step}: delta SI-SDR={score:.6f}", flush=True)
            if best is None or score > best["best_validation_delta_si_sdr"]:
                best = {"model_state": cpu_state(model), "step": step,
                        "best_validation_delta_si_sdr": score}
    model.load_state_dict(best["model_state"], strict=True)
    return best

"""Training objectives used for the released SARC architecture."""

from __future__ import annotations

import torch

from .audio import STFTConfig, istft


def compressed_complex_loss(
    estimate: torch.Tensor, target: torch.Tensor, exponent: float = 0.30
) -> torch.Tensor:
    def compress(value: torch.Tensor) -> torch.Tensor:
        magnitude = value.abs().clamp_min(1e-8)
        return magnitude.pow(exponent) * value / magnitude

    estimate_c = compress(estimate)
    target_c = compress(target)
    return (
        (estimate_c - target_c).abs().square().mean()
        / target_c.abs().square().mean().clamp_min(1e-8)
    )


def si_sdr(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    scale = (estimate * target).sum(dim=-1, keepdim=True) / target.square().sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    projection = scale * target
    error = estimate - projection
    return 10.0 * torch.log10(
        projection.square().sum(dim=-1).clamp_min(1e-8)
        / error.square().sum(dim=-1).clamp_min(1e-8)
    )


def waveform_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    window: torch.Tensor,
    config: STFTConfig,
    length: int,
) -> torch.Tensor:
    estimate_wave = istft(estimate, window, config, length)
    target_wave = istft(target, window, config, length)
    return -si_sdr(estimate_wave, target_wave).mean() / 20.0


def precision_regularizer(
    output: dict[str, torch.Tensor], target: torch.Tensor,
    mixture: torch.Tensor | None = None, transfer: torch.Tensor | None = None,
) -> torch.Tensor:
    if mixture is not None and transfer is not None:
        # Preserve the arithmetic order of the successful experiment:
        # subtract the target image first, then divide by the RTF.
        transfer = transfer[None].expand(mixture.shape[0], -1, -1)
        target_sensor = transfer.permute(0, 2, 1)[:, :, :, None] * target[:, None]
        noise = mixture - target_sensor
        aligned_noise = (noise / transfer.permute(0, 2, 1)[:, :, :, None]).permute(0, 2, 3, 1)
    else:
        aligned_noise = (
            output["aligned_observation"] - target[:, None]
        ).permute(0, 2, 3, 1)
    scale = aligned_noise.abs().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    column = (aligned_noise / scale)[..., None]
    microphones = aligned_noise.shape[-1]
    quadratic = (
        column.conj().transpose(-1, -2) @ output["precision"] @ column
    ).real.squeeze(-1).squeeze(-1) / microphones
    return (
        quadratic - output["log_determinant_precision"] / microphones
    ).mean()


def spatial_objective(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    window: torch.Tensor,
    config: STFTConfig,
    length: int,
    mixture: torch.Tensor | None = None,
    transfer: torch.Tensor | None = None,
) -> torch.Tensor:
    if "spatial_mean" in output:
        raise ValueError("stage 1 requires spatial_model(mixture), not the complete SARC output")
    weight_norm = output["aligned_weight"].abs().square().sum(dim=-1).mean()
    return (
        compressed_complex_loss(output["mean"], target)
        + 0.10 * waveform_loss(output["mean"], target, window, config, length)
        + 0.02 * precision_regularizer(output, target, mixture, transfer)
        + 0.001 * weight_norm
    )


def correction_objective(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    window: torch.Tensor,
    config: STFTConfig,
    length: int,
) -> torch.Tensor:
    # The Table 1 experiment uses 1e-8 here, but 1e-10 in joint scale calibration.
    variance = output["likelihood_variance"].clamp_min(1e-8)
    target_power = target.abs().square().mean(dim=1, keepdim=True)
    correction_target = target - output["spatial_mean"]
    correction_error = (
        (output["correction"] - correction_target).abs().square()
        / (variance + 1e-3 * target_power)
    ).mean()
    correction_size = (
        output["correction"].abs().square()
        / (output["spatial_mean"].abs().square() + variance + 1e-8)
    ).mean()
    return (
        compressed_complex_loss(output["mean"], target)
        + 0.03 * correction_error
        + 0.12 * waveform_loss(output["mean"], target, window, config, length)
        + 0.001 * correction_size
    )


def joint_objective(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    window: torch.Tensor,
    config: STFTConfig,
    length: int,
    mixture: torch.Tensor | None = None,
    transfer: torch.Tensor | None = None,
) -> torch.Tensor:
    correction = correction_objective(output, target, window, config, length)
    precision = precision_regularizer(output, target, mixture, transfer)
    spatial_complex = compressed_complex_loss(output["spatial_mean"], target)
    variance = output["likelihood_variance"].clamp_min(1e-10)
    target_power = target.abs().square().mean(dim=1, keepdim=True).clamp_min(1e-8)
    scalar_calibration = (
        (output["spatial_mean"] - target).abs().square() / variance
        + torch.log(variance / target_power).clamp(-12.0, 12.0)
    ).mean()
    regularizer = (
        0.10 * spatial_complex
        + 0.02 * precision
        + 0.01 * scalar_calibration
    )
    return correction + regularizer


"""SARC model used for the four-microphone experiments.

The implementation is self-contained and keeps the parameter names of the
frozen experimental checkpoints so that they can be loaded with ``strict=True``.
Complex tensors use shape [batch, microphone, frequency, frame] at the input
and [batch, frequency, frame] at the output.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CausalBlock(nn.Module):
    """Depthwise-separable convolution, causal along the frame axis."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=(3, 3),
            dilation=(1, dilation),
            groups=channels,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.PReLU(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = F.pad(value, (2 * self.dilation, 0, 1, 1))
        update = self.pointwise(self.depthwise(update))
        return value + self.activation(update)


def complex_cholesky_precision(
    raw: torch.Tensor, microphones: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert M^2 real coordinates into an M-by-M HPD precision matrix."""
    if raw.shape[-1] != microphones * microphones:
        raise ValueError("expected M^2 precision coordinates")
    batch, frequency, frames, _ = raw.shape
    diagonal_log = 1.5 * torch.tanh(raw[..., :microphones])
    diagonal = torch.exp(diagonal_log)
    lower = torch.zeros(
        batch,
        frequency,
        frames,
        microphones,
        microphones,
        dtype=torch.complex64,
        device=raw.device,
    )
    for microphone in range(microphones):
        lower[..., microphone, microphone] = diagonal[..., microphone].to(
            torch.complex64
        )
    cursor = microphones
    for row in range(1, microphones):
        for column in range(row):
            real = 0.70 * torch.tanh(raw[..., cursor])
            imag = 0.70 * torch.tanh(raw[..., cursor + 1])
            lower[..., row, column] = torch.complex(real, imag)
            cursor += 2
    precision = lower @ lower.conj().transpose(-1, -2)
    log_determinant = 2.0 * diagonal_log.sum(dim=-1)
    return precision, log_determinant


class SARCSpatial(nn.Module):
    """RTF canonicalization, learned precision, and analytic unit-response estimate."""

    def __init__(
        self,
        target_transfer: torch.Tensor,
        hidden: int = 32,
        gru_hidden: int = 48,
    ) -> None:
        super().__init__()
        if target_transfer.ndim != 2 or target_transfer.shape[1] < 2:
            raise ValueError("target_transfer must be complex [F,M], M>=2")
        if not torch.is_complex(target_transfer):
            raise ValueError("target_transfer must be complex")
        if torch.any(target_transfer.abs() <= 0):
            raise ValueError("RTF entries must be nonzero")
        self.sensor_count = int(target_transfer.shape[1])
        self.feature_count = 3 * self.sensor_count + 4
        self.precision_parameter_count = self.sensor_count**2
        self.register_buffer("target_transfer", target_transfer.to(torch.complex64))
        self.input_projection = nn.Conv2d(self.feature_count, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [CausalBlock(hidden, dilation) for dilation in (1, 2, 4, 8)]
        )
        self.gru = nn.GRU(hidden, gru_hidden, batch_first=True)
        self.head = nn.Linear(gru_hidden, self.precision_parameter_count)

    def target_normalize(self, mixture: torch.Tensor) -> torch.Tensor:
        """Divide out the measured target RTF so target speech is common across mics."""
        return mixture / self.target_transfer.T[None, :, :, None]

    def _features_from_aligned(self, aligned: torch.Tensor) -> torch.Tensor:
        scale = aligned.abs().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        normalized = aligned / scale
        log_relative_magnitude = (
            torch.log(normalized.abs().clamp_min(1e-6)).clamp(-6.0, 4.0) / 6.0
        )
        matched = aligned.mean(dim=1)
        matched_unit = matched / matched.abs().clamp_min(1e-8)
        split = max(1, self.sensor_count // 2)
        first_power = aligned[:, :split].abs().square().mean(dim=1)
        second_power = aligned[:, split:].abs().square().mean(dim=1)
        partition_balance = (
            torch.log(first_power + 1e-10) - torch.log(second_power + 1e-10)
        ).clamp(-8.0, 8.0) / 8.0
        spread = (
            torch.log((aligned - matched[:, None]).abs().square().mean(dim=1) + 1e-10)
            - torch.log(matched.abs().square() + 1e-10)
        ).clamp(-8.0, 8.0) / 8.0
        return torch.cat(
            [
                normalized.real,
                normalized.imag,
                log_relative_magnitude,
                matched_unit.real[:, None],
                matched_unit.imag[:, None],
                partition_balance[:, None],
                spread[:, None],
            ],
            dim=1,
        )

    def forward(self, mixture: torch.Tensor) -> dict[str, torch.Tensor]:
        if (
            mixture.ndim != 4
            or mixture.shape[1] != self.sensor_count
            or not torch.is_complex(mixture)
        ):
            raise ValueError("mixture must be complex [B,M,F,T]")

        aligned = self.target_normalize(mixture)
        value = F.leaky_relu(
            self.input_projection(self._features_from_aligned(aligned)),
            negative_slope=0.2,
        )
        for block in self.blocks:
            value = block(value)
        batch, channels, frequency, frames = value.shape
        sequence = value.permute(0, 2, 3, 1).reshape(
            batch * frequency, frames, channels
        )
        sequence, _ = self.gru(sequence)
        raw = self.head(sequence).reshape(
            batch, frequency, frames, self.precision_parameter_count
        )
        precision, log_determinant = complex_cholesky_precision(
            raw, self.sensor_count
        )

        ones = torch.ones(
            batch,
            frequency,
            frames,
            self.sensor_count,
            1,
            dtype=torch.complex64,
            device=mixture.device,
        )
        precision_times_ones = precision @ ones
        denominator = (
            ones.conj().transpose(-1, -2) @ precision_times_ones
        ).real.squeeze(-1).squeeze(-1).clamp_min(1e-8)
        aligned_weight = precision_times_ones.squeeze(-1) / denominator[..., None]
        observation = aligned.permute(0, 2, 3, 1)
        estimate = (aligned_weight.conj() * observation).sum(dim=-1)

        transfer = self.target_transfer[None, :, None, :]
        original_weight = aligned_weight / transfer.conj()
        constraint = (original_weight.conj() * transfer).sum(dim=-1)
        residual = observation - estimate[..., None]
        residual_column = residual[..., None]
        residual_quadratic = (
            residual_column.conj().transpose(-1, -2)
            @ precision
            @ residual_column
        ).real.squeeze(-1).squeeze(-1)
        residual_scale = (residual_quadratic / (self.sensor_count - 1)).clamp_min(
            1e-10
        )
        variance = residual_scale / denominator
        relative_floor = (
            1e-6 * estimate.abs().square().mean(dim=1, keepdim=True).clamp_min(1e-10)
        )
        variance = variance + relative_floor
        return {
            "mean": estimate,
            "likelihood_variance": variance,
            "weight": original_weight,
            "aligned_weight": aligned_weight,
            "precision": precision,
            "log_determinant_precision": log_determinant,
            "constraint": constraint,
            "aligned_observation": aligned,
            "residual": residual.permute(0, 3, 1, 2),
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class SARCCorrector(nn.Module):
    """Predict a bounded complex residual from (z, r, v).

    The frozen main-result checkpoints use |z|, not sqrt(v), as the output
    scale. The variance v remains part of the input conditioning features.
    """

    def __init__(
        self,
        sensor_count: int,
        hidden: int = 32,
        frequency_hidden: int = 24,
        temporal_hidden: int = 48,
        correction_limit: float = 4.0,
    ) -> None:
        super().__init__()
        self.sensor_count = sensor_count
        self.standardized_correction_limit = correction_limit
        self.input_projection = nn.Conv2d(6 + 3 * sensor_count, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [CausalBlock(hidden, dilation) for dilation in (1, 2, 4, 8)]
        )
        self.frequency_gru = nn.GRU(
            hidden, frequency_hidden, batch_first=True, bidirectional=True
        )
        self.frequency_projection = nn.Linear(2 * frequency_hidden, hidden)
        self.temporal_gru = nn.GRU(hidden, temporal_hidden, batch_first=True)
        self.head = nn.Linear(temporal_hidden, 2)

    def forward(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        magnitude = mean.abs().clamp_min(1e-8)
        unit = mean / magnitude
        relative_variance = variance / (magnitude.square() + 1e-8)
        scalar_features = torch.stack(
            [
                torch.log(magnitude).clamp(-12.0, 4.0) / 8.0,
                unit.real,
                unit.imag,
                torch.log(relative_variance.clamp_min(1e-8)).clamp(-12.0, 12.0)
                / 12.0,
                torch.log(variance.clamp_min(1e-10)).clamp(-16.0, 4.0) / 10.0,
                torch.log1p(relative_variance).clamp(0.0, 12.0) / 12.0,
            ],
            dim=1,
        )
        scale = residual.abs().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-7)
        normalized = residual / scale
        residual_features = torch.cat(
            [
                normalized.real,
                normalized.imag,
                torch.log(normalized.abs().clamp_min(1e-6)).clamp(-6.0, 4.0) / 6.0,
            ],
            dim=1,
        )
        value = F.leaky_relu(
            self.input_projection(torch.cat([scalar_features, residual_features], dim=1)),
            negative_slope=0.2,
        )
        for block in self.blocks:
            value = block(value)
        batch, channels, frequency, frames = value.shape
        frequency_sequence = value.permute(0, 3, 2, 1).reshape(
            batch * frames, frequency, channels
        )
        frequency_sequence, _ = self.frequency_gru(frequency_sequence)
        frequency_sequence = self.frequency_projection(frequency_sequence)
        value = value + frequency_sequence.reshape(
            batch, frames, frequency, channels
        ).permute(0, 3, 2, 1)
        temporal_sequence = value.permute(0, 2, 3, 1).reshape(
            batch * frequency, frames, channels
        )
        temporal_sequence, _ = self.temporal_gru(temporal_sequence)
        raw = self.head(temporal_sequence).reshape(batch, frequency, frames, 2)
        relative_correction = torch.complex(
            self.standardized_correction_limit * torch.tanh(raw[..., 0]),
            self.standardized_correction_limit * torch.tanh(raw[..., 1]),
        )
        correction = magnitude * relative_correction
        return correction, relative_correction


class SARC(nn.Module):
    """Complete Structured Analytic-Residual Correction model."""

    def __init__(self, target_transfer: torch.Tensor, freeze_spatial: bool = False) -> None:
        super().__init__()
        self.spatial_model = SARCSpatial(target_transfer)
        self.score_model = SARCCorrector(self.spatial_model.sensor_count)
        self.freeze_spatial = freeze_spatial
        if freeze_spatial:
            for parameter in self.spatial_model.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "SARC":
        super().train(mode)
        if self.freeze_spatial:
            self.spatial_model.eval()
        return self

    def forward(self, mixture: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.freeze_spatial:
            with torch.no_grad():
                spatial = self.spatial_model(mixture)
        else:
            spatial = self.spatial_model(mixture)
        correction, relative_correction = self.score_model(
            spatial["mean"],
            spatial["likelihood_variance"],
            spatial["residual"],
        )
        return {
            **spatial,
            "spatial_mean": spatial["mean"],
            "mean": spatial["mean"] + correction,
            "correction": correction,
            "relative_correction": relative_correction,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


"""STFT helpers used by SARC inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class STFTConfig:
    sample_rate: int = 16_000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    center: bool = True


def make_window(config: STFTConfig, device: torch.device) -> torch.Tensor:
    return torch.hann_window(config.win_length, periodic=True, device=device).sqrt()


def stft(audio: torch.Tensor, window: torch.Tensor, config: STFTConfig) -> torch.Tensor:
    """Transform [M,N] or [B,M,N] real audio to complex STFT."""
    original_shape = audio.shape
    if audio.ndim == 2:
        audio = audio[None]
    if audio.ndim != 3:
        raise ValueError("audio must have shape [M,N] or [B,M,N]")
    batch, microphones, samples = audio.shape
    spectrum = torch.stft(
        audio.reshape(batch * microphones, samples),
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=config.center,
        return_complex=True,
    )
    spectrum = spectrum.reshape(batch, microphones, spectrum.shape[-2], spectrum.shape[-1])
    return spectrum[0] if len(original_shape) == 2 else spectrum


def istft(
    spectrum: torch.Tensor,
    window: torch.Tensor,
    config: STFTConfig,
    length: int,
) -> torch.Tensor:
    """Transform [F,T] or [B,F,T] complex STFT to real audio."""
    original_shape = spectrum.shape
    if spectrum.ndim == 2:
        spectrum = spectrum[None]
    if spectrum.ndim != 3:
        raise ValueError("spectrum must have shape [F,T] or [B,F,T]")
    audio = torch.istft(
        spectrum,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=config.center,
        length=length,
    )
    return audio[0] if len(original_shape) == 2 else audio


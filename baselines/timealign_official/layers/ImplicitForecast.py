"""Source-informed implicit trajectory controls for Stage C D19."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TwoLayerHead(nn.Module):
    """Two-layer MLP used by the amplitude and phase heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_width: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_width)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_width, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout(F.gelu(self.input(features))))


class ImplicitFrequencyReadout(nn.Module):
    """Predict a polar spectrum and synthesize one full future trajectory."""

    def __init__(
        self,
        readout_dim: int,
        history_length: int = 720,
        series_length: int = 720,
        hidden_width: int = 2048,
        dropout: float = 0.1,
        fourier_norm: str = "ortho",
        use_input_spectrum: bool = True,
    ) -> None:
        super().__init__()
        if readout_dim <= 0 or history_length <= 0 or series_length <= 0:
            raise ValueError("readout and trajectory dimensions must be positive")
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if fourier_norm not in {"backward", "forward", "ortho"}:
            raise ValueError("unsupported Fourier normalization")
        self.readout_dim = int(readout_dim)
        self.history_length = int(history_length)
        self.series_length = int(series_length)
        self.history_spectrum_bins = self.history_length // 2 + 1
        self.spectrum_bins = self.series_length // 2 + 1
        self.hidden_width = int(hidden_width)
        self.dropout = float(dropout)
        self.fourier_norm = fourier_norm
        self.use_input_spectrum = bool(use_input_spectrum)
        input_dim = self.readout_dim + self.history_spectrum_bins
        self.amplitude_head = _TwoLayerHead(
            input_dim,
            self.hidden_width,
            self.spectrum_bins,
            self.dropout,
        )
        self.phase_sine_head = _TwoLayerHead(
            input_dim,
            self.hidden_width,
            self.spectrum_bins,
            self.dropout,
        )
        self.phase_cosine_head = _TwoLayerHead(
            input_dim,
            self.hidden_width,
            self.spectrum_bins,
            self.dropout,
        )

    @property
    def decoder_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def history_spectrum(
        self,
        normalized_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if normalized_history.ndim != 3:
            raise ValueError("normalized_history must have shape [B, L, C]")
        if normalized_history.shape[1] != self.history_length:
            raise ValueError(
                "history length mismatch: "
                f"expected {self.history_length}, got {normalized_history.shape[1]}"
            )
        history = normalized_history.permute(0, 2, 1)
        spectrum = torch.fft.rfft(
            history,
            n=self.history_length,
            dim=-1,
            norm=self.fourier_norm,
        )
        return spectrum.abs(), torch.angle(spectrum)

    def polar_spectrum(
        self,
        hidden: torch.Tensor,
        normalized_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.readout_dim:
            raise ValueError("hidden must have shape [B, C, readout_dim]")
        history_amplitude, history_phase = self.history_spectrum(
            normalized_history
        )
        if not self.use_input_spectrum:
            history_amplitude = torch.zeros_like(history_amplitude)
            history_phase = torch.zeros_like(history_phase)
        amplitude_logits = self.amplitude_head(
            torch.cat([hidden, history_amplitude], dim=-1)
        )
        amplitude = F.leaky_relu(
            amplitude_logits,
            negative_slope=0.5,
        ).abs()
        phase_features = torch.cat([hidden, history_phase], dim=-1)
        phase_sine = torch.tanh(self.phase_sine_head(phase_features))
        phase_cosine = torch.tanh(self.phase_cosine_head(phase_features))
        phase = torch.atan2(phase_sine, phase_cosine)
        return amplitude, phase, phase_sine, phase_cosine

    def full_forecast(
        self,
        hidden: torch.Tensor,
        normalized_history: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        amplitude, phase, phase_sine, phase_cosine = self.polar_spectrum(
            hidden,
            normalized_history,
        )
        spectrum = torch.polar(amplitude, phase)
        forecast = torch.fft.irfft(
            spectrum,
            n=self.series_length,
            dim=-1,
            norm=self.fourier_norm,
        )
        diagnostics = {
            "amplitude": amplitude,
            "phase": phase,
            "phase_sine": phase_sine,
            "phase_cosine": phase_cosine,
        }
        return forecast, diagnostics

    def forward(
        self,
        hidden: torch.Tensor,
        normalized_history: torch.Tensor,
        target_prefix: int | None = None,
    ) -> torch.Tensor:
        horizon = self.series_length if target_prefix is None else int(target_prefix)
        if horizon <= 0 or horizon > self.series_length:
            raise ValueError("target_prefix must lie in [1, series_length]")
        forecast, _diagnostics = self.full_forecast(
            hidden,
            normalized_history,
        )
        return forecast[:, :, :horizon].permute(0, 2, 1)


class DirectNonlinearMatchedReadout(nn.Module):
    """Parameter-matched nonlinear control using the same history information."""

    def __init__(
        self,
        readout_dim: int,
        hidden_width: int,
        history_length: int = 720,
        series_length: int = 720,
        dropout: float = 0.1,
        fourier_norm: str = "ortho",
    ) -> None:
        super().__init__()
        if readout_dim <= 0 or hidden_width <= 0:
            raise ValueError("readout_dim and hidden_width must be positive")
        if history_length <= 0 or series_length <= 0:
            raise ValueError("trajectory dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.readout_dim = int(readout_dim)
        self.hidden_width = int(hidden_width)
        self.history_length = int(history_length)
        self.series_length = int(series_length)
        self.history_spectrum_bins = self.history_length // 2 + 1
        self.fourier_norm = fourier_norm
        input_dim = self.readout_dim + 2 * self.history_spectrum_bins
        self.network = _TwoLayerHead(
            input_dim,
            self.hidden_width,
            self.series_length,
            dropout,
        )

    @property
    def decoder_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def history_features(self, normalized_history: torch.Tensor) -> torch.Tensor:
        if normalized_history.ndim != 3:
            raise ValueError("normalized_history must have shape [B, L, C]")
        if normalized_history.shape[1] != self.history_length:
            raise ValueError("history length does not match the frozen contract")
        spectrum = torch.fft.rfft(
            normalized_history.permute(0, 2, 1),
            n=self.history_length,
            dim=-1,
            norm=self.fourier_norm,
        )
        return torch.cat([spectrum.abs(), torch.angle(spectrum)], dim=-1)

    def forward(
        self,
        hidden: torch.Tensor,
        normalized_history: torch.Tensor,
        target_prefix: int | None = None,
    ) -> torch.Tensor:
        horizon = self.series_length if target_prefix is None else int(target_prefix)
        if horizon <= 0 or horizon > self.series_length:
            raise ValueError("target_prefix must lie in [1, series_length]")
        features = torch.cat(
            [hidden, self.history_features(normalized_history)],
            dim=-1,
        )
        forecast = self.network(features)
        return forecast[:, :, :horizon].permute(0, 2, 1)

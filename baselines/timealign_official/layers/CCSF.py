"""Contrast-conditioned scope fusion and competence calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from layers.PCC import prefix_measure
from layers.SIFF import SIFFCouplingFieldReadout, siff_parameter_count


CCSF_CORRECTION_MODES = frozenset({"true", "zero", "permuted"})
CCSF_OBJECTIVE_MODES = frozenset(
    {
        "ccsf_relative_calibration",
        "ccsf_standardized_calibration",
    }
)
CCSF_CONTRAST_DIMENSION = 6
CCSF_CALIBRATION_WEIGHT = 0.1
CCSF_RELATIVE_EPSILON = 1e-6


def ccsf_parameter_count(
    readout_dim: int,
    series_length: int = 720,
    scale_count: int = 5,
    coordinate_dim: int = 4,
    mode_rank: int = 256,
    scale_components: int = 2,
    policy_history_dim: int = 32,
    policy_hidden_dim: int = 64,
    contrast_dimension: int = CCSF_CONTRAST_DIMENSION,
    correction_hidden_dim: int = 64,
) -> int:
    """Return the SIFF base plus shared contrast-correction parameters."""
    base = siff_parameter_count(
        readout_dim=readout_dim,
        series_length=series_length,
        scale_count=scale_count,
        coordinate_dim=coordinate_dim,
        mode_rank=mode_rank,
        scale_components=scale_components,
        policy_history_dim=policy_history_dim,
        policy_hidden_dim=policy_hidden_dim,
    )
    correction_input = (
        policy_history_dim + coordinate_dim + 1 + contrast_dimension
    )
    correction = correction_input * correction_hidden_dim
    correction += correction_hidden_dim
    correction += correction_hidden_dim + 1
    return base + correction


class CCSFCouplingFieldReadout(SIFFCouplingFieldReadout):
    """Fuse SIFF scope forecasts using their target-wise contrast field.

    The parent SIFF logits remain active. A scope-shared scorer observes the
    existing history state, target coordinate, scale coordinate, and a
    six-dimensional descriptor constructed only from same-forward arm
    forecasts. The complete T-domain policy is computed before prefix crop.
    """

    def __init__(
        self,
        *args: Any,
        correction_mode: str = "true",
        contrast_dimension: int = CCSF_CONTRAST_DIMENSION,
        correction_hidden_dim: int = 64,
        contrast_epsilon: float = CCSF_RELATIVE_EPSILON,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.correction_mode = str(correction_mode)
        self.contrast_dimension = int(contrast_dimension)
        self.correction_hidden_dim = int(correction_hidden_dim)
        self.contrast_epsilon = float(contrast_epsilon)
        if self.correction_mode not in CCSF_CORRECTION_MODES:
            raise ValueError(
                f"unsupported CCSF correction mode: {self.correction_mode}"
            )
        if self.policy_mode != "direct":
            raise ValueError("CCSF v1 requires the learned direct policy")
        if self.contrast_dimension != CCSF_CONTRAST_DIMENSION:
            raise ValueError("CCSF v1 requires a six-dimensional contrast")
        if self.correction_hidden_dim <= 0 or self.contrast_epsilon <= 0.0:
            raise ValueError("CCSF correction dimensions must be positive")

        scale_coordinate = torch.log(
            torch.tensor(self.scales, dtype=torch.float64)
        ) / torch.log(
            torch.tensor(float(self.series_length), dtype=torch.float64)
        )
        permutation = torch.roll(
            torch.arange(len(self.scales), dtype=torch.long),
            shifts=-1,
        )
        self.register_buffer("ccsf_scale_coordinate", scale_coordinate.float())
        self.register_buffer("ccsf_contrast_permutation", permutation)

        correction_input = (
            self.policy_history_dim
            + self.coordinate_dim
            + 1
            + self.contrast_dimension
        )
        self.correction_hidden = nn.Linear(
            correction_input,
            self.correction_hidden_dim,
        )
        self.correction_output = nn.Linear(self.correction_hidden_dim, 1)
        self.correction_hidden.reset_parameters()
        nn.init.zeros_(self.correction_output.weight)
        nn.init.zeros_(self.correction_output.bias)

    def _true_contrast_descriptor(self, arms: Tensor) -> Tensor:
        """Map ``[B,C,S,T]`` arms to ``[B,C,T,S,6]`` descriptors."""
        if arms.ndim != 4:
            raise ValueError(
                f"expected arms [B,C,S,T], got {tuple(arms.shape)}"
            )
        expected = (
            arms.shape[0],
            arms.shape[1],
            len(self.scales),
            self.series_length,
        )
        if tuple(arms.shape) != expected:
            raise ValueError(
                "expected arms [B,C,S,T] with frozen scope/domain sizes; "
                f"got {tuple(arms.shape)}"
            )
        consensus = arms.mean(dim=2, keepdim=True)
        centered = arms - consensus
        disagreement = torch.sqrt(
            centered.square().mean(dim=2, keepdim=True)
            + self.contrast_epsilon
        )
        normalized = centered / disagreement
        relative = disagreement.squeeze(2) / (
            arms.abs().mean(dim=2) + self.contrast_epsilon
        )
        log_relative = torch.log1p(relative)

        descriptors = []
        for scale_index in range(len(self.scales)):
            indices = self.group_indices(scale_index)
            values = normalized[:, :, scale_index, :][:, :, indices]
            group_mean = values.mean(dim=-1)
            group_rms = values.square().mean(dim=-1).sqrt()
            endpoint = values[..., -1] - values[..., 0]
            labels = self.target_group_labels(scale_index)
            descriptors.append(
                torch.stack(
                    (
                        normalized[:, :, scale_index, :],
                        normalized[:, :, scale_index, :].abs(),
                        group_mean.index_select(-1, labels),
                        group_rms.index_select(-1, labels),
                        endpoint.index_select(-1, labels),
                        log_relative,
                    ),
                    dim=-1,
                )
            )
        return torch.stack(descriptors, dim=3)

    def contrast_descriptor(self, arms: Tensor) -> Tensor:
        """Return the true, zero, or fixed-permutation descriptor control."""
        descriptor = self._true_contrast_descriptor(arms)
        if self.correction_mode == "zero":
            return torch.zeros_like(descriptor)
        if self.correction_mode == "permuted":
            return descriptor.index_select(3, self.ccsf_contrast_permutation)
        return descriptor

    def correction_logits(
        self,
        hidden: Tensor,
        descriptor: Tensor,
    ) -> Tensor:
        """Return scope-shared logit corrections with shape ``[B,C,T,S]``."""
        history_state = self.history_projection(hidden)
        target_coordinate = self.coordinate_field.to(dtype=hidden.dtype)
        scale_coordinate = self.ccsf_scale_coordinate.to(dtype=hidden.dtype)
        chunks = []
        for start in range(0, self.series_length, self.target_chunk_size):
            end = min(start + self.target_chunk_size, self.series_length)
            target_count = end - start
            history_chunk = history_state[:, :, None, None, :].expand(
                -1,
                -1,
                target_count,
                len(self.scales),
                -1,
            )
            target_chunk = target_coordinate[start:end].view(
                1,
                1,
                target_count,
                1,
                self.coordinate_dim,
            ).expand(
                hidden.shape[0],
                hidden.shape[1],
                -1,
                len(self.scales),
                -1,
            )
            scale_chunk = scale_coordinate.view(
                1,
                1,
                1,
                len(self.scales),
                1,
            ).expand(
                hidden.shape[0],
                hidden.shape[1],
                target_count,
                -1,
                -1,
            )
            correction_input = torch.cat(
                (
                    history_chunk,
                    target_chunk,
                    scale_chunk,
                    descriptor[:, :, start:end],
                ),
                dim=-1,
            )
            chunks.append(
                self.correction_output(
                    F.gelu(self.correction_hidden(correction_input))
                ).squeeze(-1)
            )
        return torch.cat(chunks, dim=2)

    def policy_tensors(
        self,
        hidden: Tensor,
        arms: Tensor,
    ) -> dict[str, Tensor]:
        """Return base/correction/final policy tensors for one arm field."""
        base_logits = super()._learned_policy_logits(hidden)
        descriptor = self.contrast_descriptor(arms)
        correction = self.correction_logits(hidden, descriptor)
        policy = torch.softmax(base_logits + correction, dim=-1)
        return {
            "base_logits": base_logits,
            "base_policy": torch.softmax(base_logits, dim=-1),
            "contrast_descriptor": descriptor,
            "correction_logits": correction,
            "policy": policy,
        }

    def policy_weights(
        self,
        hidden: Tensor,
        arms: Tensor | None = None,
    ) -> Tensor:
        """Return final policy, computing arms only when not already supplied."""
        if self.policy_mode != "direct":
            return super().policy_weights(hidden)
        actual_arms = self.arm_forecasts(hidden) if arms is None else arms
        return self.policy_tensors(hidden, actual_arms)["policy"]

    def forward_with_ccsf_diagnostics(
        self,
        hidden: Tensor,
        target_prefix: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Return forecast, arms, policy, and contrast-policy diagnostics."""
        horizon = self.series_length if target_prefix is None else int(target_prefix)
        if horizon <= 0 or horizon > self.series_length:
            raise ValueError("target_prefix must lie in [1, series_length]")
        arms = self.arm_forecasts(hidden)
        tensors = self.policy_tensors(hidden, arms)
        weights = tensors["policy"]
        full = (arms * weights.permute(0, 1, 3, 2)).sum(dim=2)
        return full[:, :, :horizon].permute(0, 2, 1), arms, weights, tensors

    def forward_with_diagnostics(
        self,
        hidden: Tensor,
        target_prefix: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        output, arms, weights, _tensors = self.forward_with_ccsf_diagnostics(
            hidden,
            target_prefix,
        )
        return output, arms, weights

    @property
    def correction_parameters(self) -> int:
        """Return trainable parameters added beyond the SIFF parent."""
        return sum(
            parameter.numel()
            for module in (self.correction_hidden, self.correction_output)
            for parameter in module.parameters()
        )


@dataclass
class CCSFObjectiveResult:
    """Loss decomposition and stopped competence tensors."""

    total_loss: Tensor
    fused_loss: Tensor
    skill_loss: Tensor
    calibration_loss: Tensor
    weighted_calibration_loss: Tensor
    teacher: Tensor
    teacher_confidence: Tensor
    measure: Tensor
    diagnostics: dict[str, Tensor]


def _relative_teacher(
    arm_error: Tensor,
    temperature: float,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    mean_error = arm_error.mean(dim=-1, keepdim=True)
    relative_regret = (arm_error - mean_error) / mean_error.clamp_min(epsilon)
    teacher = torch.softmax(-relative_regret.detach() / temperature, dim=-1)
    scopes = arm_error.shape[-1]
    normalized_entropy = -(
        teacher * teacher.clamp_min(1e-12).log()
    ).sum(dim=-1) / torch.log(teacher.new_tensor(float(scopes)))
    return teacher, (1.0 - normalized_entropy).detach()


def _standardized_teacher(
    arm_error: Tensor,
    temperature: float,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    source = arm_error.detach()
    centered = source - source.mean(dim=-1, keepdim=True)
    scale = source.var(dim=-1, keepdim=True, unbiased=False).sqrt()
    teacher = torch.softmax(
        -centered / (scale.clamp_min(epsilon) * temperature),
        dim=-1,
    )
    confidence = teacher.new_ones(teacher.shape[:-1])
    return teacher, confidence


def contrast_scope_calibration_loss(
    fused_forecast: Tensor,
    arm_forecasts: Tensor,
    policy: Tensor,
    target: Tensor,
    *,
    mode: str,
    progress: float,
    temperature: float,
    calibration_weight: float = CCSF_CALIBRATION_WEIGHT,
    epsilon: float = CCSF_RELATIVE_EPSILON,
) -> CCSFObjectiveResult:
    """Compute equal-skill forecasting plus stopped competence calibration."""
    del progress
    if mode not in CCSF_OBJECTIVE_MODES:
        raise ValueError(f"unsupported CCSF objective mode: {mode}")
    if temperature <= 0.0 or calibration_weight < 0.0 or epsilon <= 0.0:
        raise ValueError("CCSF objective scalars are outside their domain")
    if fused_forecast.shape != target.shape or target.ndim != 3:
        raise ValueError("fused_forecast and target must share shape [B,T,C]")
    expected = (
        target.shape[0],
        target.shape[2],
        target.shape[1],
        policy.shape[-1],
    )
    if tuple(arm_forecasts.shape) != expected or tuple(policy.shape) != expected:
        raise ValueError(
            "arm_forecasts and policy must share [B,C,T,S]; "
            f"expected {expected}"
        )

    length = target.shape[1]
    measure = prefix_measure(length, device=target.device, dtype=target.dtype)
    target_bct = target.permute(0, 2, 1)
    fused_error = (fused_forecast - target).abs().permute(0, 2, 1)
    arm_error = (arm_forecasts - target_bct.unsqueeze(-1)).abs()
    fused_loss = (
        fused_error * measure.view(1, 1, length)
    ).sum(dim=-1).mean()
    skill_loss = (
        arm_error.mean(dim=-1) * measure.view(1, 1, length)
    ).sum(dim=-1).mean()

    if mode == "ccsf_relative_calibration":
        teacher, confidence = _relative_teacher(
            arm_error,
            temperature,
            epsilon,
        )
    else:
        teacher, confidence = _standardized_teacher(
            arm_error,
            temperature,
            epsilon,
        )
    route_kl = (
        teacher
        * (
            teacher.clamp_min(1e-12).log()
            - policy.clamp_min(1e-12).log()
        )
    ).sum(dim=-1)
    calibration_loss = (
        route_kl * confidence * measure.view(1, 1, length)
    ).sum(dim=-1).mean()
    weighted_calibration = calibration_weight * calibration_loss
    total = fused_loss + skill_loss + weighted_calibration

    scopes = policy.shape[-1]
    entropy_denominator = torch.log(policy.new_tensor(float(scopes)))
    policy_entropy = -(
        policy * policy.clamp_min(1e-12).log()
    ).sum(dim=-1) / entropy_denominator
    teacher_entropy = -(
        teacher * teacher.clamp_min(1e-12).log()
    ).sum(dim=-1) / entropy_denominator
    coordinate_weight = measure.view(1, 1, length)
    accuracy = (policy.argmax(dim=-1) == teacher.argmax(dim=-1)).to(policy)
    diagnostics = {
        "ccsf_total_loss": total.detach(),
        "ccsf_fused_measure_l1": fused_loss.detach(),
        "ccsf_equal_skill_l1": skill_loss.detach(),
        "ccsf_calibration_kl": calibration_loss.detach(),
        "ccsf_weighted_calibration_kl": weighted_calibration.detach(),
        "ccsf_calibration_weight": total.new_tensor(calibration_weight),
        "ccsf_temperature": total.new_tensor(temperature),
        "ccsf_teacher_normalized_entropy": (
            teacher_entropy * coordinate_weight
        ).sum(dim=-1).mean().detach(),
        "ccsf_teacher_confidence": (
            confidence * coordinate_weight
        ).sum(dim=-1).mean().detach(),
        "ccsf_policy_normalized_entropy": (
            policy_entropy * coordinate_weight
        ).sum(dim=-1).mean().detach(),
        "ccsf_teacher_policy_argmax_accuracy": (
            accuracy * coordinate_weight
        ).sum(dim=-1).mean().detach(),
    }
    arm_measure_l1 = (
        arm_error * measure.view(1, 1, length, 1)
    ).sum(dim=-2).mean(dim=(0, 1))
    for scope_index, value in enumerate(arm_measure_l1):
        diagnostics[f"ccsf_arm_s{scope_index}_measure_l1"] = value.detach()

    return CCSFObjectiveResult(
        total_loss=total,
        fused_loss=fused_loss,
        skill_loss=skill_loss,
        calibration_loss=calibration_loss,
        weighted_calibration_loss=weighted_calibration,
        teacher=teacher,
        teacher_confidence=confidence,
        measure=measure,
        diagnostics=diagnostics,
    )

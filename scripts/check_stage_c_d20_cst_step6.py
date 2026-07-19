#!/usr/bin/env python3
"""Validate the frozen SC-D20-CST Step 6 diagnostic design."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


EXPECTED_ARMS = {
    "A6_MEASURE_RETRAIN",
    "A6_CST_SPEC",
    "A6_CST_RANDOM",
}
EXPECTED_DATASETS = {"Weather", "ETTm1", "ETTh1", "ETTh2", "ETTm2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def check(condition: bool, name: str, details: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(condition), "details": details}


def real_fourier_projection(length: int, max_frequency: int) -> torch.Tensor:
    steps = torch.arange(length, dtype=torch.float64)
    columns = []
    scale = math.sqrt(2.0 / float(length))
    for frequency in range(1, max_frequency + 1):
        angle = 2.0 * math.pi * float(frequency) * steps / float(length)
        columns.extend((scale * torch.cos(angle), scale * torch.sin(angle)))
    return torch.stack(columns, dim=1)


def random_orthogonal_projection(
    length: int,
    dimension: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(
        length,
        dimension,
        generator=generator,
        dtype=torch.float64,
    )
    q_matrix, r_matrix = torch.linalg.qr(matrix, mode="reduced")
    signs = torch.where(
        torch.diagonal(r_matrix) < 0.0,
        -torch.ones(dimension, dtype=torch.float64),
        torch.ones(dimension, dtype=torch.float64),
    )
    return q_matrix * signs.unsqueeze(0)


def orthogonality_gap(projection: torch.Tensor) -> float:
    identity = torch.eye(projection.shape[1], dtype=projection.dtype)
    return float((projection.T @ projection - identity).abs().max().item())


def decoder_parameter_count(
    readout_dim: int,
    basis_rank: int,
    prediction_length: int,
    summary_dim: int,
) -> int:
    coefficient = (readout_dim + summary_dim) * basis_rank + basis_rank
    temporal_basis = prediction_length * basis_rank
    temporal_bias = prediction_length
    return coefficient + temporal_basis + temporal_bias


def synthetic_initialization_audit(
    readout_dim: int,
    summary_dim: int,
    basis_rank: int,
    prediction_length: int,
    spectrum_projection: torch.Tensor,
    random_projection: torch.Tensor,
) -> dict[str, float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260719)
    batch, channels = 2, 3
    hidden = torch.randn(
        batch,
        channels,
        readout_dim,
        generator=generator,
        dtype=torch.float64,
    )
    history = torch.randn(
        batch,
        channels,
        prediction_length,
        generator=generator,
        dtype=torch.float64,
    )
    weight = torch.randn(
        basis_rank,
        readout_dim,
        generator=generator,
        dtype=torch.float64,
    )
    bias = torch.randn(
        basis_rank,
        generator=generator,
        dtype=torch.float64,
    )
    basis = torch.randn(
        prediction_length,
        basis_rank,
        generator=generator,
        dtype=torch.float64,
    )
    temporal_bias = torch.randn(
        prediction_length,
        generator=generator,
        dtype=torch.float64,
    )
    zero_summary_weight = torch.zeros(
        basis_rank,
        summary_dim,
        dtype=torch.float64,
    )
    augmented_weight = torch.cat((weight, zero_summary_weight), dim=1)

    base_coeff = torch.einsum("bcr,kr->bck", hidden, weight) + bias
    base_output = torch.einsum("tk,bck->btc", basis, base_coeff)
    base_output = base_output + temporal_bias.view(1, -1, 1)

    outputs = {}
    for name, projection in {
        "spectrum": spectrum_projection,
        "random": random_projection,
    }.items():
        summary = torch.einsum("bct,tq->bcq", history, projection)
        augmented = torch.cat((hidden, summary), dim=-1)
        coeff = torch.einsum("bcr,kr->bck", augmented, augmented_weight) + bias
        outputs[name] = torch.einsum("tk,bck->btc", basis, coeff)
        outputs[name] = outputs[name] + temporal_bias.view(1, -1, 1)

    active_summary_weight = torch.randn(
        basis_rank,
        summary_dim,
        generator=generator,
        dtype=torch.float64,
    )
    spectrum_summary = torch.einsum("bct,tq->bcq", history, spectrum_projection)
    active_coeff = base_coeff + torch.einsum(
        "bcq,kq->bck",
        spectrum_summary,
        active_summary_weight,
    )
    active_output = torch.einsum("tk,bck->btc", basis, active_coeff)
    active_output = active_output + temporal_bias.view(1, -1, 1)

    prefix_gaps = []
    for horizon in (96, 192, 336, 720):
        cropped = active_output[:, :horizon]
        direct = torch.einsum("tk,bck->btc", basis[:horizon], active_coeff)
        direct = direct + temporal_bias[:horizon].view(1, -1, 1)
        prefix_gaps.append(float((cropped - direct).abs().max().item()))

    zero_weight_for_gradient = torch.zeros(
        basis_rank,
        summary_dim,
        dtype=torch.float64,
        requires_grad=True,
    )
    gradient_coeff = base_coeff + torch.einsum(
        "bcq,kq->bck",
        spectrum_summary,
        zero_weight_for_gradient,
    )
    gradient_output = torch.einsum("tk,bck->btc", basis, gradient_coeff)
    gradient_loss = gradient_output.square().mean()
    gradient_loss.backward()
    deformation_nrmse = float(
        (active_output - base_output).square().mean().sqrt().item()
        / base_output.square().mean().sqrt().clamp_min(1e-12).item()
    )

    return {
        "a6_vs_spectrum_max_abs": float(
            (base_output - outputs["spectrum"]).abs().max().item()
        ),
        "a6_vs_random_max_abs": float(
            (base_output - outputs["random"]).abs().max().item()
        ),
        "spectrum_vs_random_max_abs": float(
            (outputs["spectrum"] - outputs["random"]).abs().max().item()
        ),
        "prefix_max_abs": max(prefix_gaps),
        "active_prediction_deformation_nrmse": deformation_nrmse,
        "zero_init_summary_weight_gradient_norm": float(
            zero_weight_for_gradient.grad.norm().item()
        ),
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    profile_path = Path(config["carrier"]["profiles_path"])
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))
    length = int(config["carrier"]["history_length"])
    dimension = int(config["summary_contract"]["dimension"])
    max_frequency = int(
        config["summary_contract"]["spectrum"]["frequencies"][
            "end_inclusive"
        ]
    )
    random_seed = int(config["summary_contract"]["random_control"]["seed"])
    basis_rank = int(config["carrier"]["basis_rank"])
    prediction_length = int(config["carrier"]["full_prediction_length"])
    tolerance = float(
        config["internal_health_gates"][
            "projection_orthogonality_max_abs_max"
        ]
    )

    spectrum = real_fourier_projection(length, max_frequency)
    random_projection = random_orthogonal_projection(
        length,
        dimension,
        random_seed,
    )
    spectrum_gap = orthogonality_gap(spectrum)
    random_gap = orthogonality_gap(random_projection)
    spectrum_dc = float(spectrum.sum(dim=0).abs().max().item())

    results = []
    results.append(
        check(
            config["candidate_version"] == "SC-D20-CST-v1-step6"
            and config["role"] == "diagnostic_only"
            and config["paper_method_authorized"] is False,
            "diagnostic_identity_and_boundary",
            {
                "candidate_version": config["candidate_version"],
                "role": config["role"],
                "paper_method_authorized": config["paper_method_authorized"],
            },
        )
    )
    results.append(
        check(
            sha256(profile_path) == config["carrier"]["profiles_sha256"],
            "profile_contract_hash",
            {
                "computed": sha256(profile_path),
                "frozen": config["carrier"]["profiles_sha256"],
            },
        )
    )
    results.append(
        check(
            set(config["training"]["datasets"]) == EXPECTED_DATASETS,
            "five_dataset_contract",
            config["training"]["datasets"],
        )
    )
    results.append(
        check(
            {arm["id"] for arm in config["arms"]} == EXPECTED_ARMS,
            "three_arm_contract",
            [arm["id"] for arm in config["arms"]],
        )
    )
    results.append(
        check(
            dimension == 2 * max_frequency
            and spectrum.shape == (length, dimension)
            and random_projection.shape == (length, dimension),
            "projection_shapes",
            {
                "spectrum": list(spectrum.shape),
                "random": list(random_projection.shape),
                "dimension": dimension,
            },
        )
    )
    results.append(
        check(
            spectrum_gap <= tolerance and random_gap <= tolerance,
            "projection_orthogonality",
            {
                "spectrum_max_abs": spectrum_gap,
                "random_max_abs": random_gap,
                "tolerance": tolerance,
            },
        )
    )
    results.append(
        check(
            spectrum_dc
            <= config["internal_health_gates"][
                "projection_dc_leakage_max_abs_max"
            ],
            "spectrum_dc_exclusion",
            {
                "column_sum_max_abs": spectrum_dc,
                "tolerance": config["internal_health_gates"][
                    "projection_dc_leakage_max_abs_max"
                ],
            },
        )
    )

    parameter_rows = []
    parameter_match = []
    for dataset, profile in profiles["dataset_profiles"].items():
        readout_dim = int(profile["state_width"])
        base = decoder_parameter_count(
            readout_dim,
            basis_rank,
            prediction_length,
            summary_dim=0,
        )
        augmented = decoder_parameter_count(
            readout_dim,
            basis_rank,
            prediction_length,
            summary_dim=dimension,
        )
        row = {
            "dataset": dataset,
            "readout_dim": readout_dim,
            "a6_decoder_parameters": base,
            "spectrum_decoder_parameters": augmented,
            "random_decoder_parameters": augmented,
            "augmentation_parameters": augmented - base,
        }
        parameter_rows.append(row)
        parameter_match.append(augmented - base == dimension * basis_rank)
    results.append(
        check(
            all(parameter_match),
            "spec_random_parameter_and_shape_match",
            parameter_rows,
        )
    )

    initialization = synthetic_initialization_audit(
        readout_dim=768,
        summary_dim=dimension,
        basis_rank=basis_rank,
        prediction_length=prediction_length,
        spectrum_projection=spectrum,
        random_projection=random_projection,
    )
    initialization_tolerance = float(
        config["internal_health_gates"]["initial_output_max_abs_max"]
    )
    results.append(
        check(
            initialization["a6_vs_spectrum_max_abs"]
            <= initialization_tolerance
            and initialization["a6_vs_random_max_abs"]
            <= initialization_tolerance
            and initialization["spectrum_vs_random_max_abs"]
            <= initialization_tolerance,
            "function_preserving_paired_initialization",
            initialization,
        )
    )
    results.append(
        check(
            initialization["prefix_max_abs"]
            <= config["internal_health_gates"]["prefix_max_abs_max"],
            "active_summary_full_trajectory_prefix_crop",
            initialization,
        )
    )
    results.append(
        check(
            initialization["zero_init_summary_weight_gradient_norm"]
            >= config["internal_health_gates"]["summary_gradient_norm_min"]
            and initialization["active_prediction_deformation_nrmse"]
            >= config["internal_health_gates"][
                "prediction_deformation_nrmse_min"
            ],
            "summary_path_trainability_and_deformation",
            initialization,
        )
    )
    results.append(
        check(
            config["matrix"]["expected_runs"] == 15
            and config["matrix"]["new_training_runs"] == 15
            and config["matrix"]["reused_runs"] == 0
            and config["matrix"]["official_test_cells"] == 60,
            "matrix_accounting",
            config["matrix"],
        )
    )
    comparison_ids = {item["id"] for item in config["primary_comparisons"]}
    results.append(
        check(
            comparison_ids == set(config["primary_gates"])
            == {"transfer_spec_vs_a6", "specificity_spec_vs_random"},
            "transfer_and_specificity_gates",
            sorted(comparison_ids),
        )
    )
    results.append(
        check(
            config["authorization"]["step7a_local_implementation"] is True
            and config["authorization"]["remote_training"] is False
            and config["authorization"]["official_test_access"] is False
            and config["authorization"]["confirmation"] is False
            and config["authorization"]["paper_method"] is False,
            "authorization_boundary",
            config["authorization"],
        )
    )

    summary = {
        "candidate_version": config["candidate_version"],
        "config_path": str(args.config),
        "config_sha256": sha256(args.config),
        "checks_passed": sum(item["pass"] for item in results),
        "checks_total": len(results),
        "overall_pass": all(item["pass"] for item in results),
        "projection_diagnostics": {
            "spectrum_orthogonality_max_abs": spectrum_gap,
            "random_orthogonality_max_abs": random_gap,
            "spectrum_dc_leakage_max_abs": spectrum_dc,
            "cross_subspace_singular_value_max": float(
                torch.linalg.svdvals(spectrum.T @ random_projection).max().item()
            ),
        },
        "parameter_rows": parameter_rows,
        "initialization_diagnostics": initialization,
        "checks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the frozen SC-D19-IFC Step 6 control design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_ARMS = {
    "A6_MEASURE",
    "IF_MEASURE",
    "IF_NOSKIP_MEASURE",
    "DIRECT_NONLINEAR_MATCHED_MEASURE",
}
EXPECTED_DATASETS = {"Weather", "ETTm1", "ETTh1", "ETTh2", "ETTm2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def if_parameter_count(
    readout_dim: int,
    history_bins: int,
    spectrum_bins: int,
    hidden_width: int,
) -> int:
    input_dim = readout_dim + history_bins
    single_head = (
        input_dim * hidden_width
        + hidden_width
        + hidden_width * spectrum_bins
        + spectrum_bins
    )
    return 3 * single_head


def direct_parameter_count(
    readout_dim: int,
    history_bins: int,
    prediction_length: int,
    hidden_width: int,
) -> int:
    input_dim = readout_dim + 2 * history_bins
    return (
        input_dim * hidden_width
        + hidden_width
        + hidden_width * prediction_length
        + prediction_length
    )


def check(condition: bool, name: str, details: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(condition), "details": details}


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    implicit = config["implicit_forecaster"]
    results = []
    results.append(
        check(
            config["candidate_version"] == "SC-D19-IFC-control-v1",
            "candidate_identity",
            config["candidate_version"],
        )
    )
    results.append(
        check(
            config["role"] == "control_only"
            and config["method_claim_authorized"] is False,
            "control_only_boundary",
            {
                "role": config["role"],
                "method_claim_authorized": config[
                    "method_claim_authorized"
                ],
            },
        )
    )
    results.append(
        check(
            set(config["data"]["datasets"]) == EXPECTED_DATASETS,
            "five_dataset_matrix",
            config["data"]["datasets"],
        )
    )
    results.append(
        check(
            {arm["id"] for arm in config["arms"]} == EXPECTED_ARMS,
            "four_arm_matrix",
            [arm["id"] for arm in config["arms"]],
        )
    )
    results.append(
        check(
            config["data"]["full_prediction_length"]
            == implicit["spectrum_size"]
            == implicit["irfft_length_explicit"]
            == 720,
            "full_trajectory_contract",
            {
                "prediction_length": config["data"][
                    "full_prediction_length"
                ],
                "spectrum_size": implicit["spectrum_size"],
                "irfft_length": implicit["irfft_length_explicit"],
            },
        )
    )
    results.append(
        check(
            config["shared_training"]["prediction_loss_mode"]
            == "measure_only"
            and config["shared_training"]["checkpoint_rule"]
            == "mean_validation_mse_h96_h192_h336_h720",
            "shared_measure_and_checkpoint",
            config["shared_training"],
        )
    )

    parameter_rows = []
    parameter_checks = []
    for profile, contract in config["profile_contracts"].items():
        readout_dim = int(contract["readout_dim"])
        if_parameters = if_parameter_count(
            readout_dim=readout_dim,
            history_bins=int(implicit["history_spectrum_bins"]),
            spectrum_bins=int(implicit["spectrum_bins"]),
            hidden_width=int(implicit["hidden_width"]),
        )
        direct_parameters = direct_parameter_count(
            readout_dim=readout_dim,
            history_bins=int(implicit["history_spectrum_bins"]),
            prediction_length=int(
                config["data"]["full_prediction_length"]
            ),
            hidden_width=int(contract["direct_hidden_width"]),
        )
        gap = (
            100.0
            * abs(direct_parameters - if_parameters)
            / if_parameters
        )
        row = {
            "profile": profile,
            "readout_dim": readout_dim,
            "if_parameters_computed": if_parameters,
            "if_parameters_frozen": contract["if_decoder_parameters"],
            "direct_hidden_width": contract["direct_hidden_width"],
            "direct_parameters_computed": direct_parameters,
            "direct_parameters_frozen": contract[
                "direct_decoder_parameters"
            ],
            "relative_gap_percent": gap,
        }
        parameter_rows.append(row)
        parameter_checks.append(
            if_parameters == contract["if_decoder_parameters"]
            and direct_parameters
            == contract["direct_decoder_parameters"]
            and gap
            <= config["hard_gates"]["protocol"][
                "direct_parameter_gap_percent_max"
            ]
        )
    results.append(
        check(
            all(parameter_checks),
            "parameter_formula_and_tolerance",
            parameter_rows,
        )
    )
    results.append(
        check(
            config["phase_a"]["new_training_runs"] == 15
            and config["phase_a"]["reused_reference_runs"] == 5
            and config["phase_a"]["artifact_units"] == 20
            and config["phase_a"]["official_test_cells"] == 80,
            "phase_a_matrix_accounting",
            config["phase_a"],
        )
    )
    results.append(
        check(
            config["authorization"]["step7a_local"] is True
            and config["authorization"]["remote"] is False
            and config["authorization"]["official_test"] is False
            and config["authorization"]["paper_method"] is False,
            "authorization_boundary",
            config["authorization"],
        )
    )
    summary = {
        "candidate_version": config["candidate_version"],
        "checks_passed": sum(item["pass"] for item in results),
        "checks_total": len(results),
        "overall_pass": all(item["pass"] for item in results),
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

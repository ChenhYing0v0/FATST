#!/usr/bin/env python3
"""Statically audit the SC1-SIFF-v2-CCSF Step6 design contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPECTED_ARMS = {
    "a6_measure",
    "siff_v1_equal",
    "siff_v1_relcal",
    "ccsf_equal",
    "ccsf_relcal",
    "ccsf_stdcal",
    "ccsf_no_contrast_equal",
    "ccsf_no_contrast_relcal",
    "ccsf_permuted_contrast_relcal",
    "ccsf_independent_relcal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_siff_ccsf_step6.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parameter_count(config: dict[str, Any]) -> int:
    scorer = config["method"]["correction_scorer"]
    input_dimension = (
        int(scorer["history_dimension"])
        + int(scorer["target_coordinate_dimension"])
        + int(scorer["scale_coordinate_dimension"])
        + int(scorer["contrast_dimension"])
    )
    hidden = int(scorer["hidden_dimension"])
    output = int(scorer["output_dimension"])
    return input_dimension * hidden + hidden + hidden * output + output


def audit_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    arms = {arm["id"]: arm for arm in config["arms"]}
    comparisons = {row["id"]: row for row in config["comparisons"]}
    hard = set(config["hard_gates"]["comparison_ids"])
    rows = [
        {
            "check": "exact_arm_set",
            "actual": len(arms),
            "expected": len(EXPECTED_ARMS),
            "pass": set(arms) == EXPECTED_ARMS,
        },
        {
            "check": "comparison_references_exist",
            "actual": len(comparisons),
            "expected": len(config["comparisons"]),
            "pass": all(
                row["candidate"] in arms and row["reference"] in arms
                for row in comparisons.values()
            ),
        },
        {
            "check": "hard_comparisons_exist",
            "actual": len(hard),
            "expected": 10,
            "pass": hard.issubset(comparisons) and len(hard) == 10,
        },
        {
            "check": "phase_a_runs",
            "actual": config["matrix"]["phase_a_expected_runs"],
            "expected": len(arms)
            * len(config["datasets"])
            * len(config["phase_a_seeds"]),
            "pass": config["matrix"]["phase_a_expected_runs"]
            == len(arms)
            * len(config["datasets"])
            * len(config["phase_a_seeds"]),
        },
        {
            "check": "phase_a_test_cells",
            "actual": config["matrix"]["phase_a_expected_test_cells"],
            "expected": config["matrix"]["phase_a_expected_runs"]
            * len(config["matrix"]["horizons"]),
            "pass": config["matrix"]["phase_a_expected_test_cells"]
            == config["matrix"]["phase_a_expected_runs"]
            * len(config["matrix"]["horizons"]),
        },
        {
            "check": "temperature_pilot_runs",
            "actual": config["objective"]["temperature_selection"][
                "expected_runs"
            ],
            "expected": len(
                config["objective"]["temperature_selection"]["grid"]
            )
            * len(config["datasets"]),
            "pass": config["objective"]["temperature_selection"][
                "expected_runs"
            ]
            == len(config["objective"]["temperature_selection"]["grid"])
            * len(config["datasets"]),
        },
        {
            "check": "horizon_not_model_input",
            "actual": config["method"]["contrast_descriptor"][
                "requested_horizon_input"
            ],
            "expected": False,
            "pass": config["method"]["contrast_descriptor"][
                "requested_horizon_input"
            ]
            is False,
        },
        {
            "check": "benchmark_bins_not_model_input",
            "actual": config["method"]["contrast_descriptor"][
                "benchmark_horizon_bin_input"
            ],
            "expected": False,
            "pass": config["method"]["contrast_descriptor"][
                "benchmark_horizon_bin_input"
            ]
            is False,
        },
        {
            "check": "remote_not_authorized",
            "actual": config["authorization"]["remote_training_authorized"],
            "expected": False,
            "pass": config["authorization"]["remote_training_authorized"]
            is False,
        },
    ]
    return rows


def audit_parameters(config: dict[str, Any]) -> list[dict[str, Any]]:
    correction = parameter_count(config)
    expected = config["method"]["correction_scorer"]["expected_parameters"]
    if correction != expected:
        raise ValueError(
            f"correction parameter mismatch: {correction} != {expected}"
        )
    reference = config["parameter_reference"]
    rows = []
    for dataset in config["datasets"]:
        ordered = int(reference["ordered_base_total"][dataset]) + correction
        independent = (
            int(reference["independent_matched_base_total"][dataset])
            + correction
        )
        gap = abs(independent - ordered) / ordered
        rows.append(
            {
                "dataset": dataset,
                "correction_parameters": correction,
                "ordered_ccsf_parameters": ordered,
                "independent_ccsf_parameters": independent,
                "relative_gap": gap,
                "threshold": reference["matched_relative_gap_max"],
                "pass": gap <= reference["matched_relative_gap_max"],
            }
        )
    return rows


def audit_claims(config: dict[str, Any]) -> list[dict[str, Any]]:
    comparisons = {row["id"] for row in config["comparisons"]}
    hard = set(config["hard_gates"]["comparison_ids"])
    rows = []
    for claim, controls in config["claim_control_map"].items():
        rows.append(
            {
                "claim": claim,
                "controls": ",".join(controls),
                "control_count": len(controls),
                "all_exist": set(controls).issubset(comparisons),
                "hard_control_count": len(set(controls) & hard),
                "pass": set(controls).issubset(comparisons),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    matrix_rows = audit_matrix(config)
    parameter_rows = audit_parameters(config)
    claim_rows = audit_claims(config)
    gates = {
        "matrix_contract": all(bool(row["pass"]) for row in matrix_rows),
        "parameter_contract": all(
            bool(row["pass"]) for row in parameter_rows
        ),
        "claim_control_contract": all(bool(row["pass"]) for row in claim_rows),
        "narrative_conditional": config["narrative_gate"]["status"]
        == "conditional_pass_step7a_local_only",
        "step7a_local_only": bool(
            config["authorization"]["step7a_local_implementation_authorized"]
            and not config["authorization"]["remote_training_authorized"]
            and not config["authorization"]["formal_test_access_authorized"]
        ),
    }
    payload = {
        "candidate_version": config["candidate_version"],
        "current_step": 6,
        "gate_results": gates,
        "passed_gates": sum(gates.values()),
        "total_gates": len(gates),
        "decision": (
            "step6_pass_step7a_local_only"
            if all(gates.values())
            else "step6_fail_return_design"
        ),
        "implementation_authorized": all(gates.values()),
        "remote_authorized": False,
        "formal_test_authorized": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "matrix_audit.csv", matrix_rows)
    write_csv(args.output_dir / "parameter_audit.csv", parameter_rows)
    write_csv(args.output_dir / "claim_control_matrix.csv", claim_rows)
    (args.output_dir / "step6_gate.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check the frozen SIFF_EQUAL attribution Step 6 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_DATASETS = ["Weather", "ETTm1", "ETTm2", "ETTh1", "ETTh2"]
EXPECTED_ARMS = {
    "a6_full": ("learned-basis-forecast-operator", "off", "fixed_256"),
    "a6_measure": (
        "learned-basis-forecast-operator",
        "measure_only",
        "fixed_256",
    ),
    "pcsd_measure": ("pcsd-coupling-field", "measure_only", "fixed_256"),
    "pcsd_equal": ("pcsd-coupling-field", "equal_skill", "fixed_256"),
    "siff_measure": ("siff-coupling-field", "measure_only", "fixed_256"),
    "siff_equal": ("siff-coupling-field", "equal_skill", "fixed_256"),
    "siff_constant_equal": (
        "siff-constant-control",
        "equal_skill",
        "fixed_256",
    ),
    "siff_permuted_equal": (
        "siff-permuted-scale-control",
        "equal_skill",
        "fixed_256",
    ),
    "siff_q1_wide_equal": (
        "siff-q1-wide-control",
        "equal_skill",
        "q1_dataset_matched",
    ),
    "siff_independent_equal": (
        "siff-independent-scope-control",
        "equal_skill",
        "independent_dataset_matched",
    ),
}
EXPECTED_HARD_COMPARISONS = {
    "siff_equal_over_a6_full",
    "siff_equal_over_a6_measure",
    "siff_equal_over_pcsd_equal",
    "ordered_over_constant_equal",
    "ordered_over_permuted_equal",
    "ordered_over_q1_wide_equal",
    "ordered_over_independent_equal",
}
REQUIRED_LAYERS = {
    "paper_facing_effectiveness",
    "matched_mechanism_attribution",
    "internal_mechanism_health",
    "failure_attribution",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_siff_equal_attribution_v2.json"),
    )
    parser.add_argument(
        "--evaluation-protocol",
        type=Path,
        default=Path("configs/paper_facing_evaluation_protocol.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/stage_c_siff_equal_attribution_step6_20260718/"
            "step6_gate.json"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_case(
    cases: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> None:
    cases.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        }
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = json.loads(args.evaluation_protocol.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []

    add_case(
        cases,
        "candidate_identity",
        config["candidate_version"] == "SC1-SIFF-v2-EQ-ATTR-v1",
        config["candidate_version"],
        "SC1-SIFF-v2-EQ-ATTR-v1",
    )
    add_case(
        cases,
        "test_informed",
        config["test_informed"] is True,
        config["test_informed"],
        True,
    )
    add_case(
        cases,
        "datasets",
        config["datasets"] == EXPECTED_DATASETS,
        config["datasets"],
        EXPECTED_DATASETS,
    )

    arms = {arm["id"]: arm for arm in config["arms"]}
    observed_arms = {
        arm_id: (
            arm["readout_mode"],
            arm["objective_mode"],
            arm["rank_rule"],
        )
        for arm_id, arm in arms.items()
    }
    add_case(
        cases,
        "arm_contracts",
        observed_arms == EXPECTED_ARMS,
        observed_arms,
        EXPECTED_ARMS,
    )
    add_case(
        cases,
        "no_pcc_or_mcca_objective",
        all(
            "pcc" not in arm["objective_mode"]
            and "mcca" not in arm["objective_mode"]
            for arm in config["arms"]
        ),
        sorted({arm["objective_mode"] for arm in config["arms"]}),
        ["equal_skill", "measure_only", "off"],
    )

    profile_path = Path(config["profiles"]["path"])
    observed_profile_hash = sha256(profile_path)
    add_case(
        cases,
        "profile_hash",
        observed_profile_hash == config["profiles"]["sha256"],
        observed_profile_hash,
        config["profiles"]["sha256"],
    )

    matrix = config["matrix"]
    phase_a_runs = len(config["datasets"]) * len(config["arms"]) * len(
        config["seeds"]
    )
    phase_a_cells = phase_a_runs * len(matrix["horizons"])
    confirmation_runs = (
        len(config["datasets"])
        * len(config["arms"])
        * len(config["confirmation_seeds"])
    )
    confirmation_cells = confirmation_runs * len(matrix["horizons"])
    add_case(
        cases,
        "phase_a_matrix",
        (
            phase_a_runs == matrix["phase_a_expected_runs"]
            and phase_a_cells == matrix["phase_a_expected_test_cells"]
        ),
        [phase_a_runs, phase_a_cells],
        [
            matrix["phase_a_expected_runs"],
            matrix["phase_a_expected_test_cells"],
        ],
    )
    add_case(
        cases,
        "confirmation_matrix",
        (
            confirmation_runs == matrix["confirmation_expected_runs"]
            and confirmation_cells
            == matrix["confirmation_expected_test_cells"]
        ),
        [confirmation_runs, confirmation_cells],
        [
            matrix["confirmation_expected_runs"],
            matrix["confirmation_expected_test_cells"],
        ],
    )

    comparisons = {row["id"]: row for row in config["comparisons"]}
    hard_ids = set(config["effectiveness_gates"]["hard_comparison_ids"])
    add_case(
        cases,
        "hard_comparison_set",
        hard_ids == EXPECTED_HARD_COMPARISONS,
        sorted(hard_ids),
        sorted(EXPECTED_HARD_COMPARISONS),
    )
    add_case(
        cases,
        "hard_comparisons_defined",
        hard_ids.issubset(comparisons),
        sorted(comparisons),
        sorted(hard_ids),
    )
    add_case(
        cases,
        "all_comparison_arms_defined",
        all(
            row["candidate"] in arms and row["reference"] in arms
            for row in config["comparisons"]
        ),
        True,
        True,
    )

    training = config["training"]
    add_case(
        cases,
        "checkpoint_contract",
        (
            training["validation_horizons"] == [96, 192, 336, 720]
            and training["validation_checkpoint_score"]
            == "mean_mse_h96_h192_h336_h720"
            and training["test_labels_select_checkpoint"] is False
        ),
        [
            training["validation_horizons"],
            training["validation_checkpoint_score"],
            training["test_labels_select_checkpoint"],
        ],
        [
            [96, 192, 336, 720],
            "mean_mse_h96_h192_h336_h720",
            False,
        ],
    )
    add_case(
        cases,
        "joint_from_scratch",
        training["from_scratch"] and training["joint_encoder_decoder"],
        [training["from_scratch"], training["joint_encoder_decoder"]],
        [True, True],
    )

    protocol_layers = set(
        protocol["mechanism_evaluation_layers"]["required"]
    )
    add_case(
        cases,
        "four_layer_protocol",
        protocol_layers == REQUIRED_LAYERS,
        sorted(protocol_layers),
        sorted(REQUIRED_LAYERS),
    )
    health = config["internal_mechanism_health"]
    add_case(
        cases,
        "internal_health_cannot_rescue_effectiveness",
        health["diagnostics_cannot_override_effectiveness"] is True,
        health["diagnostics_cannot_override_effectiveness"],
        True,
    )
    authorization = config["authorization"]
    add_case(
        cases,
        "authorization_boundary",
        (
            authorization["step7a_local_implementation_authorized"] is True
            and authorization["remote_training_authorized"] is False
            and authorization["formal_test_access_authorized"] is False
        ),
        authorization,
        "Step7A local only",
    )

    payload = {
        "candidate_version": config["candidate_version"],
        "current_step": config["current_step"],
        "cases": cases,
        "checks_passed": sum(case["passed"] for case in cases),
        "checks_total": len(cases),
        "passed_cases": sum(case["passed"] for case in cases),
        "total_cases": len(cases),
        "passed": all(case["passed"] for case in cases),
        "profile_contract_hash": observed_profile_hash,
        "phase_a_runs": phase_a_runs,
        "phase_a_test_cells": phase_a_cells,
        "confirmation_runs": confirmation_runs,
        "confirmation_test_cells": confirmation_cells,
        "next_step": (
            "Step7A local implementation"
            if all(case["passed"] for case in cases)
            else "repair Step6 protocol"
        ),
        "remote_authorized": False,
        "test_authorized": False,
        "remote_training_authorized": False,
        "formal_test_access_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not payload["passed"]:
        raise SystemExit("SIFF_EQUAL attribution Step6 gate failed")


if __name__ == "__main__":
    main()

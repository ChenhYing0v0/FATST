#!/usr/bin/env python3
"""Compare the frozen PCSD-CF validation screen with its milestone test audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _comparison_map(root: Path) -> dict[tuple[str, str], float]:
    rows = _read_csv(root / "direct_comparisons.csv")
    return {
        (row["reference"], row["dataset"]): float(row["direct_gain_percent"])
        for row in rows
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    validation = _comparison_map(args.validation_root)
    test = _comparison_map(args.test_root)
    if validation.keys() != test.keys():
        raise ValueError("Validation and test comparison matrices differ")

    rows: list[dict[str, Any]] = []
    for reference, dataset in sorted(validation):
        validation_gain = validation[(reference, dataset)]
        test_gain = test[(reference, dataset)]
        rows.append(
            {
                "reference": reference,
                "dataset": dataset,
                "validation_gain_percent": validation_gain,
                "test_gain_percent": test_gain,
                "test_minus_validation_pp": test_gain - validation_gain,
                "validation_win": validation_gain > 0,
                "test_win": test_gain > 0,
                "sign_reversal": (validation_gain > 0) != (test_gain > 0),
            }
        )

    test_gate = json.loads((args.test_root / "gate.json").read_text())
    deep_gate = json.loads((args.test_root / "deep_dive_gate.json").read_text())
    invariants = list((args.test_root / "raw").rglob("test_audit_invariants.json"))
    invariant_rows = [json.loads(path.read_text()) for path in invariants]

    a6_macro = next(
        row for row in rows if row["reference"] == "a6" and row["dataset"] == "macro"
    )
    positive_oracle_datasets = int(
        deep_gate["positive_same_run_oracle_headroom_datasets"]
    )
    decision = (
        "test_fail_with_arm_headroom"
        if not test_gate["method_pass"] and positive_oracle_datasets > 0
        else "test_method_pass"
        if test_gate["method_pass"]
        else "test_fail_without_arm_headroom"
    )
    summary = {
        "audit_id": "SC-D15-T1",
        "candidate_version": "SC1-PCSD-CF-v1",
        "validation_run_count": int(test_gate["expected_runs"]),
        "test_run_count": int(test_gate["valid_runs"]),
        "test_matrix_complete": bool(test_gate["complete"]),
        "test_invariant_file_count": len(invariants),
        "all_test_invariants_pass": len(invariants) == 60
        and all(row["pass"] for row in invariant_rows),
        "all_checkpoints_frozen": len(invariants) == 60
        and all(not row["checkpoint_retrained"] for row in invariant_rows),
        "all_test_access_authorized": len(invariants) == 60
        and all(row["test_access_authorized"] for row in invariant_rows),
        "direct_over_a6_validation_macro_percent": a6_macro[
            "validation_gain_percent"
        ],
        "direct_over_a6_test_macro_percent": a6_macro["test_gain_percent"],
        "direct_over_a6_test_dataset_wins": int(
            test_gate["macro_comparisons"]["a6"]["dataset_wins"]
        ),
        "method_pass": bool(test_gate["method_pass"]),
        "same_run_oracle_headroom_test_macro_percent": 100.0
        * float(test_gate["same_run_oracle_headroom_macro"]),
        "positive_same_run_oracle_headroom_datasets": positive_oracle_datasets,
        "direct_arm_undertrained_pairs": int(deep_gate["direct_arm_undertrained_pairs"]),
        "direct_arm_total_pairs": int(deep_gate["direct_arm_total_pairs"]),
        "direct_arm_degradation_test_median_percent": float(
            deep_gate["direct_arm_degradation_percent_median"]
        ),
        "validation_test_sign_reversal_cells": sum(
            bool(row["sign_reversal"]) for row in rows if row["dataset"] != "macro"
        ),
        "decision": decision,
        "pcc_step6_status": "authorized_test_informed_design_only"
        if decision == "test_fail_with_arm_headroom"
        else "held",
        "paper_claim_status": "pcsd_cf_v1_not_supported",
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_root / "validation_test_comparison.csv", rows)
    (args.output_root / "audit_decision.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

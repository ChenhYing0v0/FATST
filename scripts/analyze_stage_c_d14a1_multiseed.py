#!/usr/bin/env python3
"""Apply the preregistered three-seed D14-A1 dual-carrier confirmation gate."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
CARRIERS = ("neutral_raw", "a6_natural")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"invalid boolean value: {value}")
    return value == "True"


def load_rows(
    raw_root: Path,
    carrier: str,
    seeds: list[int],
) -> dict[tuple[str, int], dict[str, str]]:
    payload: dict[tuple[str, int], dict[str, str]] = {}
    for seed in seeds:
        path = raw_root / f"_analysis_{carrier}_seed{seed}" / "dataset_metrics.csv"
        rows = read_csv(path)
        if {row["dataset"] for row in rows} != set(DATASETS):
            raise RuntimeError(f"dataset coverage mismatch: {path}")
        for row in rows:
            payload[(row["dataset"], seed)] = row
    return payload


def carrier_confirmation(
    carrier: str,
    rows: dict[tuple[str, int], dict[str, str]],
    seeds: list[int],
    design: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    confirmation = design["confirmation_gates"]
    remote = design["remote_gates"]
    stable_seed_required = int(confirmation["stable_seed_required"])
    dataset_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        units = [rows[(dataset, seed)] for seed in seeds]
        pair_counts: Counter[str] = Counter()
        for unit in units:
            pair_counts.update(filter(None, unit["crossing_pairs"].split(";")))
        stable_pairs = sorted(
            pair
            for pair, count in pair_counts.items()
            if count >= stable_seed_required
        )
        separation_seed_count = sum(
            float(unit["prediction_disagreement_median"])
            >= float(remote["minimum_prediction_disagreement"])
            for unit in units
        )
        skill_seed_count = sum(
            float(unit["carrier_skill_relative_gain"])
            >= float(remote["carrier_skill_gain_min"])
            for unit in units
        )
        contiguity_seed_count = sum(
            float(unit["canonical_vs_random_oracle_relative_gain"]) > 0.0
            for unit in units
        )
        dataset_rows.append(
            {
                "carrier": carrier,
                "dataset": dataset,
                "seed_count": len(seeds),
                "function_separation_seed_count": separation_seed_count,
                "carrier_skill_seed_count": skill_seed_count,
                "stable_crossing": bool(stable_pairs),
                "stable_crossing_pairs": ";".join(stable_pairs),
                "strict_oracle_macro_gain": mean(
                    float(unit["oracle_vs_validation_best_fixed_gain"])
                    for unit in units
                ),
                "sample_over_bin_policy_macro_gain": mean(
                    float(unit["sample_oracle_vs_validation_bin_policy_gain"])
                    for unit in units
                ),
                "contiguity_seed_count": contiguity_seed_count,
                "contiguity_macro_gain": mean(
                    float(unit["canonical_vs_random_oracle_relative_gain"])
                    for unit in units
                ),
                "invariants_pass": all(
                    as_bool(unit["invariants_pass"])
                    and not as_bool(unit["severe_degradation"])
                    for unit in units
                ),
                "train_selected_vs_a6_lbf_h720_macro_gain": (
                    mean(
                        float(unit["train_selected_vs_a6_lbf_h720_gain"])
                        for unit in units
                    )
                    if carrier == "a6_natural"
                    else None
                ),
            }
        )
    dataset_required = int(confirmation["dataset_required"])
    separation_count = sum(
        row["function_separation_seed_count"] >= stable_seed_required
        for row in dataset_rows
    )
    skill_count = sum(
        row["carrier_skill_seed_count"] >= stable_seed_required
        for row in dataset_rows
    )
    crossing_count = sum(row["stable_crossing"] for row in dataset_rows)
    contiguity_count = sum(
        row["contiguity_seed_count"] >= stable_seed_required
        for row in dataset_rows
    )
    strict_oracle_macro = mean(
        row["strict_oracle_macro_gain"] for row in dataset_rows
    )
    instance_macro = mean(
        row["sample_over_bin_policy_macro_gain"] for row in dataset_rows
    )
    contiguity_macro = mean(
        row["contiguity_macro_gain"] for row in dataset_rows
    )
    invariants_pass = all(row["invariants_pass"] for row in dataset_rows)
    passed = bool(
        invariants_pass
        and separation_count >= dataset_required
        and skill_count >= dataset_required
        and crossing_count >= dataset_required
        and strict_oracle_macro
        >= float(confirmation["strict_oracle_macro_gain_min"])
        and instance_macro
        >= float(confirmation["sample_over_bin_policy_macro_gain_min"])
        and contiguity_count >= dataset_required
        and contiguity_macro
        >= float(confirmation["contiguity_macro_gain_min"])
    )
    gate = {
        "carrier": carrier,
        "seed_count": len(seeds),
        "stable_seed_required": stable_seed_required,
        "function_separation_dataset_count": separation_count,
        "carrier_skill_dataset_count": skill_count,
        "stable_crossing_dataset_count": crossing_count,
        "strict_oracle_macro_gain": strict_oracle_macro,
        "sample_over_bin_policy_macro_gain": instance_macro,
        "contiguity_dataset_count": contiguity_count,
        "contiguity_macro_gain": contiguity_macro,
        "invariants_pass": invariants_pass,
        "confirmation_pass": passed,
        "a6_lbf_performance_is_problem_gate": False,
    }
    return dataset_rows, gate


def combined_decision(
    neutral_gate: dict[str, Any],
    a6_gate: dict[str, Any],
) -> tuple[str, str]:
    if neutral_gate["confirmation_pass"] and a6_gate["confirmation_pass"]:
        return (
            "dual_carrier_confirmation_pass_authorize_d14b_design",
            "return to D14-B Step 4-6; method implementation and test remain held",
        )
    if neutral_gate["confirmation_pass"]:
        return (
            "neutral_confirmation_pass_a6_nonconfirming",
            "retain scale problem; audit carrier compatibility before D14-B",
        )
    return (
        "neutral_confirmation_fail_or_invalid",
        "apply failure attribution before any direction-level decision",
    )


def analyze(args: argparse.Namespace) -> None:
    if args.raw_root is None or args.design is None or args.output_dir is None:
        raise ValueError("raw-root, design, and output-dir are required")
    design = json.loads(args.design.read_text(encoding="utf-8"))
    seeds = [int(value) for value in design["confirmation_gates"]["seeds"]]
    all_dataset_rows: list[dict[str, Any]] = []
    gates: dict[str, dict[str, Any]] = {}
    for carrier in CARRIERS:
        rows = load_rows(args.raw_root, carrier, seeds)
        dataset_rows, gate = carrier_confirmation(
            carrier,
            rows,
            seeds,
            design,
        )
        all_dataset_rows.extend(dataset_rows)
        gates[carrier] = gate
    decision, next_action = combined_decision(
        gates["neutral_raw"],
        gates["a6_natural"],
    )
    combined_gate = {
        "diagnostic_id": design["diagnostic_id"],
        "seeds": seeds,
        "neutral": gates["neutral_raw"],
        "a6": gates["a6_natural"],
        "a6_failure_can_reject_scale_hypothesis": False,
        "test_used": False,
        "decision": decision,
        "next_action": next_action,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "multiseed_dataset_metrics.csv", all_dataset_rows)
    (args.output_dir / "gate.json").write_text(
        json.dumps(combined_gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# StageC D14-A1 Multi-Seed Confirmation",
        "",
        f"- `decision`: `{decision}`",
        f"- seeds: {seeds}",
        f"- neutral stable crossing: {gates['neutral_raw']['stable_crossing_dataset_count']}/5",
        f"- A6 stable crossing: {gates['a6_natural']['stable_crossing_dataset_count']}/5",
        f"- neutral strict oracle: {gates['neutral_raw']['strict_oracle_macro_gain']:.6%}",
        f"- A6 strict oracle: {gates['a6_natural']['strict_oracle_macro_gain']:.6%}",
        "- A6-LBF performance is descriptive, not a scale-problem gate.",
        "- test=false；D14-B implementation is not automatic.",
        "",
    ]
    (args.output_dir / "research_interpretation.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(f"d14a1_multiseed=pass decision={decision}")


def synthetic_smoke() -> None:
    neutral = {"confirmation_pass": True}
    a6 = {"confirmation_pass": True}
    decision, _next = combined_decision(neutral, a6)
    if decision != "dual_carrier_confirmation_pass_authorize_d14b_design":
        raise RuntimeError(f"synthetic dual pass failed: {decision}")
    a6["confirmation_pass"] = False
    decision, _next = combined_decision(neutral, a6)
    if decision != "neutral_confirmation_pass_a6_nonconfirming":
        raise RuntimeError(f"synthetic A6 boundary failed: {decision}")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.json"
        path.write_text(json.dumps({"decision": decision}), encoding="utf-8")
        json.loads(path.read_text(encoding="utf-8"))
    print("stage_c_d14a1_multiseed_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
    else:
        analyze(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the frozen SIFF-v2 FCC config, references, and launch tooling."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_siff_v2_fcc_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_post_d21_unconstrained_reset_20260720/"
            "siff_v2_fcc_v1_prelaunch"
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_command(command: list[str], env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    arm_ids = [arm["id"] for arm in config["arms"]]
    expected_arms = {
        "siff_equal",
        "a6_full",
        "siff_independent_equal",
    }
    launch_rows = [tuple(row) for row in config["launch_order"]]
    expected_jobs = {
        (seed, dataset, arm)
        for seed in config["seeds"]
        for dataset in config["datasets"]
        for arm in expected_arms
    }

    static_checks = {
        "candidate_version": config["candidate_version"]
        == "SC1-SIFF-v2-FCC-v1",
        "status_authorized_prelaunch": config["status"]
        == "authorized_prelaunch",
        "arms_exact": set(arm_ids) == expected_arms and len(arm_ids) == 3,
        "a6_measure_absent": "a6_measure"
        not in json.dumps(config["arms"]),
        "seeds_exact": config["seeds"] == [2022, 2023],
        "datasets_exact": len(config["datasets"]) == 5
        and len(set(config["datasets"])) == 5,
        "launch_order_complete": len(launch_rows) == 30
        and len(set(launch_rows)) == 30
        and set(launch_rows) == expected_jobs,
        "matrix_contract": config["matrix"]["expected_runs"] == 30
        and config["matrix"]["historical_reused_runs"] == 15
        and config["matrix"]["effective_runs"] == 45
        and config["matrix"]["effective_test_cells"] == 180,
        "comparisons_exact": {
            row["id"] for row in config["comparisons"]
        }
        == {"siff_over_a6_full", "ordered_over_independent_equal"},
        "margin_not_relaxed": config["effectiveness_gates"][
            "mse_macro_gain_percent_min"
        ]
        == 0.3,
        "user_change_recorded": config["comparator_change"][
            "requested_by_user"
        ]
        is True
        and config["comparator_change"]["removed_from_fcc"]
        == "a6_measure"
        and config["comparator_change"]["replacement"] == "a6_full",
        "claim_boundary_recorded": config["comparator_change"][
            "historical_a6_measure_negative_evidence_retained"
        ]
        is True,
        "remote_authorized": config["authorization"][
            "remote_training_authorized"
        ]
        is True,
        "test_authorized": config["authorization"][
            "formal_test_access_authorized"
        ]
        is True,
        "test_tuning_forbidden": config["authorization"][
            "per_dataset_horizon_or_cell_tuning_allowed"
        ]
        is False,
    }

    reference_path = ROOT / config["historical_reference"]["local_run_audit"]
    reference_rows = [
        row
        for row in read_csv(reference_path)
        if row["arm"] in expected_arms
    ]
    reference_checks = {
        "reference_count": len(reference_rows) == 15,
        "reference_cells_complete": {
            (row["dataset"], row["arm"]) for row in reference_rows
        }
        == {
            (dataset, arm)
            for dataset in config["datasets"]
            for arm in expected_arms
        },
        "reference_protocol_pass": all(
            row["status"] == "ok" and row["protocol_pass"] == "True"
            for row in reference_rows
        ),
        "reference_checkpoints_unique": len(
            {row["checkpoint_sha256"] for row in reference_rows}
        )
        == 15,
        "reference_initialization_paired": all(
            len(
                {
                    row["encoder_initialization_hash"]
                    for row in reference_rows
                    if row["dataset"] == dataset
                }
            )
            == 1
            for dataset in config["datasets"]
        ),
    }

    environment = dict(os.environ)
    environment.update(
        {
            "CONFIG": str(config_path.relative_to(ROOT)),
            "DRY_RUN": "1",
            "PYTHON_BIN": sys.executable,
        }
    )
    dry_run_output = run_command(
        ["bash", "scripts/remote/run_stage_c_siff_v2_fcc_v1.sh"],
        environment,
    )
    dry_lines = [
        line for line in dry_run_output.splitlines() if line.count("\t") == 10
    ]
    runner_checks = {
        "runner_dry_run_pass": "siff_fcc_dry_run=pass jobs=30"
        in dry_run_output,
        "runner_jobs_exact": len(dry_lines) == 30,
        "runner_excludes_a6_measure": "a6_measure" not in dry_run_output,
        "runner_includes_both_seeds": all(
            any(line.startswith(f"{seed}\t") for line in dry_lines)
            for seed in config["seeds"]
        ),
    }
    analyzer_output = run_command(
        [
            sys.executable,
            "scripts/analyze_stage_c_siff_v2_fcc.py",
            "--config",
            str(config_path.relative_to(ROOT)),
            "--synthetic-smoke",
        ]
    )
    runner_checks["analyzer_synthetic_smoke"] = (
        "siff_v2_fcc_analyzer_synthetic_smoke=pass" in analyzer_output
    )

    all_checks = {**static_checks, **reference_checks, **runner_checks}
    overall_pass = all(all_checks.values())
    jobs = []
    arms = {arm["id"]: arm for arm in config["arms"]}
    profiles = json.loads(
        (ROOT / config["profiles"]["path"]).read_text(encoding="utf-8")
    )["dataset_profiles"]
    for seed, dataset, arm_id in launch_rows:
        arm = arms[arm_id]
        rule = arm["rank_rule"]
        rank = (
            256
            if rule == "fixed_256"
            else config["matched_ranks"][dataset][rule]
        )
        jobs.append(
            {
                "seed": seed,
                "dataset": dataset,
                "arm": arm_id,
                "role": arm["role"],
                "profile": profiles[dataset]["profile"],
                "rank": rank,
                "readout_mode": arm["readout_mode"],
                "objective_mode": arm["objective_mode"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "jobs.csv", jobs)
    write_csv(output_dir / "historical_reference_audit.csv", reference_rows)
    (output_dir / "prelaunch_gate.json").write_text(
        json.dumps(
            {
                "candidate_version": config["candidate_version"],
                "overall_pass": overall_pass,
                "checks": all_checks,
                "check_count": len(all_checks),
                "passed_checks": sum(all_checks.values()),
                "new_runs": len(jobs),
                "historical_runs": len(reference_rows),
                "a6_measure_in_fcc": False,
                "remote_training_authorized": config["authorization"][
                    "remote_training_authorized"
                ],
                "formal_test_authorized": config["authorization"][
                    "formal_test_access_authorized"
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not overall_pass:
        failed = [name for name, passed in all_checks.items() if not passed]
        raise RuntimeError(f"FCC prelaunch failed: {failed}")
    print(
        "siff_v2_fcc_prelaunch=pass "
        f"checks={len(all_checks)}/{len(all_checks)} jobs={len(jobs)}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the D19 Step 7B formal prelaunch gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_repo as training_adapter  # noqa: E402
from evaluate_stage_c_pcsd_cf_checkpoint import (  # noqa: E402
    test_audit_authorized,
)


RUNNER = ROOT / "scripts/remote/run_stage_c_d19_if_control_v1_1.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_d19_if_control_step7b.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_post_ccsf_step24_reset_20260719/"
            "d19_step7b_prelaunch"
        ),
    )
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def add(
    rows: list[dict[str, Any]],
    category: str,
    case: str,
    passed: bool,
    value: Any,
    threshold: Any,
) -> None:
    rows.append(
        {
            "category": category,
            "case": case,
            "value": value,
            "threshold": threshold,
            "pass": bool(passed),
        }
    )


def manifest_rows(
    config: dict[str, Any],
    profiles: dict[str, Any],
) -> list[dict[str, Any]]:
    arms = {arm["id"]: arm for arm in config["arms"]}
    rows = []
    for dataset, arm_id in config["launch_order"]:
        arm = arms[arm_id]
        profile = profiles[dataset]
        contract = config["profile_contracts"][profile["profile"]]
        rows.append(
            {
                "job_index": len(rows) + 1,
                "dataset": dataset,
                "arm": arm_id,
                "readout_mode": arm["readout_mode"],
                "profile": profile["profile"],
                "patch_num": profile["patch_num"],
                "d_model": profile["d_model"],
                "d_ff": profile["d_ff"],
                "direct_hidden_width": contract["direct_hidden_width"],
                "seed": config["matrix"]["seed"],
            }
        )
    return rows


def training_argv(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    return [
        "train_repo.py",
        "--dataset-root",
        "/home/yingch/dataset",
        "--dataset",
        row["dataset"],
        "--mode",
        "unified",
        "--seq-len",
        "720",
        "--pred-len",
        "720",
        "--target-horizons",
        "720",
        "--validation-horizons",
        "96,192,336,720",
        "--evaluation-horizons",
        "96,192,336,720",
        "--segment-horizons",
        "96,192,336,720",
        "--evaluation-prefix-mode",
        "full-crop",
        "--e-layers",
        "2",
        "--batch-size",
        str(config["training"]["batch_size"]),
        "--epochs",
        str(config["training"]["max_epochs"]),
        "--patience",
        str(config["training"]["early_stopping_patience"]),
        "--enable-early-stopping",
        "--learning-rate",
        str(config["training"]["learning_rate"]),
        "--seed",
        str(row["seed"]),
        "--run-name",
        f"D19_{row['arm']}",
        "--output-dir",
        f"/tmp/d19_{row['dataset']}_{row['arm']}",
        "--device",
        "cuda",
        "--checkpoint-policy",
        "best-val",
        "--protocol-class",
        "method_screening",
        "--protocol-profile",
        "stage_c_d19_if_control_v1_1",
        "--profile-hash",
        config["profiles"]["sha256"],
        "--legacy-patch-num",
        str(row["patch_num"]),
        "--legacy-d-model",
        str(row["d_model"]),
        "--legacy-d-ff",
        str(row["d_ff"]),
        "--legacy-dropout",
        "0.1",
        "--legacy-layer-norm",
        "1",
        "--readout-mode",
        row["readout_mode"],
        "--if-hidden-width",
        "2048",
        "--if-direct-hidden-width",
        str(row["direct_hidden_width"]),
        "--if-head-dropout",
        "0.1",
        "--if-fourier-norm",
        "ortho",
        "--pcc-objective-mode",
        "measure_only",
        "--pred-loss-mode",
        "full",
        "--final-evaluation-split",
        "val",
        "--no-save-predictions",
    ]


def subprocess_pass(command: list[str], env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profile_path = ROOT / config["profiles"]["path"]
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))[
        "dataset_profiles"
    ]
    rows: list[dict[str, Any]] = []

    for name, value in config["contracts"].items():
        if not name.endswith("_path"):
            continue
        prefix = name.removesuffix("_path")
        contract_path = ROOT / value
        actual = file_hash(contract_path)
        expected = config["contracts"][f"{prefix}_sha256"]
        add(rows, "contracts", prefix, actual == expected, actual, expected)
    profile_hash = file_hash(profile_path)
    add(
        rows,
        "contracts",
        "dataset_profiles",
        profile_hash == config["profiles"]["sha256"],
        profile_hash,
        config["profiles"]["sha256"],
    )

    manifest = manifest_rows(config, profiles)
    new_arms = [arm for arm in config["arms"] if arm["training_new"]]
    add(
        rows,
        "matrix",
        "new_training_runs",
        len(manifest) == 15 == config["matrix"]["new_training_runs"],
        len(manifest),
        15,
    )
    add(
        rows,
        "matrix",
        "artifact_units",
        len(config["datasets"]) * len(config["arms"])
        == config["matrix"]["expected_runs"]
        == 20,
        config["matrix"]["expected_runs"],
        20,
    )
    add(
        rows,
        "matrix",
        "official_test_cells",
        20 * len(config["matrix"]["horizons"])
        == config["matrix"]["official_test_cells"]
        == 80,
        config["matrix"]["official_test_cells"],
        80,
    )
    add(
        rows,
        "matrix",
        "three_new_arms",
        len(new_arms) == 3,
        len(new_arms),
        3,
    )

    original = sys.argv
    try:
        for manifest_row in manifest:
            sys.argv = training_argv(manifest_row, config)
            parsed = training_adapter.parse_args()
            passed = bool(
                parsed.dataset == manifest_row["dataset"]
                and parsed.readout_mode == manifest_row["readout_mode"]
                and parsed.seq_len == parsed.pred_len == 720
                and parsed.validation_horizons == [96, 192, 336, 720]
                and parsed.checkpoint_policy == "best-val"
                and parsed.if_hidden_width == 2048
                and parsed.if_direct_hidden_width
                == manifest_row["direct_hidden_width"]
                and parsed.pcc_objective_mode == "measure_only"
            )
            add(
                rows,
                "cli",
                f"{manifest_row['dataset']}_{manifest_row['arm']}",
                passed,
                manifest_row["readout_mode"],
                "frozen D19 training contract",
            )
    finally:
        sys.argv = original

    authorization = config["authorization"]
    add(
        rows,
        "authorization",
        "formal_test_audit",
        test_audit_authorized(config),
        json.dumps(authorization, sort_keys=True),
        "authorized_prelaunch",
    )
    add(
        rows,
        "authorization",
        "control_only_boundary",
        config["role"] == "control_only"
        and config["paper_method_authorized"] is False
        and authorization["confirmation_seeds_authorized"] is False,
        config["role"],
        "control_only",
    )

    subprocess_pass(["bash", "-n", str(RUNNER)])
    add(rows, "runner", "bash_syntax", True, "pass", "pass")
    dry_env = dict(os.environ)
    dry_env["DRY_RUN"] = "1"
    dry_env["CONFIG"] = str(config_path)
    dry_output = subprocess_pass(["bash", str(RUNNER)], env=dry_env)
    add(
        rows,
        "runner",
        "dry_run_15_jobs",
        "d19_dry_run=pass jobs=15" in dry_output,
        dry_output.splitlines()[-1],
        "jobs=15",
    )
    evaluator_output = subprocess_pass(
        [
            sys.executable,
            "scripts/evaluate_stage_c_pcsd_cf_checkpoint.py",
            "--synthetic-smoke",
        ]
    )
    add(
        rows,
        "evaluator",
        "d19_internal_tensor_smoke",
        "d19_checkpoint_evaluator_synthetic_smoke=pass" in evaluator_output,
        evaluator_output.splitlines()[-1],
        "pass",
    )
    analyzer_output = subprocess_pass(
        [
            sys.executable,
            "scripts/analyze_stage_c_d19_if_control.py",
            "--config",
            str(config_path),
            "--synthetic-smoke",
        ]
    )
    add(
        rows,
        "analyzer",
        "four_layer_synthetic_smoke",
        analyzer_output.endswith("d19_analyzer_synthetic_smoke=pass"),
        analyzer_output,
        "pass",
    )

    category_summary: dict[str, dict[str, Any]] = {}
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        category_summary[category] = {
            "passed": sum(bool(row["pass"]) for row in selected),
            "total": len(selected),
            "overall_pass": all(bool(row["pass"]) for row in selected),
        }
    overall = all(bool(row["pass"]) for row in rows)
    payload = {
        "candidate_version": config["candidate_version"],
        "role": "control_only",
        "overall_pass": overall,
        "checks_passed": sum(bool(row["pass"]) for row in rows),
        "checks_total": len(rows),
        "manifest_rows": len(manifest),
        "artifact_units": config["matrix"]["expected_runs"],
        "official_test_cells": config["matrix"]["official_test_cells"],
        "categories": category_summary,
        "authorization_after_gate": {
            "remote": overall,
            "official_test": overall,
            "paper_method": False,
            "confirmation_seeds": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "prelaunch_gate_cases.csv", rows)
    write_csv(args.output_dir / "manifest.csv", manifest)
    (args.output_dir / "gate_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# D19 IF control Step 7B prelaunch gate",
        "",
        f"- `overall_pass`: `{overall}`",
        f"- `checks`: `{payload['checks_passed']}/{payload['checks_total']}`",
        "- `matrix`: 15 new training runs + 5 reused A6 controls; "
        "80 official-test cells",
        "- `checkpoint`: validation mean MSE over H96/H192/H336/H720",
        "- `role`: `control_only`; confirmation seeds and paper-method "
        "promotion remain unauthorized",
        "",
        "通过本 gate 只授权一次冻结的 Phase-A remote/test audit。",
    ]
    (args.output_dir / "prelaunch_gate_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    if not overall:
        raise RuntimeError("D19 Step 7B prelaunch gate failed")
    print(
        f"d19_step7b_prelaunch=pass checks={len(rows)}/{len(rows)} "
        f"jobs={len(manifest)} cells={config['matrix']['official_test_cells']}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the SC-D23-FCMI Step 7B local prelaunch gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from models import TimeAlign  # noqa: E402
import train_repo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_d23_fcmi_step7b.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_post_d21_unconstrained_reset_20260720/"
            "d23_step7b_prelaunch"
        ),
    )
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def subprocess_run(
    command: list[str],
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def manifest_rows(
    config: dict[str, Any],
    profiles: dict[str, Any],
) -> list[dict[str, Any]]:
    arms = {arm["id"]: arm for arm in config["arms"]}
    dense_ranks = config["fcmi_contract"]["dense_ranks"]
    rows = []
    for dataset, arm_id in config["launch_order"]:
        arm = arms[arm_id]
        profile = profiles[dataset]
        rows.append(
            {
                "job_index": len(rows) + 1,
                "dataset": dataset,
                "arm": arm_id,
                "readout_mode": arm["readout_mode"],
                "evaluation_role": arm["evaluation_role"],
                "objective_mode": arm["objective_mode"],
                "profile": profile["profile"],
                "patch_num": profile["patch_num"],
                "d_model": profile["d_model"],
                "d_ff": profile["d_ff"],
                "dense_rank": (
                    dense_ranks[dataset]
                    if arm_id == "DENSE_DUAL_MATCHED"
                    else 0
                ),
                "seed": config["seeds"][0],
            }
        )
    return rows


def training_argv(
    row: dict[str, Any],
    config: dict[str, Any],
    dataset_root: Path,
) -> list[str]:
    return [
        "train_repo.py",
        "--dataset-root",
        str(dataset_root),
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
        "1",
        "--patience",
        "1",
        "--enable-early-stopping",
        "--learning-rate",
        str(config["training"]["learning_rate"]),
        "--seed",
        str(row["seed"]),
        "--num-workers",
        "0",
        "--run-name",
        f"D23_{row['arm']}_{row['dataset']}",
        "--output-dir",
        str(dataset_root / "output" / row["arm"] / row["dataset"]),
        "--device",
        "cpu",
        "--checkpoint-policy",
        "best-val",
        "--protocol-class",
        "method_screening",
        "--protocol-profile",
        "stage_c_d23_fcmi_v1",
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
        "--basis-rank",
        "256",
        "--fcmi-n-heads",
        "8",
        "--fcmi-dropout",
        "0",
        "--fcmi-permutation-seed",
        "20260720",
        "--fcmi-dense-rank",
        str(row["dense_rank"]),
        "--pcc-objective-mode",
        row["objective_mode"],
        "--pred-loss-mode",
        "full",
        "--final-evaluation-split",
        "val",
        "--no-save-predictions",
    ]


def parse_cli(arguments: list[str]) -> argparse.Namespace:
    original = sys.argv
    try:
        sys.argv = arguments
        return train_repo.parse_args()
    finally:
        sys.argv = original


def model_from_parsed(
    parsed: argparse.Namespace,
    dataset_root: Path,
) -> TimeAlign.Model:
    preset = train_repo.OFFICIAL_PRESETS[parsed.dataset][720]
    (dataset_root / preset.data_path).touch()
    official = train_repo.build_official_args(parsed, preset)
    torch.manual_seed(parsed.seed)
    return TimeAlign.Model(official).float()


def module_gradient_norm(module: torch.nn.Module) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += parameter.grad.detach().double().square().sum()
    return float(squared.sqrt().item())


def dense_gate(
    config: dict[str, Any],
    profiles: dict[str, Any],
    manifest: list[dict[str, Any]],
) -> tuple[bool, list[dict[str, Any]]]:
    rows = []
    overall = True
    with tempfile.TemporaryDirectory(prefix="fatst_d23_dense_") as temp:
        dataset_root = Path(temp)
        for dataset in config["datasets"]:
            source_row = next(
                row
                for row in manifest
                if row["dataset"] == dataset
                and row["arm"] == "DENSE_DUAL_MATCHED"
            )
            standard_row = dict(source_row)
            standard_row.update(
                {
                    "arm": "STANDARD_DUAL_MATCHED",
                    "readout_mode": "fcmi-standard-dual-matched",
                    "dense_rank": 0,
                }
            )
            a6_row = dict(source_row)
            a6_row.update(
                {
                    "arm": "A6_MEASURE",
                    "readout_mode": "learned-basis-forecast-operator",
                    "dense_rank": 0,
                }
            )
            dense = model_from_parsed(
                parse_cli(training_argv(source_row, config, dataset_root)),
                dataset_root,
            ).eval()
            standard = model_from_parsed(
                parse_cli(training_argv(standard_row, config, dataset_root)),
                dataset_root,
            ).eval()
            a6 = model_from_parsed(
                parse_cli(training_argv(a6_row, config, dataset_root)),
                dataset_root,
            ).eval()
            dense_diagnostics = train_repo.model_diagnostics(dense)
            a6_diagnostics = train_repo.model_diagnostics(a6)
            parameter_gap = abs(
                dense_diagnostics["active_forward_parameters"]
                - a6_diagnostics["active_forward_parameters"]
            ) / a6_diagnostics["active_forward_parameters"]

            generator = torch.Generator().manual_seed(20260720)
            history = torch.randn(1, 720, 2, generator=generator)
            with torch.no_grad():
                dense_memory = dense.encode_history(history)
                standard_memory = standard.encode_history(history)
                dense_output, dense_details = dense.fcmi_readout(
                    dense_memory,
                    720,
                    return_details=True,
                )
                standard_output = standard.fcmi_readout(
                    standard_memory,
                    720,
                )
            initial_gap = float(
                (dense_output - standard_output).abs().max().item()
            )
            initial_residual = float(
                dense_details["dense_residual"].abs().max().item()
            )

            dense.train()
            dense.zero_grad(set_to_none=True)
            memory = dense.encode_history(history).detach()
            target = torch.randn(
                1,
                720,
                2,
                generator=generator,
            )
            output = dense.fcmi_readout(memory, 720)
            (output - target).square().mean().backward()
            coefficient_gradient = module_gradient_norm(
                dense.fcmi_readout.dense_coefficient
            )
            optimizer = torch.optim.SGD(
                dense.fcmi_readout.dense_coefficient.parameters(),
                lr=0.01,
            )
            optimizer.step()
            dense.zero_grad(set_to_none=True)
            output, details = dense.fcmi_readout(
                memory,
                720,
                return_details=True,
            )
            (output - target).square().mean().backward()
            basis_gradient = float(
                dense.fcmi_readout.dense_temporal_basis.grad.norm().item()
            )
            residual_std = float(
                details["dense_residual"].std(unbiased=False).item()
            )
            passed = bool(
                parameter_gap
                <= config["fcmi_contract"][
                    "dense_a6_active_parameter_relative_gap_max"
                ]
                and initial_gap <= 1e-6
                and initial_residual == 0.0
                and coefficient_gradient > 1e-10
                and basis_gradient > 1e-10
                and residual_std > 1e-10
            )
            rows.append(
                {
                    "dataset": dataset,
                    "dense_rank": source_row["dense_rank"],
                    "a6_active_parameters": a6_diagnostics[
                        "active_forward_parameters"
                    ],
                    "dense_active_parameters": dense_diagnostics[
                        "active_forward_parameters"
                    ],
                    "parameter_relative_gap": parameter_gap,
                    "initial_standard_output_max_abs": initial_gap,
                    "initial_dense_residual_max_abs": initial_residual,
                    "coefficient_first_step_gradient_norm": (
                        coefficient_gradient
                    ),
                    "basis_second_step_gradient_norm": basis_gradient,
                    "dense_second_step_residual_std": residual_std,
                    "pass": passed,
                }
            )
            overall = overall and passed
    return overall, rows


def main() -> None:
    args = parse_args()
    config_path = (
        args.config if args.config.is_absolute() else ROOT / args.config
    )
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
        actual = file_hash(ROOT / value)
        expected = config["contracts"][f"{prefix}_sha256"]
        add(
            rows,
            "contracts",
            prefix,
            actual == expected,
            actual,
            expected,
        )
    profile_hash = file_hash(profile_path)
    add(
        rows,
        "contracts",
        "profiles",
        profile_hash == config["profiles"]["sha256"],
        profile_hash,
        config["profiles"]["sha256"],
    )

    step7a_gate = json.loads(
        (ROOT / config["contracts"]["step7a_gate_path"]).read_text(
            encoding="utf-8"
        )
    )
    add(
        rows,
        "dependency",
        "step7a_committed_gate",
        step7a_gate["all_pass"] is True
        and step7a_gate["pass_count"] == step7a_gate["gate_count"] == 11,
        step7a_gate["decision"],
        "step7a_local_pass_step7b_design_freeze_next",
    )
    regression_path = args.output_dir / "step7a_regression.json"
    regression = subprocess_run(
        [
            sys.executable,
            "scripts/check_stage_c_d23_fcmi_step7a.py",
            "--config",
            "configs/stage_c_d23_fcmi_step7a.json",
            "--output",
            str(regression_path),
        ]
    )
    regression_payload = json.loads(
        regression_path.read_text(encoding="utf-8")
    )
    add(
        rows,
        "dependency",
        "step7a_current_code_regression",
        regression_payload["all_pass"] is True
        and "11/11 gates passed" in regression.stdout,
        regression_payload["decision"],
        "11/11 pass",
    )

    manifest = manifest_rows(config, profiles)
    formal_count = sum(
        row["evaluation_role"] == "formal_test" for row in manifest
    )
    validation_only_count = sum(
        row["evaluation_role"] == "validation_only" for row in manifest
    )
    unique_pairs = {
        (row["dataset"], row["arm"]) for row in manifest
    }
    add(
        rows,
        "matrix",
        "complete_unique_matrix",
        len(manifest)
        == len(unique_pairs)
        == config["matrix"]["expected_training_runs"]
        == 40
        and formal_count == config["matrix"]["formal_test_runs"] == 40
        and validation_only_count
        == config["matrix"]["validation_only_runs"]
        == 0
        and config["matrix"]["official_test_cells"] == 160
        and config["matrix"]["validation_cells"] == 160,
        {
            "training": len(manifest),
            "formal": formal_count,
            "validation_only": validation_only_count,
        },
        "40/40/0 runs and 160/160 cells",
    )

    cli_pass = True
    with tempfile.TemporaryDirectory(prefix="fatst_d23_cli_") as temp:
        dataset_root = Path(temp)
        for row in manifest:
            parsed = parse_cli(training_argv(row, config, dataset_root))
            cli_pass = cli_pass and bool(
                parsed.dataset == row["dataset"]
                and parsed.readout_mode == row["readout_mode"]
                and parsed.fcmi_dense_rank == row["dense_rank"]
                and parsed.pcc_objective_mode == "measure_only"
                and parsed.validation_horizons == [96, 192, 336, 720]
                and parsed.final_evaluation_split == "val"
            )
    add(
        rows,
        "cli",
        "production_40_cases",
        cli_pass,
        len(manifest),
        40,
    )

    dense_pass, dense_rows = dense_gate(config, profiles, manifest)
    add(
        rows,
        "dense_control",
        "five_profile_capacity_and_gradient_gate",
        dense_pass,
        dense_rows,
        "gap<=0.2%, initial equality, two-step active path",
    )

    runner = ROOT / config["contracts"]["runner_path"]
    subprocess_run(["bash", "-n", str(runner)])
    dry_env = dict(os.environ)
    dry_env["DRY_RUN"] = "1"
    dry_env["CONFIG"] = str(config_path)
    dry = subprocess_run(["bash", str(runner)], env=dry_env)
    dry_lines = [
        line for line in dry.stdout.splitlines() if "\t" in line
    ]
    add(
        rows,
        "runner",
        "syntax_and_dry_run",
        len(dry_lines) == 40
        and "d23_dry_run=pass jobs=40" in dry.stdout
        and "remote_authorized=false" in dry.stdout
        and "test_authorized=false" in dry.stdout,
        dry.stdout.splitlines()[-1],
        "40 jobs; authorization false",
    )
    blocked_env = dict(os.environ)
    blocked_env["CONFIG"] = str(config_path)
    blocked = subprocess_run(
        ["bash", str(runner)],
        env=blocked_env,
        check=False,
    )
    add(
        rows,
        "authorization",
        "unauthorized_launch_blocked",
        blocked.returncode == 3
        and "not authorized" in blocked.stderr,
        blocked.returncode,
        3,
    )

    evaluator = subprocess_run(
        [
            sys.executable,
            config["contracts"]["evaluator_path"],
            "--synthetic-smoke",
        ]
    )
    add(
        rows,
        "evaluator",
        "fcmi_checkpoint_smoke",
        "d23_fcmi_checkpoint_evaluator_synthetic_smoke=pass"
        in evaluator.stdout,
        evaluator.stdout.splitlines()[-1],
        "pass",
    )
    analyzer = subprocess_run(
        [
            sys.executable,
            config["contracts"]["analyzer_path"],
            "--config",
            str(config_path),
            "--synthetic-smoke",
        ]
    )
    add(
        rows,
        "analyzer",
        "four_layer_decision_smoke",
        analyzer.stdout.strip().endswith(
            "d23_fcmi_analyzer_synthetic_smoke=pass"
        ),
        analyzer.stdout.strip(),
        "pass",
    )

    expected_comparisons = {
        "effectiveness_fcmi_vs_a6",
        "decomposition_fcmi_vs_standard_dual",
        "interaction_fcmi_vs_generic_dual",
        "order_fcmi_vs_order_shuffled",
        "capacity_fcmi_vs_dense_dual",
        "target_coordinate_fcmi_vs_target_shuffle",
    }
    add(
        rows,
        "governance",
        "gates_and_rollback",
        set(config["comparison_gates"]) == expected_comparisons
        and set(config["decision_map"])
        == {
            "numeric_or_protocol_invalid",
            "fails_a6_internal_valid",
            "beats_a6_but_attribution_fails",
            "attribution_pass_internal_fails",
            "all_seed2021_gates_pass",
        },
        {
            "comparisons": sorted(config["comparison_gates"]),
            "decisions": sorted(config["decision_map"]),
        },
        "complete four-layer mapping",
    )
    authorization = config["authorization"]
    add(
        rows,
        "authorization",
        "local_only_boundary",
        config["status"] == "prelaunch_ready_waiting_authorization"
        and authorization["user_authorized"] is False
        and authorization["remote_training_authorized"] is False
        and authorization["formal_test_access_authorized"] is False
        and authorization["confirmation_seeds_authorized"] is False
        and authorization["paper_method"] is False,
        authorization,
        "all external execution and promotion false",
    )

    summary = {
        "candidate_version": config["candidate_version"],
        "config_path": str(config_path),
        "config_sha256": file_hash(config_path),
        "checks_passed": sum(row["pass"] for row in rows),
        "checks_total": len(rows),
        "overall_pass": all(row["pass"] for row in rows),
        "decision": (
            "step7b_prelaunch_pass_waiting_remote_test_authorization"
            if all(row["pass"] for row in rows)
            else "step7b_prelaunch_fail_return_step6_7"
        ),
        "matrix": config["matrix"],
        "authorization": authorization,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "prelaunch_cases.csv", rows)
    write_csv(args.output_dir / "manifest.csv", manifest)
    write_csv(args.output_dir / "dense_control_audit.csv", dense_rows)
    (args.output_dir / "prelaunch_gate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"SC-D23-FCMI Step7B: {summary['checks_passed']}/"
        f"{summary['checks_total']} gates passed; "
        f"decision={summary['decision']}"
    )
    if not summary["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

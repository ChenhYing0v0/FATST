#!/usr/bin/env python3
"""Run the SC1-SIFF-v3-TSAF Step 7B local prelaunch gate."""

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
from evaluate_stage_c_pcsd_cf_checkpoint import (  # noqa: E402
    test_audit_authorized,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_siff_v3_tsaf_step7b.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_post_d21_unconstrained_reset_20260720/"
            "siff_v3_tsaf_step7b_prelaunch"
        ),
    )
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def add_case(
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
    arms = {arm["id"]: arm for arm in config["effective_arms"]}
    rows = []
    for dataset, arm_id in config["launch_order"]:
        arm = arms[arm_id]
        profile = profiles[dataset]
        rule = arm["rank_rule"]
        rank = (
            256
            if rule == "fixed_256"
            else config["matched_ranks"][dataset][rule]
        )
        rows.append(
            {
                "job_index": len(rows) + 1,
                "dataset": dataset,
                "arm": arm_id,
                "readout_mode": arm["readout_mode"],
                "policy_mode": arm["policy_mode"],
                "objective_mode": arm["objective_mode"],
                "mode_rank": rank,
                "profile": profile["profile"],
                "patch_num": profile["patch_num"],
                "d_model": profile["d_model"],
                "d_ff": profile["d_ff"],
                "seed": config["seeds"][0],
                "checkpoint_score": config["training"]["checkpoint_score"],
                "training_evaluation_split": "val",
                "formal_test_executed_by_runner": False,
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
        f"TSAF_{row['arm']}_{row['dataset']}",
        "--output-dir",
        str(dataset_root / "output" / row["arm"] / row["dataset"]),
        "--device",
        "cpu",
        "--checkpoint-policy",
        "best-val",
        "--protocol-class",
        "method_screening",
        "--protocol-profile",
        "stage_c_siff_v3_tsaf_v1",
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
        "--pcsd-coordinate-dim",
        "4",
        "--pcsd-mode-rank",
        str(row["mode_rank"]),
        "--pcsd-policy-history-dim",
        "32",
        "--pcsd-policy-hidden-dim",
        "64",
        "--pcsd-policy-mode",
        row["policy_mode"],
        "--pcsd-fixed-scale",
        "720",
        "--pcsd-partition",
        "canonical",
        "--pcsd-partition-seed",
        "15101",
        "--pcsd-group-chunk-size",
        "64",
        "--pcsd-target-chunk-size",
        "128",
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


def reference_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    contract = config["reference_contract"]
    audit_path = ROOT / contract["source_run_audit_path"]
    source_rows = read_csv(audit_path)
    source_lookup = {
        (row["arm"], row["dataset"]): row for row in source_rows
    }
    rows = []
    all_pass = True
    for arm in config["effective_arms"]:
        if arm["source"] != "reused_reference":
            continue
        for dataset in config["datasets"]:
            source = source_lookup[(arm["source_arm"], dataset)]
            expected_hash = contract["checkpoint_sha256"][arm["id"]][dataset]
            directory = (
                ROOT
                / contract["local_artifact_root"]
                / arm["source_arm"]
                / dataset
                / "h720_full"
                / "seed2021"
            )
            required_local = [
                "test_audit_metrics_by_target_horizon.csv",
                "test_audit_invariants.json",
                "effective_config.json",
                "initialization_contract.json",
            ]
            local_complete = all(
                (directory / name).is_file() for name in required_local
            )
            passed = bool(
                source["status"] == "ok"
                and source["protocol_pass"] == "True"
                and source["checkpoint_sha256"] == expected_hash
                and local_complete
            )
            all_pass = all_pass and passed
            rows.append(
                {
                    "dataset": dataset,
                    "effective_arm": arm["id"],
                    "source_arm": arm["source_arm"],
                    "checkpoint_sha256": expected_hash,
                    "encoder_initialization_hash": source[
                        "encoder_initialization_hash"
                    ],
                    "local_lite_artifacts_complete": local_complete,
                    "checkpoint_hash_from_frozen_audit": True,
                    "pass": passed,
                }
            )
    return rows, all_pass


def gradient_norm(parameters: list[torch.Tensor]) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().double().square().sum()
    return float(squared.sqrt().item())


def two_step_gradient_case(
    model: TimeAlign.Model,
    arm: str,
) -> dict[str, Any]:
    readout = model.pcsd_readout
    generator = torch.Generator().manual_seed(20260721)
    hidden = torch.randn(
        1,
        2,
        readout.readout_dim,
        generator=generator,
    )
    target = torch.randn(1, 720, 2, generator=generator)
    optimizer = torch.optim.SGD(readout.parameters(), lr=0.01)

    optimizer.zero_grad(set_to_none=True)
    prediction = readout(hidden, 720)
    loss = (prediction - target).square().mean()
    loss.backward()
    mode_step1 = float(readout.mode_weight.grad.norm().item())
    if readout.policy_mode.startswith("target-scale-"):
        output_parameters = list(
            readout.target_scale_allocation_output.parameters()
        )
    else:
        output_parameters = list(readout.policy_output.parameters())
    policy_output_step1 = gradient_norm(output_parameters)
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    prediction = readout(hidden, 720)
    loss = (prediction - target).square().mean()
    loss.backward()
    if readout.policy_mode.startswith("target-scale-"):
        input_parameters = list(
            readout.scale_allocation_projection.parameters()
        )
        if readout.policy_mode != "target-scale-global":
            input_parameters += list(
                readout.target_allocation_projection.parameters()
            )
    else:
        input_parameters = list(readout.policy_hidden.parameters())
    policy_input_step2 = gradient_norm(input_parameters)
    finite = all(
        torch.isfinite(parameter.grad).all().item()
        for parameter in readout.parameters()
        if parameter.grad is not None
    )
    passed = bool(
        torch.isfinite(loss).item()
        and finite
        and mode_step1 > 0.0
        and policy_output_step1 > 0.0
        and policy_input_step2 > 0.0
    )
    return {
        "arm": arm,
        "policy_mode": readout.policy_mode,
        "loss_step2": float(loss.item()),
        "mode_gradient_step1": mode_step1,
        "policy_output_gradient_step1": policy_output_step1,
        "policy_input_gradient_step2": policy_input_step2,
        "all_gradients_finite": finite,
        "pass": passed,
    }


def runner_checks(
    config_path: Path,
    authorized: bool,
) -> dict[str, Any]:
    runner = ROOT / "scripts/remote/run_stage_c_siff_v3_tsaf_v1.sh"
    syntax = subprocess.run(
        ["bash", "-n", str(runner)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    env = os.environ.copy()
    env.update({"CONFIG": str(config_path), "DRY_RUN": "1"})
    dry = subprocess.run(
        ["bash", str(runner)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    blocked_returncode = None
    blocked_stderr = "not run because remote training is authorized"
    if not authorized:
        blocked_env = os.environ.copy()
        blocked_env.update({"CONFIG": str(config_path), "DRY_RUN": "0"})
        blocked = subprocess.run(
            ["bash", str(runner)],
            cwd=ROOT,
            env=blocked_env,
            check=False,
            capture_output=True,
            text=True,
        )
        blocked_returncode = blocked.returncode
        blocked_stderr = blocked.stderr.strip()
    return {
        "syntax_pass": syntax.returncode == 0,
        "executable": os.access(runner, os.X_OK),
        "dry_run_pass": dry.returncode == 0
        and "tsaf_dry_run=pass jobs=25" in dry.stdout,
        "authorization_mode": "authorized" if authorized else "held",
        "authorization_boundary_pass": authorized
        or blocked_returncode == 3,
        "dry_run_last_line": dry.stdout.strip().splitlines()[-1]
        if dry.stdout.strip()
        else "",
        "blocked_stderr": blocked_stderr,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else ROOT / args.output_dir
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profile_path = ROOT / config["profiles"]["path"]
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))[
        "dataset_profiles"
    ]
    manifest = manifest_rows(config, profiles)
    cases: list[dict[str, Any]] = []

    step7a_path = ROOT / config["contracts"]["step7a_gate_path"]
    step7a = json.loads(step7a_path.read_text(encoding="utf-8"))
    add_case(
        cases,
        "evidence",
        "step7a_gate",
        step7a["all_passed"] and step7a["case_count"] == 26,
        f"{step7a['passed_count']}/{step7a['case_count']}",
        "26/26",
    )
    add_case(
        cases,
        "contract",
        "profile_hash",
        file_hash(profile_path) == config["profiles"]["sha256"],
        file_hash(profile_path),
        config["profiles"]["sha256"],
    )
    reference_contract = config["reference_contract"]
    source_config_path = ROOT / reference_contract["source_config_path"]
    source_audit_path = ROOT / reference_contract["source_run_audit_path"]
    add_case(
        cases,
        "reference",
        "source_config_hash",
        file_hash(source_config_path)
        == reference_contract["source_config_sha256"],
        file_hash(source_config_path),
        reference_contract["source_config_sha256"],
    )
    add_case(
        cases,
        "reference",
        "source_run_audit_hash",
        file_hash(source_audit_path)
        == reference_contract["source_run_audit_sha256"],
        file_hash(source_audit_path),
        reference_contract["source_run_audit_sha256"],
    )

    arm_ids = {arm["id"] for arm in config["effective_arms"]}
    new_arm_ids = {
        arm["id"]
        for arm in config["effective_arms"]
        if arm["source"] == "new_training"
    }
    reference_arm_ids = arm_ids - new_arm_ids
    launch_pairs = {(row["dataset"], row["arm"]) for row in manifest}
    expected_pairs = {
        (dataset, arm) for dataset in config["datasets"] for arm in new_arm_ids
    }
    add_case(
        cases,
        "matrix",
        "effective_and_new_runs",
        len(arm_ids) == 9
        and len(new_arm_ids) == 5
        and len(reference_arm_ids) == 4
        and len(manifest) == 25
        and launch_pairs == expected_pairs,
        (
            f"arms={len(arm_ids)} new={len(new_arm_ids)} "
            f"refs={len(reference_arm_ids)} jobs={len(manifest)}"
        ),
        "9/5/4/25 exact",
    )
    matrix = config["matrix"]
    add_case(
        cases,
        "matrix",
        "paper_facing_cells",
        matrix["effective_runs"] == 45
        and matrix["new_training_runs"] == 25
        and matrix["reused_reference_runs"] == 20
        and matrix["new_official_test_cells"] == 100
        and matrix["effective_official_test_cells"] == 180,
        matrix,
        "45 effective runs / 100 new cells / 180 effective cells",
    )
    add_case(
        cases,
        "matrix",
        "new_arm_semantics",
        {
            (row["arm"], row["readout_mode"], row["policy_mode"])
            for row in manifest
        }
        == {
            ("siff_categorical_target_only", "siff-coupling-field", "static-target"),
            ("tsaf", "siff-coupling-field", "target-scale-field"),
            ("tsaf_permuted_scale", "siff-coupling-field", "target-scale-field-permuted"),
            ("tsaf_no_target_global", "siff-coupling-field", "target-scale-global"),
            ("siff_independent_target_only", "siff-independent-scope-control", "static-target"),
        },
        sorted(
            {
                (row["arm"], row["readout_mode"], row["policy_mode"])
                for row in manifest
            }
        ),
        "five frozen arm semantics",
    )

    references, references_pass = reference_rows(config)
    add_case(
        cases,
        "reference",
        "twenty_reusable_runs",
        len(references) == 20 and references_pass,
        f"{sum(int(row['pass']) for row in references)}/{len(references)}",
        "20/20",
    )
    add_case(
        cases,
        "reference",
        "direct_independent_not_reused",
        "siff_independent_target_only" not in reference_arm_ids
        and all(
            row["effective_arm"] != "siff_independent_target_only"
            for row in references
        ),
        sorted(reference_arm_ids),
        "direct-policy independent excluded",
    )

    parsed_rows = []
    initialization_rows = []
    gradient_rows = []
    with tempfile.TemporaryDirectory(prefix="fatst_tsaf_step7b_") as temp:
        dataset_root = Path(temp)
        models_by_dataset: dict[str, dict[str, TimeAlign.Model]] = {}
        for row in manifest:
            parsed = parse_cli(training_argv(row, config, dataset_root))
            parsed_ok = bool(
                parsed.readout_mode == row["readout_mode"]
                and parsed.pcsd_policy_mode == row["policy_mode"]
                and parsed.pcsd_mode_rank == row["mode_rank"]
                and parsed.pcc_objective_mode == row["objective_mode"]
                and parsed.validation_horizons == [96, 192, 336, 720]
                and parsed.final_evaluation_split == "val"
            )
            parsed_rows.append(parsed_ok)
            model = model_from_parsed(parsed, dataset_root)
            models_by_dataset.setdefault(row["dataset"], {})[
                row["arm"]
            ] = model

        for dataset, models in models_by_dataset.items():
            contracts = {
                arm: train_repo.initialization_contract(model)
                for arm, model in models.items()
            }
            diagnostics = {
                arm: train_repo.model_diagnostics(model)
                for arm, model in models.items()
            }
            encoder_hashes = {
                contract["encoder_initialization_hash"]
                for contract in contracts.values()
            }
            tsaf_hashes = {
                contracts[arm]["pcsd_initialization_hash"]
                for arm in (
                    "tsaf",
                    "tsaf_permuted_scale",
                    "tsaf_no_target_global",
                )
            }
            candidate_parameters = diagnostics["tsaf"][
                "active_forward_parameters"
            ]
            independent_parameters = diagnostics[
                "siff_independent_target_only"
            ]["active_forward_parameters"]
            relative_gap = abs(
                candidate_parameters - independent_parameters
            ) / candidate_parameters
            passed = bool(
                len(encoder_hashes) == 1
                and len(tsaf_hashes) == 1
                and relative_gap
                <= config["capacity_control"][
                    "candidate_independent_relative_gap_max"
                ]
            )
            initialization_rows.append(
                {
                    "dataset": dataset,
                    "encoder_hashes": len(encoder_hashes),
                    "tsaf_parameter_hashes": len(tsaf_hashes),
                    "candidate_active_parameters": candidate_parameters,
                    "independent_active_parameters": independent_parameters,
                    "relative_gap": relative_gap,
                    "matched_rank": config["matched_ranks"][dataset][
                        "independent_dataset_matched"
                    ],
                    "pass": passed,
                }
            )

        for arm, model in models_by_dataset["Weather"].items():
            gradient_rows.append(two_step_gradient_case(model, arm))

    add_case(
        cases,
        "cli",
        "all_new_jobs_parse",
        all(parsed_rows) and len(parsed_rows) == 25,
        f"{sum(parsed_rows)}/{len(parsed_rows)}",
        "25/25",
    )
    add_case(
        cases,
        "initialization",
        "paired_encoder_tsaf_and_capacity",
        all(row["pass"] for row in initialization_rows),
        (
            f"{sum(int(row['pass']) for row in initialization_rows)}/"
            f"{len(initialization_rows)} max_gap="
            f"{max(row['relative_gap'] for row in initialization_rows):.6f}"
        ),
        "5/5 and gap<=0.004",
    )
    add_case(
        cases,
        "gradient",
        "two_step_policy_and_field",
        all(row["pass"] for row in gradient_rows)
        and len(gradient_rows) == 5,
        f"{sum(int(row['pass']) for row in gradient_rows)}/{len(gradient_rows)}",
        "5/5 finite nonzero",
    )

    authorization = config["authorization"]
    remote_authorized = authorization["remote_training_authorized"] is True
    runner = runner_checks(config_path, remote_authorized)
    add_case(
        cases,
        "runner",
        "syntax_dry_run_and_authorization_refusal",
        all(
            runner[key]
            for key in (
                "syntax_pass",
                "executable",
                "dry_run_pass",
                "authorization_boundary_pass",
            )
        ),
        runner,
        "syntax/executable/dry-run true and authorization boundary honored",
    )
    add_case(
        cases,
        "authorization",
        "seed2021_remote_and_single_test_authorized",
        authorization["step7b_design_and_prelaunch_authorized"] is True
        and authorization["user_authorized"] is True
        and authorization["remote_training_authorized"] is True
        and authorization["formal_test_access_authorized"] is True
        and authorization["confirmation_seeds_authorized"] is False
        and authorization["formal_test_access_count_for_version"] == 1
        and test_audit_authorized(config),
        authorization,
        "seed2021 remote/test true; confirmation false; evaluator accepts",
    )
    schema = config["artifact_schema"]
    add_case(
        cases,
        "artifact",
        "checkpoint_preserving_schema",
        "checkpoint.pt" in schema["training_required"]
        and "initialization_contract.json" in schema["training_required"]
        and "test_audit_invariants.json" in schema["formal_test_required"]
        and schema["checkpoint_nonmutation_required"] is True,
        schema,
        "training/test split and checkpoint nonmutation frozen",
    )

    categories = sorted({row["category"] for row in cases})
    category_pass = {
        category: all(
            row["pass"] for row in cases if row["category"] == category
        )
        for category in categories
    }
    overall_pass = all(row["pass"] for row in cases)
    summary = {
        "gate_id": "SC1-SIFF-v3-TSAF-v1-Step7B-prelaunch",
        "candidate_version": config["candidate_version"],
        "config_sha256": file_hash(config_path),
        "profile_sha256": file_hash(profile_path),
        "categories": category_pass,
        "categories_passed": sum(category_pass.values()),
        "categories_total": len(category_pass),
        "cases_passed": sum(int(row["pass"]) for row in cases),
        "cases_total": len(cases),
        "new_training_runs": len(manifest),
        "reused_reference_runs": len(references),
        "effective_runs": 45,
        "effective_official_test_cells": 180,
        "overall_pass": overall_pass,
        "remote_training_authorized": remote_authorized,
        "formal_test_access_authorized": bool(
            authorization["formal_test_access_authorized"]
        ),
        "confirmation_seeds_authorized": False,
        "decision": (
            "step8_seed2021_remote_and_test_authorized"
            if overall_pass
            else "repair_step7b_prelaunch"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "cases.csv", cases)
    write_csv(output_dir / "jobs_seed2021.csv", manifest)
    write_csv(output_dir / "reference_runs.csv", references)
    write_csv(output_dir / "initialization_pairing.csv", initialization_rows)
    write_csv(output_dir / "early_gradient.csv", gradient_rows)
    (output_dir / "prelaunch_gate.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if not overall_pass:
        raise RuntimeError(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

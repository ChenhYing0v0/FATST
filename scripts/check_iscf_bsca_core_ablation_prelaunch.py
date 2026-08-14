#!/usr/bin/env python3
"""Run the local prelaunch gate for the frozen Core-Ablation matrix."""

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
from layers.PCC import projective_coupling_credit_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/iscf_bsca_core_ablation_protocol.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parsed_model(
    dataset: str,
    arm: dict[str, Any],
    rank: int,
    profile: dict[str, Any],
    profile_hash: str,
    temp_root: Path,
) -> TimeAlign.Model:
    argv = [
        "train_repo.py",
        "--dataset-root", str(temp_root),
        "--dataset", dataset,
        "--mode", "unified",
        "--seq-len", "720",
        "--pred-len", "720",
        "--target-horizons", "720",
        "--validation-horizons", "96,192,336,720",
        "--evaluation-horizons", "96,192,336,720",
        "--segment-horizons", "96,192,336,720",
        "--evaluation-prefix-mode", "full-crop",
        "--e-layers", "2",
        "--batch-size", "2",
        "--epochs", "1",
        "--patience", "1",
        "--seed", "2021",
        "--num-workers", "0",
        "--run-name", f"CORE_ABLATION_PRELAUNCH_{arm['id']}",
        "--output-dir", str(temp_root / arm["id"]),
        "--device", "cpu",
        "--checkpoint-policy", "best-val",
        "--protocol-class", "method_screening",
        "--protocol-profile", "iscf_bsca_core_ablation_20260814",
        "--profile-hash", profile_hash,
        "--legacy-patch-num", str(profile["patch_num"]),
        "--legacy-d-model", str(profile["d_model"]),
        "--legacy-d-ff", str(profile["d_ff"]),
        "--legacy-dropout", "0.1",
        "--legacy-layer-norm", "1",
        "--learning-rate", "0.0001",
        "--readout-mode", arm["readout_mode"],
        "--basis-rank", "256",
        "--pcsd-coordinate-dim", "4",
        "--pcsd-mode-rank", str(rank),
        "--pcsd-policy-history-dim", "32",
        "--pcsd-policy-hidden-dim", "64",
        "--pcsd-policy-mode", arm["policy_mode"],
        "--pcsd-fixed-scale", str(arm["fixed_scale"]),
        "--pcsd-partition", arm["partition"],
        "--pcsd-partition-seed", str(arm["partition_seed"]),
        "--pcc-objective-mode", arm["objective_mode"],
        "--pred-loss-mode", "full",
        "--final-evaluation-split", "val",
        "--no-save-predictions",
    ]
    original = sys.argv
    try:
        sys.argv = argv
        parsed = train_repo.parse_args()
    finally:
        sys.argv = original
    preset = train_repo.OFFICIAL_PRESETS[dataset][720]
    (temp_root / preset.data_path).parent.mkdir(parents=True, exist_ok=True)
    (temp_root / preset.data_path).touch()
    official = train_repo.build_official_args(parsed, preset)
    torch.manual_seed(2021)
    return TimeAlign.Model(official).float().eval()


def run_command(
    command: list[str], extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, **(extra_env or {})},
        text=True,
        capture_output=True,
        check=False,
    )


def objective_semantics(rows: list[dict[str, Any]]) -> None:
    torch.manual_seed(20260814)
    fused = torch.randn(2, 12, 3)
    arms = torch.randn(2, 3, 12, 5)
    target = torch.randn(2, 12, 3)
    direct = torch.softmax(torch.randn(2, 3, 12, 5), dim=-1)
    equal = torch.full_like(direct, 0.2)
    fixed = torch.zeros_like(direct)
    fixed[..., 2] = 1.0
    without_bsca = projective_coupling_credit_loss(
        fused, arms, direct, target, mode="measure_only", progress=0.5
    )
    fixed_result = projective_coupling_credit_loss(
        fused, arms, fixed, target, mode="measure_only", progress=0.5
    )
    equal_result = projective_coupling_credit_loss(
        fused, arms, equal, target, mode="equal_uniform_scope_anchor", progress=0.5
    )
    add(
        rows,
        "objective",
        "without_bsca_prefix_only",
        bool(
            without_bsca.skill_loss.item() == 0.0
            and without_bsca.route_loss.item() == 0.0
            and torch.equal(without_bsca.total_loss, without_bsca.fused_loss)
        ),
        f"skill={without_bsca.skill_loss.item():.6g},route={without_bsca.route_loss.item():.6g}",
        "skill=0,route=0,total=fused",
    )
    add(
        rows,
        "objective",
        "fixed_scope_prefix_only",
        bool(
            fixed_result.skill_loss.item() == 0.0
            and fixed_result.route_loss.item() == 0.0
        ),
        f"skill={fixed_result.skill_loss.item():.6g},route={fixed_result.route_loss.item():.6g}",
        "skill=0,route=0",
    )
    add(
        rows,
        "objective",
        "equal_allocation_balance_zero",
        abs(equal_result.route_loss.item()) <= 1e-7,
        equal_result.route_loss.item(),
        "<=1e-7",
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    profiles_path = ROOT / config["profiles"]["path"]
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))["dataset_profiles"]
    rows: list[dict[str, Any]] = []
    add(rows, "contract", "profile_hash", file_hash(profiles_path) == config["profiles"]["sha256"], file_hash(profiles_path), config["profiles"]["sha256"])
    add(rows, "contract", "five_variants", len(config["arms"]) == 5, len(config["arms"]), 5)
    add(rows, "contract", "effective_100_cells", config["matrix"]["effective_test_cells"] == 100, config["matrix"]["effective_test_cells"], 100)
    add(rows, "contract", "new_20_runs", len(config["launch_order"]) == 20 and len(set(map(tuple, config["launch_order"]))) == 20, len(config["launch_order"]), 20)
    add(rows, "contract", "full_reuse_only", config["reference_contract"]["full_retraining_forbidden"] is True and all("full_iscf_bsca" not in pair for pair in config["launch_order"]), config["reference_contract"]["full_retraining_forbidden"], True)
    add(rows, "contract", "historical_equal_excluded", config["reference_contract"]["historical_iscf_equal_reusable_as_without_bsca"] is False, config["reference_contract"]["historical_iscf_equal_reusable_as_without_bsca"], False)
    add(rows, "governance", "end_to_end_joint_training", config["training"]["from_scratch"] is True and config["training"]["joint_encoder_decoder"] is True, "from_scratch+joint", "true+true")
    add(rows, "governance", "no_hpo", config["training"]["hyperparameter_search_allowed"] is False, config["training"]["hyperparameter_search_allowed"], False)
    add(rows, "governance", "single_formal_access", config["authorization"]["formal_test_access_count_for_version"] == 1, config["authorization"]["formal_test_access_count_for_version"], 1)
    objective_semantics(rows)

    arm_lookup = {arm["id"]: arm for arm in config["arms"]}
    with tempfile.TemporaryDirectory(prefix="fatst_core_ablation_") as temp:
        temp_root = Path(temp)
        diagnostics: dict[str, dict[str, Any]] = {}
        encoder_hashes = []
        for arm_id in config["table"]["row_order"]:
            arm = arm_lookup[arm_id]
            rank = config["matched_ranks"]["ETTh1"][arm["rank_rule"]]
            model = parsed_model(
                "ETTh1", arm, rank, profiles["ETTh1"], config["profiles"]["sha256"], temp_root
            )
            initialization = train_repo.initialization_contract(model)
            diagnostics[arm_id] = train_repo.model_diagnostics(model)
            encoder_hashes.append(initialization["encoder_initialization_hash"])
            x = torch.randn(1, 720, 7)
            with torch.no_grad():
                output = model(x, torch.zeros_like(x), is_training=False, target_prefix=720)[0]
            add(rows, "model", arm_id, tuple(output.shape) == (1, 720, 7) and torch.isfinite(output).all().item(), tuple(output.shape), (1, 720, 7))
        add(rows, "model", "matched_encoder_initialization", len(set(encoder_hashes)) == 1, len(set(encoder_hashes)), 1)
        full_parameters = diagnostics["full_iscf_bsca"]["active_forward_parameters"]
        shared_parameters = diagnostics["shared_scope_projection"]["active_forward_parameters"]
        gap = abs(shared_parameters - full_parameters) / full_parameters
        add(rows, "model", "shared_projection_capacity_match", gap <= 0.005, gap, "<=0.005")

    syntax = run_command(["bash", "-n", "scripts/remote/run_iscf_bsca_core_ablation.sh"])
    dry = run_command(["bash", "scripts/remote/run_iscf_bsca_core_ablation.sh"], {"DRY_RUN": "1"})
    with tempfile.TemporaryDirectory(prefix="fatst_core_ablation_empty_") as temp:
        blocked = run_command(
            ["bash", "scripts/remote/run_iscf_bsca_core_ablation.sh"],
            {
                "FORMAL_TEST_ONLY": "1",
                "OUTPUT_ROOT": temp,
                "CONDA_BIN": sys.executable,
                "CONDA_ENV": "unused",
            },
        )
    add(rows, "execution", "runner_syntax", syntax.returncode == 0, syntax.returncode, 0)
    add(rows, "execution", "runner_dry_run_20", dry.returncode == 0 and "jobs=20" in dry.stdout, dry.stdout.strip().splitlines()[-1] if dry.stdout.strip() else dry.stderr, "returncode=0 and jobs=20")
    add(rows, "execution", "formal_test_manifest_block", blocked.returncode != 0, blocked.returncode, "nonzero before manifest")
    runner_text = (ROOT / "scripts/remote/run_iscf_bsca_core_ablation.sh").read_text(encoding="utf-8")
    add(rows, "execution", "remote_gpu_audit_present", "nvidia-smi --query-gpu" in runner_text, "present" if "nvidia-smi --query-gpu" in runner_text else "missing", "present")
    add(rows, "execution", "manifest_verification_present", "--verify-manifest" in runner_text, "present" if "--verify-manifest" in runner_text else "missing", "present")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "prelaunch_checks.csv", rows)
    jobs = []
    for index, (dataset, arm_id) in enumerate(config["launch_order"], start=1):
        arm = arm_lookup[arm_id]
        jobs.append(
            {
                "job": index,
                "dataset": dataset,
                "arm_id": arm_id,
                "readout_mode": arm["readout_mode"],
                "policy_mode": arm["policy_mode"],
                "objective_mode": arm["objective_mode"],
                "fixed_scale": arm["fixed_scale"],
                "rank": config["matched_ranks"][dataset][arm["rank_rule"]],
                "seed": 2021,
            }
        )
    write_csv(args.output_dir / "training_jobs.csv", jobs)
    passed = sum(bool(row["pass"]) for row in rows)
    summary = {
        "candidate_version": config["candidate_version"],
        "checks": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "pass": passed == len(rows),
        "training_jobs": len(jobs),
        "decision": "remote_training_ready" if passed == len(rows) else "prelaunch_blocked",
    }
    (args.output_dir / "prelaunch_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"core_ablation_prelaunch={'pass' if summary['pass'] else 'fail'} "
        f"checks={len(rows)} failed={summary['failed']} jobs={len(jobs)}"
    )
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

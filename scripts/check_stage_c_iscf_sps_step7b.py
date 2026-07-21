#!/usr/bin/env python3
"""Run the local ISCF-SPS Step 7B prelaunch gate."""

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
from evaluate_stage_c_pcsd_cf_checkpoint import sps_tensors  # noqa: E402
from analyze_stage_c_iscf_sps import expected_projection_contract  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_sps_step7b.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_post_d21_unconstrained_reset_20260720/"
            "iscf_sps_step7b_prelaunch_20260721"
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parsed_model(
    projection: str,
    partition: str,
    rank: int,
    profile: dict[str, Any],
    profile_hash: str,
    temp_root: Path,
) -> TimeAlign.Model:
    argv = [
        "train_repo.py",
        "--dataset-root", str(temp_root),
        "--dataset", "ETTh1",
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
        "--run-name", f"SPS_PRELAUNCH_{projection}_{partition}",
        "--output-dir", str(temp_root / "output"),
        "--device", "cpu",
        "--checkpoint-policy", "best-val",
        "--protocol-class", "method_screening",
        "--protocol-profile", "stage_c_iscf_sps_v0_step7b",
        "--profile-hash", profile_hash,
        "--legacy-patch-num", str(profile["patch_num"]),
        "--legacy-d-model", str(profile["d_model"]),
        "--legacy-d-ff", str(profile["d_ff"]),
        "--readout-mode", "iscf-scope-projected-synthesis",
        "--pcsd-coordinate-dim", "4",
        "--pcsd-mode-rank", str(rank),
        "--pcsd-policy-mode", "direct",
        "--pcsd-partition", partition,
        "--pcsd-partition-seed", "15101",
        "--sps-projection-mode", projection,
        "--pcc-objective-mode", "equal_skill",
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
    preset = train_repo.OFFICIAL_PRESETS["ETTh1"][720]
    (temp_root / preset.data_path).parent.mkdir(parents=True, exist_ok=True)
    (temp_root / preset.data_path).touch()
    official = train_repo.build_official_args(parsed, preset)
    torch.manual_seed(2021)
    return TimeAlign.Model(official).float().eval()


def run_command(command: list[str], extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, **(extra_env or {})},
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    profile_path = ROOT / config["profiles"]["path"]
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))[
        "dataset_profiles"
    ]
    add(rows, "contract", "profile_hash", file_hash(profile_path) == config["profiles"]["sha256"], file_hash(profile_path), config["profiles"]["sha256"])
    add(rows, "contract", "matrix_20", config["matrix"]["expected_runs"] == 20, config["matrix"]["expected_runs"], 20)
    add(rows, "contract", "unique_launch_20", len(config["launch_order"]) == 20 and len(set(map(tuple, config["launch_order"]))) == 20, len(config["launch_order"]), 20)
    remote_authorized = config["authorization"]["remote_training_authorized"]
    add(rows, "governance", "remote_authorization_boolean", isinstance(remote_authorized, bool), remote_authorized, "boolean")
    add(rows, "governance", "test_disabled", config["authorization"]["formal_test_access_authorized"] is False, config["authorization"]["formal_test_access_authorized"], False)
    add(rows, "governance", "no_new_loss_router_h", config["training"]["new_auxiliary_loss"] is False and config["training"]["new_router"] is False and config["training"]["requested_h_input"] is False, "loss=false,router=false,H=false", "all false")
    add(rows, "governance", "random_attribution_only", config["validation_gates"]["random_partition_is_attribution_only"] is True, config["validation_gates"]["random_partition_is_attribution_only"], True)

    with tempfile.TemporaryDirectory(prefix="fatst_sps_step7b_") as temp:
        temp_root = Path(temp)
        parameter_hashes = []
        x = torch.randn(2, 720, 7)
        for arm in config["arms"]:
            mode_rank = config["matched_ranks"]["ETTh1"][arm["rank_rule"]]
            model = parsed_model(
                arm["projection_mode"],
                arm["partition"],
                mode_rank,
                profiles["ETTh1"],
                config["profiles"]["sha256"],
                temp_root,
            )
            initialization = train_repo.initialization_contract(model)
            diagnostics = train_repo.model_diagnostics(model)
            parameter_hashes.append(initialization["pcsd_initialization_hash"])
            with torch.no_grad():
                output = model(
                    x,
                    torch.zeros_like(x),
                    is_training=False,
                    target_prefix=720,
                )[0]
                health = sps_tensors(model, x)
            expected_ranks, expected_degrees = expected_projection_contract(
                config,
                arm["projection_mode"],
                mode_rank,
            )
            passed = bool(
                tuple(output.shape) == (2, 720, 7)
                and health is not None
                and all(torch.isfinite(value).all() for value in health.values())
                and diagnostics["sps_projection_ranks"] == expected_ranks
                and diagnostics["sps_projected_degrees"] == expected_degrees
            )
            add(rows, "model", arm["id"], passed, f"ranks={diagnostics['sps_projection_ranks']},degrees={diagnostics['sps_projected_degrees']}", f"ranks={expected_ranks},degrees={expected_degrees}")
        add(rows, "model", "paired_trainable_initialization", len(set(parameter_hashes)) == 1, len(set(parameter_hashes)), 1)

    syntax = run_command(["bash", "-n", "scripts/remote/run_stage_c_iscf_sps_step7b.sh"])
    dry = run_command(["bash", "scripts/remote/run_stage_c_iscf_sps_step7b.sh"], {"DRY_RUN": "1"})
    test_blocked = run_command(["bash", "scripts/remote/run_stage_c_iscf_sps_step7b.sh"], {"EVALUATION_SPLIT": "test", "DRY_RUN": "1"})
    analyzer = run_command([sys.executable, "scripts/analyze_stage_c_iscf_sps.py", "--synthetic-smoke"])
    add(rows, "execution", "runner_syntax", syntax.returncode == 0, syntax.returncode, 0)
    dry_summary = dry.stdout.strip().splitlines()[-1] if dry.stdout.strip() else ""
    add(rows, "execution", "runner_dry_run_20", dry.returncode == 0 and "jobs=20" in dry.stdout, dry_summary, "returncode=0 and jobs=20")
    add(rows, "execution", "remote_state_reflected_in_dry_run", f"remote_authorized={str(remote_authorized).lower()}" in dry.stdout, dry_summary, f"remote_authorized={str(remote_authorized).lower()}")
    add(rows, "execution", "test_split_blocked", test_blocked.returncode == 3 and "validation-only" in test_blocked.stderr, f"returncode={test_blocked.returncode}:{test_blocked.stderr.strip()}", "returncode=3")
    add(rows, "execution", "analyzer_smoke", analyzer.returncode == 0, analyzer.stdout.strip()[-200:], "returncode=0")
    runner_text = (ROOT / "scripts/remote/run_stage_c_iscf_sps_step7b.sh").read_text(encoding="utf-8")
    add(rows, "execution", "remote_gpu_audit_present", "nvidia-smi --query-gpu" in runner_text, "nvidia-smi" if "nvidia-smi --query-gpu" in runner_text else "missing", "present")
    add(rows, "execution", "log_scanner_fallback", "command -v rg" in runner_text and "grep -Ein" in runner_text, "rg+grep" if "grep -Ein" in runner_text else "incomplete", "rg+grep")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "checks.csv", rows)
    jobs = []
    arm_lookup = {arm["id"]: arm for arm in config["arms"]}
    for index, (dataset, arm_id) in enumerate(config["launch_order"], start=1):
        arm = arm_lookup[arm_id]
        profile = profiles[dataset]
        jobs.append(
            {
                "job": index,
                "dataset": dataset,
                "arm": arm_id,
                "projection_mode": arm["projection_mode"],
                "partition": arm["partition"],
                "mode_rank": config["matched_ranks"][dataset][arm["rank_rule"]],
                "profile": profile["profile"],
                "seed": 2021,
                "split": "validation",
            }
        )
    write_csv(args.output_dir / "jobs.csv", jobs)
    passed = sum(bool(row["pass"]) for row in rows)
    summary = {
        "candidate": config["candidate_version"],
        "checks": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "pass": passed == len(rows),
        "jobs": len(jobs),
        "remote_training_authorized": remote_authorized,
        "formal_test_authorized": False,
        "decision": (
            "step8_remote_validation_authorized"
            if passed == len(rows) and remote_authorized
            else "step7b_prelaunch_pass_wait_remote_authorization"
            if passed == len(rows)
            else "step7b_blocked"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not summary["pass"]:
        failed = [row["case"] for row in rows if not row["pass"]]
        raise RuntimeError(f"SPS Step7B gate failed: {failed}")
    print(f"iscf_sps_step7b_prelaunch=pass checks={len(rows)} jobs={len(jobs)}")


if __name__ == "__main__":
    main()

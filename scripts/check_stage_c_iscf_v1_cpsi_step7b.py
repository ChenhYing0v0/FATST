#!/usr/bin/env python3
"""Run the ISCF-v1-CPSI Step 7B prelaunch gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
    cpsi_tensors,
    test_audit_authorized,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_v1_cpsi_step7b.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_post_d21_unconstrained_reset_20260720/"
            "iscf_v1_cpsi_step7b_prelaunch_20260721"
        ),
    )
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    readout: str,
    rank: int,
    profile: dict[str, Any],
    profile_hash: str,
    dataset_root: Path,
) -> TimeAlign.Model:
    argv = [
        "train_repo.py",
        "--dataset-root", str(dataset_root),
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
        "--run-name", f"CPSI_PRELAUNCH_{readout}",
        "--output-dir", str(dataset_root / "output"),
        "--device", "cpu",
        "--checkpoint-policy", "best-val",
        "--protocol-class", "method_screening",
        "--protocol-profile", "stage_c_iscf_v1_cpsi_v1",
        "--profile-hash", profile_hash,
        "--legacy-patch-num", str(profile["patch_num"]),
        "--legacy-d-model", str(profile["d_model"]),
        "--legacy-d-ff", str(profile["d_ff"]),
        "--readout-mode", readout,
        "--cpsi-rank", "32",
        "--pcsd-coordinate-dim", "4",
        "--pcsd-mode-rank", str(rank),
        "--pcsd-policy-mode", "direct",
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
    (dataset_root / preset.data_path).parent.mkdir(parents=True, exist_ok=True)
    (dataset_root / preset.data_path).touch()
    official = train_repo.build_official_args(parsed, preset)
    torch.manual_seed(2021)
    return TimeAlign.Model(official).float().eval()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    profiles_path = ROOT / config["profiles"]["path"]
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))[
        "dataset_profiles"
    ]
    add(rows, "contract", "profile_hash", file_hash(profiles_path) == config["profiles"]["sha256"], file_hash(profiles_path), config["profiles"]["sha256"])
    add(rows, "contract", "matrix_35", config["matrix"]["expected_runs"] == 35, config["matrix"]["expected_runs"], 35)
    add(rows, "contract", "launch_25", len(config["launch_order"]) == 25 and len(set(map(tuple, config["launch_order"]))) == 25, len(config["launch_order"]), 25)
    add(rows, "contract", "test_authorized", test_audit_authorized(config), test_audit_authorized(config), True)
    add(rows, "governance", "controls_intermediate", config["severity"]["controls_are_intermediate_diagnostics"] is True, config["severity"]["controls_are_intermediate_diagnostics"], True)
    add(rows, "governance", "controls_do_not_block_test", config["severity"]["controls_can_block_mechanism_claim_but_not_test_access"] is True, config["severity"]["controls_can_block_mechanism_claim_but_not_test_access"], True)

    contract = config["reference_contract"]
    source_config = ROOT / contract["source_config_path"]
    source_audit = ROOT / contract["source_run_audit_path"]
    add(rows, "reference", "source_config_hash", file_hash(source_config) == contract["source_config_sha256"], file_hash(source_config), contract["source_config_sha256"])
    add(rows, "reference", "source_audit_hash", file_hash(source_audit) == contract["source_run_audit_sha256"], file_hash(source_audit), contract["source_run_audit_sha256"])
    lookup = {(row["arm"], row["dataset"]): row for row in read_csv(source_audit)}
    reference_ok = True
    for arm in config["effective_arms"]:
        if arm["source"] != "reused_reference":
            continue
        for dataset in config["datasets"]:
            source = lookup[(arm["source_arm"], dataset)]
            expected = contract["checkpoint_sha256"][arm["id"]][dataset]
            reference_ok = reference_ok and bool(
                source["status"] == "ok"
                and source["protocol_pass"] == "True"
                and source["checkpoint_sha256"] == expected
            )
    add(rows, "reference", "ten_frozen_runs", reference_ok, reference_ok, True)

    new_arms = [
        arm for arm in config["effective_arms"] if arm["source"] == "new_training"
    ]
    with tempfile.TemporaryDirectory(prefix="fatst_cpsi_step7b_") as temp:
        temp_root = Path(temp)
        parent_hashes = []
        for arm in new_arms:
            model = parsed_model(
                arm["readout_mode"],
                config["matched_ranks"]["ETTh1"][arm["rank_rule"]],
                profiles["ETTh1"],
                config["profiles"]["sha256"],
                temp_root,
            )
            initialization = train_repo.initialization_contract(model)
            diagnostics = train_repo.model_diagnostics(model)
            parent_hashes.append(initialization["cpsi_parent_initialization_hash"])
            x = torch.randn(2, 720, 7)
            with torch.no_grad():
                output = model(x, torch.zeros_like(x), is_training=False, target_prefix=720)[0]
                health = cpsi_tensors(model, x)
            passed = bool(
                tuple(output.shape) == (2, 720, 7)
                and health is not None
                and all(torch.isfinite(value).all() for value in health.values())
                and diagnostics["cpsi_interaction_rank"] == 32
                and diagnostics["cpsi_interaction_parameters"] > 0
            )
            add(rows, "model", arm["id"], passed, diagnostics["cpsi_interaction_parameters"], ">0 and finite")
        add(rows, "model", "paired_parent_hash", len(set(parent_hashes)) == 1, len(set(parent_hashes)), 1)

    commands = [
        ["bash", "-n", "scripts/remote/run_stage_c_iscf_v1_cpsi.sh"],
        ["bash", "scripts/remote/run_stage_c_iscf_v1_cpsi.sh"],
        [sys.executable, "scripts/analyze_stage_c_iscf_v1_cpsi.py", "--synthetic-smoke"],
    ]
    environments = [{}, {"DRY_RUN": "1"}, {}]
    names = ["runner_syntax", "runner_dry_run", "analyzer_smoke"]
    for name, command, extra_env in zip(names, commands, environments):
        result = subprocess.run(
            command,
            cwd=ROOT,
            env={**dict(__import__("os").environ), **extra_env},
            text=True,
            capture_output=True,
            check=False,
        )
        add(rows, "execution", name, result.returncode == 0, result.stdout.strip()[-200:], "returncode=0")
    runner_text = (ROOT / "scripts/remote/run_stage_c_iscf_v1_cpsi.sh").read_text(
        encoding="utf-8"
    )
    add(
        rows,
        "execution",
        "remote_log_scanner_fallback",
        "command -v rg" in runner_text and "grep -Ein" in runner_text,
        "rg+grep" if "grep -Ein" in runner_text else "rg-only",
        "rg+grep",
    )

    passed = sum(bool(row["pass"]) for row in rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "checks.csv", rows)
    summary = {
        "candidate": config["candidate_version"],
        "checks": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "pass": passed == len(rows),
        "decision": "step7b_prelaunch_pass" if passed == len(rows) else "step7b_blocked",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not summary["pass"]:
        failed = [row["case"] for row in rows if not row["pass"]]
        raise RuntimeError(f"CPSI Step7B gate failed: {failed}")
    print(f"cpsi_step7b_prelaunch=pass checks={len(rows)}")


if __name__ == "__main__":
    main()

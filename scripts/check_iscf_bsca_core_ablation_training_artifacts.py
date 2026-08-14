#!/usr/bin/env python3
"""Validate and freeze the Core-Ablation checkpoints before formal test."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/iscf_bsca_core_ablation_protocol.json"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--verify-manifest", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def default_path(value: str, override: Path | None) -> Path:
    return override if override is not None else Path(value)


def run_dir(output_root: Path, arm_id: str, dataset: str, seed: int) -> Path:
    return output_root / arm_id / dataset / "h720_full" / f"seed{seed}"


def expected_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    arms = {arm["id"]: arm for arm in config["arms"]}
    profiles = load_json(Path(config["profiles"]["path"]))["dataset_profiles"]
    rows = []
    for dataset, arm_id in config["launch_order"]:
        arm = arms[arm_id]
        rows.append(
            {
                "dataset": dataset,
                "arm_id": arm_id,
                "seed": config["seeds"][0],
                "readout_mode": arm["readout_mode"],
                "policy_mode": arm["policy_mode"],
                "objective_mode": arm["objective_mode"],
                "fixed_scale": arm["fixed_scale"],
                "rank": config["matched_ranks"][dataset][arm["rank_rule"]],
                "patch_num": profiles[dataset]["patch_num"],
                "d_model": profiles[dataset]["d_model"],
                "d_ff": profiles[dataset]["d_ff"],
            }
        )
    return rows


def validate_row(
    config: dict[str, Any], output_root: Path, row: dict[str, Any]
) -> dict[str, Any]:
    path = run_dir(output_root, row["arm_id"], row["dataset"], row["seed"])
    missing = [
        name
        for name in config["artifact_contract"]["training_required_files"]
        if not (path / name).is_file() or (path / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"missing training artifacts in {path}: {missing}")
    if any((path / name).exists() for name in config["artifact_contract"]["formal_test_required_files"]):
        raise RuntimeError(f"formal-test artifact exists before manifest freeze: {path}")

    effective = load_json(path / "effective_config.json")["adapter"]
    invariants = load_json(path / "trained_invariants.json")
    initialization = load_json(path / "initialization_contract.json")
    expected = {
        "dataset": row["dataset"],
        "seed": row["seed"],
        "readout_mode": row["readout_mode"],
        "pcsd_policy_mode": row["policy_mode"],
        "pcc_objective_mode": row["objective_mode"],
        "pcsd_fixed_scale": row["fixed_scale"],
        "pcsd_mode_rank": row["rank"],
        "legacy_patch_num": row["patch_num"],
        "legacy_d_model": row["d_model"],
        "legacy_d_ff": row["d_ff"],
        "checkpoint_policy": "best-val",
        "evaluation_prefix_mode": "full-crop",
        "target_horizons": [720],
        "validation_horizons": [96, 192, 336, 720],
        "final_evaluation_split": "val",
        "profile_hash": config["profiles"]["sha256"],
    }
    mismatches = {
        key: {"expected": value, "actual": effective.get(key)}
        for key, value in expected.items()
        if effective.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"effective-config mismatch in {path}: {mismatches}")
    if invariants.get("pass") is not True or invariants.get("all_finite") is not True:
        raise RuntimeError(f"validation invariant failure in {path}")
    if invariants.get("evaluation_split") != "val":
        raise RuntimeError(f"wrong validation artifact role in {path}")
    if initialization.get("encoder_initialization_hash") is None:
        raise RuntimeError(f"missing encoder initialization hash in {path}")

    checkpoint = path / config["artifact_contract"]["checkpoint_file"]
    return {
        **row,
        "run_dir": str(path),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "encoder_initialization_hash": initialization["encoder_initialization_hash"],
        "readout_initialization_hash": initialization.get("pcsd_initialization_hash"),
        "validation_invariant_checkpoint_sha256": invariants.get("checkpoint_sha256"),
    }


def verify_manifest(
    config: dict[str, Any], config_path: Path, manifest_path: Path
) -> None:
    manifest = load_json(manifest_path)
    if manifest.get("candidate_version") != config["candidate_version"]:
        raise RuntimeError("manifest candidate version mismatch")
    if manifest.get("protocol_sha256") != file_sha256(config_path):
        raise RuntimeError("manifest protocol hash mismatch")
    rows = manifest.get("rows", [])
    if len(rows) != config["matrix"]["new_training_runs"]:
        raise RuntimeError("manifest row count mismatch")
    for row in rows:
        checkpoint = Path(row["run_dir"]) / config["artifact_contract"]["checkpoint_file"]
        if file_sha256(checkpoint) != row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed after manifest freeze: {checkpoint}")
    print(
        f"core_ablation_manifest_verify=pass rows={len(rows)} "
        f"manifest={manifest_path}"
    )


def main(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    output_root = default_path(
        config["artifact_contract"]["remote_output_root"], args.output_root
    )
    manifest_path = default_path(
        config["artifact_contract"]["training_manifest"], args.manifest
    )
    if args.verify_manifest:
        verify_manifest(config, args.config, manifest_path)
        return

    rows = [validate_row(config, output_root, row) for row in expected_rows(config)]
    expected_count = config["matrix"]["new_training_runs"]
    if len(rows) != expected_count:
        raise RuntimeError(f"expected {expected_count} rows, found {len(rows)}")
    hashes = [row["checkpoint_sha256"] for row in rows]
    required_unique = config["gates"]["unique_new_checkpoint_hashes_required"]
    if len(set(hashes)) != required_unique:
        raise RuntimeError(
            f"expected {required_unique} unique checkpoints, found {len(set(hashes))}"
        )

    for dataset in config["datasets"]:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        encoder_hashes = {row["encoder_initialization_hash"] for row in dataset_rows}
        if len(encoder_hashes) != 1:
            raise RuntimeError(
                f"matched encoder initialization failed for {dataset}: {encoder_hashes}"
            )

    payload = {
        "manifest_version": 1,
        "candidate_version": config["candidate_version"],
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol_path": str(args.config),
        "protocol_sha256": file_sha256(args.config),
        "profile_sha256": file_sha256(Path(config["profiles"]["path"])),
        "row_count": len(rows),
        "unique_checkpoint_hashes": len(set(hashes)),
        "formal_test_artifacts_present_at_freeze": False,
        "rows": rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"core_ablation_training_gate=pass rows={len(rows)} "
        f"unique_hashes={len(set(hashes))} manifest={manifest_path}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)

#!/usr/bin/env python3
"""Freeze or verify the complete Decoder-Transfer checkpoint manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/iscf_bsca_decoder_transfer_protocol.json"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--verify-manifest", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    arms = {row["id"]: row for row in config["arms"]}
    profiles = load_json(Path(config["profiles"]["path"]))
    rows = []
    for backbone, dataset, arm_id in config["launch_order"]:
        arm = arms[arm_id]
        common = profiles["common"]
        family = profiles["backbones"][backbone]
        profile = family["dataset_profiles"][dataset]
        rows.append(
            {
                "backbone": backbone,
                "dataset": dataset,
                "arm_id": arm_id,
                "seed": config["seeds"][0],
                "encoder_mode": family["encoder_mode"],
                "readout_mode": arm["readout_mode"],
                "policy_mode": arm["policy_mode"],
                "objective_mode": arm["objective_mode"],
                "seq_len": common["seq_len"],
                "learning_rate": profile["learning_rate"],
                "batch_size": profile["batch_size"],
                "rank": (
                    config["matched_ranks"][dataset]
                    if arm["readout_mode"] == "siff-independent-scope-control"
                    else 256
                ),
            }
        )
    return rows


def run_dir(root: Path, row: dict[str, Any]) -> Path:
    return root / row["backbone"] / row["arm_id"] / row["dataset"] / "seed2021"


def validate(config: dict[str, Any], root: Path, row: dict[str, Any]) -> dict[str, Any]:
    directory = run_dir(root, row)
    missing = [
        name
        for name in config["artifact_contract"]["training_required_files"]
        if not (directory / name).is_file() or not (directory / name).stat().st_size
    ]
    if missing:
        raise RuntimeError(f"missing artifacts in {directory}: {missing}")
    if any(
        (directory / name).exists()
        for name in config["artifact_contract"]["formal_test_required_files"]
    ):
        raise RuntimeError(f"formal-test artifact exists before freeze: {directory}")
    adapter = load_json(directory / "effective_config.json")["adapter"]
    invariants = load_json(directory / "trained_invariants.json")
    initialization = load_json(directory / "initialization_contract.json")
    expected = {
        "dataset": row["dataset"],
        "seed": row["seed"],
        "seq_len": row["seq_len"],
        "encoder_mode": row["encoder_mode"],
        "readout_mode": row["readout_mode"],
        "pcsd_policy_mode": row["policy_mode"],
        "pcc_objective_mode": row["objective_mode"],
        "pcsd_mode_rank": row["rank"],
        "learning_rate": row["learning_rate"],
        "batch_size": row["batch_size"],
        "checkpoint_policy": "best-val",
        "validation_horizons": [96, 192, 336, 720],
        "final_evaluation_split": "val",
        "profile_hash": config["profiles"]["sha256"],
    }
    mismatch = {
        key: {"expected": value, "actual": adapter.get(key)}
        for key, value in expected.items()
        if adapter.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"effective-config mismatch in {directory}: {mismatch}")
    if invariants.get("pass") is not True or invariants.get("evaluation_split") != "val":
        raise RuntimeError(f"validation invariant failure: {directory}")
    if not initialization.get("encoder_initialization_hash"):
        raise RuntimeError(f"missing encoder hash: {directory}")
    checkpoint = directory / config["artifact_contract"]["checkpoint_file"]
    return {
        **row,
        "run_dir": str(directory),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "encoder_initialization_hash": initialization["encoder_initialization_hash"],
        "readout_initialization_hash": initialization.get(
            "pcsd_initialization_hash",
            initialization.get("direct_readout_initialization_hash"),
        ),
    }


def main(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    root = args.output_root or Path(config["artifact_contract"]["remote_output_root"])
    manifest_path = args.manifest or Path(config["artifact_contract"]["training_manifest"])
    if args.verify_manifest:
        manifest = load_json(manifest_path)
        if manifest["protocol_sha256"] != sha256(args.config):
            raise RuntimeError("protocol hash changed after manifest freeze")
        for row in manifest["rows"]:
            checkpoint = Path(row["run_dir"]) / config["artifact_contract"]["checkpoint_file"]
            if sha256(checkpoint) != row["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint mutated: {checkpoint}")
        print(f"decoder_transfer_manifest_verify=pass rows={len(manifest['rows'])}")
        return

    rows = [validate(config, root, row) for row in expected_rows(config)]
    if len(rows) != config["matrix"]["new_training_runs"]:
        raise RuntimeError("training row count mismatch")
    hashes = {row["checkpoint_sha256"] for row in rows}
    if len(hashes) != config["gates"]["unique_checkpoint_hashes_required"]:
        raise RuntimeError("checkpoint hashes are not unique")
    for backbone in config["backbones"]:
        for dataset in config["datasets"]:
            subset = [
                row for row in rows
                if row["backbone"] == backbone["id"] and row["dataset"] == dataset
            ]
            if len({row["encoder_initialization_hash"] for row in subset}) != 1:
                raise RuntimeError(f"unmatched encoder initialization: {backbone['id']} {dataset}")
    payload = {
        "manifest_version": 1,
        "candidate_version": config["candidate_version"],
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol_sha256": sha256(args.config),
        "profile_sha256": sha256(Path(config["profiles"]["path"])),
        "row_count": len(rows),
        "unique_checkpoint_hashes": len(hashes),
        "formal_test_artifacts_present_at_freeze": False,
        "rows": rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"decoder_transfer_training_gate=pass rows={len(rows)} hashes={len(hashes)}")


if __name__ == "__main__":
    main(parse_args())

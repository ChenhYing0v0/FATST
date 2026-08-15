#!/usr/bin/env python3
"""Freeze or verify the PatchTST decoder-transfer v2.1 manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/iscf_bsca_decoder_transfer_patchtst_v2p1_formal.json"
        ),
    )
    parser.add_argument("--hpo-root", type=Path)
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def expected_rows(
    config: dict[str, Any], hpo_root: Path, output_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selected in config["selected_profiles"]:
        dataset = selected["dataset"]
        for arm in config["arms"]:
            if arm["source"] == "selected_parent_hpo_checkpoint":
                run_dir = (
                    hpo_root
                    / dataset
                    / selected["profile_id"]
                    / "seed2021"
                )
            else:
                run_dir = (
                    output_root
                    / "training"
                    / arm["id"]
                    / dataset
                    / "seed2021"
                )
            rows.append(
                {
                    "backbone": config["backbone"]["id"],
                    "dataset": dataset,
                    "arm_id": arm["id"],
                    "seed": config["seeds"][0],
                    "profile_id": selected["profile_id"],
                    "mode_rank": selected["mode_rank"],
                    "readout_learning_rate_multiplier": selected[
                        "readout_learning_rate_multiplier"
                    ],
                    "readout_weight_decay": selected[
                        "readout_weight_decay"
                    ],
                    "expected_selected_checkpoint_sha256": selected[
                        "bsca_checkpoint_sha256"
                    ]
                    if arm["source"] == "selected_parent_hpo_checkpoint"
                    else None,
                    "readout_mode": arm["readout_mode"],
                    "policy_mode": arm["policy_mode"],
                    "objective_mode": arm["objective_mode"],
                    "source": arm["source"],
                    "run_dir": str(run_dir),
                    "formal_artifact_dir": str(
                        output_root
                        / "formal_test"
                        / arm["id"]
                        / dataset
                        / "seed2021"
                    ),
                }
            )
    return rows


def numeric_training_ok(run_dir: Path) -> tuple[bool, int, float]:
    training_rows = read_csv(run_dir / "training_log.csv")
    values = [float(row["val_mean_mse"]) for row in training_rows]
    if not values or not all(math.isfinite(value) for value in values):
        return False, -1, float("nan")
    best_index = min(range(len(values)), key=values.__getitem__)
    metric_rows = read_csv(run_dir / "metrics_by_target_horizon.csv")
    metric_values = [
        float(row[key])
        for row in metric_rows
        for key in ("mse", "mae")
    ]
    metric_ok = (
        {int(row["target_horizon"]) for row in metric_rows}
        == {96, 192, 336, 720}
        and all(math.isfinite(value) for value in metric_values)
    )
    return (
        metric_ok,
        int(training_rows[best_index]["epoch"]),
        values[best_index],
    )


def validate_row(
    config: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    run_dir = Path(row["run_dir"])
    missing = [
        name
        for name in config["artifact_contract"]["training_required_files"]
        if not (run_dir / name).is_file()
        or not (run_dir / name).stat().st_size
    ]
    if missing:
        raise RuntimeError(f"missing training artifacts in {run_dir}: {missing}")
    formal_dir = Path(row["formal_artifact_dir"])
    existing_test = [
        name
        for name in config["artifact_contract"]["formal_test_required_files"]
        if (formal_dir / name).exists()
    ]
    if existing_test:
        raise RuntimeError(
            f"formal-test artifacts exist before manifest freeze in "
            f"{formal_dir}: {existing_test}"
        )

    effective = load_json(run_dir / "effective_config.json")
    adapter = effective["adapter"]
    expected = {
        "dataset": row["dataset"],
        "seed": row["seed"],
        "seq_len": 336,
        "encoder_mode": config["backbone"]["encoder_mode"],
        "readout_mode": row["readout_mode"],
        "pcsd_policy_mode": row["policy_mode"],
        "pcc_objective_mode": row["objective_mode"],
        "pcsd_mode_rank": row["mode_rank"],
        "checkpoint_policy": "best-val",
        "validation_horizons": [96, 192, 336, 720],
        "final_evaluation_split": "val",
        "official_test_mode": False,
        "profile_hash": config["profiles"]["sha256"],
    }
    mismatch = {
        key: {"expected": value, "actual": adapter.get(key)}
        for key, value in expected.items()
        if adapter.get(key) != value
    }
    for key, expected_float in (
        (
            "readout_learning_rate_multiplier",
            row["readout_learning_rate_multiplier"],
        ),
        ("readout_weight_decay", row["readout_weight_decay"]),
    ):
        actual = float(adapter.get(key, float("nan")))
        if not math.isclose(actual, expected_float, rel_tol=0.0, abs_tol=1e-15):
            mismatch[key] = {"expected": expected_float, "actual": actual}
    if mismatch:
        raise RuntimeError(f"effective-config mismatch in {run_dir}: {mismatch}")

    numeric_ok, best_epoch, validation_mean_mse = numeric_training_ok(run_dir)
    if not numeric_ok:
        raise RuntimeError(f"non-finite or incomplete validation metrics: {run_dir}")
    initialization = load_json(run_dir / "initialization_contract.json")
    checkpoint = run_dir / config["artifact_contract"]["checkpoint_file"]
    checkpoint_hash = sha256(checkpoint)
    expected_hash = row["expected_selected_checkpoint_sha256"]
    if expected_hash is not None and checkpoint_hash != expected_hash:
        raise RuntimeError(
            f"selected checkpoint hash mismatch: {checkpoint} "
            f"expected={expected_hash} actual={checkpoint_hash}"
        )
    if not initialization.get("encoder_initialization_hash"):
        raise RuntimeError(f"missing encoder initialization hash: {run_dir}")
    readout_hash = initialization.get("pcsd_initialization_hash")
    if not readout_hash:
        raise RuntimeError(f"missing readout initialization hash: {run_dir}")
    return {
        **row,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "best_epoch": best_epoch,
        "validation_mean_mse": validation_mean_mse,
        "encoder_initialization_hash": initialization[
            "encoder_initialization_hash"
        ],
        "readout_initialization_hash": readout_hash,
    }


def verify_manifest(
    config_path: Path,
    config: dict[str, Any],
    manifest_path: Path,
) -> None:
    manifest = load_json(manifest_path)
    if manifest["protocol_sha256"] != sha256(config_path):
        raise RuntimeError("protocol hash changed after manifest freeze")
    selection = Path(config["selection_artifact"]["path"])
    if sha256(selection) != manifest["selection_artifact_sha256"]:
        raise RuntimeError("selection artifact changed after manifest freeze")
    for row in manifest["rows"]:
        checkpoint = Path(row["run_dir"]) / config["artifact_contract"][
            "checkpoint_file"
        ]
        if sha256(checkpoint) != row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint mutated: {checkpoint}")
    if manifest["row_count"] != 10 or manifest["unique_checkpoint_hashes"] != 10:
        raise RuntimeError("immutable manifest completeness mismatch")
    print("patchtst_v2p1_manifest_verify=pass rows=10 hashes=10")


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    hpo_root = args.hpo_root or Path(
        config["artifact_contract"]["hpo_output_root"]
    )
    output_root = args.output_root or Path(
        config["artifact_contract"]["remote_output_root"]
    )
    manifest_path = args.manifest or Path(
        config["artifact_contract"]["training_manifest"]
    )
    if args.verify_manifest:
        verify_manifest(args.config, config, manifest_path)
        return

    profile_path = Path(config["profiles"]["path"])
    selection_path = Path(config["selection_artifact"]["path"])
    if sha256(profile_path) != config["profiles"]["sha256"]:
        raise RuntimeError("backbone profile hash mismatch")
    if sha256(selection_path) != config["selection_artifact"]["sha256"]:
        raise RuntimeError("selected profile artifact hash mismatch")

    rows = [
        validate_row(config, row)
        for row in expected_rows(config, hpo_root, output_root)
    ]
    if len(rows) != config["matrix"]["expected_runs"]:
        raise RuntimeError("manifest row count mismatch")
    hashes = {row["checkpoint_sha256"] for row in rows}
    if len(hashes) != config["gates"]["unique_checkpoint_hashes_required"]:
        raise RuntimeError(
            f"checkpoint uniqueness gate failed: {len(hashes)}/10"
        )
    for dataset in config["datasets"]:
        pair = [row for row in rows if row["dataset"] == dataset]
        if len(pair) != 2:
            raise RuntimeError(f"matched pair missing: {dataset}")
        if len({row["mode_rank"] for row in pair}) != 1:
            raise RuntimeError(f"rank mismatch: {dataset}")
        if len({row["encoder_initialization_hash"] for row in pair}) != 1:
            raise RuntimeError(f"encoder initialization mismatch: {dataset}")
        if len({row["readout_initialization_hash"] for row in pair}) != 1:
            raise RuntimeError(f"readout initialization mismatch: {dataset}")

    payload = {
        "manifest_version": 1,
        "candidate_version": config["candidate_version"],
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol_sha256": sha256(args.config),
        "profile_sha256": sha256(profile_path),
        "selection_artifact_sha256": sha256(selection_path),
        "row_count": len(rows),
        "unique_checkpoint_hashes": len(hashes),
        "formal_test_artifacts_present_at_freeze": False,
        "matched_initialization_pairs": len(config["datasets"]),
        "rows": rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("patchtst_v2p1_training_gate=pass rows=10 hashes=10 pairs=5")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit PatchTST decoder-HPO v2 training artifacts without test access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/iscf_bsca_decoder_transfer_patchtst_hpo_v2.json"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    reference_ranks = config["reference_profile"]["mode_rank_by_dataset"]
    required = config["artifact_contract"]["required_files"]
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        for profile in config["search_profiles"]:
            run_dir = args.output_root / dataset / profile["id"] / "seed2021"
            missing = [name for name in required if not (run_dir / name).is_file()]
            validation_mse = float("nan")
            best_epoch = -1
            effective_ok = False
            checkpoint_hash = ""
            if not missing:
                training_rows = read_rows(run_dir / "training_log.csv")
                values = [float(row["val_mean_mse"]) for row in training_rows]
                if values and all(math.isfinite(value) for value in values):
                    index = min(range(len(values)), key=values.__getitem__)
                    validation_mse = values[index]
                    best_epoch = int(training_rows[index]["epoch"])
                effective = json.loads((run_dir / "effective_config.json").read_text())
                adapter = effective["adapter"]
                expected_rank = max(1, round(reference_ranks[dataset] * profile["rank_scale"]))
                effective_ok = (
                    adapter["dataset"] == dataset
                    and adapter["hpo_profile_id"] == profile["id"]
                    and adapter["final_evaluation_split"] == "val"
                    and adapter["official_test_mode"] is False
                    and adapter["pcsd_mode_rank"] == expected_rank
                    and abs(adapter["readout_learning_rate_multiplier"] - profile["readout_learning_rate_multiplier"]) < 1e-15
                    and abs(adapter["readout_weight_decay"] - profile["readout_weight_decay"]) < 1e-15
                )
                invariants = json.loads((run_dir / "trained_invariants.json").read_text())
                effective_ok = effective_ok and invariants.get("pass") is True
                checkpoint_hash = sha256(run_dir / "checkpoint.pt")
            passed = not missing and effective_ok and math.isfinite(validation_mse)
            rows.append(
                {
                    "dataset": dataset,
                    "profile_id": profile["id"],
                    "run_dir": str(run_dir),
                    "mode_rank": max(1, round(reference_ranks[dataset] * profile["rank_scale"])),
                    "readout_lr_multiplier": profile["readout_learning_rate_multiplier"],
                    "readout_weight_decay": profile["readout_weight_decay"],
                    "best_epoch": best_epoch,
                    "validation_mean_mse": validation_mse,
                    "checkpoint_sha256": checkpoint_hash,
                    "missing_files": ";".join(missing),
                    "effective_config_ok": effective_ok,
                    "pass": passed,
                }
            )
            if passed:
                manifest_rows.append(
                    {
                        "dataset": dataset,
                        "profile_id": profile["id"],
                        "checkpoint": str(run_dir / "checkpoint.pt"),
                        "checkpoint_sha256": checkpoint_hash,
                    }
                )

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.audit_dir / "trial_scorecard.csv", rows)
    complete = sum(bool(row["pass"]) for row in rows)
    unique_hashes = len({row["checkpoint_sha256"] for row in rows if row["pass"]})
    summary = {
        "pass": complete == 50 and unique_hashes == 50,
        "complete_runs": complete,
        "expected_runs": 50,
        "unique_checkpoint_hashes": unique_hashes,
        "formal_test_jobs": 0,
    }
    (args.audit_dir / "artifact_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.manifest is not None and summary["pass"]:
        write_rows(args.manifest, manifest_rows)
    if not summary["pass"]:
        raise SystemExit(
            f"artifact gate failed: complete={complete}/50 unique_hashes={unique_hashes}/50"
        )
    print("patchtst_decoder_hpo_v2_artifacts=pass runs=50 hashes=50 test=0")


if __name__ == "__main__":
    main()

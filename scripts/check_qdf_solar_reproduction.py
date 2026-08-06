#!/usr/bin/env python3
"""Audit the frozen QDF Solar reproduction contract before remote launch."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "qdf_solar_reproduction.json"
PUBLISHED = (
    ROOT
    / "analysis"
    / "iscf_bsca_paper_experiment_consolidation_20260731"
    / "qdf_main_i_20260806"
    / "qdf_table6_published.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    errors: list[str] = []
    contract = config["evaluation_contract"]
    if contract["horizons"] != [96, 192, 336, 720]:
        errors.append("horizon contract is not the frozen four-horizon matrix")
    if contract["seed"] != 2023:
        errors.append("seed must match the released QDF scripts (2023)")
    if not contract["test_access_once_after_training"]:
        errors.append("formal test must run once after training")

    source_root = ROOT / config["candidate"]["local_source_root"]
    for relative, expected in config["source_hashes"]["executed_local"].items():
        actual = sha256(source_root / relative)
        if actual != expected:
            errors.append(f"executed source hash mismatch: {relative}")

    command = ["bash", str(source_root / "scripts" / "Solar.sh")]
    environment = {"MODE": "dry-run", "SEED": str(contract["seed"])}
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**__import__("os").environ, **environment},
        check=True,
        text=True,
        capture_output=True,
    )
    dry_rows = [line.split("\t") for line in completed.stdout.splitlines() if line]
    if len(dry_rows) != 4:
        errors.append(f"dry-run expected 4 rows, got {len(dry_rows)}")
    expected_profiles = config["profiles"]
    for values in dry_rows:
        if len(values) != 8:
            errors.append(f"malformed dry-run row: {values}")
            continue
        horizon = values[0].split("__H", 1)[1].split("__", 1)[0]
        profile = expected_profiles[horizon]
        observed = {
            "learning_rate": float(values[1]),
            "inner_lr": float(values[2]),
            "meta_lr": float(values[3]),
            "warmup_steps": int(values[4]),
            "num_tasks": int(values[5]),
            "meta_inner_steps": int(values[6]),
            "batch_size": int(values[7]),
        }
        if observed != profile:
            errors.append(f"profile mismatch at H{horizon}: {observed}")

    with PUBLISHED.open(encoding="utf-8", newline="") as handle:
        published = list(csv.DictReader(handle))
    keys = {(row["dataset"], int(row["horizon"])) for row in published}
    if len(published) != 24 or len(keys) != 24:
        errors.append("published QDF transcription must contain 24 unique rows")
    expected_datasets = {"ETTm1", "ETTm2", "ETTh1", "ETTh2", "ECL", "Weather"}
    if {row["dataset"] for row in published} != expected_datasets:
        errors.append("published QDF dataset set mismatch")

    if errors:
        raise SystemExit("QDF Solar prelaunch audit failed:\n- " + "\n- ".join(errors))
    print(
        json.dumps(
            {
                "protocol_id": config["protocol_id"],
                "source_commit": config["candidate"]["source_commit"],
                "dry_run_jobs": len(dry_rows),
                "published_rows": len(published),
                "prelaunch_gate": "pass",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

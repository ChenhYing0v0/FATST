#!/usr/bin/env python3
"""Audit the frozen QDF L336 Main I reproduction before remote launch."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "qdf_main_i_seq336_reproduction.json"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Solar", "Exchange")
HORIZONS = (96, 192, 336, 720)


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
    if contract["datasets"] != list(DATASETS) or contract["horizons"] != list(HORIZONS):
        errors.append("dataset or horizon matrix does not match 8x4 contract")
    if contract["seq_len"] != 336 or contract["label_len"] != 48:
        errors.append("lookback contract must be seq_len=336,label_len=48")
    if contract["seed"] != 2023 or contract["expected_jobs"] != 32:
        errors.append("seed/job-count mismatch")
    if not contract["test_access_once_after_training"]:
        errors.append("formal test must be post-training only")

    source_root = ROOT / config["candidate"]["local_source_root"]
    for relative, expected in config["source_hashes"].items():
        if sha256(source_root / relative) != expected:
            errors.append(f"source hash mismatch: {relative}")

    completed = subprocess.run(
        ["bash", str(source_root / "scripts" / "MainI_L336.sh")],
        cwd=ROOT,
        env={**os.environ, "MODE": "dry-run", "SEED": "2023"},
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [line.split("\t") for line in completed.stdout.splitlines() if line]
    if len(rows) != 32:
        errors.append(f"dry-run expected 32 rows, got {len(rows)}")
    observed_keys: set[tuple[str, int]] = set()
    for row in rows:
        if len(row) != 12:
            errors.append(f"malformed dry-run row: {row}")
            continue
        job_id = row[0]
        dataset = job_id.split("__")[1]
        horizon = int(job_id.split("__H")[1].split("__")[0])
        observed_keys.add((dataset, horizon))
        spec = config["dataset_contracts"][dataset]
        observed_dataset = {
            "data": row[1], "channels": int(row[2]), "cycle": int(row[3]), "dropout": float(row[4])
        }
        expected_dataset = {
            "data": spec["data"], "channels": spec["channels"], "cycle": spec["cycle"], "dropout": spec["dropout"]
        }
        if observed_dataset != expected_dataset:
            errors.append(f"dataset contract mismatch: {dataset}")
        observed_profile = [float(row[5]), float(row[6]), float(row[7]), int(row[8]), int(row[9]), int(row[10]), int(row[11])]
        expected_profile = config["profiles"][dataset][str(horizon)]
        if observed_profile != expected_profile:
            errors.append(f"profile mismatch: {dataset} H{horizon}")
    expected_keys = {(dataset, horizon) for dataset in DATASETS for horizon in HORIZONS}
    if observed_keys != expected_keys:
        errors.append(f"dry-run key mismatch: missing={sorted(expected_keys - observed_keys)}")

    roles = {dataset: config["dataset_contracts"][dataset]["role"] for dataset in DATASETS}
    if roles["Solar"] != "source_informed_ecl_profile" or roles["Exchange"] != "source_informed_etth1_profile":
        errors.append("source-informed role disclosure missing")
    if errors:
        raise SystemExit("QDF L336 prelaunch audit failed:\n- " + "\n- ".join(errors))
    print(json.dumps({"protocol_id": config["protocol_id"], "dry_run_jobs": len(rows), "matrix": "8x4", "seq_len": 336, "prelaunch_gate": "pass"}, indent=2))


if __name__ == "__main__":
    main()

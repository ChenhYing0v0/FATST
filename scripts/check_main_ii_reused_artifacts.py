#!/usr/bin/env python3
"""Resolve and hash the 42 non-ISCF reused Main II checkpoint evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Solar")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def exactly_one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"expected one {label}, found {len(paths)}")
    return paths[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--timealign-main-root", type=Path, required=True)
    parser.add_argument("--timealign-reuse-root", type=Path, required=True)
    parser.add_argument("--qdf-root", type=Path, required=True)
    parser.add_argument("--amd-simpletm-root", type=Path, required=True)
    parser.add_argument("--iscf-reuse-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    master_rows = read_csv(args.checkpoint_manifest)
    master = {(row["system"], row["dataset"]): row for row in master_rows}
    iscf = json.loads(args.iscf_reuse_json.read_text(encoding="utf-8"))
    if iscf.get("gate") != "pass" or iscf.get("checkpoint_evaluations") != 7:
        raise RuntimeError("seven-checkpoint ISCF reuse audit is incomplete")

    jobs: list[dict[str, object]] = []
    for dataset in DATASETS:
        timealign_parent = (
            args.timealign_reuse_root
            if dataset in {"ETTm2", "Weather"}
            else args.timealign_main_root
        )
        run_dir = timealign_parent / f"TimeAlign__{dataset}__H720__seed2021"
        checkpoint = run_dir / "checkpoint.pt"
        effective_config = run_dir / "effective_config.json"
        anchor_metrics = run_dir / "metrics_by_target_horizon.csv"
        for path in (checkpoint, effective_config, anchor_metrics):
            if not path.is_file():
                raise FileNotFoundError(path)
        expected = master[("TimeAlign", dataset)]["checkpoint_hashes"]
        actual = sha256(checkpoint)
        if actual != expected:
            raise RuntimeError(f"TimeAlign checkpoint mismatch: {dataset}")
        jobs.append(
            {
                "system": "TimeAlign",
                "dataset": dataset,
                "repeat": 0,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": actual,
                "effective_config": str(effective_config),
                "anchor_metrics": str(anchor_metrics),
            }
        )

        qdf_run = args.qdf_root / "runs" / f"QDF__{dataset}__H720__seed2023"
        qdf_checkpoint = exactly_one(
            list(qdf_run.glob("checkpoints/**/checkpoint.pth")),
            f"QDF {dataset} checkpoint",
        )
        qdf_config = exactly_one(
            list(qdf_run.glob("results/**/config.yaml")), f"QDF {dataset} config"
        )
        qdf_anchor = exactly_one(
            list(qdf_run.glob("results/**/metrics.npy")), f"QDF {dataset} metrics"
        )
        expected = master[("QDF", dataset)]["checkpoint_hashes"]
        actual = sha256(qdf_checkpoint)
        if actual != expected:
            raise RuntimeError(f"QDF checkpoint mismatch: {dataset}")
        jobs.append(
            {
                "system": "QDF",
                "dataset": dataset,
                "repeat": 0,
                "checkpoint": str(qdf_checkpoint),
                "checkpoint_sha256": actual,
                "config_yaml": str(qdf_config),
                "anchor_metrics": str(qdf_anchor),
            }
        )

        amd_run = args.amd_simpletm_root / "runs" / f"AMD__{dataset}"
        amd_rows = [
            row
            for row in read_csv(amd_run / "metrics.csv")
            if int(row["horizon"]) == 720
        ]
        if len(amd_rows) != 1:
            raise RuntimeError(f"AMD H720 row mismatch: {dataset}")
        expected = master[("AMD", dataset)]["checkpoint_hashes"]
        if amd_rows[0]["checkpoint_sha256"] != expected:
            raise RuntimeError(f"AMD manifest mismatch: {dataset}")
        if sha256(Path(amd_rows[0]["checkpoint"])) != expected:
            raise RuntimeError(f"AMD checkpoint mismatch: {dataset}")
        jobs.append(
            {
                "system": "AMD",
                "dataset": dataset,
                "repeat": 0,
                "checkpoint": amd_rows[0]["checkpoint"],
                "checkpoint_sha256": expected,
                "run_dir": str(amd_run),
            }
        )

        simple_run = args.amd_simpletm_root / "runs" / f"SimpleTM__{dataset}"
        simple_rows = sorted(
            (
                row
                for row in read_csv(simple_run / "metrics.csv")
                if int(row["horizon"]) == 720
            ),
            key=lambda row: int(row["repeat"]),
        )
        if len(simple_rows) != 3:
            raise RuntimeError(f"SimpleTM H720 repeats mismatch: {dataset}")
        expected_hashes = master[("SimpleTM", dataset)]["checkpoint_hashes"].split(
            ";"
        )
        actual_hashes = [row["checkpoint_sha256"] for row in simple_rows]
        if actual_hashes != expected_hashes:
            raise RuntimeError(f"SimpleTM manifest mismatch: {dataset}")
        for row in simple_rows:
            if sha256(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
                raise RuntimeError(
                    f"SimpleTM checkpoint mismatch: {dataset}/{row['repeat']}"
                )
            jobs.append(
                {
                    "system": "SimpleTM",
                    "dataset": dataset,
                    "repeat": int(row["repeat"]),
                    "checkpoint": row["checkpoint"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "run_dir": str(simple_run),
                }
            )

    if len(jobs) != 42:
        raise RuntimeError(f"expected 42 reused evaluations, found {len(jobs)}")
    counts = {
        system: sum(job["system"] == system for job in jobs)
        for system in ("TimeAlign", "QDF", "AMD", "SimpleTM")
    }
    if counts != {"TimeAlign": 7, "QDF": 7, "AMD": 7, "SimpleTM": 21}:
        raise RuntimeError(f"reused evaluation count mismatch: {counts}")
    payload = {
        "gate": "pass",
        "new_test_access_planned": 42,
        "iscf_completed_reuse_evaluations": 7,
        "total_reused_checkpoint_evaluations": 49,
        "counts": counts,
        "jobs": jobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "main_ii_reused_artifacts=pass TimeAlign=7 QDF=7 AMD=7 "
        "SimpleTM=21 ISCF_completed=7 total=49"
    )


if __name__ == "__main__":
    main()

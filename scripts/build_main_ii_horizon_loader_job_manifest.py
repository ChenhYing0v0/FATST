#!/usr/bin/env python3
"""Freeze the complete external-baseline Main II horizon-loader test queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Solar")
HORIZONS = (96, 192, 336, 720)
UPSTREAM = ("iTransformer", "PatchTST", "DLinear")
EXPECTED_DROP_LAST = {
    "TimeAlign": False,
    "QDF": True,
    "AMD": False,
    "SimpleTM": True,
    "iTransformer": True,
    "PatchTST": True,
    "DLinear": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reused-manifest", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reused = json.loads(args.reused_manifest.read_text(encoding="utf-8"))
    if reused.get("gate") != "pass" or len(reused.get("jobs", [])) != 42:
        raise RuntimeError("reused H720 checkpoint manifest is incomplete")
    config = json.loads(args.execution_config.read_text(encoding="utf-8"))
    jobs: list[dict[str, object]] = []
    checkpoint_objects: dict[tuple[str, str, int], str] = {}

    for source in reused["jobs"]:
        key = (source["system"], source["dataset"], int(source["repeat"]))
        checkpoint = Path(source["checkpoint"])
        observed = sha256(checkpoint)
        if observed != source["checkpoint_sha256"]:
            raise RuntimeError(f"reused checkpoint hash mismatch: {key}")
        checkpoint_objects[key] = observed
        for horizon in HORIZONS:
            jobs.append(
                {
                    **source,
                    "loader_horizon": horizon,
                    "model_horizon": 720,
                    "expected_drop_last": EXPECTED_DROP_LAST[source["system"]],
                    "evaluator_family": "reused_native",
                }
            )

    for system in UPSTREAM:
        workspace = args.workspace_root / system
        if not (workspace / "fatst_runtime_patch_manifest.json").is_file():
            raise FileNotFoundError(workspace)
        for dataset in DATASETS:
            training = args.training_root / f"{system}__{dataset}"
            artifact_path = training / "artifact_manifest.json"
            command_path = training / "effective_command.json"
            done_path = training / "DONE"
            if not all(path.is_file() for path in (artifact_path, command_path, done_path)):
                raise RuntimeError(f"incomplete H720 training artifact: {system}/{dataset}")
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            checkpoint = Path(artifact["checkpoint"])
            observed = sha256(checkpoint)
            if observed != artifact["checkpoint_sha256"]:
                raise RuntimeError(f"training checkpoint hash mismatch: {system}/{dataset}")
            checkpoint_objects[(system, dataset, 0)] = observed
            for horizon in HORIZONS:
                jobs.append(
                    {
                        "system": system,
                        "dataset": dataset,
                        "repeat": 0,
                        "loader_horizon": horizon,
                        "model_horizon": 720,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": observed,
                        "training_dir": str(training),
                        "workspace": str(workspace),
                        "expected_drop_last": EXPECTED_DROP_LAST[system],
                        "evaluator_family": "runpy_upstream",
                    }
                )

    if len(checkpoint_objects) != 63 or len(set(checkpoint_objects.values())) != 63:
        raise RuntimeError("expected 63 unique external H720 checkpoint objects")
    if len(jobs) != 252:
        raise RuntimeError(f"expected 252 formal evaluations, found {len(jobs)}")
    counts = Counter(str(job["system"]) for job in jobs)
    expected = {
        "TimeAlign": 28,
        "QDF": 28,
        "AMD": 28,
        "SimpleTM": 84,
        "iTransformer": 28,
        "PatchTST": 28,
        "DLinear": 28,
    }
    if dict(counts) != expected:
        raise RuntimeError(f"job count mismatch: {dict(counts)}")
    dataset_priority = {name: index for index, name in enumerate(
        ("ECL", "Weather", "Solar", "ETTm1", "ETTm2", "ETTh1", "ETTh2")
    )}
    jobs.sort(
        key=lambda job: (
            dataset_priority[str(job["dataset"])],
            -int(job["loader_horizon"]),
            str(job["system"]),
            int(job["repeat"]),
        )
    )
    payload = {
        "gate": "pass",
        "protocol": "horizon_specific_loader_H720_checkpoint",
        "external_systems": list(EXPECTED_DROP_LAST),
        "datasets": list(DATASETS),
        "horizons": list(HORIZONS),
        "checkpoint_objects": len(checkpoint_objects),
        "unique_checkpoint_hashes": len(set(checkpoint_objects.values())),
        "formal_evaluations": len(jobs),
        "expected_drop_last": EXPECTED_DROP_LAST,
        "counts": expected,
        "execution_config_sha256": sha256(args.execution_config),
        "reused_manifest_sha256": sha256(args.reused_manifest),
        "jobs": jobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "main_ii_horizon_loader_manifest=pass "
        "checkpoints=63 unique_hashes=63 evaluations=252"
    )


if __name__ == "__main__":
    main()

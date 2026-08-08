#!/usr/bin/env python3
"""Static and command-level gate for Main II Tier A."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_main_ii_h720_training_job import DATASETS, effective_command


BASELINES = ("iTransformer", "PatchTST", "DLinear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--itransformer-workspace", type=Path, required=True)
    parser.add_argument("--patchtst-workspace", type=Path, required=True)
    parser.add_argument("--dlinear-workspace", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    authorization = protocol["authorization"]
    for key in ("local_protocol_patch", "remote_training", "formal_prefix_test"):
        if authorization.get(key) is not True:
            raise RuntimeError(f"authorization missing: {key}")

    workspaces = {
        "iTransformer": args.itransformer_workspace,
        "PatchTST": args.patchtst_workspace,
        "DLinear": args.dlinear_workspace,
    }
    jobs: list[dict[str, object]] = []
    for baseline in BASELINES:
        workspace = workspaces[baseline]
        manifest_path = workspace / "fatst_runtime_patch_manifest.json"
        patch_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if patch_manifest["training_test_access"] != "disabled":
            raise RuntimeError(f"training test hygiene failed: {baseline}")
        exp_relative = (
            "experiments/exp_long_term_forecasting.py"
            if baseline == "iTransformer"
            else "exp/exp_main.py"
        )
        exp_text = (workspace / exp_relative).read_text(encoding="utf-8")
        train_text = exp_text.split("    def train(", 1)[1].split("    def test(", 1)[0]
        if "flag='test'" in train_text or "Test Loss" in train_text:
            raise RuntimeError(f"training-time test access remains: {baseline}")
        if baseline in {"PatchTST", "DLinear"}:
            if "class Dataset_Solar(Dataset):" not in exp_text and not (
                "class Dataset_Solar(Dataset):" in (
                    workspace / "data_provider/data_loader.py"
                ).read_text(encoding="utf-8")
            ):
                raise RuntimeError(f"Solar loader missing: {baseline}")
        for dataset in DATASETS:
            command = effective_command(
                config,
                baseline,
                dataset,
                workspace,
                args.data_root,
                Path("/tmp/main_ii_dry_run") / baseline / dataset,
                "dry-run",
            )
            if command[command.index("--pred_len") + 1] != "720":
                raise RuntimeError(f"non-H720 command: {baseline}/{dataset}")
            if command[command.index("--is_training") + 1] != "1":
                raise RuntimeError(f"training command mismatch: {baseline}/{dataset}")
            jobs.append(
                {
                    "baseline": baseline,
                    "dataset": dataset,
                    "command": command,
                    "role": (
                        "source_informed_not_official"
                        if dataset == "Solar" and baseline != "iTransformer"
                        else "official_released_H720"
                    ),
                }
            )
    if len(jobs) != 21:
        raise RuntimeError(f"expected 21 training jobs, got {len(jobs)}")

    with args.checkpoint_manifest.open(encoding="utf-8", newline="") as handle:
        checkpoint_rows = list(csv.DictReader(handle))
    if len(checkpoint_rows) != 56:
        raise RuntimeError(f"expected 56 checkpoint rows, got {len(checkpoint_rows)}")
    reused = sum(int(row["checkpoint_count"]) for row in checkpoint_rows)
    planned_evaluations = reused + len(jobs)
    if reused != 49 or planned_evaluations != 70:
        raise RuntimeError(
            f"evaluation count mismatch: reused={reused} total={planned_evaluations}"
        )

    report = {
        "gate": "pass",
        "authorization": "Tier_A_B_C_explicit_2026-08-08",
        "training_jobs": len(jobs),
        "checkpoint_manifest_rows": len(checkpoint_rows),
        "reused_checkpoint_objects": reused,
        "formal_checkpoint_evaluations": planned_evaluations,
        "raw_prefix_rows": planned_evaluations * 4,
        "aggregate_cells": 224,
        "training_time_test_access": 0,
        "jobs": jobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "main_ii_prelaunch=pass training_jobs=21 reused=49 "
        "formal_evaluations=70 raw_prefix_rows=280 aggregate_cells=224"
    )


if __name__ == "__main__":
    main()

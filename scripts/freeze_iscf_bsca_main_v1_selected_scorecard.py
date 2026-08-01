#!/usr/bin/env python3
"""Build the frozen 32-cell scorecard for selected main-model profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis" / "iscf_bsca_main_v1_hpo_20260731"
HORIZONS = (96, 192, 336, 720)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "iscf_bsca_main_v1_selected_profiles.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scorecards = []
    for relative in (
        "test_audit_result/all_trial_scorecard.csv",
        "h3a_test_result/all_trial_scorecard.csv",
        "h3b_test_result/all_trial_scorecard.csv",
    ):
        scorecards.extend(read_csv(BASE / relative))
    manifests = []
    for name in (
        "combined_checkpoint_manifest.csv",
        "h3a_checkpoint_manifest.csv",
        "h3b_checkpoint_manifest.csv",
    ):
        manifests.extend(read_csv(BASE / name))
    checkpoint_by_trial = {
        row["trial_id"]: row["checkpoint_sha256_before_test"]
        for row in manifests
    }
    metric_by_key = {
        (row["trial_id"], int(row["horizon"])): row
        for row in scorecards
    }
    output: list[dict[str, Any]] = []
    for dataset, profile in config["profiles"].items():
        trial_id = profile["trial_id"]
        if checkpoint_by_trial.get(trial_id) != profile["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {trial_id}")
        rows = []
        for horizon in HORIZONS:
            source = metric_by_key.get((trial_id, horizon))
            if source is None or source["dataset"] != dataset:
                raise ValueError(f"missing scorecard cell: {dataset}/{horizon}")
            mse = float(source["test_mse"])
            mae = float(source["test_mae"])
            if not math.isfinite(mse) or not math.isfinite(mae):
                raise ValueError(f"non-finite scorecard cell: {dataset}/{horizon}")
            rows.append((mse, mae))
            output.append(
                {
                    "dataset": dataset,
                    "trial_id": trial_id,
                    "phase": profile["phase"],
                    "seed": config["seed"],
                    "horizon": horizon,
                    "test_mse": mse,
                    "test_mae": mae,
                    "checkpoint_sha256": profile["checkpoint_sha256"],
                    "test_tuned": True,
                    "test_informed": dataset in {"ECL", "Solar"},
                }
            )
        mean_mse = sum(pair[0] for pair in rows) / 4
        mean_mae = sum(pair[1] for pair in rows) / 4
        if not math.isclose(
            mean_mse,
            float(profile["test_mean_mse_4h"]),
            abs_tol=1e-12,
        ):
            raise ValueError(f"mean MSE mismatch: {dataset}")
        if not math.isclose(
            mean_mae,
            float(profile["test_mean_mae_4h"]),
            abs_tol=1e-12,
        ):
            raise ValueError(f"mean MAE mismatch: {dataset}")
    if len(output) != 32:
        raise ValueError(f"expected 32 selected cells, found {len(output)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(output[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output)
    print(
        json.dumps(
            {
                "datasets": len(config["profiles"]),
                "horizons": list(HORIZONS),
                "cells": len(output),
                "checkpoint_hashes": len(checkpoint_by_trial),
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

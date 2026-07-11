#!/usr/bin/env python3
"""Evaluate SC0 fixed-20 best/last checkpoints on test without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from data_provider.data_factory import data_provider  # noqa: E402
from models.TimeAlign import Model  # noqa: E402
from train_repo import evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["Weather", "ETTm1", "ETTh2"], required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def arm_name(run_dir: Path) -> str:
    name = run_dir.parents[2].name
    return name.removeprefix("SC0_").removesuffix("_validation_only")


def evaluate_run(run_dir: Path, device: str) -> list[dict[str, Any]]:
    effective = json.loads((run_dir / "effective_config.json").read_text(encoding="utf-8"))
    adapter = effective["adapter"]
    official_payload = effective["official_args"]
    official_payload["device"] = torch.device(device)
    official_payload["use_gpu"] = device.startswith("cuda")
    official_args = SimpleNamespace(**official_payload)
    horizons = [int(value) for value in adapter["evaluation_horizons"]]
    _test_data, test_loader = data_provider(official_args, "test")
    model = Model(official_args).float().to(official_args.device)

    test_metrics: dict[str, dict[int, dict[str, Any]]] = {}
    for selector, checkpoint in (
        ("best_val", run_dir / "checkpoint_best_val.pt"),
        ("last", run_dir / "checkpoint_last.pt"),
    ):
        model.load_state_dict(load_state(checkpoint), strict=True)
        rows, _segments, _preds, _trues = evaluate(
            model,
            test_loader,
            official_args,
            horizons,
            max_batches=0,
            is_training_flag=bool(adapter["official_test_mode"]),
        )
        test_metrics[selector] = {int(row["target_horizon"]): row for row in rows}

    validation = {
        selector: {
            int(row["target_horizon"]): row
            for row in read_csv(run_dir / f"metrics_{selector}_by_target_horizon.csv")
        }
        for selector in ("best_val", "last")
    }
    results = []
    for horizon in horizons:
        val_best = validation["best_val"][horizon]
        val_last = validation["last"][horizon]
        test_best = test_metrics["best_val"][horizon]
        test_last = test_metrics["last"][horizon]
        results.append(
            {
                "dataset": adapter["dataset"],
                "arm": arm_name(run_dir),
                "seed": adapter["seed"],
                "target_horizon": horizon,
                "validation_best_mse": float(val_best["mse"]),
                "validation_last_mse": float(val_last["mse"]),
                "validation_last_vs_best_mse": float(val_last["mse"])
                / float(val_best["mse"])
                - 1.0,
                "test_best_mse": float(test_best["mse"]),
                "test_last_mse": float(test_last["mse"]),
                "test_last_vs_best_mse": float(test_last["mse"])
                / float(test_best["mse"])
                - 1.0,
                "test_best_mae": float(test_best["mae"]),
                "test_last_mae": float(test_last["mae"]),
                "test_last_vs_best_mae": float(test_last["mae"])
                / float(test_best["mae"])
                - 1.0,
                "evaluation_split": "test",
                "selection_role": "diagnostic_only_after_profile_freeze",
            }
        )
    return results


def main() -> None:
    args = parse_args()
    paths = sorted(
        args.raw_root.glob(
            f"SC0_*/{args.dataset}/h720_full/seed2021"
        )
    )
    if len(paths) != 3:
        raise RuntimeError(f"expected 3 SC0 runs for {args.dataset}, found {len(paths)}")
    rows = []
    for path in paths:
        rows.extend(evaluate_run(path, args.device))
    output = args.output_dir / f"{args.dataset}_checkpoint_test_comparisons.csv"
    write_csv(output, rows)
    print(f"stage_c_sc0_checkpoint_test_done dataset={args.dataset} rows={len(rows)} output={output}")


if __name__ == "__main__":
    main()

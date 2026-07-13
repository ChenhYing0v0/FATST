#!/usr/bin/env python3
"""Evaluate frozen A6-LBF natural-profile checkpoints on the test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--phase-b-root", type=Path, required=True)
    parser.add_argument("--phase-c-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["Weather", "ETTm1", "ETTh2"], required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


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


def run_dir(args: argparse.Namespace, profile: dict[str, Any], seed: int) -> Path:
    name = profile["profile"]
    if seed == 2021 and name.endswith("_medium"):
        run_name = (
            f"SC0DAP_R2A_r2a_p{profile['patch_num']}_"
            f"d{profile['d_model']}_ff{profile['d_ff']}"
        )
        root = args.phase_a_root
    elif seed == 2021:
        run_name = f"SC0DAP_R2B_{name}"
        root = args.phase_b_root
    else:
        run_name = f"SC0DAP_R2C_{name}"
        root = args.phase_c_root
    return root / run_name / args.dataset / "h720_full" / f"seed{seed}"


def evaluate_seed(
    args: argparse.Namespace,
    contract: dict[str, Any],
    contract_hash: str,
    seed: int,
) -> list[dict[str, Any]]:
    profile = contract["dataset_profiles"][args.dataset]
    directory = run_dir(args, profile, seed)
    effective = json.loads((directory / "effective_config.json").read_text(encoding="utf-8"))
    adapter = effective["adapter"]
    official_payload = effective["official_args"]
    expected = {
        "patch_num": int(profile["patch_num"]),
        "d_model": int(profile["d_model"]),
        "d_ff": int(profile["d_ff"]),
    }
    observed = {key: int(official_payload[key]) for key in expected}
    if observed != expected or int(adapter["seed"]) != seed:
        raise ValueError(f"frozen contract mismatch at {directory}: {observed} != {expected}")
    official_payload["device"] = torch.device(args.device)
    official_payload["use_gpu"] = args.device.startswith("cuda")
    official_args = SimpleNamespace(**official_payload)
    _test_data, test_loader = data_provider(official_args, "test")
    model = Model(official_args).float().to(official_args.device)
    model.load_state_dict(load_state(directory / "checkpoint.pt"), strict=True)
    horizons = [int(value) for value in contract["global_fields"]["dense_validation_horizons"]]
    metrics, _segments, _preds, _trues = evaluate(
        model,
        test_loader,
        official_args,
        horizons,
        max_batches=0,
        is_training_flag=bool(adapter["official_test_mode"]),
    )
    return [
        {
            "dataset": args.dataset,
            "profile": profile["profile"],
            "contract_hash": contract_hash,
            "seed": seed,
            "target_horizon": int(row["target_horizon"]),
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
            "evaluation_split": "test",
            "checkpoint_selector": "frozen_best_validation",
            "run_dir": str(directory),
        }
        for row in metrics
    ]


def main() -> None:
    args = parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    contract_hash = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    for seed in contract["global_fields"]["default_seeds"]:
        rows.extend(evaluate_seed(args, contract, contract_hash, int(seed)))
    output = args.output_dir / f"{args.dataset}_natural_baseline_test_metrics.csv"
    write_csv(output, rows)
    print(f"natural_baseline_test_done dataset={args.dataset} rows={len(rows)} output={output}")


if __name__ == "__main__":
    main()

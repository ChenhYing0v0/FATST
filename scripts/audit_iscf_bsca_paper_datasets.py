#!/usr/bin/env python3
"""Audit paper-facing dataset identity and train/validation construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from data_provider.data_factory import data_provider  # noqa: E402
from train_repo import OFFICIAL_PRESETS, resolve_dataset_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "iscf_bsca_main_v1_hpo.json",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ECL", "Solar", "Exchange"],
    )
    parser.add_argument("--construct-loaders", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_file(dataset_root: Path, dataset: str) -> tuple[Path, Any]:
    preset = OFFICIAL_PRESETS[dataset][720]
    root_path = resolve_dataset_root(dataset_root, preset)
    return root_path / preset.data_path, preset


def split_rows(dataset: str, rows: int, seq_len: int) -> dict[str, Any]:
    if dataset in {"ETTh1", "ETTh2"}:
        boundaries = {
            "train": [0, 12 * 30 * 24],
            "val": [
                12 * 30 * 24 - seq_len,
                12 * 30 * 24 + 4 * 30 * 24,
            ],
            "test": [
                12 * 30 * 24 + 4 * 30 * 24 - seq_len,
                12 * 30 * 24 + 8 * 30 * 24,
            ],
        }
    elif dataset in {"ETTm1", "ETTm2"}:
        boundaries = {
            "train": [0, 12 * 30 * 24 * 4],
            "val": [
                12 * 30 * 24 * 4 - seq_len,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            ],
            "test": [
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            ],
        }
    else:
        train_rows = int(rows * 0.7)
        test_rows = int(rows * 0.2)
        validation_rows = rows - train_rows - test_rows
        boundaries = {
            "train": [0, train_rows],
            "val": [train_rows - seq_len, train_rows + validation_rows],
            "test": [rows - test_rows - seq_len, rows],
        }
    return {
        "boundaries": boundaries,
        "history_overlap_only": True,
        "scaler_fit_rows": boundaries["train"],
    }


def cadence_summary(dates: pd.Series) -> dict[str, Any]:
    parsed = pd.to_datetime(dates, errors="coerce")
    deltas = parsed.diff().dropna()
    counts = deltas.value_counts()
    return {
        "parse_failures": int(parsed.isna().sum()),
        "monotonic_increasing": bool(parsed.is_monotonic_increasing),
        "duplicate_count": int(parsed.duplicated().sum()),
        "most_common_delta": str(counts.index[0]) if len(counts) else None,
        "most_common_delta_count": int(counts.iloc[0]) if len(counts) else 0,
        "unique_delta_count": int(len(counts)),
    }


def audit_csv(
    path: Path,
    expected_channels: int,
) -> tuple[int, int, dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {"date", "OT"}
    missing = sorted(required.difference(frame.columns))
    value_columns = [column for column in frame.columns if column != "date"]
    numeric = frame[value_columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    metadata = {
        "columns": list(frame.columns),
        "required_columns_missing": missing,
        "channel_count": len(value_columns),
        "expected_channel_count": expected_channels,
        "channel_count_match": len(value_columns) == expected_channels,
        "nonfinite_count": int((~finite).sum()),
        "constant_channel_count": int(
            sum(
                math.isclose(float(numeric[column].std(ddof=0)), 0.0)
                for column in value_columns
                if numeric[column].notna().all()
            )
        ),
        "cadence": cadence_summary(frame["date"])
        if "date" in frame.columns
        else None,
    }
    return len(frame), len(value_columns), metadata


def audit_solar(
    path: Path,
    expected_channels: int,
) -> tuple[int, int, dict[str, Any]]:
    values = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Solar matrix must be two-dimensional: {values.shape}")
    metadata = {
        "channel_count": int(values.shape[1]),
        "expected_channel_count": expected_channels,
        "channel_count_match": int(values.shape[1]) == expected_channels,
        "nonfinite_count": int((~np.isfinite(values)).sum()),
        "constant_channel_count": int(
            np.isclose(values.std(axis=0), 0.0).sum()
        ),
        "cadence": "not_observable_no_timestamp",
    }
    return int(values.shape[0]), int(values.shape[1]), metadata


def loader_args(
    dataset_root: Path,
    dataset: str,
    batch_size: int = 2,
) -> SimpleNamespace:
    preset = OFFICIAL_PRESETS[dataset][720]
    root_path = resolve_dataset_root(dataset_root, preset)
    return SimpleNamespace(
        data=preset.data,
        root_path=str(root_path),
        data_path=preset.data_path,
        seq_len=720,
        label_len=48,
        pred_len=720,
        features="M",
        target="OT",
        timeenc=1,
        embed="timeF",
        freq=preset.freq,
        seasonal_patterns="Monthly",
        batch_size=batch_size,
        num_workers=0,
        augmentation_ratio=0,
    )


def audit_loader(dataset_root: Path, dataset: str) -> dict[str, Any]:
    args = loader_args(dataset_root, dataset)
    outputs: dict[str, Any] = {}
    for split in ("train", "val"):
        data_set, loader = data_provider(args, split)
        batch = next(iter(loader))
        shapes = [list(tensor.shape) for tensor in batch]
        outputs[split] = {
            "rows": len(data_set),
            "batch_shapes": shapes,
            "finite": all(bool(np.isfinite(tensor.numpy()).all()) for tensor in batch),
        }
    return outputs


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    contracts = config["dataset_contracts"]
    records = []
    for dataset in args.datasets:
        if dataset not in contracts:
            raise KeyError(f"dataset is not in config: {dataset}")
        path, preset = resolve_file(args.dataset_root, dataset)
        contract = contracts[dataset]
        if preset.data == "Solar":
            rows, channels, metadata = audit_solar(path, contract["channels"])
        else:
            rows, channels, metadata = audit_csv(path, contract["channels"])
        record = {
            "dataset": dataset,
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "rows": rows,
            "channels": channels,
            "expected_rows": contract.get("expected_rows"),
            "expected_rows_match": (
                rows == contract["expected_rows"]
                if "expected_rows" in contract
                else None
            ),
            "frequency_contract": contract["frequency"],
            "split": split_rows(dataset, rows, 720),
            "raw_audit": metadata,
        }
        if args.construct_loaders:
            record["loader_audit"] = audit_loader(args.dataset_root, dataset)
        failures = [
            bool(metadata["channel_count_match"]),
            metadata["nonfinite_count"] == 0,
            not metadata.get("required_columns_missing"),
        ]
        if args.construct_loaders:
            failures.extend(
                entry["finite"] for entry in record["loader_audit"].values()
            )
        record["pass"] = all(failures)
        records.append(record)
    result = {
        "protocol_id": config["protocol_id"],
        "datasets": records,
        "test_loader_constructed": False,
        "overall_pass": all(record["pass"] for record in records),
    }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not result["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

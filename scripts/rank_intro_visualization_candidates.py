#!/usr/bin/env python3
"""Rank validation-only Introduction figure candidates across datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
SHARING_KEYS = (
    "supported_winner_count",
    "distinct_winner_count",
    "winner_entropy",
    "qualified_crossing_pair_count",
    "mean_winner_margin",
    "sample_oracle_headroom",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def sharing_key(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float(row[key]) for key in SHARING_KEYS)


def main() -> None:
    args = parse_args()
    prefix_rows = []
    sharing_rows = []
    for dataset in DATASETS:
        prefix = load_json(
            args.analysis_root / dataset / "prefix" / "summary.json"
        )
        if prefix["selection_mode"] != "maximum":
            raise RuntimeError(
                f"{dataset} prefix candidate was not selected by maximum"
            )
        prefix_rows.append(
            {
                "dataset": dataset,
                "selected_origin_index": prefix["selected_origin_index"],
                "selected_channel_index": prefix["selected_channel_index"],
                "selected_joint_score": prefix["selected_joint_score"],
                "macro_nchpd_l1": prefix["macro_nchpd_l1"],
                "macro_rda": prefix["macro_rda"],
                "searched_origin_channel_cells": (
                    prefix["searched_origin_channel_cells"]
                ),
            }
        )

        sharing = load_json(
            args.analysis_root / dataset / "sharing_sample" / "summary.json"
        )
        selected = sharing["selected"]
        sharing_rows.append(
            {
                "dataset": dataset,
                "selected_origin_index": selected["origin_index"],
                **{key: selected[key] for key in SHARING_KEYS},
                "best_fixed_scale": selected["best_fixed_scale"],
            }
        )

    prefix_rows.sort(
        key=lambda row: float(row["selected_joint_score"]),
        reverse=True,
    )
    sharing_rows.sort(key=sharing_key, reverse=True)
    for rank, row in enumerate(prefix_rows, start=1):
        row["rank"] = rank
    for rank, row in enumerate(sharing_rows, start=1):
        row["rank"] = rank

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "prefix_candidate_ranking.csv", prefix_rows)
    write_csv(args.output_dir / "sharing_candidate_ranking.csv", sharing_rows)
    summary = {
        "split": "validation",
        "test_accessed": False,
        "datasets": list(DATASETS),
        "prefix_ranking_policy": "descending selected_joint_score",
        "sharing_ranking_policy": {
            "type": "lexicographic descending",
            "keys": list(SHARING_KEYS),
        },
        "selected_prefix_dataset": prefix_rows[0]["dataset"],
        "selected_sharing_dataset": sharing_rows[0]["dataset"],
        "claim_role": "exploratory_visualization_only",
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

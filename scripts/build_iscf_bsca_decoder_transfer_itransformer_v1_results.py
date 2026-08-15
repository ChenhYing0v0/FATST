#!/usr/bin/env python3
"""Audit iTransformer transfer formal artifacts and build its result block."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


HORIZONS = (96, 192, 336, 720)
ARM_ORDER = (
    "itransformer_original",
    "itransformer_iscf",
    "itransformer_iscf_bsca",
)
ARM_LABELS = {
    "itransformer_original": "Original Decoder",
    "itransformer_iscf": "+ISCF",
    "itransformer_iscf_bsca": "+ISCF-BSCA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/iscf_bsca_decoder_transfer_itransformer_v1_formal.json"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    cells: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        groups[tuple(cell[key] for key in keys)].append(cell)
    return [
        {
            **dict(zip(keys, key)),
            "mean_mse": sum(row["mse"] for row in rows) / len(rows),
            "mean_mae": sum(row["mae"] for row in rows) / len(rows),
            "cell_count": len(rows),
        }
        for key, rows in sorted(groups.items())
    ]


def build_table(
    datasets: list[str],
    dataset_means: list[dict[str, Any]],
    overall: list[dict[str, Any]],
) -> str:
    dataset_lookup = {
        (row["arm"], row["dataset"]): row for row in dataset_means
    }
    overall_lookup = {row["arm"]: row for row in overall}
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Decoder transfer on the iTransformer-style carrier. "
        "Each dataset entry averages the four standard horizons. Lower is better.}",
        "\\label{tab:decoder-transfer-itransformer}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "cc" * (len(datasets) + 1) + "}",
        "\\toprule",
        "Decoder & "
        + " & ".join(
            f"\\multicolumn{{2}}{{c}}{{{dataset}}}" for dataset in datasets
        )
        + " & \\multicolumn{2}{c}{Avg.} \\\\",
        " & " + " & ".join(["MSE & MAE"] * (len(datasets) + 1)) + " \\\\",
        "\\midrule",
    ]
    for arm in ARM_ORDER:
        values = []
        for dataset in datasets:
            row = dataset_lookup[(arm, dataset)]
            values.extend((f'{row["mean_mse"]:.3f}', f'{row["mean_mae"]:.3f}'))
        row = overall_lookup[arm]
        values.extend((f'{row["mean_mse"]:.3f}', f'{row["mean_mae"]:.3f}'))
        lines.append(f"{ARM_LABELS[arm]} & " + " & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    expected_manifest_hash = config["artifact_contract"][
        "training_manifest_sha256"
    ]
    if sha256(args.manifest) != expected_manifest_hash:
        raise RuntimeError("training manifest hash mismatch")
    manifest_rows = read_csv(args.manifest)
    if len(manifest_rows) != 15:
        raise RuntimeError(f"manifest incomplete: {len(manifest_rows)}/15")
    if len({row["checkpoint_sha256"] for row in manifest_rows}) != 15:
        raise RuntimeError("checkpoint uniqueness mismatch")

    cells: list[dict[str, Any]] = []
    test_dates: set[str] = set()
    for manifest in manifest_rows:
        checkpoint = Path(manifest["checkpoint"])
        if sha256(checkpoint) != manifest["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint mutated: {checkpoint}")
        artifact = (
            args.output_root
            / "formal_test"
            / manifest["arm"]
            / manifest["dataset"]
            / "seed2021"
        )
        invariants = load_json(artifact / "test_audit_invariants.json")
        if not (
            invariants.get("pass") is True
            and invariants.get("evaluation_split") == "test"
            and invariants.get("test_access_authorized") is True
            and invariants.get("checkpoint_sha256")
            == manifest["checkpoint_sha256"]
            and invariants.get("checkpoint_retrained") is False
        ):
            raise RuntimeError(f"formal invariant failed: {artifact}")
        test_dates.add(invariants["test_access_date"])
        dense_rows = read_csv(
            artifact / "test_audit_metrics_by_target_horizon.csv"
        )
        if len(dense_rows) != 720:
            raise RuntimeError(f"dense test rows incomplete: {artifact}")
        by_horizon = {int(row["target_horizon"]): row for row in dense_rows}
        for horizon in HORIZONS:
            row = by_horizon[horizon]
            mse, mae = float(row["mse"]), float(row["mae"])
            if not math.isfinite(mse) or not math.isfinite(mae):
                raise RuntimeError(f"non-finite formal metric: {artifact} H{horizon}")
            cells.append(
                {
                    "backbone": "itransformer_style",
                    "arm": manifest["arm"],
                    "dataset": manifest["dataset"],
                    "horizon": horizon,
                    "mse": mse,
                    "mae": mae,
                    "seed": int(manifest["seed"]),
                    "checkpoint_sha256": manifest["checkpoint_sha256"],
                    "checkpoint_selector": manifest["checkpoint_selector"],
                    "test_access_date": invariants["test_access_date"],
                    "test_role": config["authorization"]["test_role"],
                    "test_tuned": False,
                    "checkpoint_retrained": False,
                }
            )
    if len(cells) != 60:
        raise RuntimeError(f"formal matrix incomplete: {len(cells)}/60")

    dataset_means = aggregate(cells, ("arm", "dataset"))
    overall = aggregate(cells, ("arm",))
    overall_lookup = {row["arm"]: row for row in overall}
    dataset_lookup = {
        (row["arm"], row["dataset"]): row for row in dataset_means
    }
    original = overall_lookup["itransformer_original"]
    iscf = overall_lookup["itransformer_iscf"]
    bsca = overall_lookup["itransformer_iscf_bsca"]

    def gain(base: dict[str, Any], candidate: dict[str, Any], metric: str) -> float:
        return 100.0 * (
            base[f"mean_{metric}"] - candidate[f"mean_{metric}"]
        ) / base[f"mean_{metric}"]

    bsca_original_dataset_wins = sum(
        dataset_lookup[("itransformer_iscf_bsca", dataset)]["mean_mse"]
        < dataset_lookup[("itransformer_original", dataset)]["mean_mse"]
        for dataset in config["datasets"]
    )
    bsca_iscf_dataset_wins = sum(
        dataset_lookup[("itransformer_iscf_bsca", dataset)]["mean_mse"]
        < dataset_lookup[("itransformer_iscf", dataset)]["mean_mse"]
        for dataset in config["datasets"]
    )
    bsca_original_cell_wins = sum(
        row["mse"]
        < next(
            other["mse"]
            for other in cells
            if other["arm"] == "itransformer_original"
            and other["dataset"] == row["dataset"]
            and other["horizon"] == row["horizon"]
        )
        for row in cells
        if row["arm"] == "itransformer_iscf_bsca"
    )
    bsca_iscf_cell_wins = sum(
        row["mse"]
        < next(
            other["mse"]
            for other in cells
            if other["arm"] == "itransformer_iscf"
            and other["dataset"] == row["dataset"]
            and other["horizon"] == row["horizon"]
        )
        for row in cells
        if row["arm"] == "itransformer_iscf_bsca"
    )
    mse_gain = gain(original, bsca, "mse")
    mae_gain = gain(original, bsca, "mae")
    gate_pass = (
        mse_gain
        > config["gates"][
            "bsca_vs_original_macro_mse_gain_percent_min_exclusive"
        ]
        and mae_gain
        > config["gates"][
            "bsca_vs_original_macro_mae_gain_percent_min_exclusive"
        ]
        and bsca_original_dataset_wins
        >= config["gates"]["bsca_vs_original_dataset_mse_wins_min"]
    )
    summary = {
        "pass": gate_pass,
        "candidate_version": config["candidate_version"],
        "formal_checkpoint_jobs": 15,
        "formal_cells": 60,
        "unique_checkpoint_hashes": 15,
        "test_access_dates": sorted(test_dates),
        "bsca_vs_original_macro_mse_gain_percent": mse_gain,
        "bsca_vs_original_macro_mae_gain_percent": mae_gain,
        "bsca_vs_original_dataset_mse_wins": bsca_original_dataset_wins,
        "bsca_vs_original_mse_cell_wins": bsca_original_cell_wins,
        "bsca_vs_iscf_macro_mse_gain_percent": gain(iscf, bsca, "mse"),
        "bsca_vs_iscf_macro_mae_gain_percent": gain(iscf, bsca, "mae"),
        "bsca_vs_iscf_dataset_mse_wins": bsca_iscf_dataset_wins,
        "bsca_vs_iscf_mse_cell_wins": bsca_iscf_cell_wins,
        "checkpoint_nonmutation": True,
        "table_mutation_authorized": False,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.results_dir / "itransformer_transfer_60_cells.csv", cells)
    write_csv(
        args.results_dir / "itransformer_transfer_dataset_means.csv",
        dataset_means,
    )
    write_csv(args.results_dir / "itransformer_transfer_overall.csv", overall)
    (args.results_dir / "itransformer_transfer_result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.results_dir / "table_iscf_bsca_decoder_transfer_itransformer.tex").write_text(
        build_table(config["datasets"], dataset_means, overall)
    )
    print(
        "itransformer_transfer_results=pass "
        f"cells=60 gate_pass={str(gate_pass).lower()}"
    )


if __name__ == "__main__":
    main()

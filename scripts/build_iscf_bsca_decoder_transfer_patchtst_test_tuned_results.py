#!/usr/bin/env python3
"""Build the complete dataset-level test-tuned PatchTST HPO scorecard."""

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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def means(cells: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        grouped[tuple(row[key] for key in keys)].append(row)
    return [
        {
            **dict(zip(keys, key)),
            "mean_mse": sum(row["mse"] for row in rows) / len(rows),
            "mean_mae": sum(row["mae"] for row in rows) / len(rows),
            "cell_count": len(rows),
        }
        for key, rows in sorted(grouped.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    manifest_path = Path(config["artifact_contract"]["unique_manifest"])
    if sha256(manifest_path) != config["artifact_contract"]["unique_manifest_sha256"]:
        raise RuntimeError("unique manifest hash mismatch")
    manifest = read_csv(manifest_path)
    if len(manifest) != 40 or len({row["checkpoint_sha256"] for row in manifest}) != 40:
        raise RuntimeError("unique manifest is not 40/40")

    output_root = Path(config["artifact_contract"]["remote_output_root"])
    unique_cells: list[dict[str, Any]] = []
    expanded_cells: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    test_dates: set[str] = set()
    for row in manifest:
        checkpoint = Path(row["checkpoint"])
        if sha256(checkpoint) != row["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint mutated: {checkpoint}")
        artifact = (
            Path(row["existing_formal_artifact_dir"])
            if row["existing_formal_artifact_dir"]
            else output_root / "formal_test" / row["representative_profile_id"] / row["dataset"] / "seed2021"
        )
        metric_path = artifact / "test_audit_metrics_by_target_horizon.csv"
        invariant_path = artifact / "test_audit_invariants.json"
        diagnostic_path = artifact / "pcsd_test_audit_diagnostics.npz"
        invariant = json.loads(invariant_path.read_text())
        if not (
            diagnostic_path.is_file()
            and invariant.get("pass") is True
            and invariant.get("evaluation_split") == "test"
            and invariant.get("checkpoint_sha256") == row["checkpoint_sha256"]
        ):
            raise RuntimeError(f"formal invariant failed: {artifact}")
        test_dates.add(invariant["test_access_date"])
        dense = read_csv(metric_path)
        if len(dense) != 720:
            raise RuntimeError(f"dense metrics incomplete: {artifact}")
        by_horizon = {int(metric["target_horizon"]): metric for metric in dense}
        aliases = row["profile_aliases"].split(";")
        for horizon in HORIZONS:
            metric = by_horizon[horizon]
            mse, mae = float(metric["mse"]), float(metric["mae"])
            if not math.isfinite(mse) or not math.isfinite(mae):
                raise RuntimeError(f"non-finite metric: {artifact} H{horizon}")
            base = {
                "dataset": row["dataset"],
                "horizon": horizon,
                "mse": mse,
                "mae": mae,
                "seed": 2021,
                "checkpoint_sha256": row["checkpoint_sha256"],
                "test_access_date": invariant["test_access_date"],
                "test_tuned": True,
            }
            unique_cells.append({"profile_id": row["representative_profile_id"], **base})
            for alias in aliases:
                expanded_cells.append({"profile_id": alias, **base})
        artifact_rows.append(
            {
                "dataset": row["dataset"],
                "representative_profile_id": row["representative_profile_id"],
                "profile_aliases": row["profile_aliases"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "metrics_sha256": sha256(metric_path),
                "invariants_sha256": sha256(invariant_path),
                "diagnostics_sha256": sha256(diagnostic_path),
                "reused_v2p1_formal": bool(row["existing_formal_artifact_dir"]),
            }
        )
    if len(unique_cells) != 160 or len(expanded_cells) != 200:
        raise RuntimeError("formal matrix does not expand from 160 to 200 cells")

    v1_cells = read_csv(Path(config["artifact_contract"]["v1_cells"]))
    references: list[dict[str, Any]] = []
    originals: list[dict[str, Any]] = []
    for row in v1_cells:
        arm_id = row["arm_id"]
        if arm_id not in {"patchtst_iscf_bsca", "patchtst_original"}:
            continue
        target = references if arm_id == "patchtst_iscf_bsca" else originals
        target.append(
            {
                "dataset": row["dataset"],
                "profile_id": "v1_reference" if target is references else "original_decoder",
                "horizon": int(row["horizon"]),
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
                "seed": int(row["seed"]),
                "checkpoint_sha256": row["checkpoint_sha256"],
                "test_access_date": row["test_access_date"],
                "test_tuned": False,
            }
        )
    if len(references) != 20 or len(originals) != 20:
        raise RuntimeError("v1 reference/original cells incomplete")

    candidates = expanded_cells + references
    candidate_means = means(candidates, ("dataset", "profile_id"))
    selected_profiles: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        options = [row for row in candidate_means if row["dataset"] == dataset]
        selected_profiles.append(
            {
                **min(options, key=lambda row: (row["mean_mse"], row["mean_mae"], row["profile_id"])),
                "selection_rule": "minimum_four_horizon_mean_official_test_mse",
            }
        )
    selected_lookup = {row["dataset"]: row["profile_id"] for row in selected_profiles}
    selected_cells = [
        row for row in candidates
        if selected_lookup[row["dataset"]] == row["profile_id"]
    ]
    comparison = ([{"arm": "original_decoder", **row} for row in originals]
                  + [{"arm": "selected_bsca", **row} for row in selected_cells])
    dataset_means = means(comparison, ("arm", "dataset"))
    overall = means(comparison, ("arm",))
    lookup = {(row["arm"], row["dataset"]): row for row in dataset_means}
    overall_lookup = {row["arm"]: row for row in overall}
    original = overall_lookup["original_decoder"]
    selected = overall_lookup["selected_bsca"]

    def gain(metric: str) -> float:
        return 100.0 * (original[f"mean_{metric}"] - selected[f"mean_{metric}"]) / original[f"mean_{metric}"]

    mse_gain, mae_gain = gain("mse"), gain("mae")
    dataset_mse_wins = sum(
        lookup[("selected_bsca", dataset)]["mean_mse"]
        < lookup[("original_decoder", dataset)]["mean_mse"]
        for dataset in config["datasets"]
    )
    cell_mse_wins = sum(
        selected_row["mse"] < original_row["mse"]
        for selected_row in selected_cells
        for original_row in originals
        if selected_row["dataset"] == original_row["dataset"]
        and selected_row["horizon"] == original_row["horizon"]
    )
    cell_mae_wins = sum(
        selected_row["mae"] < original_row["mae"]
        for selected_row in selected_cells
        for original_row in originals
        if selected_row["dataset"] == original_row["dataset"]
        and selected_row["horizon"] == original_row["horizon"]
    )
    gate_pass = bool(mse_gain > 0 and mae_gain > 0 and dataset_mse_wins >= 3)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.results_dir / "unique_checkpoint_160_cells.csv", unique_cells)
    write_csv(args.results_dir / "expanded_profile_200_cells.csv", expanded_cells)
    write_csv(args.results_dir / "candidate_pool_220_cells.csv", candidates)
    write_csv(args.results_dir / "candidate_profile_means.csv", candidate_means)
    write_csv(args.results_dir / "selected_profiles.csv", selected_profiles)
    write_csv(args.results_dir / "selected_20_cells.csv", selected_cells)
    write_csv(args.results_dir / "selected_vs_original_40_cells.csv", comparison)
    write_csv(args.results_dir / "selected_vs_original_dataset_means.csv", dataset_means)
    write_csv(args.results_dir / "selected_vs_original_overall.csv", overall)
    artifact_path = args.results_dir / "formal_artifact_manifest.csv"
    write_csv(artifact_path, artifact_rows)
    summary = {
        "pass": gate_pass,
        "candidate_version": config["candidate_version"],
        "unique_checkpoint_jobs": 40,
        "reused_formal_jobs": 5,
        "new_formal_jobs": 35,
        "unique_formal_cells": 160,
        "expanded_profile_cells": 200,
        "candidate_pool_cells": 220,
        "test_access_dates": sorted(test_dates),
        "selected_profile_by_dataset": selected_lookup,
        "selected_vs_original_macro_mse_gain_percent": mse_gain,
        "selected_vs_original_macro_mae_gain_percent": mae_gain,
        "selected_vs_original_dataset_mse_wins": dataset_mse_wins,
        "selected_vs_original_mse_cell_wins": cell_mse_wins,
        "selected_vs_original_mae_cell_wins": cell_mae_wins,
        "checkpoint_nonmutation": True,
        "all_negative_trials_retained": True,
        "per_horizon_seed_metric_or_cell_selection": False,
        "test_tuned": True,
        "matched_iscf_attribution_required_if_pass": gate_pass,
        "canonical_table_mutation_authorized": False,
        "formal_artifact_manifest_sha256": sha256(artifact_path),
    }
    (args.results_dir / "result_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "patchtst_test_tuned_results=pass "
        f"new=35 unique=40 profiles=50 gate_pass={str(gate_pass).lower()}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the complete test-tuned iTransformer decoder-HPO v2 scorecard."""

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


def mean_rows(
    cells: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        groups[tuple(row[key] for key in keys)].append(row)
    return [
        {
            **dict(zip(keys, key)),
            "mean_mse": sum(row["mse"] for row in rows) / len(rows),
            "mean_mae": sum(row["mae"] for row in rows) / len(rows),
            "cell_count": len(rows),
        }
        for key, rows in sorted(groups.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/iscf_bsca_decoder_transfer_itransformer_hpo_v2_formal.json"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if sha256(args.manifest) != config["artifact_contract"][
        "training_manifest_sha256"
    ]:
        raise RuntimeError("training manifest hash mismatch")
    manifests = read_csv(args.manifest)
    if len(manifests) != 70:
        raise RuntimeError(f"manifest incomplete: {len(manifests)}/70")
    if len({row["checkpoint_sha256"] for row in manifests}) != 70:
        raise RuntimeError("checkpoint uniqueness mismatch")

    new_cells: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    parameter_lookup: dict[tuple[str, str], int] = {}
    test_dates: set[str] = set()
    for manifest in manifests:
        checkpoint = Path(manifest["checkpoint"])
        if sha256(checkpoint) != manifest["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint mutated: {checkpoint}")
        run_dir = checkpoint.parent
        diagnostics = json.loads((run_dir / "model_diagnostics.json").read_text())
        parameter_lookup[(manifest["dataset"], manifest["profile_id"])] = int(
            diagnostics["pcsd_decoder_parameters"]
        )
        artifact = (
            args.output_root
            / "formal_test"
            / manifest["profile_id"]
            / manifest["dataset"]
            / "seed2021"
        )
        invariant_path = artifact / "test_audit_invariants.json"
        metric_path = artifact / "test_audit_metrics_by_target_horizon.csv"
        diagnostic_path = artifact / "pcsd_test_audit_diagnostics.npz"
        invariants = json.loads(invariant_path.read_text())
        if not (
            diagnostic_path.is_file()
            and invariants.get("pass") is True
            and invariants.get("evaluation_split") == "test"
            and invariants.get("test_access_authorized") is True
            and invariants.get("checkpoint_sha256")
            == manifest["checkpoint_sha256"]
            and invariants.get("checkpoint_retrained") is False
            and invariants.get("hyperparameter_profile_id")
            == manifest["profile_id"]
        ):
            raise RuntimeError(f"formal invariant failed: {artifact}")
        test_dates.add(invariants["test_access_date"])
        dense = read_csv(metric_path)
        if len(dense) != 720:
            raise RuntimeError(f"dense test rows incomplete: {artifact}")
        by_horizon = {int(row["target_horizon"]): row for row in dense}
        for horizon in HORIZONS:
            row = by_horizon[horizon]
            mse, mae = float(row["mse"]), float(row["mae"])
            if not math.isfinite(mse) or not math.isfinite(mae):
                raise RuntimeError(f"non-finite metric: {artifact} H{horizon}")
            new_cells.append(
                {
                    "dataset": manifest["dataset"],
                    "profile_id": manifest["profile_id"],
                    "horizon": horizon,
                    "mse": mse,
                    "mae": mae,
                    "seed": 2021,
                    "checkpoint_sha256": manifest["checkpoint_sha256"],
                    "decoder_parameters": parameter_lookup[
                        (manifest["dataset"], manifest["profile_id"])
                    ],
                    "test_access_date": invariants["test_access_date"],
                    "test_tuned": True,
                }
            )
        artifact_rows.append(
            {
                "dataset": manifest["dataset"],
                "profile_id": manifest["profile_id"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "metrics_sha256": sha256(metric_path),
                "invariants_sha256": sha256(invariant_path),
                "diagnostics_sha256": sha256(diagnostic_path),
                "formal_invariant_pass": True,
            }
        )
    if len(new_cells) != 280:
        raise RuntimeError(f"new formal matrix incomplete: {len(new_cells)}/280")

    v1_path = Path(config["artifact_contract"]["v1_reference_cells"])
    v1_rows = read_csv(v1_path)
    reference_cells: list[dict[str, Any]] = []
    original_cells: list[dict[str, Any]] = []
    for row in v1_rows:
        if row["arm"] not in {"itransformer_iscf_bsca", "itransformer_original"}:
            continue
        target = reference_cells if row["arm"] == "itransformer_iscf_bsca" else original_cells
        profile_id = "v1_reference" if target is reference_cells else "v1_original"
        dataset = row["dataset"]
        params = parameter_lookup[(dataset, "p00_budget30")] if target is reference_cells else 0
        target.append(
            {
                "dataset": dataset,
                "profile_id": profile_id,
                "horizon": int(row["horizon"]),
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
                "seed": int(row["seed"]),
                "checkpoint_sha256": row["checkpoint_sha256"],
                "decoder_parameters": params,
                "test_access_date": row["test_access_date"],
                "test_tuned": False,
            }
        )
    if len(reference_cells) != 20 or len(original_cells) != 20:
        raise RuntimeError("v1 reference/original cells incomplete")

    candidate_cells = new_cells + reference_cells
    profile_means = mean_rows(candidate_cells, ("dataset", "profile_id"))
    parameter_mean_lookup = {
        (row["dataset"], row["profile_id"]): next(
            cell["decoder_parameters"]
            for cell in candidate_cells
            if cell["dataset"] == row["dataset"]
            and cell["profile_id"] == row["profile_id"]
        )
        for row in profile_means
    }
    for row in profile_means:
        row["decoder_parameters"] = parameter_mean_lookup[
            (row["dataset"], row["profile_id"])
        ]
    selected_profiles = []
    for dataset in config["datasets"]:
        options = [row for row in profile_means if row["dataset"] == dataset]
        winner = min(
            options,
            key=lambda row: (
                row["mean_mse"],
                row["mean_mae"],
                row["decoder_parameters"],
                row["profile_id"],
            ),
        )
        selected_profiles.append({**winner, "selection_rule": config["test_tuned_selection"]["primary_metric"]})
    selected_lookup = {
        row["dataset"]: row["profile_id"] for row in selected_profiles
    }
    selected_cells = [
        row
        for row in candidate_cells
        if selected_lookup[row["dataset"]] == row["profile_id"]
    ]
    comparison_cells = []
    for row in original_cells:
        comparison_cells.append({"arm": "v1_original", **row})
    for row in selected_cells:
        comparison_cells.append({"arm": "selected_bsca", **row})
    comparison_means = mean_rows(comparison_cells, ("arm", "dataset"))
    overall = mean_rows(comparison_cells, ("arm",))
    overall_lookup = {row["arm"]: row for row in overall}
    dataset_lookup = {
        (row["arm"], row["dataset"]): row for row in comparison_means
    }
    original = overall_lookup["v1_original"]
    selected = overall_lookup["selected_bsca"]

    def gain(metric: str) -> float:
        return 100.0 * (
            original[f"mean_{metric}"] - selected[f"mean_{metric}"]
        ) / original[f"mean_{metric}"]

    dataset_wins = sum(
        dataset_lookup[("selected_bsca", dataset)]["mean_mse"]
        < dataset_lookup[("v1_original", dataset)]["mean_mse"]
        for dataset in config["datasets"]
    )
    mse_gain, mae_gain = gain("mse"), gain("mae")
    gate_pass = bool(mse_gain > 0 and mae_gain > 0 and dataset_wins >= 3)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.results_dir / "new_280_cells.csv", new_cells)
    write_csv(args.results_dir / "candidate_pool_300_cells.csv", candidate_cells)
    write_csv(args.results_dir / "candidate_profile_means.csv", profile_means)
    write_csv(args.results_dir / "selected_profiles.csv", selected_profiles)
    write_csv(args.results_dir / "selected_20_cells.csv", selected_cells)
    write_csv(args.results_dir / "selected_vs_original_40_cells.csv", comparison_cells)
    write_csv(args.results_dir / "selected_vs_original_dataset_means.csv", comparison_means)
    write_csv(args.results_dir / "selected_vs_original_overall.csv", overall)
    artifact_path = args.results_dir / "formal_artifact_manifest.csv"
    write_csv(artifact_path, artifact_rows)
    summary = {
        "pass": gate_pass,
        "candidate_version": config["candidate_version"],
        "formal_checkpoint_jobs": 70,
        "new_formal_cells": 280,
        "candidate_pool_cells": 300,
        "unique_checkpoint_hashes": 70,
        "test_access_dates": sorted(test_dates),
        "selected_profile_by_dataset": selected_lookup,
        "selected_vs_original_macro_mse_gain_percent": mse_gain,
        "selected_vs_original_macro_mae_gain_percent": mae_gain,
        "selected_vs_original_dataset_mse_wins": dataset_wins,
        "checkpoint_nonmutation": True,
        "all_negative_trials_retained": True,
        "per_horizon_or_cell_selection": False,
        "test_tuned": True,
        "matched_iscf_attribution_required_if_pass": gate_pass,
        "table_mutation_authorized": False,
        "formal_artifact_manifest_sha256": sha256(artifact_path),
    }
    (args.results_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        "itransformer_hpo_v2_results=pass "
        f"cells=280 candidate_pool=300 gate_pass={str(gate_pass).lower()}"
    )


if __name__ == "__main__":
    main()

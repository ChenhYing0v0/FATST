#!/usr/bin/env python3
"""Build the frozen validation-only ISCF-BSCA Figure 5 evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
]
plt.rcParams["svg.fonttype"] = "none"


SCOPE_COLORS = ["#484878", "#7884B4", "#B4C0E4", "#E4CCD8", "#D8899D"]
FULL_COLOR = "#0F4D92"
FIXED_COLOR = "#A8A8A8"
TARGET_COLOR = "#272727"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/iscf_bsca_figure5_diagnostic_protocol.json"),
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def normalize_svg(path: Path) -> None:
    """Remove generator-added trailing whitespace for clean version control."""
    text = path.read_text(encoding="utf-8")
    normalized = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    path.write_text(normalized, encoding="utf-8")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def artifact_dir(raw_root: Path, role: str, dataset: str) -> Path:
    return raw_root / role / dataset


def load_artifact(
    raw_root: Path,
    role: str,
    dataset: str,
    expected_hash: str,
    required_files: list[str],
    required_arrays: list[str],
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    directory = artifact_dir(raw_root, role, dataset)
    missing = [name for name in required_files if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory} missing {missing}")

    invariant = json.loads(
        (directory / "trained_invariants.json").read_text(encoding="utf-8")
    )
    if invariant.get("checkpoint_sha256") != expected_hash:
        raise ValueError(f"checkpoint hash mismatch for {role}/{dataset}")
    if (
        invariant.get("dataset") != dataset
        or invariant.get("evaluation_split") != "val"
        or invariant.get("uses_test_split") is not False
        or invariant.get("pass") is not True
    ):
        raise ValueError(f"invalid validation invariant for {role}/{dataset}")

    with np.load(directory / "pcsd_validation_diagnostics.npz") as source:
        missing_arrays = [name for name in required_arrays if name not in source]
        if missing_arrays:
            raise KeyError(f"{directory} missing arrays {missing_arrays}")
        arrays = {name: source[name].copy() for name in source.files}

    if any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in "fiu"):
        raise ValueError(f"non-finite array in {role}/{dataset}")

    manifest_rows = []
    for name in required_files:
        path = directory / name
        manifest_rows.append(
            {
                "role": role,
                "dataset": dataset,
                "file": name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "checkpoint_sha256": expected_hash,
                "evaluation_split": "val",
            }
        )
    return invariant, arrays, manifest_rows


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.10,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def add_heatmap_annotations(axis: plt.Axes, values: np.ndarray, fmt: str) -> None:
    midpoint = float(np.nanmin(values) + np.nanmax(values)) / 2.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            axis.text(
                column,
                row,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=5.4,
                color="white" if value > midpoint else "#272727",
            )


def build_figure(
    output_dir: Path,
    config: dict[str, Any],
    chpc_rows: list[dict[str, Any]],
    macro_usage: np.ndarray,
    macro_excess_mse: np.ndarray,
    selected: dict[str, Any],
) -> dict[str, str]:
    configure_style()
    regions = [entry["name"] for entry in config["future_regions"]]
    scopes = config["scopes"]
    horizons = config["horizons"]
    datasets = config["datasets"]

    figure = plt.figure(figsize=(180 / 25.4, 160 / 25.4), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=[0.78, 1.05, 1.12])
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 0])
    axis_d = figure.add_subplot(grid[1, 1])
    axis_e = figure.add_subplot(grid[2, :])

    dataset_max = []
    for dataset in datasets:
        values = [
            row["max_abs_chpd"]
            for row in chpc_rows
            if row["dataset"] == dataset and row["horizon"] in horizons[:-1]
        ]
        dataset_max.append(max(values))
    y_positions = np.arange(len(datasets))
    axis_a.scatter(dataset_max, y_positions, color=FULL_COLOR, s=22, zorder=3)
    tolerance = config["statistics"]["chpc"]["max_abs_tolerance"]
    axis_a.axvline(tolerance, color="#B64342", linestyle="--", linewidth=0.9)
    axis_a.set_xlim(-0.04 * tolerance, 1.08 * tolerance)
    axis_a.set_yticks(y_positions, labels=datasets)
    axis_a.set_xlabel("Maximum absolute CHPD")
    axis_a.set_title("Prefix consistency", loc="left", fontsize=8)
    axis_a.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    axis_a.text(
        0.98,
        0.04,
        f"tolerance = {tolerance:.0e}",
        transform=axis_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color="#767676",
    )
    add_panel_label(axis_a, "a")

    policy = selected["policy"].T
    image_b = axis_b.imshow(
        policy,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=max(0.25, float(policy.max())),
    )
    axis_b.set_yticks(np.arange(len(scopes)), labels=[f"s={scope}" for scope in scopes])
    axis_b.set_xticks([0, 95, 191, 335, 511, 719], labels=[1, 96, 192, 336, 512, 720])
    axis_b.set_xlabel("Future step")
    axis_b.set_title(
        f"Learned Scope Probability ({selected['dataset']}, selected row)",
        loc="left",
        fontsize=8,
    )
    colorbar_b = figure.colorbar(image_b, ax=axis_b, fraction=0.034, pad=0.02)
    colorbar_b.set_label("Probability", fontsize=6)
    colorbar_b.ax.tick_params(labelsize=5.5)
    add_panel_label(axis_b, "b")

    image_c = axis_c.imshow(
        macro_usage.T,
        aspect="auto",
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=max(0.25, float(macro_usage.max())),
    )
    axis_c.set_yticks(np.arange(len(scopes)), labels=[f"s={scope}" for scope in scopes])
    axis_c.set_xticks(np.arange(len(regions)), labels=regions, rotation=36, ha="right")
    axis_c.set_title("Aggregate scope utilization", loc="left", fontsize=8)
    add_heatmap_annotations(axis_c, macro_usage.T, ".2f")
    colorbar_c = figure.colorbar(image_c, ax=axis_c, fraction=0.034, pad=0.02)
    colorbar_c.set_label("Dataset-macro probability", fontsize=6)
    colorbar_c.ax.tick_params(labelsize=5.5)
    add_panel_label(axis_c, "c")

    image_d = axis_d.imshow(
        macro_excess_mse.T,
        aspect="auto",
        interpolation="nearest",
        cmap="YlOrRd",
        vmin=0.0,
    )
    axis_d.set_yticks(np.arange(len(scopes)), labels=[f"s={scope}" for scope in scopes])
    axis_d.set_xticks(np.arange(len(regions)), labels=regions, rotation=36, ha="right")
    axis_d.set_title("Scope-wise regional error", loc="left", fontsize=8)
    add_heatmap_annotations(axis_d, macro_excess_mse.T, ".1f")
    colorbar_d = figure.colorbar(image_d, ax=axis_d, fraction=0.034, pad=0.02)
    colorbar_d.set_label("Excess MSE above region best (%)", fontsize=6)
    colorbar_d.ax.tick_params(labelsize=5.5)
    add_panel_label(axis_d, "d")

    steps = np.arange(1, 721)
    axis_e.plot(steps, selected["target"], color=TARGET_COLOR, linewidth=1.15, label="Ground truth")
    axis_e.plot(steps, selected["full"], color=FULL_COLOR, linewidth=1.0, label="Full ISCF-BSCA")
    axis_e.plot(steps, selected["fixed"], color=FIXED_COLOR, linewidth=0.9, label="Fixed Scope (s=144)")
    for horizon, alpha in zip(horizons[:-1], (0.75, 0.55, 0.35)):
        axis_e.axvline(horizon, color="#9A4D8E", linewidth=0.7, linestyle="--", alpha=alpha)
        axis_e.text(
            horizon,
            0.98,
            f"H{horizon}",
            transform=axis_e.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=5.5,
            color="#9A4D8E",
        )
    axis_e.set_xlim(1, 720)
    axis_e.set_xlabel("Future step")
    axis_e.set_ylabel("Target value")
    axis_e.set_title(
        "Performance-selected validation trajectory "
        f"({selected['dataset']}, row {selected['row_index']}; "
        f"MSE reduction {selected['mse_gain_percent']:.1f}%)",
        loc="left",
        fontsize=8,
    )
    axis_e.grid(color="#EEEEEE", linewidth=0.5)
    axis_e.legend(ncol=3, loc="upper right", fontsize=6.2)
    add_panel_label(axis_e, "e")

    stem = output_dir / "figure_5_iscf_bsca_mechanism"
    outputs = {
        "svg": str(stem.with_suffix(".svg")),
        "pdf": str(stem.with_suffix(".pdf")),
        "png": str(stem.with_suffix(".png")),
        "tiff": str(stem.with_suffix(".tiff")),
    }
    figure.savefig(outputs["svg"])
    normalize_svg(Path(outputs["svg"]))
    figure.savefig(outputs["pdf"])
    figure.savefig(outputs["png"], dpi=300)
    figure.savefig(
        outputs["tiff"],
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)
    return outputs


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    raw_root = args.raw_root or Path(config["artifact_sources"]["local_raw_root"])
    output_dir = args.output_dir or raw_root.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    required_files = config["required_artifacts"]
    required_arrays = config["required_npz_arrays"]
    datasets = config["datasets"]
    scopes = np.asarray(config["scopes"], dtype=np.int64)
    expected_bins = np.asarray(
        [entry["artifact_name"] for entry in config["future_regions"]]
    )
    display_bins = np.asarray(
        [entry["name"] for entry in config["future_regions"]]
    )

    loaded: dict[str, dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]]] = {
        "full": {},
        "fixed_scope_s144": {},
    }
    artifact_rows: list[dict[str, Any]] = []
    chpc_rows: list[dict[str, Any]] = []
    utilization_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    preference_rows: list[dict[str, Any]] = []
    dataset_usage = []
    dataset_mse = []

    for role in loaded:
        source = config["artifact_sources"][role]
        for dataset in datasets:
            invariant, arrays, rows = load_artifact(
                raw_root,
                role,
                dataset,
                source["checkpoint_sha256"][dataset],
                required_files,
                required_arrays,
            )
            if not np.array_equal(arrays["scales"], scopes):
                raise ValueError(f"scope mismatch for {role}/{dataset}")
            if not np.array_equal(arrays["bin_names"].astype(str), expected_bins):
                raise ValueError(f"region mismatch for {role}/{dataset}")
            loaded[role][dataset] = (invariant, arrays)
            artifact_rows.extend(rows)

    tolerance = float(config["statistics"]["chpc"]["max_abs_tolerance"])
    for dataset in datasets:
        invariant, full = loaded["full"][dataset]
        fixed_invariant, fixed = loaded["fixed_scope_s144"][dataset]
        if not np.array_equal(full["probe_targets"], fixed["probe_targets"]):
            raise ValueError(f"Full/fixed probe target mismatch for {dataset}")
        if full["probe_fused"].shape[0] != config["matrix"]["probe_rows_per_dataset"]:
            raise ValueError(f"unexpected probe count for {dataset}")
        if full["probe_fused"].shape != fixed["probe_fused"].shape:
            raise ValueError(f"Full/fixed probe shape mismatch for {dataset}")
        policy = full["probe_direct_policy"]
        probability_gap = float(np.abs(policy.sum(axis=-1) - 1.0).max())
        if probability_gap > config["gates"]["scope_probabilities_sum_to_one_max_abs"]:
            raise ValueError(f"policy normalization failed for {dataset}")

        for entry in invariant["prefix_rows"]:
            horizon = int(entry["horizon"])
            if horizon not in config["horizons"]:
                continue
            gap = float(entry["full_prefix_max_abs"])
            if gap > tolerance:
                raise ValueError(f"CHPC failed for {dataset}/H{horizon}")
            chpc_rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "max_abs_chpd": gap,
                    "tolerance": tolerance,
                    "pass": gap <= tolerance,
                    "checkpoint_sha256": invariant["checkpoint_sha256"],
                    "evaluation_split": "val",
                }
            )

        usage = full["policy_row_bin_usage"].mean(axis=0, dtype=np.float64)
        mse = full["arm_row_bin_mse"].mean(axis=0, dtype=np.float64)
        mae = full["arm_row_bin_mae"].mean(axis=0, dtype=np.float64)
        dataset_usage.append(usage)
        dataset_mse.append(mse)
        for region_index, region in enumerate(display_bins):
            allocated_index = int(np.argmax(usage[region_index]))
            best_mse_index = int(np.argmin(mse[region_index]))
            best_mae_index = int(np.argmin(mae[region_index]))
            preference_rows.append(
                {
                    "dataset": dataset,
                    "region": region,
                    "highest_utilization_scope": int(scopes[allocated_index]),
                    "lowest_mse_scope": int(scopes[best_mse_index]),
                    "lowest_mae_scope": int(scopes[best_mae_index]),
                    "utilization_mse_agreement": allocated_index == best_mse_index,
                    "validation_series_rows": int(full["arm_row_bin_mse"].shape[0]),
                }
            )
            for scope_index, scope in enumerate(scopes):
                utilization_rows.append(
                    {
                        "dataset": dataset,
                        "region": region,
                        "scope": int(scope),
                        "mean_probability": float(usage[region_index, scope_index]),
                        "validation_series_rows": int(full["policy_row_bin_usage"].shape[0]),
                    }
                )
                error_rows.append(
                    {
                        "dataset": dataset,
                        "region": region,
                        "scope": int(scope),
                        "mean_mse": float(mse[region_index, scope_index]),
                        "mean_mae": float(mae[region_index, scope_index]),
                        "is_region_best_mse": scope_index == best_mse_index,
                        "is_region_best_mae": scope_index == best_mae_index,
                        "validation_series_rows": int(full["arm_row_bin_mse"].shape[0]),
                    }
                )
        if fixed_invariant["fixed_scale"] != 144:
            raise ValueError(f"fixed control mismatch for {dataset}")

    selection_rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, float, int, int, dict[str, Any]]] = []
    for dataset_index, dataset in enumerate(datasets):
        full = loaded["full"][dataset][1]
        fixed = loaded["fixed_scope_s144"][dataset][1]
        for row_index in range(full["probe_fused"].shape[0]):
            target = full["probe_targets"][row_index].astype(np.float64)
            full_prediction = full["probe_fused"][row_index].astype(np.float64)
            fixed_prediction = fixed["probe_fused"][row_index].astype(np.float64)
            full_mse = float(np.mean((full_prediction - target) ** 2))
            fixed_mse = float(np.mean((fixed_prediction - target) ** 2))
            full_mae = float(np.mean(np.abs(full_prediction - target)))
            fixed_mae = float(np.mean(np.abs(fixed_prediction - target)))
            mse_gain = 100.0 * (fixed_mse - full_mse) / max(fixed_mse, 1e-12)
            mae_gain = 100.0 * (fixed_mae - full_mae) / max(fixed_mae, 1e-12)
            row = {
                "dataset": dataset,
                "probe_row_index": row_index,
                "full_mse": full_mse,
                "fixed_scope_s144_mse": fixed_mse,
                "mse_gain_percent": mse_gain,
                "full_mae": full_mae,
                "fixed_scope_s144_mae": fixed_mae,
                "mae_gain_percent": mae_gain,
            }
            selection_rows.append(row)
            candidates.append((mse_gain, mae_gain, -dataset_index, -row_index, row))

    _mse, _mae, _dataset_order, _row_order, selected_row = max(candidates)
    selected_dataset = selected_row["dataset"]
    selected_index = int(selected_row["probe_row_index"])
    selected_full = loaded["full"][selected_dataset][1]
    selected_fixed = loaded["fixed_scope_s144"][selected_dataset][1]
    selected = {
        "dataset": selected_dataset,
        "row_index": selected_index,
        "target": selected_full["probe_targets"][selected_index],
        "full": selected_full["probe_fused"][selected_index],
        "fixed": selected_fixed["probe_fused"][selected_index],
        "arms": selected_full["probe_arms"][selected_index],
        "policy": selected_full["probe_direct_policy"][selected_index],
        "mse_gain_percent": float(selected_row["mse_gain_percent"]),
        "mae_gain_percent": float(selected_row["mae_gain_percent"]),
    }

    macro_usage = np.mean(np.stack(dataset_usage), axis=0)
    macro_mse = np.mean(np.stack(dataset_mse), axis=0)
    region_best = macro_mse.min(axis=1, keepdims=True)
    macro_excess_mse = 100.0 * (macro_mse - region_best) / np.maximum(
        region_best,
        1e-12,
    )

    write_csv(output_dir / "artifact_manifest.csv", artifact_rows)
    write_csv(output_dir / "chpc_verification.csv", chpc_rows)
    write_csv(output_dir / "scope_utilization.csv", utilization_rows)
    write_csv(output_dir / "scope_regional_error.csv", error_rows)
    write_csv(output_dir / "scope_preference_summary.csv", preference_rows)
    write_csv(output_dir / "qualitative_selection_pool.csv", selection_rows)

    source_rows = []
    for step in range(720):
        row: dict[str, Any] = {
            "dataset": selected_dataset,
            "probe_row_index": selected_index,
            "future_step": step + 1,
            "target": float(selected["target"][step]),
            "full_iscf_bsca": float(selected["full"][step]),
            "fixed_scope_s144": float(selected["fixed"][step]),
        }
        for scope_index, scope in enumerate(scopes):
            row[f"scope_{scope}_probability"] = float(
                selected["policy"][step, scope_index]
            )
            row[f"scope_{scope}_forecast"] = float(
                selected["arms"][scope_index, step]
            )
        source_rows.append(row)
    write_csv(output_dir / "qualitative_source_data.csv", source_rows)

    outputs = build_figure(
        output_dir,
        config,
        chpc_rows,
        macro_usage,
        macro_excess_mse,
        selected,
    )

    agreement = sum(bool(row["utilization_mse_agreement"]) for row in preference_rows)
    summary = {
        "diagnostic_id": config["diagnostic_id"],
        "status": "complete",
        "datasets": datasets,
        "checkpoint_objects": config["matrix"]["checkpoint_objects"],
        "new_training_runs": 0,
        "formal_test_runs": 0,
        "validation_only": True,
        "chpc_rows": len(chpc_rows),
        "chpc_max_abs": max(float(row["max_abs_chpd"]) for row in chpc_rows),
        "policy_sum_max_abs": max(
            float(
                np.abs(
                    loaded["full"][dataset][1]["probe_direct_policy"].sum(axis=-1)
                    - 1.0
                ).max()
            )
            for dataset in datasets
        ),
        "utilization_mse_agreement_cells": agreement,
        "utilization_mse_agreement_total": len(preference_rows),
        "selected_qualitative": {
            key: value for key, value in selected_row.items()
        },
        "selected_example_interpretation": config["statistics"][
            "qualitative_selection"
        ]["interpretation"],
        "figure_outputs": outputs,
        "claim_boundary": config["claim_boundaries"],
        "decision": "figure5_validation_diagnostics_complete_pending_visual_and_claim_audit",
    }
    (output_dir / "figure5_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

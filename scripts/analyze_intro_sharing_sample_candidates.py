#!/usr/bin/env python3
"""Select a sample with strongly heterogeneous future-region sharing demand."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCALES = (1, 8, 32, 128, 720)
REGION_LENGTH = 60
SCALE_COLORS = {
    1: "#D55E00",
    8: "#E69F00",
    32: "#0072B2",
    128: "#CC79A7",
    720: "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="NeutralSharingExtent")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--crossing-margin", type=float, default=0.005)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    artifacts: dict[int, dict[str, Any]] = {}
    for scale in SCALES:
        run_dir = (
            args.input_root
            / args.run_name
            / args.dataset
            / f"s{scale}"
            / f"seed{args.seed}"
        )
        prediction_path = run_dir / "predictions_val.npz"
        config_path = run_dir / "effective_config.json"
        if not prediction_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                prediction_path
                if not prediction_path.is_file()
                else config_path
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with np.load(prediction_path) as archive:
            artifacts[scale] = {
                "pred": archive["pred"].copy(),
                "true": archive["true"].copy(),
                "history": archive["history"].copy(),
                "origin_index": archive["origin_index"].copy(),
                "parameter_count": int(config["parameter_count"]),
            }
    return artifacts


def validate_artifacts(
    artifacts: dict[int, dict[str, Any]],
) -> dict[str, int | float]:
    reference = artifacts[SCALES[0]]
    parameter_counts = {
        artifacts[scale]["parameter_count"] for scale in SCALES
    }
    if len(parameter_counts) != 1:
        raise RuntimeError(f"parameter mismatch: {sorted(parameter_counts)}")
    maximum_target_gap = 0.0
    maximum_history_gap = 0.0
    maximum_origin_gap = 0.0
    for scale in SCALES:
        item = artifacts[scale]
        if item["pred"].shape != reference["pred"].shape:
            raise RuntimeError(f"prediction shape mismatch at scale {scale}")
        maximum_target_gap = max(
            maximum_target_gap,
            float(np.max(np.abs(item["true"] - reference["true"]))),
        )
        maximum_history_gap = max(
            maximum_history_gap,
            float(np.max(np.abs(item["history"] - reference["history"]))),
        )
        maximum_origin_gap = max(
            maximum_origin_gap,
            float(
                np.max(
                    np.abs(
                        item["origin_index"] - reference["origin_index"]
                    )
                )
            ),
        )
    if max(maximum_target_gap, maximum_history_gap, maximum_origin_gap) > 1e-6:
        raise RuntimeError(
            "artifact alignment failed: "
            f"target_gap={maximum_target_gap}, "
            f"history_gap={maximum_history_gap}, "
            f"origin_gap={maximum_origin_gap}"
        )
    return {
        "parameter_count": parameter_counts.pop(),
        "origin_count": int(reference["pred"].shape[0]),
        "channel_count": int(reference["pred"].shape[2]),
        "maximum_target_alignment_gap": maximum_target_gap,
        "maximum_history_alignment_gap": maximum_history_gap,
        "maximum_origin_alignment_gap": maximum_origin_gap,
    }


def normalized_entropy(counts: np.ndarray) -> float:
    probabilities = counts[counts > 0] / counts.sum()
    if probabilities.size <= 1:
        return 0.0
    return float(
        -np.sum(probabilities * np.log(probabilities)) / np.log(len(SCALES))
    )


def qualified_crossing_count(
    region_risk: np.ndarray,
    margin: float,
) -> int:
    count = 0
    for left in range(len(SCALES)):
        for right in range(left + 1, len(SCALES)):
            relative = (
                region_risk[left] - region_risk[right]
            ) / np.maximum(region_risk[right], 1e-12)
            if (
                float(np.max(relative)) >= margin
                and float(np.min(relative)) <= -margin
            ):
                count += 1
    return count


def compute_candidates(
    artifacts: dict[int, dict[str, Any]],
    crossing_margin: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    sample_step_risks = []
    for scale in SCALES:
        error = artifacts[scale]["pred"] - artifacts[scale]["true"]
        sample_step_risks.append(
            np.mean(error * error, axis=2).astype(np.float64)
        )
    step_risk = np.stack(sample_step_risks, axis=1)
    origin_count, _, pred_len = step_risk.shape
    if pred_len % REGION_LENGTH:
        raise RuntimeError(
            f"pred_len={pred_len} is not divisible by {REGION_LENGTH}"
        )
    region_count = pred_len // REGION_LENGTH
    region_risk = step_risk.reshape(
        origin_count,
        len(SCALES),
        region_count,
        REGION_LENGTH,
    ).mean(axis=3)

    rows: list[dict[str, Any]] = []
    for origin in range(origin_count):
        current = region_risk[origin]
        winners = np.argmin(current, axis=0)
        winner_counts = np.bincount(
            winners,
            minlength=len(SCALES),
        )
        ordered = np.partition(current, kth=1, axis=0)
        winner_margin = (
            ordered[1] - ordered[0]
        ) / np.maximum(ordered[1], 1e-12)
        fixed_risks = np.mean(current, axis=1)
        best_fixed_risk = float(np.min(fixed_risks))
        oracle_risk = float(np.mean(np.min(current, axis=0)))
        row: dict[str, Any] = {
            "origin_index": origin,
            "supported_winner_count": int(np.sum(winner_counts >= 2)),
            "distinct_winner_count": int(np.sum(winner_counts > 0)),
            "winner_entropy": normalized_entropy(winner_counts),
            "qualified_crossing_pair_count": qualified_crossing_count(
                current,
                crossing_margin,
            ),
            "mean_winner_margin": float(np.mean(winner_margin)),
            "sample_oracle_headroom": (
                best_fixed_risk - oracle_risk
            ) / max(best_fixed_risk, 1e-12),
            "best_fixed_scale": SCALES[int(np.argmin(fixed_risks))],
            "best_fixed_validation_mse": best_fixed_risk,
            "region_oracle_validation_mse": oracle_risk,
        }
        for index, scale in enumerate(SCALES):
            row[f"s{scale}_winner_regions"] = int(winner_counts[index])
        rows.append(row)
    return step_risk, region_risk, rows


def selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["supported_winner_count"]),
        float(row["distinct_winner_count"]),
        float(row["winner_entropy"]),
        float(row["qualified_crossing_pair_count"]),
        float(row["mean_winner_margin"]),
        float(row["sample_oracle_headroom"]),
        -float(row["origin_index"]),
    )


def moving_average(values: np.ndarray, window: int = 31) -> np.ndarray:
    padding = window // 2
    padded = np.pad(values, (padding, padding), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def plot_candidate(
    args: argparse.Namespace,
    step_risk: np.ndarray,
    region_risk: np.ndarray,
    selected: dict[str, Any],
) -> tuple[Path, Path]:
    origin = int(selected["origin_index"])
    current_step = step_risk[origin]
    current_region = region_risk[origin]
    fixed_index = SCALES.index(int(selected["best_fixed_scale"]))
    fixed_region = current_region[fixed_index]
    relative_region = (
        current_region - fixed_region[None, :]
    ) / np.maximum(fixed_region[None, :], 1e-12)
    winners = np.argmin(current_region, axis=0)
    winner_counts = np.bincount(winners, minlength=len(SCALES))

    figure = plt.figure(figsize=(15.8, 4.8), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.3, 1.65, 0.85))
    heatmap_axis = figure.add_subplot(grid[0, 0])
    curve_axis = figure.add_subplot(grid[0, 1])
    count_axis = figure.add_subplot(grid[0, 2])

    magnitude = max(float(np.max(np.abs(relative_region))), 1e-6)
    image = heatmap_axis.imshow(
        relative_region * 100.0,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-100.0 * magnitude,
        vmax=100.0 * magnitude,
    )
    heatmap_axis.plot(
        np.arange(current_region.shape[1]),
        winners,
        color="#111111",
        linewidth=1.8,
        marker="o",
        markersize=3.4,
    )
    heatmap_axis.set_yticks(range(len(SCALES)), labels=SCALES)
    heatmap_axis.set_xticks(
        range(current_region.shape[1]),
        labels=range(1, current_region.shape[1] + 1),
    )
    heatmap_axis.set_xlabel("60-step future region")
    heatmap_axis.set_ylabel("Sharing extent")
    heatmap_axis.set_title("(a) Sample-level sharing-risk landscape")
    colorbar = figure.colorbar(image, ax=heatmap_axis, shrink=0.82)
    colorbar.set_label("MSE change vs sample-best fixed scope (%)")

    future_steps = np.arange(1, current_step.shape[1] + 1)
    fixed_step = current_step[fixed_index]
    for scale_index, scale in enumerate(SCALES):
        relative_curve = (
            current_step[scale_index] - fixed_step
        ) / np.maximum(fixed_step, 1e-12)
        curve_axis.plot(
            future_steps,
            moving_average(relative_curve) * 100.0,
            color=SCALE_COLORS[scale],
            linewidth=1.25,
            label=f"s={scale}",
        )
    curve_axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
    for boundary in range(REGION_LENGTH, 720, REGION_LENGTH):
        curve_axis.axvline(boundary, color="#BBBBBB", linewidth=0.35)
    curve_axis.set_xlabel("Future step")
    curve_axis.set_ylabel("MSE change vs sample-best fixed scope (%)")
    curve_axis.set_title("(b) Step-wise risk differences")
    curve_axis.legend(ncol=3, frameon=False, fontsize=8.5)
    curve_axis.grid(alpha=0.15)

    positions = np.arange(len(SCALES))
    count_axis.bar(
        positions,
        winner_counts,
        color=[SCALE_COLORS[scale] for scale in SCALES],
    )
    count_axis.set_xticks(positions, labels=SCALES)
    count_axis.set_ylim(0, current_region.shape[1])
    count_axis.set_xlabel("Sharing extent")
    count_axis.set_ylabel("Winning 60-step regions")
    count_axis.set_title("(c) Region-winner distribution")
    for index, count in enumerate(winner_counts):
        count_axis.text(index, count + 0.15, str(count), ha="center")
    count_axis.grid(axis="y", alpha=0.15)

    figure.suptitle(
        f"{args.dataset}: maximum-heterogeneity validation candidate "
        f"(origin {origin})",
        fontsize=12,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = args.output_dir / "sharing_sample_candidate.svg"
    png_path = args.output_dir / "sharing_sample_candidate.png"
    figure.savefig(svg_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return svg_path, png_path


def main() -> None:
    args = parse_args()
    if args.crossing_margin <= 0:
        raise ValueError("crossing_margin must be positive")
    artifacts = load_predictions(args)
    alignment = validate_artifacts(artifacts)
    step_risk, region_risk, rows = compute_candidates(
        artifacts,
        args.crossing_margin,
    )
    selected = max(rows, key=selection_key)
    selected_origin = int(selected["origin_index"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "sample_candidates.csv", rows)
    write_csv(
        args.output_dir / "selected_region_risk.csv",
        [
            {
                "scale": scale,
                "region": region + 1,
                "start_step": region * REGION_LENGTH + 1,
                "end_step": (region + 1) * REGION_LENGTH,
                "mse": float(
                    region_risk[selected_origin, scale_index, region]
                ),
                "is_region_winner": bool(
                    scale_index
                    == np.argmin(region_risk[selected_origin, :, region])
                ),
            }
            for scale_index, scale in enumerate(SCALES)
            for region in range(region_risk.shape[2])
        ],
    )
    write_csv(
        args.output_dir / "selected_step_risk.csv",
        [
            {
                "scale": scale,
                "future_step": step + 1,
                "mse": float(step_risk[selected_origin, scale_index, step]),
            }
            for scale_index, scale in enumerate(SCALES)
            for step in range(step_risk.shape[2])
        ],
    )
    svg_path, png_path = plot_candidate(
        args,
        step_risk,
        region_risk,
        selected,
    )
    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "split": "validation",
        "test_accessed": False,
        **alignment,
        "selection_policy": {
            "unit": "one origin aggregated over all channels",
            "region_length": REGION_LENGTH,
            "crossing_margin": args.crossing_margin,
            "lexicographic_order": [
                "supported_winner_count",
                "distinct_winner_count",
                "winner_entropy",
                "qualified_crossing_pair_count",
                "mean_winner_margin",
                "sample_oracle_headroom",
            ],
            "disclosure": "maximum heterogeneity candidate",
        },
        "selected": selected,
        "figures": {
            "svg": str(svg_path),
            "png": str(png_path),
        },
        "claim_role": "exploratory_visualization_only",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

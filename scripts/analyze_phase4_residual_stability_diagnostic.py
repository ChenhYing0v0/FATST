from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baselines" / "patch_encoder_target_set_decoder"
sys.path.insert(0, str(BASELINE_ROOT))

from dataset import ForecastDataset  # noqa: E402


DATASETS = ["ETTh2", "Weather"]
EPS = 1e-8


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def block_range(block_index: int, block_size: int, pred_len: int) -> tuple[int, int]:
    start = block_index * block_size
    end = min(start + block_size, pred_len)
    return start + 1, end


def block_group(block_index: int, block_size: int, pred_len: int) -> str:
    start, _ = block_range(block_index, block_size, pred_len)
    if start <= 96:
        return "early_1_96"
    if start <= 336:
        return "middle_97_336"
    return f"late_337_{pred_len}"


def chunk_targets(data: np.ndarray, indices: np.ndarray, seq_len: int, pred_len: int) -> np.ndarray:
    targets = [data[index + seq_len : index + seq_len + pred_len] for index in indices]
    return np.stack(targets, axis=0).astype(np.float64, copy=False)


def seasonal_reference(
    data: np.ndarray,
    indices: np.ndarray,
    seq_len: int,
    pred_len: int,
    period: int,
) -> np.ndarray:
    offsets = seq_len - period + (np.arange(pred_len) % period)
    if np.min(offsets) < 0:
        raise ValueError("seasonal period must be <= seq_len.")
    ref_indices = indices[:, None] + offsets[None, :]
    return data[ref_indices].astype(np.float64, copy=False)


def block_metrics(
    block: np.ndarray,
    persistence_ref: np.ndarray,
    seasonal_refs: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    baseline_names = ["persistence"] + [f"seasonal_{period}" for period in seasonal_refs]
    refs = [np.broadcast_to(persistence_ref[:, None, :], block.shape)]
    refs.extend(seasonal_refs.values())
    ref_stack = np.stack(refs, axis=1)
    residual_stack = block[:, None, :, :] - ref_stack
    mse_stack = np.mean(residual_stack * residual_stack, axis=(2, 3))
    best_index = np.argmin(mse_stack, axis=1)
    row_index = np.arange(block.shape[0])
    best_residual = residual_stack[row_index, best_index]
    best_mse = mse_stack[row_index, best_index]
    novelty_mse = mse_stack[:, 0]

    if block.shape[1] <= 1:
        local_variation = np.zeros(block.shape[0], dtype=np.float64)
        residual_first_diff = np.zeros(block.shape[0], dtype=np.float64)
    else:
        local_variation = np.mean(np.diff(block, axis=1) ** 2, axis=(1, 2))
        residual_first_diff = np.mean(np.diff(best_residual, axis=1) ** 2, axis=(1, 2))

    residual_bias = np.mean(best_residual, axis=(1, 2))
    return {
        "novelty_mse": novelty_mse,
        "best_baseline_mse": best_mse,
        "best_gain_over_persistence": novelty_mse / np.maximum(best_mse, EPS),
        "local_variation": local_variation,
        "residual_smoothness": residual_first_diff / np.maximum(best_mse, EPS),
        "residual_bias_share": (residual_bias * residual_bias) / np.maximum(best_mse, EPS),
        "best_baseline_index": best_index.astype(np.int64),
        "best_baseline_name": np.array([baseline_names[index] for index in best_index], dtype=object),
    }


def accumulate_metric_sums(
    totals: dict[str, float],
    metrics: dict[str, np.ndarray],
    mask: np.ndarray | None = None,
) -> int:
    count = int(metrics["novelty_mse"].shape[0] if mask is None else np.sum(mask))
    active = slice(None) if mask is None else mask
    for name in [
        "novelty_mse",
        "best_baseline_mse",
        "best_gain_over_persistence",
        "local_variation",
        "residual_smoothness",
        "residual_bias_share",
    ]:
        totals[name] += float(np.sum(metrics[name][active]))
    return count


def classify_selected_units(records: list[dict[str, Any]]) -> None:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_dataset[row["dataset"]].append(row)
    for dataset, rows in by_dataset.items():
        smooth_values = np.array([float(row["residual_smoothness"]) for row in rows], dtype=np.float64)
        variation_values = np.array([float(row["local_variation"]) for row in rows], dtype=np.float64)
        gain_values = np.array([float(row["best_gain_over_persistence"]) for row in rows], dtype=np.float64)
        smooth_median = float(np.median(smooth_values))
        variation_median = float(np.median(variation_values))
        gain_q60 = float(np.quantile(gain_values, 0.60))
        for row in rows:
            gain = float(row["best_gain_over_persistence"])
            smooth = float(row["residual_smoothness"])
            variation = float(row["local_variation"])
            if gain >= gain_q60 and smooth <= smooth_median:
                bucket = "learnable_conflict"
            elif smooth > smooth_median and variation >= variation_median:
                bucket = "noisy_conflict"
            else:
                bucket = "ambiguous_conflict"
            row["bucket"] = bucket
            row["dataset_smooth_median"] = smooth_median
            row["dataset_variation_median"] = variation_median
            row["dataset_gain_q60"] = gain_q60


def analyze_dataset(
    dataset_root: Path,
    dataset: str,
    seq_len: int,
    pred_len: int,
    block_size: int,
    top_ratio: float,
    periods: list[int],
    chunk_windows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_set = ForecastDataset(dataset_root, dataset, "train", seq_len, pred_len)
    data = train_set.data.astype(np.float64, copy=False)
    n_windows = len(train_set)
    n_blocks = int(np.ceil(pred_len / block_size))
    top_blocks = max(1, min(n_blocks, int(round(n_blocks * top_ratio))))
    valid_periods = [period for period in periods if period <= seq_len]
    if not valid_periods:
        raise ValueError("At least one seasonal period must be <= seq_len.")

    metric_names = [
        "novelty_mse",
        "best_baseline_mse",
        "best_gain_over_persistence",
        "local_variation",
        "residual_smoothness",
        "residual_bias_share",
    ]
    block_totals = {block: {name: 0.0 for name in metric_names} for block in range(n_blocks)}
    selected_totals = {block: {name: 0.0 for name in metric_names} for block in range(n_blocks)}
    nonselected_totals = {block: {name: 0.0 for name in metric_names} for block in range(n_blocks)}
    block_counts = np.zeros(n_blocks, dtype=np.int64)
    selected_counts = np.zeros(n_blocks, dtype=np.int64)
    nonselected_counts = np.zeros(n_blocks, dtype=np.int64)
    baseline_counts = {block: Counter() for block in range(n_blocks)}
    selected_baseline_counts = {block: Counter() for block in range(n_blocks)}
    selected_records: list[dict[str, Any]] = []

    for start in range(0, n_windows, chunk_windows):
        stop = min(start + chunk_windows, n_windows)
        indices = np.arange(start, stop)
        targets = chunk_targets(data, indices, seq_len, pred_len)
        persistence_ref = data[indices + seq_len - 1]
        seasonal_refs_full = {
            period: seasonal_reference(data, indices, seq_len, pred_len, period)
            for period in valid_periods
        }
        chunk_block_metrics: list[dict[str, np.ndarray]] = []
        novelty_values = []
        for block_index in range(n_blocks):
            block_start = block_index * block_size
            block_end = min(block_start + block_size, pred_len)
            block = targets[:, block_start:block_end, :]
            seasonal_refs = {
                period: ref[:, block_start:block_end, :]
                for period, ref in seasonal_refs_full.items()
            }
            metrics = block_metrics(block, persistence_ref, seasonal_refs)
            chunk_block_metrics.append(metrics)
            novelty_values.append(metrics["novelty_mse"])
            block_counts[block_index] += len(indices)
            accumulate_metric_sums(block_totals[block_index], metrics)
            baseline_counts[block_index].update(str(item) for item in metrics["best_baseline_name"])

        novelty_matrix = np.stack(novelty_values, axis=1)
        selected = np.argpartition(-novelty_matrix, kth=top_blocks - 1, axis=1)[:, :top_blocks]
        selected_mask = np.zeros_like(novelty_matrix, dtype=bool)
        selected_mask[np.arange(selected.shape[0])[:, None], selected] = True

        for block_index, metrics in enumerate(chunk_block_metrics):
            is_selected = selected_mask[:, block_index]
            is_not_selected = ~is_selected
            selected_counts[block_index] += int(np.sum(is_selected))
            nonselected_counts[block_index] += int(np.sum(is_not_selected))
            accumulate_metric_sums(selected_totals[block_index], metrics, is_selected)
            accumulate_metric_sums(nonselected_totals[block_index], metrics, is_not_selected)
            selected_baseline_counts[block_index].update(
                str(item) for item in metrics["best_baseline_name"][is_selected]
            )
            range_start, range_end = block_range(block_index, block_size, pred_len)
            for row_index in np.where(is_selected)[0]:
                selected_records.append(
                    {
                        "dataset": dataset,
                        "block_index": block_index,
                        "block_range": f"{range_start}-{range_end}",
                        "future_region": block_group(block_index, block_size, pred_len),
                        "best_baseline": str(metrics["best_baseline_name"][row_index]),
                        **{
                            name: float(metrics[name][row_index])
                            for name in metric_names
                        },
                    }
                )

    classify_selected_units(selected_records)

    block_rows: list[dict[str, Any]] = []
    for block_index in range(n_blocks):
        range_start, range_end = block_range(block_index, block_size, pred_len)
        row: dict[str, Any] = {
            "dataset": dataset,
            "block_index": block_index,
            "block_range": f"{range_start}-{range_end}",
            "future_region": block_group(block_index, block_size, pred_len),
            "train_windows": int(block_counts[block_index]),
            "selected_count": int(selected_counts[block_index]),
            "selected_rate": float(selected_counts[block_index] / max(block_counts[block_index], 1)),
            "best_baseline_mode": baseline_counts[block_index].most_common(1)[0][0],
            "selected_best_baseline_mode": (
                selected_baseline_counts[block_index].most_common(1)[0][0]
                if selected_baseline_counts[block_index]
                else "none"
            ),
        }
        for name in metric_names:
            row[f"mean_{name}"] = block_totals[block_index][name] / max(block_counts[block_index], 1)
            row[f"selected_mean_{name}"] = (
                selected_totals[block_index][name] / max(selected_counts[block_index], 1)
            )
            row[f"nonselected_mean_{name}"] = (
                nonselected_totals[block_index][name] / max(nonselected_counts[block_index], 1)
            )
        block_rows.append(row)

    bucket_rows = summarize_selected_records(selected_records)
    selection_rows = summarize_selection(block_rows, selected_records)
    return block_rows, bucket_rows, selection_rows


def summarize_selected_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(row["dataset"], row["future_region"], row["bucket"]) for row in records})
    total_by_dataset_region = Counter((row["dataset"], row["future_region"]) for row in records)
    for dataset, region, bucket in keys:
        subset = [
            row for row in records
            if row["dataset"] == dataset and row["future_region"] == region and row["bucket"] == bucket
        ]
        baseline_counter = Counter(row["best_baseline"] for row in subset)
        output.append(
            {
                "dataset": dataset,
                "future_region": region,
                "bucket": bucket,
                "selected_units": len(subset),
                "region_bucket_share": len(subset) / max(total_by_dataset_region[(dataset, region)], 1),
                "mean_best_gain_over_persistence": mean(float(row["best_gain_over_persistence"]) for row in subset),
                "mean_residual_smoothness": mean(float(row["residual_smoothness"]) for row in subset),
                "mean_local_variation": mean(float(row["local_variation"]) for row in subset),
                "best_baseline_mode": baseline_counter.most_common(1)[0][0],
            }
        )
    return output


def summarize_selection(
    block_rows: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    region_order = {"early_1_96": 0, "middle_97_336": 1}
    regions = sorted(
        {str(row["future_region"]) for row in block_rows},
        key=lambda region: (region_order.get(region, 2), region),
    )
    for dataset in sorted({row["dataset"] for row in block_rows}):
        selected = [row for row in selected_records if row["dataset"] == dataset]
        for region in regions:
            region_blocks = [row for row in block_rows if row["dataset"] == dataset and row["future_region"] == region]
            region_selected = [row for row in selected if row["future_region"] == region]
            rows.append(
                {
                    "dataset": dataset,
                    "future_region": region,
                    "blocks": len(region_blocks),
                    "selected_units": len(region_selected),
                    "selected_share": len(region_selected) / max(len(selected), 1),
                    "mean_selected_gain_over_persistence": (
                        mean(float(row["best_gain_over_persistence"]) for row in region_selected)
                        if region_selected
                        else 0.0
                    ),
                    "mean_selected_residual_smoothness": (
                        mean(float(row["residual_smoothness"]) for row in region_selected)
                        if region_selected
                        else 0.0
                    ),
                    "mean_selected_local_variation": (
                        mean(float(row["local_variation"]) for row in region_selected)
                        if region_selected
                        else 0.0
                    ),
                }
            )
    return rows


def format_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def write_report(
    path: Path,
    selection_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
) -> None:
    bucket_lookup = {
        (row["dataset"], row["future_region"], row["bucket"]): row
        for row in bucket_rows
    }
    lines = [
        "# Phase4 Residual-Stability Diagnostic",
        "",
        "## 11-Step Record",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | Step 5/6: redefine predictability proxy and dynamic router design evidence |",
        "| `problem` | Fixed late adapter routing failed because it cannot tell learnable conflict from noisy conflict |",
        "| `existence_evidence` | S1/S2/RG-A gates, gradient conflict diagnostic, train-only label residuals |",
        "| `idea` | Use train-side baseline residual stability to decide whether a future unit should update adapter, shared path, or be shielded |",
        "| `theory_check` | Seasonal/baseline residual gain plus smoothness can separate structured residuals from high-variation noisy residuals |",
        "| `design` | Compare persistence with seasonal baselines over 48-step blocks; summarize selected high-novelty units by residual-stability buckets |",
        "| `gate` | Weather late selected units should show a high noisy-conflict share; ETTh2 should retain more learnable-conflict units |",
        f"| `artifacts` | `{path.parent}` |",
        "| `decision` | Positive design evidence; advance to residual-stability dynamic routing rather than sweeping fixed late adapter |",
        "",
        "## Selection Summary",
        "",
        "| Dataset | Region | Selected share | Gain over persistence | Residual smoothness | Local variation |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in selection_rows:
        lines.append(
            "| `{dataset}` | `{region}` | {share} | {gain:.3f} | {smooth:.3f} | {variation:.3f} |".format(
                dataset=row["dataset"],
                region=row["future_region"],
                share=format_pct(float(row["selected_share"])),
                gain=float(row["mean_selected_gain_over_persistence"]),
                smooth=float(row["mean_selected_residual_smoothness"]),
                variation=float(row["mean_selected_local_variation"]),
            )
        )

    lines += [
        "",
        "## Bucket Summary",
        "",
        "| Dataset | Region | Bucket | Share within region | Selected units | Gain | Smoothness | Variation | Baseline mode |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in bucket_rows:
        lines.append(
            "| `{dataset}` | `{region}` | `{bucket}` | {share} | {units} | {gain:.3f} | {smooth:.3f} | {variation:.3f} | `{baseline}` |".format(
                dataset=row["dataset"],
                region=row["future_region"],
                bucket=row["bucket"],
                share=format_pct(float(row["region_bucket_share"])),
                units=row["selected_units"],
                gain=float(row["mean_best_gain_over_persistence"]),
                smooth=float(row["mean_residual_smoothness"]),
                variation=float(row["mean_local_variation"]),
                baseline=row["best_baseline_mode"],
            )
        )

    late_regions = sorted({row["future_region"] for row in bucket_rows if str(row["future_region"]).startswith("late_337_")})
    late_region = late_regions[0] if late_regions else "late_337_720"
    weather_late_noisy = bucket_lookup.get(("Weather", late_region, "noisy_conflict"))
    weather_late_learnable = bucket_lookup.get(("Weather", late_region, "learnable_conflict"))
    etth2_late_noisy = bucket_lookup.get(("ETTh2", late_region, "noisy_conflict"))
    etth2_late_learnable = bucket_lookup.get(("ETTh2", late_region, "learnable_conflict"))
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if weather_late_noisy is not None and weather_late_learnable is not None:
        lines.append(
            "[Fact] Weather late selected units: noisy-conflict share `{noisy}`, learnable-conflict share `{learnable}`.".format(
                noisy=format_pct(float(weather_late_noisy["region_bucket_share"])),
                learnable=format_pct(float(weather_late_learnable["region_bucket_share"])),
            )
        )
    if etth2_late_noisy is not None and etth2_late_learnable is not None:
        lines.append(
            "[Fact] ETTh2 late selected units: noisy-conflict share `{noisy}`, learnable-conflict share `{learnable}`.".format(
                noisy=format_pct(float(etth2_late_noisy["region_bucket_share"])),
                learnable=format_pct(float(etth2_late_learnable["region_bucket_share"])),
            )
        )
    lines += [
        "",
        "[Decision Rule] If Weather late noisy-conflict share is high, a fixed late adapter is too broad; the next strategy should route only learnable-conflict units to adapter and suppress noisy-conflict units.",
        "",
        "[Decision Rule] If ETTh2 retains a material learnable-conflict share, the router must not suppress all hard or late units; otherwise it will erase the positive S1/RG-A signal.",
        "",
        "## Next Design Implication",
        "",
        "[Decision] The next method should be `dynamic_residual_stability_routing`: dense base remains full 720; selected units are bucketed by residual stability; learnable-conflict units train adapter; noisy-conflict units do not update shared path and receive reduced or zero auxiliary pressure.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_periods(value: str) -> list[int]:
    periods = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not periods:
        raise ValueError("At least one seasonal period is required.")
    return periods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose residual stability for Phase4 dynamic routing.")
    parser.add_argument("--dataset-root", default="/Users/river/PaperResearch/Project/datasets")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--analysis-root", default="analysis/phase4_residual_stability_diagnostic_20260625")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=720)
    parser.add_argument("--block-size", type=int, default=48)
    parser.add_argument("--top-ratio", type=float, default=0.25)
    parser.add_argument("--seasonal-periods", default="24,48,96,168")
    parser.add_argument("--chunk-windows", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.top_ratio <= 1:
        raise ValueError("top ratio must be in (0, 1].")
    periods = parse_periods(args.seasonal_periods)
    output_dir = Path(args.analysis_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]

    block_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        blocks, buckets, selections = analyze_dataset(
            Path(args.dataset_root),
            dataset,
            args.seq_len,
            args.pred_len,
            args.block_size,
            args.top_ratio,
            periods,
            args.chunk_windows,
        )
        block_rows.extend(blocks)
        bucket_rows.extend(buckets)
        selection_rows.extend(selections)

    write_csv(output_dir / "phase4_residual_stability_block_summary.csv", block_rows)
    write_csv(output_dir / "phase4_residual_stability_bucket_summary.csv", bucket_rows)
    write_csv(output_dir / "phase4_residual_stability_selection_summary.csv", selection_rows)
    write_report(
        output_dir / "phase4_residual_stability_diagnostic_report.md",
        selection_rows,
        bucket_rows,
    )
    print(output_dir / "phase4_residual_stability_diagnostic_report.md")


if __name__ == "__main__":
    main()

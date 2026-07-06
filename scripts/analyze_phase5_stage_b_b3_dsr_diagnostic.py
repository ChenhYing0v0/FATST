from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from analyze_phase5_stage_b_reliability_diagnostic import (
    DATASET_META,
    DATASETS,
    DEFAULT_RUN_NAME,
    PRED_LEN,
    SEQ_LEN,
    correlation,
    load_scaled_train_data,
    rank,
    residualize_by_step,
    run_dir,
    write_csv,
)

BLOCK_SIZES = (24, 48, 96)
PROXIES = ("seasonal_residual", "label_novelty", "local_variation", "step_index", "shuffled_seasonal")
RESIDUAL_LABELS = ("linear_step_residual", "rank_step_residual", "prefix_normalized_residual")
BOOTSTRAP_ITERATIONS = 1000
RANDOM_SEED = 20260706


def unit_bounds(block_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + block_size, PRED_LEN)) for start in range(0, PRED_LEN, block_size)]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def top_overlap(proxy_values: list[float], label_values: list[float]) -> float:
    count = max(1, math.ceil(len(label_values) * 0.25))
    proxy_top = set(sorted(range(len(proxy_values)), key=lambda index: proxy_values[index], reverse=True)[:count])
    label_top = set(sorted(range(len(label_values)), key=lambda index: label_values[index], reverse=True)[:count])
    return len(proxy_top & label_top) / count


def finite_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def finite_mean(values: list[float]) -> float:
    finite = finite_values(values)
    if not finite:
        return float("nan")
    return mean(finite)


def finite_min(values: list[float]) -> float:
    finite = finite_values(values)
    if not finite:
        return float("nan")
    return min(finite)


def finite_max(values: list[float]) -> float:
    finite = finite_values(values)
    if not finite:
        return float("nan")
    return max(finite)


def centered_neighbor_ratio(values: list[float]) -> list[float]:
    residuals: list[float] = []
    for index, value in enumerate(values):
        if len(values) == 1:
            baseline = value
        elif index == 0:
            baseline = values[1]
        elif index == len(values) - 1:
            baseline = values[-2]
        else:
            baseline = 0.5 * (values[index - 1] + values[index + 1])
        if baseline == 0:
            residuals.append(float("nan"))
        else:
            residuals.append((value / baseline) - 1.0)
    return residuals


def residual_labels(mses: list[float], starts: list[float]) -> dict[str, list[float]]:
    return {
        "linear_step_residual": residualize_by_step(mses, starts),
        "rank_step_residual": residualize_by_step(rank(mses), rank(starts)),
        "prefix_normalized_residual": centered_neighbor_ratio(mses),
    }


def prediction_unit_mses(raw_root: Path, run_name: str, dataset: str) -> dict[int, list[dict[str, Any]]]:
    prediction_path = run_dir(raw_root, run_name, dataset) / "predictions_test.npz"
    with np.load(prediction_path) as payload:
        pred = np.asarray(payload["pred"], dtype=np.float64)
        true = np.asarray(payload["true"], dtype=np.float64)

    rows_by_block: dict[int, list[dict[str, Any]]] = {}
    for block_size in BLOCK_SIZES:
        rows: list[dict[str, Any]] = []
        for start, end in unit_bounds(block_size):
            diff = pred[:, start:end, :] - true[:, start:end, :]
            sample_mse = np.mean(diff * diff, axis=(1, 2))
            rows.append(
                {
                    "dataset": dataset,
                    "block_size": block_size,
                    "unit_start": start,
                    "unit_end": end,
                    "unit_len": end - start,
                    "mse": float(np.mean(sample_mse)),
                    "mae": float(np.mean(np.abs(diff))),
                    "sample_error_std": float(np.std(sample_mse)),
                }
            )
        rows_by_block[block_size] = rows
    del pred, true
    return rows_by_block


def train_proxy_by_offset(train_data: np.ndarray, dataset: str) -> dict[str, np.ndarray]:
    num_windows = len(train_data) - SEQ_LEN - PRED_LEN + 1
    if num_windows <= 0:
        raise ValueError(f"{dataset} train split is shorter than seq_len + pred_len")
    period = int(DATASET_META[dataset]["period"])
    values = {proxy: np.zeros(PRED_LEN, dtype=np.float64) for proxy in PROXIES[:3]}
    history_last = train_data[SEQ_LEN - 1 : SEQ_LEN - 1 + num_windows]
    for offset in range(PRED_LEN):
        future = train_data[SEQ_LEN + offset : SEQ_LEN + offset + num_windows]
        previous = train_data[SEQ_LEN + offset - 1 : SEQ_LEN + offset - 1 + num_windows]
        seasonal_reference = train_data[SEQ_LEN + offset - period : SEQ_LEN + offset - period + num_windows]
        values["label_novelty"][offset] = float(np.mean((future - history_last) ** 2))
        values["local_variation"][offset] = float(np.mean((future - previous) ** 2))
        values["seasonal_residual"][offset] = float(np.mean((future - seasonal_reference) ** 2))
    return values


def train_proxy_units(dataset_root: Path, dataset: str) -> dict[int, dict[str, list[float]]]:
    train_data = load_scaled_train_data(dataset_root, dataset)
    proxy_by_offset = train_proxy_by_offset(train_data, dataset)
    rng = np.random.default_rng(RANDOM_SEED + DATASETS.index(dataset))
    values_by_block: dict[int, dict[str, list[float]]] = {}
    for block_size in BLOCK_SIZES:
        block_values = {proxy: [] for proxy in PROXIES}
        for start, end in unit_bounds(block_size):
            block_values["seasonal_residual"].append(float(np.mean(proxy_by_offset["seasonal_residual"][start:end])))
            block_values["label_novelty"].append(float(np.mean(proxy_by_offset["label_novelty"][start:end])))
            block_values["local_variation"].append(float(np.mean(proxy_by_offset["local_variation"][start:end])))
            block_values["step_index"].append(float(start))
        shuffled = list(block_values["seasonal_residual"])
        rng.shuffle(shuffled)
        block_values["shuffled_seasonal"] = shuffled
        values_by_block[block_size] = block_values
    return values_by_block


def bootstrap_rhos(
    proxy_values: list[float],
    label_values: list[float],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> tuple[int, float, float, float, float, float]:
    rng = np.random.default_rng(seed)
    rhos: list[float] = []
    count = len(proxy_values)
    for _ in range(iterations):
        indices = rng.integers(0, count, size=count)
        xs = [proxy_values[index] for index in indices]
        ys = [label_values[index] for index in indices]
        rho = correlation(rank(xs), rank(ys))
        if not math.isnan(rho):
            rhos.append(rho)
    if not rhos:
        return 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    arr = np.asarray(rhos, dtype=np.float64)
    return (
        len(rhos),
        float(np.mean(arr)),
        float(np.quantile(arr, 0.05)),
        float(np.quantile(arr, 0.50)),
        float(np.quantile(arr, 0.95)),
        float(np.mean(arr > 0.0)),
    )


def collect_diagnostic_rows(
    raw_root: Path,
    run_name: str,
    dataset_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    unit_rows: list[dict[str, Any]] = []
    detrending_rows: list[dict[str, Any]] = []
    blocksize_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        predictions_by_block = prediction_unit_mses(raw_root, run_name, dataset)
        proxies_by_block = train_proxy_units(dataset_root, dataset)
        for block_size in BLOCK_SIZES:
            prediction_rows = predictions_by_block[block_size]
            mses = [row["mse"] for row in prediction_rows]
            starts = [float(row["unit_start"]) for row in prediction_rows]
            labels = residual_labels(mses, starts)
            proxies = proxies_by_block[block_size]
            for unit_index, prediction_row in enumerate(prediction_rows):
                row = dict(prediction_row)
                for label_name, values in labels.items():
                    row[label_name] = values[unit_index]
                for proxy_name, values in proxies.items():
                    row[proxy_name] = values[unit_index]
                unit_rows.append(format_unit_row(row))

            seasonal_positive_count = 0
            seasonal_rhos: list[float] = []
            step_rhos: list[float] = []
            shuffled_rhos: list[float] = []
            for label_name, label_values in labels.items():
                for proxy_name, proxy_values in proxies.items():
                    spearman = correlation(rank(proxy_values), rank(label_values))
                    pearson = correlation(proxy_values, label_values)
                    overlap = top_overlap(proxy_values, label_values)
                    detrending_rows.append(
                        {
                            "dataset": dataset,
                            "block_size": block_size,
                            "residual_label": label_name,
                            "proxy": proxy_name,
                            "num_units": len(label_values),
                            "spearman_proxy_residual": f"{spearman:.6f}",
                            "pearson_proxy_residual": f"{pearson:.6f}",
                            "top_quartile_overlap": f"{overlap:.6f}",
                            "proxy_mean": f"{mean(proxy_values):.10f}",
                            "residual_mean": f"{mean(label_values):.10f}",
                        }
                    )
                    if proxy_name == "seasonal_residual":
                        seasonal_rhos.append(spearman)
                        if math.isfinite(spearman) and spearman > 0.0:
                            seasonal_positive_count += 1
                        valid, mean_rho, p05, p50, p95, positive_frac = bootstrap_rhos(
                            proxy_values,
                            label_values,
                            seed=RANDOM_SEED + block_size + DATASETS.index(dataset) * 100 + len(label_name),
                        )
                        bootstrap_rows.append(
                            {
                                "dataset": dataset,
                                "block_size": block_size,
                                "residual_label": label_name,
                                "proxy": proxy_name,
                                "num_units": len(label_values),
                                "valid_bootstrap_iterations": valid,
                                "bootstrap_mean_spearman": f"{mean_rho:.6f}",
                                "bootstrap_p05_spearman": f"{p05:.6f}",
                                "bootstrap_p50_spearman": f"{p50:.6f}",
                                "bootstrap_p95_spearman": f"{p95:.6f}",
                                "bootstrap_positive_fraction": f"{positive_frac:.6f}",
                            }
                        )
                    elif proxy_name == "step_index":
                        step_rhos.append(spearman)
                    elif proxy_name == "shuffled_seasonal":
                        shuffled_rhos.append(spearman)

            blocksize_rows.append(
                {
                    "dataset": dataset,
                    "block_size": block_size,
                    "num_units": len(mses),
                    "seasonal_positive_labels": seasonal_positive_count,
                    "seasonal_mean_spearman": f"{finite_mean(seasonal_rhos):.6f}",
                    "seasonal_min_spearman": f"{finite_min(seasonal_rhos):.6f}",
                    "seasonal_max_spearman": f"{finite_max(seasonal_rhos):.6f}",
                    "step_index_mean_abs_spearman": f"{finite_mean([abs(value) for value in step_rhos]):.6f}",
                    "shuffled_mean_abs_spearman": f"{finite_mean([abs(value) for value in shuffled_rhos]):.6f}",
                    "seasonal_beats_step_abs": int(
                        abs(finite_mean(seasonal_rhos)) > finite_mean([abs(value) for value in step_rhos])
                    ),
                    "seasonal_beats_shuffled_abs": int(
                        abs(finite_mean(seasonal_rhos)) > finite_mean([abs(value) for value in shuffled_rhos])
                    ),
                }
            )
    return unit_rows, detrending_rows, blocksize_rows, bootstrap_rows


def format_unit_row(row: dict[str, Any]) -> dict[str, Any]:
    formatted: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.10f}"
        else:
            formatted[key] = value
    return formatted


def gate_decision(
    blocksize_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    dataset_positive_counts = {dataset: 0 for dataset in DATASETS}
    dataset_blocks = {dataset: 0 for dataset in DATASETS}
    for row in blocksize_rows:
        dataset = row["dataset"]
        dataset_blocks[dataset] += 1
        if int(row["seasonal_positive_labels"]) >= 2:
            dataset_positive_counts[dataset] += 1
    robust_datasets = [
        dataset
        for dataset in DATASETS
        if dataset_blocks[dataset] and dataset_positive_counts[dataset] >= 2
    ]
    if len(robust_datasets) >= 2:
        reasons.append(f"seasonal residual is positive under at least two labels on >=2 block sizes for {robust_datasets}")
    else:
        reasons.append(f"seasonal residual sign is not block-size robust enough: {dataset_positive_counts}")

    strong_contradictions = [
        row
        for row in blocksize_rows
        if math.isfinite(float(row["seasonal_mean_spearman"]))
        and float(row["seasonal_mean_spearman"]) <= -0.20
    ]
    if strong_contradictions:
        reasons.append(
            "strong negative seasonal mean rho appears in "
            + ", ".join(f"{row['dataset']}/b{row['block_size']}" for row in strong_contradictions)
        )
    else:
        reasons.append("no block has seasonal mean rho <= -0.20")

    step_failures = [
        row
        for row in blocksize_rows
        if math.isfinite(float(row["seasonal_mean_spearman"]))
        and int(row["seasonal_beats_step_abs"]) == 0
        and abs(float(row["seasonal_mean_spearman"])) < 0.20
    ]
    if step_failures:
        reasons.append(
            "seasonal signal does not clearly beat step control in "
            + ", ".join(f"{row['dataset']}/b{row['block_size']}" for row in step_failures)
        )
    else:
        reasons.append("seasonal signal is not trivially reproduced by step control under the current summary rule")

    weak_bootstrap = [
        row
        for row in bootstrap_rows
        if row["proxy"] == "seasonal_residual" and float(row["bootstrap_positive_fraction"]) < 0.60
    ]
    if weak_bootstrap:
        reasons.append(
            "bootstrap sign stability is weak in "
            + ", ".join(
                f"{row['dataset']}/b{row['block_size']}/{row['residual_label']}"
                for row in weak_bootstrap[:8]
            )
        )
    else:
        reasons.append("all seasonal bootstrap positive fractions are >= 0.60")

    if len(robust_datasets) >= 2 and not strong_contradictions and not weak_bootstrap:
        return "pass_problem_existence", reasons
    if len(robust_datasets) >= 1 and not strong_contradictions:
        return "partial_pass_needs_stronger_proxy_or_method_boundary", reasons
    return "fail_not_stable_enough_for_stage_b_method", reasons


def render_report(
    blocksize_rows: list[dict[str, Any]],
    detrending_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
) -> str:
    decision, reasons = gate_decision(blocksize_rows, bootstrap_rows)
    seasonal_rows = [
        row
        for row in detrending_rows
        if row["proxy"] == "seasonal_residual"
    ]
    lines: list[str] = [
        "# Phase5 StageB B3 Distance-Normalized Seasonal Residual Diagnostic",
        "",
        "`current_step`: StageB Step 2/3 B3 diagnostic.",
        "",
        "## Scope",
        "",
        "[Fact] This diagnostic reuses A6-LBF-r256 `predictions_test.npz` as held-out diagnostic labels and local train split labels for train-only proxies.",
        "",
        "[Boundary] No model code, loss code, or remote training is changed. Held-out residuals are labels for diagnostic evaluation only.",
        "",
        "## 11-step Record",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | StageB Step 2/3 B3 problem-existence diagnostic |",
        "| `problem` | Test whether train-only seasonal residual explains residual difficulty after controlling forecast distance |",
        "| `existence_evidence` | detrending robustness, block-size robustness, bootstrap sign stability |",
        "| `idea` | Replace raw future-unit reliability with distance-normalized structural residual reliability |",
        "| `theory_check` | A valid proxy should remain positive after step-distance trend removal and should not be reproduced by pure step index |",
        "| `design` | Post-hoc diagnostic over block sizes 24/48/96 and residual labels linear/rank/prefix-normalized |",
        f"| `narrative_gate` | {decision} |",
        "| `effectiveness_gate` | not applicable; no new method trained |",
        "| `artifacts` | this directory |",
        f"| `decision` | {decision}; do not implement B3 method unless this is upgraded by follow-up evidence |",
        "",
        "## Seasonal Residual Alignment",
        "",
        "| Dataset | Block | Residual Label | Spearman | Pearson | Top-Q Overlap |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in seasonal_rows:
        lines.append(
            f"| {row['dataset']} | {row['block_size']} | `{row['residual_label']}` | "
            f"`{float(row['spearman_proxy_residual']):.2f}` | "
            f"`{float(row['pearson_proxy_residual']):.2f}` | "
            f"`{float(row['top_quartile_overlap']):.2f}` |"
        )
    lines.extend(
        [
            "",
        "## Block-Size Robustness",
        "",
        "| Dataset | Block | Positive Labels | Mean Spearman | Min Spearman | Step Abs Mean | Shuffled Abs Mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in blocksize_rows:
        lines.append(
            f"| {row['dataset']} | {row['block_size']} | {row['seasonal_positive_labels']}/3 | "
            f"`{float(row['seasonal_mean_spearman']):.2f}` | "
            f"`{float(row['seasonal_min_spearman']):.2f}` | "
            f"`{float(row['step_index_mean_abs_spearman']):.2f}` | "
            f"`{float(row['shuffled_mean_abs_spearman']):.2f}` |"
        )
    lines.extend(
        [
            "",
            "[Note] `nan` means the residual label has no rank variance after detrending, usually because unit MSE is perfectly monotonic with step under that block size. This is treated as missing alignment evidence, not as positive support.",
            "",
            "## Bootstrap Stability",
            "",
            "| Dataset | Block | Residual Label | Mean Rho | P05 | P50 | P95 | Positive Fraction |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in bootstrap_rows:
        lines.append(
            f"| {row['dataset']} | {row['block_size']} | `{row['residual_label']}` | "
            f"`{float(row['bootstrap_mean_spearman']):.2f}` | "
            f"`{float(row['bootstrap_p05_spearman']):.2f}` | "
            f"`{float(row['bootstrap_p50_spearman']):.2f}` | "
            f"`{float(row['bootstrap_p95_spearman']):.2f}` | "
            f"`{float(row['bootstrap_positive_fraction']):.2f}` |"
        )
    lines.extend(
        [
            "",
            "## Gate Reasons",
            "",
        ]
    )
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"[Decision] B3 diagnostic decision: `{decision}`.",
            "",
            "[Interpretation] B3 should only advance if the signal is robust across detrending forms, block sizes, and bootstrap sign checks. A positive single table is not enough for a StageB method.",
            "",
            "[Rollback Point] If the decision is not `pass_problem_existence`, do not implement reliability-aware loss weighting. Either strengthen the train-only structural proxy or close StageB and move to a broader label-autocorrelation objective stage.",
            "",
            "## Output Files",
            "",
            "- `stage_b_b3_unit_residuals.csv`",
            "- `stage_b_b3_detrending_robustness.csv`",
            "- `stage_b_b3_blocksize_robustness.csv`",
            "- `stage_b_b3_bootstrap_stability.csv`",
            "- `stage_b_b3_report.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze B3 distance-normalized seasonal residual reliability.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/raw/official-last"),
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/phase5_stage_b_distance_normalized_seasonal_residual_20260706"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/Users/river/PaperResearch/Project/datasets"),
    )
    args = parser.parse_args()

    unit_rows, detrending_rows, blocksize_rows, bootstrap_rows = collect_diagnostic_rows(
        args.raw_root,
        args.run_name,
        args.dataset_root,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage_b_b3_unit_residuals.csv", unit_rows)
    write_csv(args.output_dir / "stage_b_b3_detrending_robustness.csv", detrending_rows)
    write_csv(args.output_dir / "stage_b_b3_blocksize_robustness.csv", blocksize_rows)
    write_csv(args.output_dir / "stage_b_b3_bootstrap_stability.csv", bootstrap_rows)
    report = render_report(blocksize_rows, detrending_rows, bootstrap_rows)
    (args.output_dir / "stage_b_b3_report.md").write_text(report)
    print(args.output_dir / "stage_b_b3_report.md")


if __name__ == "__main__":
    main()

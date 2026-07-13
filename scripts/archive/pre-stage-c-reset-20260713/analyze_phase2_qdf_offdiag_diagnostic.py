from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="fatst-mpl-"))

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baselines" / "patch_encoder_target_set_decoder"
sys.path.insert(0, str(BASELINE_ROOT))

from dataset import ForecastDataset  # noqa: E402


DATASETS = ["ETTh2", "ETTm1", "Weather"]
SEGMENTS = [(1, 96), (97, 192), (193, 336), (337, 720)]
SEGMENT_LABELS = [f"{start}-{end}" for start, end in SEGMENTS]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def region_target_matrix(data: np.ndarray, seq_len: int, pred_len: int) -> np.ndarray:
    n_windows = len(data) - seq_len - pred_len + 1
    if n_windows <= 0:
        raise ValueError("Dataset split is shorter than seq_len + pred_len.")

    cumsum = np.vstack([np.zeros((1, data.shape[1]), dtype=np.float64), data.cumsum(axis=0, dtype=np.float64)])
    window_index = np.arange(n_windows)
    target_start = window_index + seq_len
    regions = []
    for start, end in SEGMENTS:
        start_index = target_start + start - 1
        end_index = target_start + end
        region_sum = cumsum[end_index] - cumsum[start_index]
        regions.append(region_sum / float(end - start + 1))

    # QDF loss flattens errors from [B, P, D] to [B*D, P].
    # We mirror that axis semantics at region granularity: [B*D, 4].
    return np.stack(regions, axis=-1).reshape(-1, len(SEGMENTS))


def safe_corrcoef(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(centered * centered, axis=0, keepdims=True))
    denom[denom == 0.0] = 1.0
    normalized = centered / denom
    return normalized.T @ normalized


def matrix_rows(dataset: str, matrix_name: str, matrix: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for i, row_label in enumerate(SEGMENT_LABELS):
        for j, col_label in enumerate(SEGMENT_LABELS):
            rows.append(
                {
                    "dataset": dataset,
                    "matrix": matrix_name,
                    "row_segment": row_label,
                    "col_segment": col_label,
                    "row_index": i,
                    "col_index": j,
                    "value": float(matrix[i, j]),
                    "abs_value": abs(float(matrix[i, j])),
                    "is_offdiag": i != j,
                }
            )
    return rows


def summarize_matrix(dataset: str, region_matrix: np.ndarray, corr: np.ndarray, cov: np.ndarray) -> dict[str, object]:
    offdiag_mask = ~np.eye(corr.shape[0], dtype=bool)
    eigvals = np.linalg.eigvalsh(cov)
    positive_eigvals = eigvals[eigvals > 1e-12]
    condition_number = float(eigvals.max() / positive_eigvals.min()) if len(positive_eigvals) else float("inf")
    corr_offdiag_energy = float(np.sum(np.square(corr[offdiag_mask])))
    corr_total_energy = float(np.sum(np.square(corr)))
    cov_offdiag_energy = float(np.sum(np.square(cov[offdiag_mask])))
    cov_total_energy = float(np.sum(np.square(cov)))
    return {
        "dataset": dataset,
        "samples": int(region_matrix.shape[0]),
        "regions": int(region_matrix.shape[1]),
        "mean_abs_offdiag_corr": float(np.mean(np.abs(corr[offdiag_mask]))),
        "max_abs_offdiag_corr": float(np.max(np.abs(corr[offdiag_mask]))),
        "offdiag_corr_fro_share": corr_offdiag_energy / max(corr_total_energy, 1e-12),
        "offdiag_cov_fro_share": cov_offdiag_energy / max(cov_total_energy, 1e-12),
        "min_cov_eig": float(eigvals.min()),
        "max_cov_eig": float(eigvals.max()),
        "cov_condition_number": condition_number,
    }


def analyze_dataset(dataset_root: Path, dataset: str, seq_len: int, pred_len: int) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    train_set = ForecastDataset(dataset_root, dataset, "train", seq_len, pred_len)
    region_matrix = region_target_matrix(train_set.data.astype(np.float64), seq_len, pred_len)
    corr = safe_corrcoef(region_matrix)
    cov = np.cov(region_matrix, rowvar=False)
    summary = summarize_matrix(dataset, region_matrix, corr, cov)
    rows = matrix_rows(dataset, "correlation", corr) + matrix_rows(dataset, "covariance", cov)
    return summary, rows, corr


def plot_heatmaps(output_root: Path, matrices: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.2 * len(matrices), 3.8), squeeze=False)
    for axis_index, (ax, (dataset, corr)) in enumerate(zip(axes[0], matrices.items(), strict=True)):
        im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_title(dataset)
        ax.set_xticks(range(len(SEGMENT_LABELS)), SEGMENT_LABELS, rotation=35, ha="right")
        if axis_index == 0:
            ax.set_yticks(range(len(SEGMENT_LABELS)), SEGMENT_LABELS)
        else:
            ax.set_yticks(range(len(SEGMENT_LABELS)), [])
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
    fig.savefig(output_root / "qdf_region_correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_gate(
    dataset_summaries: list[dict[str, object]],
    step_covariance_summary: dict[str, object],
    novelty_summary: dict[str, object],
    offdiag_threshold: float,
) -> dict[str, object]:
    mean_abs_offdiag = float(np.mean([float(row["mean_abs_offdiag_corr"]) for row in dataset_summaries]))
    min_dataset_offdiag = float(min(float(row["mean_abs_offdiag_corr"]) for row in dataset_summaries))
    diagonal_gate = step_covariance_summary.get("gate", {})
    novelty_gate = novelty_summary.get("gate", {})
    diagonal_proxy_failed = isinstance(diagonal_gate, dict) and not bool(diagonal_gate.get("pass", False))
    novelty_supported_diagonal = isinstance(novelty_gate, dict) and bool(
        novelty_gate.get("supports_step_covariance_balanced", False)
    )
    offdiag_signal_strong = mean_abs_offdiag >= offdiag_threshold and min_dataset_offdiag >= offdiag_threshold * 0.5
    return {
        "mean_abs_offdiag_corr": mean_abs_offdiag,
        "min_dataset_mean_abs_offdiag_corr": min_dataset_offdiag,
        "offdiag_threshold": offdiag_threshold,
        "offdiag_signal_strong": offdiag_signal_strong,
        "diagonal_proxy_failed": diagonal_proxy_failed,
        "novelty_supported_diagonal_before_training": novelty_supported_diagonal,
        "supports_qdf_upstream_reproduction": offdiag_signal_strong
        and diagonal_proxy_failed
        and novelty_supported_diagonal,
    }


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.{digits}f}"
    return str(value)


def write_report(
    output_root: Path,
    dataset_summaries: list[dict[str, object]],
    gate: dict[str, object],
) -> None:
    decision = (
        "[Decision] 进入 QDF upstream reproduction gate：完整 QDF/off-diagonal 机制值得作为下一步实验路径。"
        if gate["supports_qdf_upstream_reproduction"]
        else "[Decision] 暂不进入 QDF upstream reproduction gate：off-diagonal 证据或前置失败链不足。"
    )
    lines = [
        "# Phase2-D QDF Off-Diagonal Diagnostic",
        "",
        "## 11-Step Loop Position",
        "",
        "- `current_step`: Step 2-3 rollback check。",
        "- `problem`: diagonal / static objective proxy 已失败，但 QDF 的完整 quadratic objective 仍可能依赖 off-diagonal future-step dependence。",
        "- `existence_evidence`: 本诊断从 train split labels 估计 future-region correlation/covariance，不训练模型。",
        "- `idea`: 若 future regions 存在强 off-diagonal dependence，且 diagonal proxy 已失败，则下一步不应继续调 diagonal weights，而应 native reproduce QDF full/off-diagonal baseline。",
        "",
        "## Decision",
        "",
        decision,
        "",
        "## Dataset Summary",
        "",
        "| Dataset | Samples `[B*D]` | Mean abs offdiag corr | Max abs offdiag corr | Offdiag corr Fro share | Cov condition number |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dataset_summaries:
        lines.append(
            "| "
            f"{row['dataset']} | {row['samples']} | "
            f"{fmt(row['mean_abs_offdiag_corr'])} | "
            f"{fmt(row['max_abs_offdiag_corr'])} | "
            f"{fmt(row['offdiag_corr_fro_share'])} | "
            f"{fmt(row['cov_condition_number'])} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- mean_abs_offdiag_corr: `{fmt(gate['mean_abs_offdiag_corr'])}`",
            f"- min_dataset_mean_abs_offdiag_corr: `{fmt(gate['min_dataset_mean_abs_offdiag_corr'])}`",
            f"- offdiag_threshold: `{fmt(gate['offdiag_threshold'])}`",
            f"- offdiag_signal_strong: `{gate['offdiag_signal_strong']}`",
            f"- diagonal_proxy_failed: `{gate['diagonal_proxy_failed']}`",
            f"- novelty_supported_diagonal_before_training: `{gate['novelty_supported_diagonal_before_training']}`",
            f"- supports_qdf_upstream_reproduction: `{gate['supports_qdf_upstream_reproduction']}`",
            "",
            "## Interpretation",
            "",
            "[Fact] QDF 的实现把误差从 `[B, P, D]` 展平成 `[B*D, P]`，再用 learned quadratic matrix 计算 loss。",
            "本诊断按相同轴语义，把 H720 target 切成四个 future regions 后形成 `[B*D, 4]` 矩阵。",
            "",
            "[Inference] 如果 off-diagonal correlation 很强，说明 future steps/regions 不是独立任务。",
            "这正是 static diagonal weighting 无法表达、而 QDF full/off-diagonal matrix 能表达的部分。",
            "",
            "[Counterargument] 该诊断只看 label-side dependence，不证明 learned QDF loss 一定提升 FATST carrier。",
            "因此下一步应先 native reproduce upstream QDF，而不是直接把 QDF module 移植进本 repo。",
            "",
            "## Artifacts",
            "",
            "- `qdf_offdiag_dataset_summary.csv`",
            "- `qdf_region_correlation_covariance.csv`",
            "- `qdf_offdiag_summary.json`",
            "- `qdf_region_correlation_heatmap.png`",
        ]
    )
    (output_root / "phase2_qdf_offdiag_diagnostic_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/Users/river/PaperResearch/Project/datasets")
    parser.add_argument("--output-root", default="analysis/phase2_qdf_offdiag_diagnostic_20260623")
    parser.add_argument("--step-covariance-summary", default="analysis/phase2_step_covariance_balanced_gate_20260623/phase2_step_covariance_balanced_summary.json")
    parser.add_argument("--novelty-summary", default="analysis/phase2_covariance_novelty_diagnostic_20260623/covariance_novelty_summary.json")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=720)
    parser.add_argument("--offdiag-threshold", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_summaries: list[dict[str, object]] = []
    matrix_rows_all: list[dict[str, object]] = []
    corr_matrices: dict[str, np.ndarray] = {}
    for dataset in DATASETS:
        summary, rows, corr = analyze_dataset(Path(args.dataset_root), dataset, args.seq_len, args.pred_len)
        dataset_summaries.append(summary)
        matrix_rows_all.extend(rows)
        corr_matrices[dataset] = corr

    step_covariance_summary = read_json(Path(args.step_covariance_summary))
    novelty_summary = read_json(Path(args.novelty_summary))
    gate = build_gate(dataset_summaries, step_covariance_summary, novelty_summary, args.offdiag_threshold)
    summary = {
        "datasets": dataset_summaries,
        "gate": gate,
        "inputs": {
            "dataset_root": str(Path(args.dataset_root)),
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "step_covariance_summary": args.step_covariance_summary,
            "novelty_summary": args.novelty_summary,
        },
    }

    write_csv(output_root / "qdf_offdiag_dataset_summary.csv", dataset_summaries)
    write_csv(output_root / "qdf_region_correlation_covariance.csv", matrix_rows_all)
    (output_root / "qdf_offdiag_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    plot_heatmaps(output_root, corr_matrices)
    write_report(output_root, dataset_summaries, gate)
    print(f"report={output_root / 'phase2_qdf_offdiag_diagnostic_report.md'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = ("ETTh2", "ETTm1", "Weather")
RANKS = (8, 16, 32, 64)
SEGMENTS = ((0, 96), (96, 192), (192, 336), (336, 720))
DEFAULT_ANALYSIS_ROOT = Path("analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707")
DEFAULT_RUN_ROOT = (
    DEFAULT_ANALYSIS_ROOT
    / "raw"
    / "TimeAlignOfficialUnified720_A6LBF_r256_main_official-last"
)


@dataclass(frozen=True)
class DatasetArtifacts:
    dataset: str
    run_dir: Path
    checkpoint: Path
    predictions: Path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def dct_basis(length: int) -> np.ndarray:
    steps = np.arange(length, dtype=np.float64) + 0.5
    freqs = np.arange(length, dtype=np.float64)
    basis = np.cos(np.pi * np.outer(steps, freqs) / length)
    basis[:, 0] *= np.sqrt(1.0 / length)
    basis[:, 1:] *= np.sqrt(2.0 / length)
    return basis


def orthonormalize(matrix: np.ndarray, rank: int) -> np.ndarray:
    if matrix.size == 0:
        return matrix.reshape(matrix.shape[0], 0)
    q, r = np.linalg.qr(matrix[:, : min(rank, matrix.shape[1])])
    diag = np.abs(np.diag(r))
    keep = diag > 1e-10
    return q[:, keep]


def learned_basis_left_space(checkpoint: Path) -> np.ndarray:
    import torch

    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    basis = state["learned_temporal_basis"].detach().cpu().numpy().astype(np.float64)
    left, _singular_values, _right = np.linalg.svd(basis, full_matrices=False)
    return left


def projection_energy(rows: np.ndarray, basis: np.ndarray) -> float:
    if basis.shape[1] == 0:
        return 0.0
    coeff = rows.astype(np.float64, copy=False) @ basis
    return float(np.sum(coeff * coeff))


def segment_projection_energy(
    residual_rows: np.ndarray,
    basis: np.ndarray,
    rank: int,
) -> tuple[float, list[dict[str, Any]]]:
    total_projected = 0.0
    segment_rows: list[dict[str, Any]] = []
    for start, end in SEGMENTS:
        segment = residual_rows[:, start:end]
        segment_basis = orthonormalize(basis[start:end], rank)
        projected = projection_energy(segment, segment_basis)
        total = float(np.sum(segment.astype(np.float64) ** 2))
        count = int(segment.size)
        residual_after = max(total - projected, 0.0)
        total_projected += projected
        segment_rows.append(
            {
                "segment_start": start,
                "segment_end": end,
                "rank": rank,
                "effective_rank": int(segment_basis.shape[1]),
                "base_mse": total / count,
                "oracle_mse": residual_after / count,
                "relative_mse_reduction_pct": 100.0 * projected / total if total > 0 else 0.0,
            }
        )
    return total_projected, segment_rows


def resolve_artifacts(run_root: Path, dataset: str) -> DatasetArtifacts:
    run_dir = run_root / dataset / "mixed_h96_h192_h336_h720" / "seed2021"
    checkpoint = run_dir / "checkpoint.pt"
    predictions = run_dir / "predictions_test.npz"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not predictions.exists():
        raise FileNotFoundError(predictions)
    return DatasetArtifacts(dataset, run_dir, checkpoint, predictions)


def analyze_dataset(run_root: Path, dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    artifacts = resolve_artifacts(run_root, dataset)
    payload = np.load(artifacts.predictions)
    residual = (payload["true"][:, :720, :] - payload["pred"][:, :720, :]).astype(np.float32)
    sample_count, horizon, channels = residual.shape
    residual_rows = np.moveaxis(residual, 2, 1).reshape(sample_count * channels, horizon)
    total_energy = float(np.sum(residual_rows.astype(np.float64) ** 2))
    total_count = int(residual_rows.size)
    base_mse = total_energy / total_count

    basis_spaces = {
        "learned_a6": learned_basis_left_space(artifacts.checkpoint),
        "dct": dct_basis(horizon),
    }

    summary_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []

    for basis_name, basis in basis_spaces.items():
        for rank in RANKS:
            global_basis = basis[:, :rank]
            global_projected = projection_energy(residual_rows, global_basis)
            global_mse = max(total_energy - global_projected, 0.0) / total_count

            segment_projected, basis_segment_rows = segment_projection_energy(residual_rows, basis, rank)
            segment_mse = max(total_energy - segment_projected, 0.0) / total_count
            incremental = max(segment_projected - global_projected, 0.0)

            summary_rows.append(
                {
                    "dataset": dataset,
                    "basis": basis_name,
                    "rank": rank,
                    "samples": sample_count,
                    "channels": channels,
                    "residual_rows": residual_rows.shape[0],
                    "base_mse": base_mse,
                    "global_oracle_mse": global_mse,
                    "segment_oracle_mse": segment_mse,
                    "global_reduction_pct": 100.0 * global_projected / total_energy if total_energy > 0 else 0.0,
                    "segment_reduction_pct": 100.0 * segment_projected / total_energy if total_energy > 0 else 0.0,
                    "segment_minus_global_reduction_pct": (
                        100.0 * incremental / total_energy if total_energy > 0 else 0.0
                    ),
                }
            )

            for row in basis_segment_rows:
                segment_rows.append({"dataset": dataset, "basis": basis_name, **row})

    return summary_rows, segment_rows


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for field in fields:
            value = row[field]
            if isinstance(value, float):
                if field.endswith("_pct"):
                    values.append(fmt_pct(value))
                else:
                    values.append(fmt_float(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path: Path, summary_rows: list[dict[str, Any]], segment_rows: list[dict[str, Any]]) -> None:
    rank32 = [row for row in summary_rows if row["rank"] == 32]
    rank64 = [row for row in summary_rows if row["rank"] == 64]
    learned_rank64 = [row for row in rank64 if row["basis"] == "learned_a6"]
    dct_rank64 = [row for row in rank64 if row["basis"] == "dct"]

    lines = [
        "# Phase5 StageB B8-OCD Coefficient-Space Oracle Diagnostic",
        "",
        "## 阶段记录",
        "",
        "| 字段 | 内容 |",
        "| --- | --- |",
        "| `candidate_id` | `B8-FQA` |",
        "| `diagnostic_id` | `B8-OCD` |",
        "| `current_step` | Step 3：problem-existence diagnostic |",
        "| `problem` | A6-LBF 的 `coeff[b,c]` 对 future positions 不变，可能限制不同 future segments 的 sample-specific 表示 |",
        "| `idea_under_test` | 固定 clean A6 prediction 与 learned temporal basis，测试 segment-specific coefficient correction 是否能显著降低 residual |",
        "| `decision` | 见文末，当前仅为 oracle diagnostic，不是 method result |",
        "",
        "## 诊断定义",
        "",
        "本诊断使用 clean A6-LBF-r256 的 `predictions_test.npz` 和 `checkpoint.pt`。对每个 dataset，先计算 denorm-space residual：",
        "",
        "```text",
        "residual[b, t, c] = true[b, t, c] - pred[b, t, c]",
        "```",
        "",
        "然后将 residual reshape 为 `[sample * channel, 720]`。诊断比较两种 oracle correction：",
        "",
        "- `global_oracle`：整段 720 steps 共用一组 correction coefficients；",
        "- `segment_oracle`：四个 segments `[0,96), [96,192), [192,336), [336,720)` 分别求 correction coefficients。",
        "",
        "为了避免 256 维 coefficient 在短 segment 上平凡拟合，本报告只使用 ranks `{8,16,32,64}`，并加入 `dct` control。",
        "",
        "## Rank 32 Summary",
        "",
        markdown_table(
            rank32,
            [
                "dataset",
                "basis",
                "rank",
                "base_mse",
                "global_reduction_pct",
                "segment_reduction_pct",
                "segment_minus_global_reduction_pct",
            ],
        ),
        "",
        "## Rank 64 Summary",
        "",
        markdown_table(
            rank64,
            [
                "dataset",
                "basis",
                "rank",
                "base_mse",
                "global_reduction_pct",
                "segment_reduction_pct",
                "segment_minus_global_reduction_pct",
            ],
        ),
        "",
        "## Learned Basis Rank 64 Segment Detail",
        "",
        markdown_table(
            [row for row in segment_rows if row["basis"] == "learned_a6" and row["rank"] == 64],
            [
                "dataset",
                "basis",
                "rank",
                "segment_start",
                "segment_end",
                "effective_rank",
                "base_mse",
                "oracle_mse",
                "relative_mse_reduction_pct",
            ],
        ),
        "",
        "## 初步判定",
        "",
    ]

    lines.extend(decision_lines(learned_rank64, dct_rank64))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def decision_lines(learned_rank64: list[dict[str, Any]], dct_rank64: list[dict[str, Any]]) -> list[str]:
    learned_by_dataset = {row["dataset"]: row for row in learned_rank64}
    dct_by_dataset = {row["dataset"]: row for row in dct_rank64}

    lines = [
        "[Fact] `segment_oracle` 在所有 dataset 上都应优于或等于 `global_oracle`，因为它放宽了 coefficient sharing；关键不是是否有提升，而是提升是否足够大、是否超过 DCT control。",
        "",
    ]
    pass_count = 0
    for dataset in DATASETS:
        learned = learned_by_dataset[dataset]
        dct = dct_by_dataset[dataset]
        learned_gap = learned["segment_minus_global_reduction_pct"]
        dct_gap = dct["segment_minus_global_reduction_pct"]
        learned_segment = learned["segment_reduction_pct"]
        dct_segment = dct["segment_reduction_pct"]
        better_than_dct = learned_gap > dct_gap + 1.0 and learned_segment >= dct_segment
        if learned_gap >= 5.0 and better_than_dct:
            pass_count += 1
        lines.append(
            f"- {dataset}: learned segment-minus-global gain = `{learned_gap:.2f}%`, "
            f"DCT control = `{dct_gap:.2f}%`; learned segment reduction = `{learned_segment:.2f}%`, "
            f"DCT segment reduction = `{dct_segment:.2f}%`."
        )

    lines.append("")
    if pass_count >= 2:
        lines.extend(
            [
                "[Decision] `B8-OCD` 暂定通过 problem-existence gate：至少两个 dataset 显示 learned basis 的 segment-specific coefficient headroom 明显超过 DCT control。",
                "",
                "[Next] 进入 Step 4-6，设计 lightweight future-query coefficient modulation，并保留零初始化/残差调制以避免破坏 A6 capacity。",
            ]
        )
    else:
        lines.extend(
            [
                "[Decision] `B8-OCD` 未通过 problem-existence gate：当前 evidence 不足以说明 B8-FQA 的 learned-basis coefficient interface 有强于 generic DCT/low-rank control 的 segment-specific headroom。",
                "",
                "[Rollback] 不实现 B8-FQA。StageB 应回到 Step 2/3，重新寻找 architecture-level problem，或将 B7-UPO 仅作为 small contribution candidate 保留。",
            ]
        )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B8-OCD coefficient-space oracle diagnostic.")
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_summary: list[dict[str, Any]] = []
    all_segments: list[dict[str, Any]] = []
    for dataset in args.datasets:
        summary_rows, segment_rows = analyze_dataset(args.run_root, dataset)
        all_summary.extend(summary_rows)
        all_segments.extend(segment_rows)

    write_csv(args.analysis_root / "b8_ocd_summary.csv", all_summary)
    write_csv(args.analysis_root / "b8_ocd_segment_detail.csv", all_segments)
    write_report(args.analysis_root / "b8_ocd_report.md", all_summary, all_segments)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = ("ETTh2", "ETTm1", "Weather")
SEGMENTS = (
    ("early_0_96", 0, 96),
    ("mid_96_192", 96, 192),
    ("late_192_336", 192, 336),
    ("tail_336_720", 336, 720),
)
RANKS = (8, 16, 32, 64)
DEFAULT_ANALYSIS_ROOT = Path("analysis/phase5_stage_b_b10_tsi_basis_geometry_20260708")
DEFAULT_CHECKPOINT_ROOT = (
    Path("analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707")
    / "raw"
    / "official-last"
    / "TimeAlignOfficialUnified720_a6_clean_official-last"
)


@dataclass(frozen=True)
class BasisGeometry:
    dataset: str
    checkpoint: Path
    basis: np.ndarray


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_path(root: Path, dataset: str) -> Path:
    return root / dataset / "mixed_h96_h192_h336_h720" / "seed2021" / "checkpoint.pt"


def load_basis(checkpoint: Path) -> np.ndarray:
    import torch

    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if "learned_temporal_basis" not in state:
        raise KeyError(f"checkpoint has no learned_temporal_basis: {checkpoint}")
    return state["learned_temporal_basis"].detach().cpu().numpy().astype(np.float64)


def load_geometry(checkpoint_root: Path, dataset: str) -> BasisGeometry:
    checkpoint = checkpoint_path(checkpoint_root, dataset)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    basis = load_basis(checkpoint)
    if basis.ndim != 2:
        raise ValueError(f"expected learned_temporal_basis to be 2D, got {basis.shape}")
    return BasisGeometry(dataset=dataset, checkpoint=checkpoint, basis=basis)


def safe_entropy(probabilities: np.ndarray) -> float:
    values = probabilities[probabilities > 0]
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


def energy_rank(singular_values: np.ndarray, threshold: float) -> int:
    energy = singular_values * singular_values
    total = float(np.sum(energy))
    if total <= 0:
        return 0
    cumulative = np.cumsum(energy) / total
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def orthonormal_row_space(segment_basis: np.ndarray, rank: int) -> np.ndarray:
    _u, singular_values, vh = np.linalg.svd(segment_basis, full_matrices=False)
    keep = min(rank, int(np.sum(singular_values > 1e-10)), vh.shape[0])
    if keep <= 0:
        return np.zeros((segment_basis.shape[1], 0), dtype=np.float64)
    return vh[:keep].T


def subspace_overlap(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    if left.shape[1] == 0 or right.shape[1] == 0:
        return 0.0, 0.0
    singular_values = np.linalg.svd(left.T @ right, compute_uv=False)
    squared = singular_values * singular_values
    return float(np.mean(squared)), float(np.min(singular_values))


def analyze_global(dataset: str, basis: np.ndarray) -> dict[str, Any]:
    singular_values = np.linalg.svd(basis, compute_uv=False)
    energy = singular_values * singular_values
    total = float(np.sum(energy))
    row_norms = np.linalg.norm(basis, axis=1)
    return {
        "dataset": dataset,
        "horizon": int(basis.shape[0]),
        "rank": int(basis.shape[1]),
        "basis_energy": total,
        "row_norm_mean": float(np.mean(row_norms)),
        "row_norm_std": float(np.std(row_norms)),
        "effective_rank_90": energy_rank(singular_values, 0.90),
        "effective_rank_95": energy_rank(singular_values, 0.95),
        "effective_rank_99": energy_rank(singular_values, 0.99),
        "top8_energy_pct": 100.0 * float(np.sum(energy[:8])) / total if total > 0 else 0.0,
        "top32_energy_pct": 100.0 * float(np.sum(energy[:32])) / total if total > 0 else 0.0,
        "top64_energy_pct": 100.0 * float(np.sum(energy[:64])) / total if total > 0 else 0.0,
    }


def analyze_segments(dataset: str, basis: np.ndarray) -> list[dict[str, Any]]:
    total_energy = float(np.sum(basis * basis))
    rows: list[dict[str, Any]] = []
    for name, start, end in SEGMENTS:
        segment = basis[start:end]
        singular_values = np.linalg.svd(segment, compute_uv=False)
        energy = singular_values * singular_values
        segment_energy = float(np.sum(energy))
        row_norms = np.linalg.norm(segment, axis=1)
        rows.append(
            {
                "dataset": dataset,
                "segment": name,
                "start": start,
                "end": end,
                "length": end - start,
                "segment_energy": segment_energy,
                "segment_energy_share_pct": (
                    100.0 * segment_energy / total_energy if total_energy > 0 else 0.0
                ),
                "row_norm_mean": float(np.mean(row_norms)),
                "row_norm_std": float(np.std(row_norms)),
                "effective_rank_90": energy_rank(singular_values, 0.90),
                "effective_rank_95": energy_rank(singular_values, 0.95),
                "effective_rank_99": energy_rank(singular_values, 0.99),
                "top8_energy_pct": (
                    100.0 * float(np.sum(energy[:8])) / segment_energy if segment_energy > 0 else 0.0
                ),
                "top32_energy_pct": (
                    100.0 * float(np.sum(energy[:32])) / segment_energy if segment_energy > 0 else 0.0
                ),
                "top64_energy_pct": (
                    100.0 * float(np.sum(energy[:64])) / segment_energy if segment_energy > 0 else 0.0
                ),
            }
        )
    return rows


def analyze_atom_stage_share(dataset: str, basis: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    segment_energy = []
    for _name, start, end in SEGMENTS:
        segment_energy.append(np.sum(basis[start:end] * basis[start:end], axis=0))
    energy_by_segment = np.stack(segment_energy, axis=0)
    total_by_atom = np.sum(energy_by_segment, axis=0)
    for atom_idx in range(basis.shape[1]):
        total = float(total_by_atom[atom_idx])
        if total <= 0:
            shares = np.zeros(len(SEGMENTS), dtype=np.float64)
        else:
            shares = energy_by_segment[:, atom_idx] / total
        row: dict[str, Any] = {
            "dataset": dataset,
            "atom_idx": atom_idx,
            "total_energy": total,
            "max_stage_share": float(np.max(shares)) if shares.size else 0.0,
            "normalized_entropy": safe_entropy(shares) / np.log(len(SEGMENTS)),
        }
        for segment_idx, (name, _start, _end) in enumerate(SEGMENTS):
            row[f"share_{name}"] = float(shares[segment_idx])
        rows.append(row)
    rows.sort(key=lambda item: item["total_energy"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["energy_rank"] = rank
    rows.sort(key=lambda item: item["atom_idx"])
    return rows


def analyze_subspaces(dataset: str, basis: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    segment_spaces: dict[int, dict[str, np.ndarray]] = {}
    for rank in RANKS:
        spaces: dict[str, np.ndarray] = {}
        for name, start, end in SEGMENTS:
            spaces[name] = orthonormal_row_space(basis[start:end], rank)
        segment_spaces[rank] = spaces
        names = [segment[0] for segment in SEGMENTS]
        for left_idx, left_name in enumerate(names):
            for right_name in names[left_idx + 1 :]:
                mean_sq_cos, min_cos = subspace_overlap(spaces[left_name], spaces[right_name])
                rows.append(
                    {
                        "dataset": dataset,
                        "rank": rank,
                        "left_segment": left_name,
                        "right_segment": right_name,
                        "mean_squared_canonical_corr": mean_sq_cos,
                        "min_canonical_corr": min_cos,
                    }
                )
    return rows


def summarize_dataset(
    global_row: dict[str, Any],
    segment_rows: list[dict[str, Any]],
    atom_rows: list[dict[str, Any]],
    subspace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset = global_row["dataset"]
    top_energy_rows = sorted(atom_rows, key=lambda item: item["total_energy"], reverse=True)[:64]
    entropy_values = np.asarray([row["normalized_entropy"] for row in top_energy_rows], dtype=np.float64)
    max_share_values = np.asarray([row["max_stage_share"] for row in top_energy_rows], dtype=np.float64)
    rank32_overlaps = [
        row["mean_squared_canonical_corr"] for row in subspace_rows if row["rank"] == 32
    ]
    rank64_overlaps = [
        row["mean_squared_canonical_corr"] for row in subspace_rows if row["rank"] == 64
    ]
    tail_row = next(row for row in segment_rows if row["segment"] == "tail_336_720")
    short_rows = [row for row in segment_rows if row["segment"] != "tail_336_720"]
    short_rank95 = float(np.mean([row["effective_rank_95"] for row in short_rows]))
    return {
        "dataset": dataset,
        "global_effective_rank_95": global_row["effective_rank_95"],
        "tail_effective_rank_95": tail_row["effective_rank_95"],
        "short_mean_effective_rank_95": short_rank95,
        "top64_atom_entropy_mean": float(np.mean(entropy_values)) if entropy_values.size else float("nan"),
        "top64_atom_max_stage_share_mean": (
            float(np.mean(max_share_values)) if max_share_values.size else float("nan")
        ),
        "top64_atom_stage_specialized_rate_0p70": (
            float(np.mean(max_share_values >= 0.70)) if max_share_values.size else float("nan")
        ),
        "rank32_pair_overlap_mean": float(np.mean(rank32_overlaps)) if rank32_overlaps else float("nan"),
        "rank64_pair_overlap_mean": float(np.mean(rank64_overlaps)) if rank64_overlaps else float("nan"),
    }


def fmt_value(value: Any) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt_value(row[field]) for field in fields) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    global_rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    subspace_rows: list[dict[str, Any]],
) -> None:
    rank32_subspace = [row for row in subspace_rows if row["rank"] == 32]
    entropy_mean = float(np.mean([row["top64_atom_entropy_mean"] for row in summary_rows]))
    specialized_rate = float(
        np.mean([row["top64_atom_stage_specialized_rate_0p70"] for row in summary_rows])
    )
    rank32_overlap = float(np.mean([row["rank32_pair_overlap_mean"] for row in summary_rows]))
    rank64_overlap = float(np.mean([row["rank64_pair_overlap_mean"] for row in summary_rows]))
    lines = [
        "# Phase5 StageB B10-TSI-A Basis Geometry Diagnostic",
        "",
        "## 阶段记录",
        "",
        "| 字段 | 内容 |",
        "| --- | --- |",
        "| `candidate_id` | `B10-TCO` |",
        "| `diagnostic_id` | `B10-TSI-A` |",
        "| `current_step` | Step 3：problem-existence diagnostic |",
        "| `problem` | A6-LBF 的 requested target set 不进入 computation graph；需要先判断 learned basis 是否已经足以承载 stage/target-set 差异 |",
        "| `scope` | checkpoint-only basis geometry audit；不训练新模型，不读取 test labels，不评估 prediction MSE |",
        "| `decision` | `partial_support_continue_tsi`; basis 不是 stage-blind，但 target set 仍缺席于 history-to-coeff/state path |",
        "",
        "## 诊断定义",
        "",
        "本诊断读取 clean A6 checkpoint 中的 `learned_temporal_basis: [720, 256]`，从三个角度检查 basis 的 stage 结构：",
        "",
        "- `segment_effective_rank`: 每个 future segment 的 basis 子矩阵需要多少 rank 才能覆盖自身能量；",
        "- `atom_stage_share`: 每个 temporal atom 的能量是否集中在单一 stage，还是横跨多个 stage；",
        "- `row_space_overlap`: 不同 stage 在 coefficient 维度上的 row subspace 是否高度重叠。",
        "",
        "如果 basis 已经强烈 stage-specialized，那么继续把 stage 信息注入 `coeff` 可能容易退化为 extra capacity。",
        "如果 basis 的主要 atoms 和 row spaces 横跨多个 stage，则更支持 B10 的问题定义：target set 应该进入",
        "`history -> target state -> basis-coeff coupling`，而不是只在 output prefix 上 slicing。",
        "",
        "## Dataset Summary",
        "",
        markdown_table(
            summary_rows,
            [
                "dataset",
                "global_effective_rank_95",
                "tail_effective_rank_95",
                "short_mean_effective_rank_95",
                "top64_atom_entropy_mean",
                "top64_atom_max_stage_share_mean",
                "top64_atom_stage_specialized_rate_0p70",
                "rank32_pair_overlap_mean",
                "rank64_pair_overlap_mean",
            ],
        ),
        "",
        "## Global Basis Rank",
        "",
        markdown_table(
            global_rows,
            [
                "dataset",
                "horizon",
                "rank",
                "effective_rank_90",
                "effective_rank_95",
                "effective_rank_99",
                "top32_energy_pct",
                "top64_energy_pct",
            ],
        ),
        "",
        "## Segment Geometry",
        "",
        markdown_table(
            segment_rows,
            [
                "dataset",
                "segment",
                "length",
                "segment_energy_share_pct",
                "effective_rank_95",
                "top32_energy_pct",
                "top64_energy_pct",
            ],
        ),
        "",
        "## Rank-32 Stage Row-Space Overlap",
        "",
        markdown_table(
            rank32_subspace,
            [
                "dataset",
                "left_segment",
                "right_segment",
                "mean_squared_canonical_corr",
                "min_canonical_corr",
            ],
        ),
        "",
        "## Observed Decision",
        "",
        f"[Fact] 三个数据集 top64 atom 的 mean normalized entropy 为 `{entropy_mean:.4f}`，",
        f"`max_stage_share >= 0.70` 的 mean rate 只有 `{specialized_rate:.4f}`。",
        "这说明高能 temporal atoms 并没有强烈局部化到单一 future stage。",
        "",
        f"[Fact] 但 stage row-space overlap 不高：rank32 mean overlap 为 `{rank32_overlap:.4f}`，",
        f"rank64 mean overlap 为 `{rank64_overlap:.4f}`。这说明不同 future segments 在 coefficient 维度上读取",
        "的是明显不同的 row subspaces。",
        "",
        "[Interpretation] 因此 B10 不能再用“basis 不包含 stage 信息”作为问题叙事。更严谨的说法是：",
        "A6 的 `learned_temporal_basis` 已经形成 stage-differentiated coefficient geometry，但",
        "`learned_basis_coeff(hidden)` 仍然只产生一个 target-set-blind coefficient vector。requested target set",
        "没有进入 `history -> coeff/state` 生成路径，模型只能让同一个 coefficient 同时服务多个 stage row subspaces。",
        "",
        "[Decision] B10-TSI-A 支持继续做 B10-TSI-B，但只支持更收窄的问题：",
        "`target-set-conditioned history readout / coefficient state`，而不是继续向现有 coefficient 后面加",
        "stage modulation。下一步必须检查真实 forward batch 中 coefficient 能量如何被不同 stage subspaces 使用，",
        "并加入 no-target-set capacity control。",
        "",
        "## Interpretation Rule",
        "",
        "- [Supports B10 problem] high `top64_atom_entropy_mean` and high `rank32_pair_overlap_mean`: basis atoms/subspaces are shared across stages, so requested target set is not natively resolved by basis alone.",
        "- [Weakens B10 problem] high `top64_atom_stage_specialized_rate_0p70` and low row-space overlap: basis already creates stage-specific coefficient axes; B10 must then show a stronger history-target readout problem.",
        "- [Boundary] This diagnostic does not compare B10 with no-target-set controls. Passing this audit only allows B10 to proceed to target-set interface diagnostic, not to implementation.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        geometry = load_geometry(args.checkpoint_root, dataset)
        global_row = analyze_global(dataset, geometry.basis)
        dataset_segment_rows = analyze_segments(dataset, geometry.basis)
        dataset_atom_rows = analyze_atom_stage_share(dataset, geometry.basis)
        dataset_subspace_rows = analyze_subspaces(dataset, geometry.basis)
        summary_rows.append(
            summarize_dataset(
                global_row,
                dataset_segment_rows,
                dataset_atom_rows,
                dataset_subspace_rows,
            )
        )
        global_rows.append(global_row)
        segment_rows.extend(dataset_segment_rows)
        atom_rows.extend(dataset_atom_rows)
        subspace_rows.extend(dataset_subspace_rows)

    write_csv(args.analysis_root / "b10_tsi_basis_global.csv", global_rows)
    write_csv(args.analysis_root / "b10_tsi_basis_segments.csv", segment_rows)
    write_csv(args.analysis_root / "b10_tsi_basis_atom_stage_share.csv", atom_rows)
    write_csv(args.analysis_root / "b10_tsi_basis_subspace_overlap.csv", subspace_rows)
    write_csv(args.analysis_root / "b10_tsi_basis_summary.csv", summary_rows)
    write_report(
        args.analysis_root / "b10_tsi_basis_geometry_report.md",
        summary_rows,
        global_rows,
        segment_rows,
        subspace_rows,
    )


if __name__ == "__main__":
    main()

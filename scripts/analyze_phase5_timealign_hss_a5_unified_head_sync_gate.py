from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]
A5_ARMS = ["a5b_r64", "a5b_r128", "a5q_seg48_small", "a5q_seg24_wide"]
REFERENCE_RUNS = {
    "official_unified": (
        "official/official-last",
        "TimeAlignOfficialUnified720_official-last",
    ),
    "a2_nested": (
        "a2/official-last",
        "TimeAlignOfficialUnified720_A2_nested_segment_decoder_multiprefix_official-last",
    ),
    "a3c_warm": (
        "a3c/official-last",
        "TimeAlignOfficialUnified720_A3C_checkpoint_initialized_nested_segment_decoder_multiprefix_official-last",
    ),
    "a3d_w03": (
        "a3d/official-last",
        "TimeAlignOfficialUnified720_A3D_teacher_preserved_nested_w03_official-last",
    ),
    "a3e_warm": (
        "a3e/official-last",
        "TimeAlignOfficialUnified720_A3E_target_conditioned_nested_warm_official-last",
    ),
    "a3e_scratch": (
        "a3e/official-last",
        "TimeAlignOfficialUnified720_A3E_target_conditioned_nested_scratch_official-last",
    ),
    "h1_target_set": (
        "h1/official-last",
        "TimeAlignOfficialUnified720_H1_target_set_decoder_multiprefix_official-last",
    ),
    "h1c_row_gated": (
        "h1c/official-last",
        "TimeAlignOfficialUnified720_H1C_row_gated_dense_head_multiprefix_official-last",
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def rel_pct(value: float, baseline: float) -> float:
    return (value / baseline - 1.0) * 100.0


def fmt(value: float) -> str:
    return f"{value:.3f}"


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def mixed_label() -> str:
    return "mixed_h96_h192_h336_h720"


def metric_path(root: Path, run_name: str, dataset: str, seed: int) -> Path:
    return root / "official-last" / run_name / dataset / mixed_label() / f"seed{seed}" / "metrics_by_target_horizon.csv"


def segment_path(root: Path, run_name: str, dataset: str, seed: int) -> Path:
    return root / "official-last" / run_name / dataset / mixed_label() / f"seed{seed}" / "metrics_by_segment.csv"


def collect_a5_metrics(raw_root: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in A5_ARMS:
        run_name = f"TimeAlignOfficialUnified720_A5_{arm}_official-last"
        for dataset in DATASETS:
            path = metric_path(raw_root, run_name, dataset, seed)
            for row in read_csv(path):
                rows.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "family": "A5-B" if arm.startswith("a5b") else "A5-Q",
                        "target_horizon": int(row["target_horizon"]),
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                        "source_path": str(path),
                    }
                )
    return rows


def collect_a5_segments(raw_root: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in A5_ARMS:
        run_name = f"TimeAlignOfficialUnified720_A5_{arm}_official-last"
        for dataset in DATASETS:
            path = segment_path(raw_root, run_name, dataset, seed)
            for row in read_csv(path):
                rows.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "target_horizon": int(row["target_horizon"]),
                        "segment_start": int(row["segment_start"]),
                        "segment_end": int(row["segment_end"]),
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                        "source_path": str(path),
                    }
                )
    return rows


def collect_reference_metrics(reference_root: Path, seed: int) -> dict[str, dict[tuple[str, int], dict[str, float]]]:
    refs: dict[str, dict[tuple[str, int], dict[str, float]]] = {}
    for name, (subroot, run_name) in REFERENCE_RUNS.items():
        ref: dict[tuple[str, int], dict[str, float]] = {}
        for dataset in DATASETS:
            path = reference_root / subroot / run_name / dataset / mixed_label() / f"seed{seed}" / "metrics_by_target_horizon.csv"
            for row in read_csv(path):
                ref[(dataset, int(row["target_horizon"]))] = {
                    "mse": float(row["mse"]),
                    "mae": float(row["mae"]),
                }
        refs[name] = ref
    refs["a3e_best"] = {}
    for key in refs["a3e_warm"]:
        warm = refs["a3e_warm"][key]
        scratch = refs["a3e_scratch"][key]
        refs["a3e_best"][key] = warm if warm["mse"] <= scratch["mse"] else scratch
    refs["best_stage_control"] = {}
    stage_names = [
        "official_unified",
        "a2_nested",
        "a3c_warm",
        "a3d_w03",
        "a3e_best",
        "h1_target_set",
        "h1c_row_gated",
    ]
    for dataset in DATASETS:
        for horizon in HORIZONS:
            key = (dataset, horizon)
            best_name = min(stage_names, key=lambda name: refs[name][key]["mse"])
            refs["best_stage_control"][key] = {
                "mse": refs[best_name][key]["mse"],
                "mae": refs[best_name][key]["mae"],
                "name": best_name,
            }
    refs["fixed"] = collect_fixed_reference(reference_root, seed)
    return refs


def collect_fixed_reference(reference_root: Path, seed: int) -> dict[tuple[str, int], dict[str, float]]:
    ref: dict[tuple[str, int], dict[str, float]] = {}
    for dataset in DATASETS:
        for horizon in HORIZONS:
            run_name = f"TimeAlignOfficialFixedH{horizon}_official-last"
            path = reference_root / "official/official-last" / run_name / dataset / f"mixed_h{horizon}" / f"seed{seed}" / "metrics_by_target_horizon.csv"
            metric_rows = read_csv(path)
            if len(metric_rows) != 1:
                raise ValueError(f"Expected one row in {path}, got {len(metric_rows)}")
            row = metric_rows[0]
            ref[(dataset, horizon)] = {
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
            }
    return ref


def add_references(
    a5_rows: list[dict[str, Any]],
    refs: dict[str, dict[tuple[str, int], dict[str, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = [
        "official_unified",
        "a2_nested",
        "a3c_warm",
        "a3d_w03",
        "a3e_best",
        "h1_target_set",
        "h1c_row_gated",
        "best_stage_control",
        "fixed",
    ]
    for row in a5_rows:
        out = dict(row)
        key = (row["dataset"], row["target_horizon"])
        for name in names:
            ref = refs[name][key]
            out[f"{name}_mse"] = ref["mse"]
            out[f"relative_mse_vs_{name}_pct"] = rel_pct(row["mse"], ref["mse"])
            out[f"beats_{name}"] = row["mse"] < ref["mse"]
            if name == "best_stage_control":
                out["best_stage_control_name"] = ref["name"]
        rows.append(out)
    return rows


def summarize_rows(compare_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS + ["ALL"]:
        ds_rows = compare_rows if dataset == "ALL" else [row for row in compare_rows if row["dataset"] == dataset]
        for arm in A5_ARMS:
            subset = [row for row in ds_rows if row["arm"] == arm]
            if not subset:
                continue
            out: dict[str, Any] = {
                "dataset": dataset,
                "arm": arm,
                "family": subset[0]["family"],
                "settings": len(subset),
                "mean_mse": mean(row["mse"] for row in subset),
            }
            for ref_name in [
                "official_unified",
                "h1_target_set",
                "h1c_row_gated",
                "a3d_w03",
                "a3e_best",
                "best_stage_control",
                "fixed",
            ]:
                key = f"relative_mse_vs_{ref_name}_pct"
                out[f"wins_vs_{ref_name}"] = sum(1 for row in subset if row[f"beats_{ref_name}"])
                out[f"mean_relative_mse_vs_{ref_name}_pct"] = mean(row[key] for row in subset)
            rows.append(out)
    return rows


def capacity_rows(compare_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (row["dataset"], row["target_horizon"], row["arm"]): row
        for row in compare_rows
    }
    pairs = [
        ("A5-B_rank_capacity", "a5b_r64", "a5b_r128"),
        ("A5-Q_query_capacity", "a5q_seg48_small", "a5q_seg24_wide"),
    ]
    rows: list[dict[str, Any]] = []
    for name, base_arm, bigger_arm in pairs:
        for dataset in DATASETS:
            for horizon in HORIZONS:
                base = by_key[(dataset, horizon, base_arm)]
                bigger = by_key[(dataset, horizon, bigger_arm)]
                rows.append(
                    {
                        "comparison": name,
                        "dataset": dataset,
                        "target_horizon": horizon,
                        "base_arm": base_arm,
                        "bigger_arm": bigger_arm,
                        "base_mse": base["mse"],
                        "bigger_mse": bigger["mse"],
                        "relative_bigger_vs_base_pct": rel_pct(bigger["mse"], base["mse"]),
                        "bigger_wins": bigger["mse"] < base["mse"],
                    }
                )
    for name, base_arm, bigger_arm in pairs:
        subset = [row for row in rows if row["comparison"] == name]
        rows.append(
            {
                "comparison": name,
                "dataset": "ALL",
                "target_horizon": "ALL",
                "base_arm": base_arm,
                "bigger_arm": bigger_arm,
                "base_mse": mean(row["base_mse"] for row in subset),
                "bigger_mse": mean(row["bigger_mse"] for row in subset),
                "relative_bigger_vs_base_pct": mean(row["relative_bigger_vs_base_pct"] for row in subset),
                "bigger_wins": sum(1 for row in subset if row["bigger_wins"]),
            }
        )
    return rows


def segment_summary(segment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        subset = [
            row for row in segment_rows
            if row["dataset"] == dataset and row["target_horizon"] == 720
        ]
        for arm in A5_ARMS:
            arm_rows = [row for row in subset if row["arm"] == arm]
            early = [row for row in arm_rows if row["segment_start"] < 192]
            late = [row for row in arm_rows if row["segment_start"] >= 336]
            rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "h720_segments": len(arm_rows),
                    "h720_mean_segment_mse": mean(row["mse"] for row in arm_rows),
                    "h720_early_mean_segment_mse": mean(row["mse"] for row in early),
                    "h720_late_mean_segment_mse": mean(row["mse"] for row in late),
                    "late_vs_early_ratio": mean(row["mse"] for row in late) / mean(row["mse"] for row in early),
                }
            )
    return rows


def best_rows(compare_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            subset = [row for row in compare_rows if row["dataset"] == dataset and row["target_horizon"] == horizon]
            best = min(subset, key=lambda row: row["mse"])
            rows.append(
                {
                    "dataset": dataset,
                    "target_horizon": horizon,
                    "best_a5_arm": best["arm"],
                    "best_a5_mse": best["mse"],
                    "best_stage_control_name": best["best_stage_control_name"],
                    "best_stage_control_mse": best["best_stage_control_mse"],
                    "relative_mse_vs_best_stage_control_pct": best["relative_mse_vs_best_stage_control_pct"],
                    "beats_best_stage_control": best["beats_best_stage_control"],
                }
            )
    return rows


def report_markdown(
    path: Path,
    summary: list[dict[str, Any]],
    best: list[dict[str, Any]],
    capacity: list[dict[str, Any]],
    segment: list[dict[str, Any]],
) -> None:
    all_rows = [row for row in summary if row["dataset"] == "ALL"]
    best_all = min(all_rows, key=lambda row: row["mean_mse"])
    wins = int(best_all["wins_vs_best_stage_control"])
    rel = float(best_all["mean_relative_mse_vs_best_stage_control_pct"])
    decision = "failed_as_core_candidate"
    if wins >= 6 and rel <= 0.3:
        decision = "partial_pass"
    if wins >= 7 and rel < 0:
        decision = "pass_candidate"

    lines = [
        "# Phase5-A5 Unified Head Sync Gate Report",
        "",
        "## Reader Path",
        "",
        "[What] 本报告评估通过 narrative gate 的两个 first-principles unified head family："
        "`A5-B_continuous_forecast_basis_operator` 与 `A5-Q_elastic_causal_target_query_decoder`。",
        "",
        "[Why] A5 的目标是判断 direct `[B,H,C]`、architecture-level prefix-consistent head 是否能替代"
        "现有 full-720 crop / dense-head controls，并成为 Stage A 的 paper-core interface。",
        "",
        "[How] 远程矩阵为 `ETTh2 + ETTm1 + Weather` × "
        "`a5b_r64/a5b_r128/a5q_seg48_small/a5q_seg24_wide`，共 12 runs。"
        "每个 run 使用 `metrics_by_target_horizon.csv` 和 `metrics_by_segment.csv` 生成本报告。",
        "",
        "[Metric] `mean_relative_mse_vs_*_pct = (A5_MSE / reference_MSE - 1) * 100`；负数表示 A5 更好。"
        "`best_stage_control` 是每个 dataset/horizon 上已有 unified controls 中 MSE 最低者，"
        "包括 official unified、A2、A3C、A3D-w03、A3E-best、H1 target-set 与 H1C row-gated。"
        "fixed per-horizon 只作为非同类参考，不纳入 `best_stage_control`。",
        "",
        "[Prefix Contract] 本轮 smoke 结果保存于 `phase5_timealign_hss_a5_smoke.json`："
        "A5-B 的 `decode(96)` vs `decode(720)[:, :96]` mismatch 为 `0.0`，"
        "A5-Q 为约 `4.77e-07`，说明 architecture-level prefix consistency 实现成立。",
        "",
        "## 结论",
        "",
        f"[Decision] 本轮 A5-Q/A5-B effectiveness gate 结论：`{decision}`。",
        "",
        (
            f"[Fact] ALL mean MSE 最优 A5 arm 是 `{best_all['arm']}`，"
            f"对 `best_stage_control` 的平均相对 MSE 为 `{fmt_pct(rel)}`，"
            f"wins 为 `{wins}/12`。"
        ),
        "",
        "[Interpretation] A5-B/A5-Q 的 architecture-level prefix consistency 已由 smoke 验证，"
        "但远程 forecasting effectiveness 没有稳定超过现有 unified controls。"
        "因此，本轮结果不能把 A5-B 或 A5-Q 直接提升为 paper-core unified head。",
        "",
        "## ALL Summary",
        "",
        "| arm | family | mean_mse | vs_best_stage_control | wins_vs_best_stage_control | vs_h1 | vs_h1c | vs_a3d_w03 | vs_fixed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in all_rows:
        lines.append(
            "| {arm} | {family} | {mean_mse:.6f} | {best_rel} | {best_wins}/12 | {h1_rel} | {h1c_rel} | {a3d_rel} | {fixed_rel} |".format(
                arm=row["arm"],
                family=row["family"],
                mean_mse=row["mean_mse"],
                best_rel=fmt_pct(row["mean_relative_mse_vs_best_stage_control_pct"]),
                best_wins=int(row["wins_vs_best_stage_control"]),
                h1_rel=fmt_pct(row["mean_relative_mse_vs_h1_target_set_pct"]),
                h1c_rel=fmt_pct(row["mean_relative_mse_vs_h1c_row_gated_pct"]),
                a3d_rel=fmt_pct(row["mean_relative_mse_vs_a3d_w03_pct"]),
                fixed_rel=fmt_pct(row["mean_relative_mse_vs_fixed_pct"]),
            )
        )

    lines.extend(
        [
            "",
            "## Best A5 Per Setting",
            "",
            "| dataset | horizon | best_a5_arm | best_a5_mse | best_stage_control | rel_vs_control | beats_control |",
            "| --- | ---: | --- | ---: | --- | ---: | --- |",
        ]
    )
    for row in best:
        lines.append(
            f"| {row['dataset']} | {row['target_horizon']} | {row['best_a5_arm']} | "
            f"{row['best_a5_mse']:.6f} | {row['best_stage_control_name']} | "
            f"{fmt_pct(row['relative_mse_vs_best_stage_control_pct'])} | {row['beats_best_stage_control']} |"
        )

    lines.extend(
        [
            "",
            "## Capacity Checks",
            "",
            "| comparison | bigger_arm | ALL relative bigger vs base | bigger wins |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in capacity:
        if row["dataset"] != "ALL":
            continue
        lines.append(
            f"| {row['comparison']} | {row['bigger_arm']} | "
            f"{fmt_pct(row['relative_bigger_vs_base_pct'])} | {row['bigger_wins']}/12 |"
        )

    lines.extend(
        [
            "",
            "## Segment Notes",
            "",
            "| dataset | arm | h720_mean_segment_mse | late_vs_early_ratio |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in segment:
        lines.append(
            f"| {row['dataset']} | {row['arm']} | {row['h720_mean_segment_mse']:.6f} | "
            f"{row['late_vs_early_ratio']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Gate 判断",
            "",
            "- A5-B: `failed_as_core_candidate`。rank 128 相比 rank 64 有稳定容量收益，但仍比 best existing unified control 差 `+14.19%`，说明当前 basis/operator class 的表达上限不足。",
            "- A5-Q: `failed_as_core_candidate`。seg24-wide 相比 seg48-small 反而 `0/12` wins，说明简单加密 target segments / FF width 未修复 query decoder 的训练与容量问题。",
            "- A5-S/A5-I/A5-M 不应自动进入远程；若继续 A5，需要先回 Step 4/5 做新的理论诊断，而不是扩 sweep。",
            "",
            "## Rollback",
            "",
            "[Decision] 按 11-step loop，本轮应回退到 Step 4/5：重新评估 first-principles unified head 的 capacity 机制。"
            "A5-Q/A5-B 的 prefix-consistency contract 成立，但当前 parameterization 的 forecasting capacity 不足。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    a5_rows = collect_a5_metrics(args.raw_root, args.seed)
    segment_rows = collect_a5_segments(args.raw_root, args.seed)
    refs = collect_reference_metrics(args.reference_root, args.seed)
    comparison = add_references(a5_rows, refs)
    summary = summarize_rows(comparison)
    capacity = capacity_rows(comparison)
    segments = segment_summary(segment_rows)
    best = best_rows(comparison)

    write_csv(args.output_dir / "phase5_timealign_hss_a5_metrics.csv", a5_rows)
    write_csv(args.output_dir / "phase5_timealign_hss_a5_comparison.csv", comparison)
    write_csv(args.output_dir / "phase5_timealign_hss_a5_summary.csv", summary)
    write_csv(args.output_dir / "phase5_timealign_hss_a5_capacity_checks.csv", capacity)
    write_csv(args.output_dir / "phase5_timealign_hss_a5_segment_summary.csv", segments)
    write_csv(args.output_dir / "phase5_timealign_hss_a5_best_per_setting.csv", best)
    dump_json(
        args.output_dir / "phase5_timealign_hss_a5_analysis_config.json",
        {
            "datasets": DATASETS,
            "horizons": HORIZONS,
            "a5_arms": A5_ARMS,
            "reference_runs": REFERENCE_RUNS,
            "seed": args.seed,
            "raw_root": str(args.raw_root),
            "reference_root": str(args.reference_root),
        },
    )
    report_markdown(
        args.output_dir / "phase5_timealign_hss_a5_unified_head_sync_gate_report.md",
        summary,
        best,
        capacity,
        segments,
    )


if __name__ == "__main__":
    main()

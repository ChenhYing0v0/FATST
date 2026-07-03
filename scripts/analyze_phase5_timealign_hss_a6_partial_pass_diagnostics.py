from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


def arm_from_path(path: Path) -> str:
    prefix = "TimeAlignOfficialUnified720_A6_"
    suffix = "_official-last"
    for part in path.parts:
        if part.startswith(prefix) and part.endswith(suffix):
            return part.removeprefix(prefix).removesuffix(suffix)
    raise ValueError(f"Cannot parse A6 arm from {path}")


def collect_training_trajectory(raw_root: Path, comparison_path: Path) -> pd.DataFrame:
    comparison = pd.read_csv(comparison_path)
    horizon_summary = (
        comparison.groupby(["dataset", "arm"], as_index=False)
        .agg(
            mean_mse=("mse", "mean"),
            mean_relative_mse_vs_best_stage_control_pct=("relative_mse_vs_best_stage_control_pct", "mean"),
            mean_relative_mse_vs_a6_der_pct=("relative_mse_vs_a6_der_pct", "mean"),
            wins_vs_best_stage_control=("beats_best_stage_control", "sum"),
        )
    )

    rows: list[dict[str, Any]] = []
    pattern = "official-last/*/*/mixed_h96_h192_h336_h720/seed2021/training_log.csv"
    for path in sorted(raw_root.glob(pattern)):
        frame = pd.read_csv(path)
        arm = arm_from_path(path)
        dataset = path.parts[-4]
        best_idx = int(frame["val_mean_mse"].idxmin())
        first = float(frame.iloc[0]["val_mean_mse"])
        best = float(frame.iloc[best_idx]["val_mean_mse"])
        last = float(frame.iloc[-1]["val_mean_mse"])
        rows.append(
            {
                "dataset": dataset,
                "arm": arm,
                "epochs": int(len(frame)),
                "best_epoch": int(frame.iloc[best_idx]["epoch"]),
                "last_epoch": int(frame.iloc[-1]["epoch"]),
                "first_val_mean_mse": first,
                "best_val_mean_mse": best,
                "last_val_mean_mse": last,
                "last_vs_best_val_mse_pct": (last / best - 1.0) * 100.0,
                "last_vs_first_val_mse_pct": (last / first - 1.0) * 100.0,
                "post_best_epochs": int(frame.iloc[-1]["epoch"] - frame.iloc[best_idx]["epoch"]),
                "source_path": str(path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No training_log.csv files found under {raw_root}")
    trajectory = pd.DataFrame(rows)
    return trajectory.merge(horizon_summary, on=["dataset", "arm"], how="left", validate="one_to_one")


def summarize_trajectory(trajectory: pd.DataFrame) -> pd.DataFrame:
    return (
        trajectory.groupby("dataset", as_index=False)
        .agg(
            arms=("arm", "nunique"),
            mean_last_vs_best_val_mse_pct=("last_vs_best_val_mse_pct", "mean"),
            max_last_vs_best_val_mse_pct=("last_vs_best_val_mse_pct", "max"),
            mean_gap_vs_best_stage_control_pct=("mean_relative_mse_vs_best_stage_control_pct", "mean"),
            max_gap_vs_best_stage_control_pct=("mean_relative_mse_vs_best_stage_control_pct", "max"),
        )
        .sort_values("dataset")
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def load_basis(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt(value: float) -> str:
    return f"{value:.2f}"


def write_report(
    output_dir: Path,
    trajectory: pd.DataFrame,
    trajectory_summary: pd.DataFrame,
    basis: pd.DataFrame,
) -> None:
    etth2 = trajectory[trajectory["dataset"].eq("ETTh2")].sort_values("arm")
    non_etth2 = trajectory[~trajectory["dataset"].eq("ETTh2")]
    etth2_mean_drift = float(etth2["last_vs_best_val_mse_pct"].mean())
    non_etth2_mean_drift = float(non_etth2["last_vs_best_val_mse_pct"].mean())

    lines = [
        "# Phase5-A6 Partial-Pass Diagnostic Report",
        "",
        "本文档记录 A6 partial-pass 后的 diagnostic-only 分析。主协议保持 `official-last` / without early stop；",
        "`best-val` 只允许作为后续 optional upper-bound audit，不能替代 paper metric 或作为 main protocol。",
        "",
        "## Diagnostic Question",
        "",
        "[Fact] A6-LBF 已恢复 dense-capacity path，但对 `best_stage_control` 仍为 `0/12` wins。",
        "",
        "[Question] 剩余差距更像是 official-last trajectory drift、learned-basis operator 结构限制，还是 objective conflict？",
        "",
        "## Official-Last Trajectory",
        "",
        (
            "[Strong Evidence] ETTh2 是主要 trajectory drift 来源："
            f"ETTh2 三个 A6 arms 的 last-vs-best validation MSE 平均漂移 `{fmt(etth2_mean_drift)}%`，"
            f"而 ETTm1/Weather 平均仅 `{fmt(non_etth2_mean_drift)}%`。"
        ),
        "",
        "| Dataset | Arms | mean last-vs-best val | max last-vs-best val | mean gap vs best control |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in trajectory_summary.iterrows():
        lines.append(
            "| {dataset} | {arms} | {mean_drift:+.2f}% | {max_drift:+.2f}% | {mean_gap:+.2f}% |".format(
                dataset=row["dataset"],
                arms=int(row["arms"]),
                mean_drift=row["mean_last_vs_best_val_mse_pct"],
                max_drift=row["max_last_vs_best_val_mse_pct"],
                mean_gap=row["mean_gap_vs_best_stage_control_pct"],
            )
        )

    lines.extend(
        [
            "",
            "### ETTh2 Arm Detail",
            "",
            "| Arm | best epoch | last-vs-best val | mean gap vs best control |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in etth2.iterrows():
        lines.append(
            "| `{arm}` | {best_epoch} | {drift:+.2f}% | {gap:+.2f}% |".format(
                arm=row["arm"],
                best_epoch=int(row["best_epoch"]),
                drift=row["last_vs_best_val_mse_pct"],
                gap=row["mean_relative_mse_vs_best_stage_control_pct"],
            )
        )

    if not basis.empty:
        lbf256 = basis[basis["arm"].eq("a6_lbf_r256")]
        lbf512 = basis[basis["arm"].eq("a6_lbf_r512")]
        op_rank99_256 = float(lbf256["operator_rank99"].mean())
        op_rank99_512 = float(lbf512["operator_rank99"].mean())
        op_eff_256 = float(lbf256["operator_effective_rank"].mean())
        op_eff_512 = float(lbf512["operator_effective_rank"].mean())
        smooth_256 = float(lbf256["basis_adjacent_cosine_mean"].mean())
        smooth_512 = float(lbf512["basis_adjacent_cosine_mean"].mean())
        lines.extend(
            [
                "",
                "## Learned-Basis Structure",
                "",
                (
                    "[Strong Evidence] `r512` 的 operator rank99 扩张没有转化为有效性能收益："
                    f"`r256` induced operator 的 mean rank99 为 `{fmt(op_rank99_256)}`，"
                    f"`r512` 为 `{fmt(op_rank99_512)}`，但 mean effective rank 仅从 "
                    f"`{fmt(op_eff_256)}` 到 `{fmt(op_eff_512)}`，且 A6 gate 中 r512 没有稳定优于 r256。"
                ),
                "",
                (
                    "[Fact] learned temporal basis 的 adjacent-row cosine 较低："
                    f"`r256` mean `{fmt(smooth_256)}`，`r512` mean `{fmt(smooth_512)}`。"
                    "这说明它不是简单平滑 Fourier-like basis，而是在学习更接近 dense row bank 的时间位置字典。"
                ),
                "",
                "| Dataset | Arm | basis rank | basis eff-rank | operator eff-rank | operator rank99 | adjacent cosine |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in basis.sort_values(["dataset", "arm"]).iterrows():
            lines.append(
                "| {dataset} | `{arm}` | {rank} | {basis_eff:.2f} | {op_eff:.2f} | {op_rank99} | {cos:.2f} |".format(
                    dataset=row["dataset"],
                    arm=row["arm"],
                    rank=int(row["basis_rank"]),
                    basis_eff=row["basis_effective_rank"],
                    op_eff=row["operator_effective_rank"],
                    op_rank99=int(row["operator_rank99"]),
                    cos=row["basis_adjacent_cosine_mean"],
                )
            )
    else:
        lines.extend(
            [
                "",
                "## Learned-Basis Structure",
                "",
                "[Fact] 本次未提供 checkpoint-level basis diagnostics；只能完成 trajectory 部分。",
            ]
        )

    lines.extend(
        [
            "",
            "## Statistic Definitions",
            "",
            "`phase5_timealign_hss_a6_official_last_trajectory_diagnostic.csv` 的来源是每个 A6 run 的 "
            "`training_log.csv` 与 A6 comparison table。`first_val_mean_mse`、`best_val_mean_mse`、"
            "`last_val_mean_mse` 分别取 validation MSE 的第 1 epoch、最小值 epoch、最后 epoch；"
            "`last_vs_best_val_mse_pct = last_val_mean_mse / best_val_mean_mse - 1`，用于衡量 official-last "
            "checkpoint 相对训练轨迹内 best validation point 的漂移；`mean_relative_mse_vs_best_stage_control_pct` "
            "来自 A6 comparison table，表示该 run 在 4 个 horizons 上相对 best stage control 的平均 test MSE gap。",
            "",
            "`phase5_timealign_hss_a6_basis_structure_diagnostic.csv` 的来源是 A6-LBF `checkpoint.pt` 中的 "
            "`learned_temporal_basis`、`learned_basis_coeff.weight` 和 `learned_temporal_bias`。"
            "`basis_effective_rank`、`operator_effective_rank` 分别对 basis matrix 与 induced operator "
            "`learned_temporal_basis @ learned_basis_coeff.weight` 的 singular-value energy 分布计算 entropy "
            "effective rank；`operator_rank99` 是累计 singular-value energy 达到 99% 所需 rank；"
            "`basis_adjacent_cosine_mean` 是相邻 future rows 的平均 cosine similarity，用于判断 learned basis "
            "更接近平滑时间函数还是 dense row dictionary。",
            "",
            "## Decision",
            "",
            (
                "[Decision] 不启动 `best-val/early-stopping` 主实验。下一步实验应继续保持 official-last，"
                "优先解决 ETTh2 上 early-best 后 drift 的 optimization/objective conflict，以及 learned-basis "
                "operator 没有转化为 best-control wins 的机制缺口。"
            ),
            "",
            (
                "[Hypothesis] A6-LBF 的问题不是 rank 不够，而是 learned-basis operator 已近 dense-equivalent ceiling，"
                "但 official-last multi-prefix objective 在 ETTh2 上把模型推离 early-best basin；Weather/ETTm1 的剩余差距"
                "更像 best controls 本身含有 teacher/nested regularization advantage。"
            ),
            "",
            "## Artifacts",
            "",
            "- `phase5_timealign_hss_a6_official_last_trajectory_diagnostic.csv`",
            "- `phase5_timealign_hss_a6_official_last_trajectory_summary.csv`",
            "- `phase5_timealign_hss_a6_basis_structure_diagnostic.csv`",
            "- `phase5_timealign_hss_a6_partial_pass_diagnostic_config.json`",
            "",
        ]
    )
    (output_dir / "phase5_timealign_hss_a6_partial_pass_diagnostic_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase5 A6 partial-pass diagnostics.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--basis-diagnostics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory = collect_training_trajectory(args.raw_root, args.comparison)
    trajectory_summary = summarize_trajectory(trajectory)
    basis = load_basis(args.basis_diagnostics)

    write_csv(
        output_dir / "phase5_timealign_hss_a6_official_last_trajectory_diagnostic.csv",
        trajectory.to_dict("records"),
    )
    write_csv(
        output_dir / "phase5_timealign_hss_a6_official_last_trajectory_summary.csv",
        trajectory_summary.to_dict("records"),
    )
    if not basis.empty:
        write_csv(
            output_dir / "phase5_timealign_hss_a6_basis_structure_diagnostic.csv",
            basis.to_dict("records"),
        )
    (output_dir / "phase5_timealign_hss_a6_partial_pass_diagnostic_config.json").write_text(
        json.dumps(
            {
                "raw_root": str(args.raw_root),
                "comparison": str(args.comparison),
                "basis_diagnostics": str(args.basis_diagnostics) if args.basis_diagnostics else None,
                "output_dir": str(output_dir),
                "protocol": "official-last / without early stop",
                "best_val_policy": "diagnostic_only_upper_bound_audit_only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(output_dir, trajectory, trajectory_summary, basis)


if __name__ == "__main__":
    main()

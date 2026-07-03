from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


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


def arm_from_path(path: Path) -> str:
    prefix = "TimeAlignOfficialUnified720_A6_"
    suffix = "_official-last"
    for part in path.parts:
        if part.startswith(prefix) and part.endswith(suffix):
            return part.removeprefix(prefix).removesuffix(suffix)
    raise ValueError(f"Cannot parse A6 arm from {path}")


def collect_metrics(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_root.glob("official-last/*/*/mixed_h96_h192_h336_h720/seed2021/metrics_by_target_horizon.csv")):
        arm = arm_from_path(path)
        frame = pd.read_csv(path)
        for record in frame.to_dict("records"):
            rows.append(
                {
                    "dataset": record["dataset"],
                    "arm": arm,
                    "target_horizon": int(record["target_horizon"]),
                    "mse": float(record["mse"]),
                    "mae": float(record["mae"]),
                    "source_path": str(path),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No metrics_by_target_horizon.csv files found under {raw_root}")
    return pd.DataFrame(rows)


def collect_training(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_root.glob("official-last/*/*/mixed_h96_h192_h336_h720/seed2021/training_log.csv")):
        arm = arm_from_path(path)
        dataset = path.parts[-4]
        frame = pd.read_csv(path)
        best_idx = frame["val_mean_mse"].idxmin()
        rows.append(
            {
                "dataset": dataset,
                "arm": arm,
                "epochs": int(len(frame)),
                "best_epoch": int(frame.loc[best_idx, "epoch"]),
                "first_val_mean_mse": float(frame.iloc[0]["val_mean_mse"]),
                "best_val_mean_mse": float(frame.loc[best_idx, "val_mean_mse"]),
                "last_val_mean_mse": float(frame.iloc[-1]["val_mean_mse"]),
                "source_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def add_comparison_columns(metrics: pd.DataFrame, prior_comparison_path: Path) -> pd.DataFrame:
    prior = pd.read_csv(prior_comparison_path)
    prior_base = prior[prior["arm"].eq("a5b_r128")][
        [
            "dataset",
            "target_horizon",
            "official_unified_mse",
            "best_stage_control_mse",
            "best_stage_control_name",
            "mse",
        ]
    ].rename(columns={"mse": "a5b_r128_mse"})
    prior_q = prior[prior["arm"].eq("a5q_seg48_small")][
        ["dataset", "target_horizon", "mse"]
    ].rename(columns={"mse": "a5q_seg48_mse"})
    merged = metrics.merge(prior_base, on=["dataset", "target_horizon"], how="left", validate="many_to_one")
    merged = merged.merge(prior_q, on=["dataset", "target_horizon"], how="left", validate="many_to_one")
    der = metrics[metrics["arm"].eq("a6_der")][["dataset", "target_horizon", "mse"]].rename(
        columns={"mse": "a6_der_mse"}
    )
    merged = merged.merge(der, on=["dataset", "target_horizon"], how="left", validate="many_to_one")
    required = ["official_unified_mse", "best_stage_control_mse", "a5b_r128_mse", "a5q_seg48_mse", "a6_der_mse"]
    if merged[required].isna().any().any():
        missing = merged[merged[required].isna().any(axis=1)][["dataset", "target_horizon", "arm"]]
        raise ValueError(f"Missing comparison rows: {missing.to_dict('records')}")
    for column in required:
        name = column.removesuffix("_mse")
        merged[f"relative_mse_vs_{name}_pct"] = (merged["mse"] / merged[column] - 1.0) * 100.0
    merged["beats_best_stage_control"] = merged["mse"] < merged["best_stage_control_mse"]
    return merged


def summarize(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [("ALL", "ALL", frame) for _, frame in comparison.groupby(lambda _: True)]
    groups.extend((dataset, arm, frame) for (dataset, arm), frame in comparison.groupby(["dataset", "arm"], sort=True))
    for dataset, arm, frame in groups:
        rows.append(
            {
                "scope": "ALL" if dataset == "ALL" else "dataset",
                "dataset": dataset,
                "arm": arm if dataset != "ALL" else "ALL",
                "n": int(len(frame)),
                "mean_mse": float(frame["mse"].mean()),
                "mean_relative_mse_vs_best_stage_control_pct": float(
                    frame["relative_mse_vs_best_stage_control_pct"].mean()
                ),
                "mean_relative_mse_vs_official_unified_pct": float(
                    frame["relative_mse_vs_official_unified_pct"].mean()
                ),
                "mean_relative_mse_vs_a5b_r128_pct": float(frame["relative_mse_vs_a5b_r128_pct"].mean()),
                "mean_relative_mse_vs_a5q_seg48_pct": float(frame["relative_mse_vs_a5q_seg48_pct"].mean()),
                "mean_relative_mse_vs_a6_der_pct": float(frame["relative_mse_vs_a6_der_pct"].mean()),
                "wins_vs_best_stage_control": int(frame["beats_best_stage_control"].sum()),
            }
        )
    for arm, frame in comparison.groupby("arm", sort=True):
        rows.append(
            {
                "scope": "arm",
                "dataset": "ALL",
                "arm": arm,
                "n": int(len(frame)),
                "mean_mse": float(frame["mse"].mean()),
                "mean_relative_mse_vs_best_stage_control_pct": float(
                    frame["relative_mse_vs_best_stage_control_pct"].mean()
                ),
                "mean_relative_mse_vs_official_unified_pct": float(
                    frame["relative_mse_vs_official_unified_pct"].mean()
                ),
                "mean_relative_mse_vs_a5b_r128_pct": float(frame["relative_mse_vs_a5b_r128_pct"].mean()),
                "mean_relative_mse_vs_a5q_seg48_pct": float(frame["relative_mse_vs_a5q_seg48_pct"].mean()),
                "mean_relative_mse_vs_a6_der_pct": float(frame["relative_mse_vs_a6_der_pct"].mean()),
                "wins_vs_best_stage_control": int(frame["beats_best_stage_control"].sum()),
            }
        )
    return pd.DataFrame(rows)


def best_per_setting(comparison: pd.DataFrame) -> pd.DataFrame:
    idx = comparison.groupby(["dataset", "target_horizon"])["mse"].idxmin()
    return comparison.loc[idx].sort_values(["dataset", "target_horizon"]).reset_index(drop=True)


def write_report(output_dir: Path, comparison: pd.DataFrame, summary: pd.DataFrame, training: pd.DataFrame) -> None:
    def fmt(value: float) -> str:
        return f"{value:.2f}"

    arm_summary = summary[summary["scope"].eq("arm")].set_index("arm")
    best = best_per_setting(comparison)
    best_mean = float(best["relative_mse_vs_best_stage_control_pct"].mean())
    best_wins = int((best["relative_mse_vs_best_stage_control_pct"] < 0).sum())
    der = arm_summary.loc["a6_der"]
    lbf256 = arm_summary.loc["a6_lbf_r256"]
    lbf512 = arm_summary.loc["a6_lbf_r512"]

    lines = [
        "# Phase5-A6 Capacity-Native Unified Head Gate Report",
        "",
        "本文档分析 A6 capacity-native remote gate。该实验验证 `A6-DER` capacity ceiling 与 "
        "`A6-LBF-r256/r512` learned-basis forecast operator。",
        "",
        "## 结论摘要",
        "",
        (
            "[Strong Evidence] A6 成功修复了 A5-Q/A5-B 暴露出的主要 capacity collapse："
            f"`A6-DER` 相对 `best_stage_control` 平均仅差 `{fmt(der['mean_relative_mse_vs_best_stage_control_pct'])}%`，"
            f"且相对 A5-B-r128 平均改善 `{fmt(der['mean_relative_mse_vs_a5b_r128_pct'])}%`。"
        ),
        "",
        (
            "[Strong Evidence] `A6-LBF-r256` 已基本贴住 dense-equivalent ceiling："
            f"相对 `A6-DER` 平均 `{fmt(lbf256['mean_relative_mse_vs_a6_der_pct'])}%`，"
            f"相对 `best_stage_control` 平均 `{fmt(lbf256['mean_relative_mse_vs_best_stage_control_pct'])}%`。"
            "`r512` 没有带来稳定收益。"
        ),
        "",
        (
            "[Fact] A6 仍未形成明确 paper-core pass：按单 arm 统计，`A6-LBF-r256/r512` 对 "
            "`best_stage_control` 的 wins 均为 `0/12`，`A6-DER` 仅在 Weather 两个 setting 上略胜。"
        ),
        "",
        (
            "[Decision] A6-LBF 应标记为 `partial_pass_capacity_recovered_not_yet_core`。"
            "它证明 learned-basis dense-capacity path 有效，但还需要 best-val/early-stopping 或 objective-level "
            "诊断来判断 ETTh2 与 long horizon 的剩余差距。"
        ),
        "",
        "## Arm Summary",
        "",
        "| Arm | mean MSE | vs best control | wins | vs official unified | vs A5-B-r128 | vs A6-DER |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ["a6_der", "a6_lbf_r256", "a6_lbf_r512"]:
        row = arm_summary.loc[arm]
        lines.append(
            "| `{arm}` | {mse:.4f} | {best:+.2f}% | {wins}/12 | {official:+.2f}% | {a5b:+.2f}% | {der_rel:+.2f}% |".format(
                arm=arm,
                mse=row["mean_mse"],
                best=row["mean_relative_mse_vs_best_stage_control_pct"],
                wins=int(row["wins_vs_best_stage_control"]),
                official=row["mean_relative_mse_vs_official_unified_pct"],
                a5b=row["mean_relative_mse_vs_a5b_r128_pct"],
                der_rel=row["mean_relative_mse_vs_a6_der_pct"],
            )
        )
    lines.extend(
        [
            "",
            "## Best A6 Per Setting",
            "",
            f"Best-of-A6 oracle 平均仍相对 best stage control `{fmt(best_mean)}%`，wins `{best_wins}/12`。",
            "",
            "| Dataset | Horizon | Best A6 | MSE | Best control | Gap |",
            "| --- | ---: | --- | ---: | --- | ---: |",
        ]
    )
    for _, row in best.iterrows():
        lines.append(
            "| {dataset} | {horizon} | `{arm}` | {mse:.4f} | `{control}` | {gap:+.2f}% |".format(
                dataset=row["dataset"],
                horizon=int(row["target_horizon"]),
                arm=row["arm"],
                mse=row["mse"],
                control=row["best_stage_control_name"],
                gap=row["relative_mse_vs_best_stage_control_pct"],
            )
        )
    lines.extend(
        [
            "",
            "## Mechanism Interpretation",
            "",
            "### A6-DER ceiling",
            "",
            (
                "A6-DER 的结果支持一个重要机制判断：head operator capacity 是 A5 collapse 的主因之一。"
                "只要保留 dense-equivalent row capacity 并改成 prefix-native invocation，性能就从 A5-B/A5-Q 的明显失败"
                "恢复到 best controls 附近。"
            ),
            "",
            "### A6-LBF learned basis",
            "",
            (
                "A6-LBF-r256 与 A6-DER 的平均差距接近 0，说明 fixed Fourier/polynomial basis 是 A5-B 的关键瓶颈，"
                "而 learned temporal basis 可以在较低 rank 下近似 dense-equivalent capacity。r512 没有稳定优于 r256，"
                "第一轮不支持继续简单扩大 rank。"
            ),
            "",
            "### Remaining gap",
            "",
            (
                "A6 尚未超过 best stage controls。ETTh2 的 training summary 显示 A6-DER 在 epoch 1 已达到 best val，"
                "随后 last-val 变差；这提示 official-last checkpoint policy 可能低估 A6 on ETTh2。"
                "但在未跑 best-val 对照前，不能把 A6-LBF 升级为 paper-core pass。"
            ),
            "",
            "## Decision",
            "",
            "- `A6-DER_prefix_native_dense_equivalent_row_bank`: `control_passed_as_capacity_ceiling`。",
            "- `A6-LBF_learned_basis_forecast_operator`: `partial_pass_capacity_recovered_not_yet_core`。",
            "- 不建议继续 rank-only sweep；下一步优先做 best-val / early-stopping diagnostic，并同步检查 whether A6-LBF 的 learned basis 具备可解释 low-rank structure。",
            "",
            "## Artifacts",
            "",
            "- `phase5_timealign_hss_a6_metrics.csv`",
            "- `phase5_timealign_hss_a6_comparison.csv`",
            "- `phase5_timealign_hss_a6_summary.csv`",
            "- `phase5_timealign_hss_a6_best_per_setting.csv`",
            "- `phase5_timealign_hss_a6_training_summary.csv`",
            "- raw metrics/logs under ignored `raw/` directory",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase5_timealign_hss_a6_capacity_native_gate_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase5 A6 capacity-native gate.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--prior-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metrics = collect_metrics(args.raw_root)
    comparison = add_comparison_columns(metrics, args.prior_comparison)
    summary = summarize(comparison)
    best = best_per_setting(comparison)
    training = collect_training(args.raw_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase5_timealign_hss_a6_metrics.csv", metrics.to_dict("records"))
    write_csv(args.output_dir / "phase5_timealign_hss_a6_comparison.csv", comparison.to_dict("records"))
    write_csv(args.output_dir / "phase5_timealign_hss_a6_summary.csv", summary.to_dict("records"))
    write_csv(args.output_dir / "phase5_timealign_hss_a6_best_per_setting.csv", best.to_dict("records"))
    write_csv(args.output_dir / "phase5_timealign_hss_a6_training_summary.csv", training.to_dict("records"))
    (args.output_dir / "phase5_timealign_hss_a6_analysis_config.json").write_text(
        json.dumps(
            {
                "raw_root": str(args.raw_root),
                "prior_comparison": str(args.prior_comparison),
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir, comparison, summary, training)


if __name__ == "__main__":
    main()

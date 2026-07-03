from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


RUN_PREFIX = "TimeAlignOfficialUnified720_A6OD_"
RUN_SUFFIX = "_official-last"


def variant_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith(RUN_PREFIX) and part.endswith(RUN_SUFFIX):
            return part.removeprefix(RUN_PREFIX).removesuffix(RUN_SUFFIX)
    raise ValueError(f"Cannot parse A6OD variant from {path}")


def read_metrics(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = "official-last/*/ETTh2/mixed_h96_h192_h336_h720/seed2021/metrics_by_target_horizon.csv"
    for path in sorted(raw_root.glob(pattern)):
        variant = variant_from_path(path)
        frame = pd.read_csv(path)
        readout = "lbf_r256" if variant.startswith("lbf_r256") else "der"
        objective = variant.removeprefix("lbf_r256_").removeprefix("der_")
        for record in frame.to_dict("records"):
            rows.append(
                {
                    "dataset": record["dataset"],
                    "variant": variant,
                    "readout_family": readout,
                    "objective_variant": objective,
                    "target_horizon": int(record["target_horizon"]),
                    "mse": float(record["mse"]),
                    "mae": float(record["mae"]),
                    "source_path": str(path),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No A6OD metrics found under {raw_root}")
    return pd.DataFrame(rows)


def read_trajectory(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = "official-last/*/ETTh2/mixed_h96_h192_h336_h720/seed2021/training_log.csv"
    for path in sorted(raw_root.glob(pattern)):
        frame = pd.read_csv(path)
        variant = variant_from_path(path)
        best_idx = int(frame["val_mean_mse"].idxmin())
        best = float(frame.iloc[best_idx]["val_mean_mse"])
        last = float(frame.iloc[-1]["val_mean_mse"])
        first = float(frame.iloc[0]["val_mean_mse"])
        rows.append(
            {
                "variant": variant,
                "best_epoch": int(frame.iloc[best_idx]["epoch"]),
                "last_epoch": int(frame.iloc[-1]["epoch"]),
                "first_val_mean_mse": first,
                "best_val_mean_mse": best,
                "last_val_mean_mse": last,
                "last_vs_best_val_mse_pct": (last / best - 1.0) * 100.0,
                "last_vs_first_val_mse_pct": (last / first - 1.0) * 100.0,
                "source_path": str(path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No A6OD training logs found under {raw_root}")
    return pd.DataFrame(rows)


def add_references(metrics: pd.DataFrame, a6_comparison_path: Path) -> pd.DataFrame:
    a6 = pd.read_csv(a6_comparison_path)
    etth2 = a6[a6["dataset"].eq("ETTh2")]
    refs = etth2[
        [
            "target_horizon",
            "arm",
            "mse",
            "best_stage_control_mse",
            "best_stage_control_name",
        ]
    ]
    lbf = refs[refs["arm"].eq("a6_lbf_r256")][["target_horizon", "mse"]].rename(
        columns={"mse": "a6_lbf_r256_mse"}
    )
    der = refs[refs["arm"].eq("a6_der")][
        ["target_horizon", "mse", "best_stage_control_mse", "best_stage_control_name"]
    ].rename(columns={"mse": "a6_der_mse"})
    merged = metrics.merge(lbf, on="target_horizon", how="left", validate="many_to_one")
    merged = merged.merge(der, on="target_horizon", how="left", validate="many_to_one")
    required = ["a6_lbf_r256_mse", "a6_der_mse", "best_stage_control_mse"]
    if merged[required].isna().any().any():
        missing = merged[merged[required].isna().any(axis=1)][["variant", "target_horizon"]]
        raise ValueError(f"Missing reference rows: {missing.to_dict('records')}")
    merged["relative_mse_vs_a6_lbf_r256_pct"] = (merged["mse"] / merged["a6_lbf_r256_mse"] - 1.0) * 100.0
    merged["relative_mse_vs_a6_der_pct"] = (merged["mse"] / merged["a6_der_mse"] - 1.0) * 100.0
    merged["relative_mse_vs_best_stage_control_pct"] = (
        merged["mse"] / merged["best_stage_control_mse"] - 1.0
    ) * 100.0
    merged["beats_best_stage_control"] = merged["mse"] < merged["best_stage_control_mse"]
    return merged


def summarize(comparison: pd.DataFrame, trajectory: pd.DataFrame) -> pd.DataFrame:
    summary = (
        comparison.groupby(["variant", "readout_family", "objective_variant"], as_index=False)
        .agg(
            mean_mse=("mse", "mean"),
            mean_relative_mse_vs_a6_lbf_r256_pct=("relative_mse_vs_a6_lbf_r256_pct", "mean"),
            mean_relative_mse_vs_a6_der_pct=("relative_mse_vs_a6_der_pct", "mean"),
            mean_relative_mse_vs_best_stage_control_pct=("relative_mse_vs_best_stage_control_pct", "mean"),
            wins_vs_best_stage_control=("beats_best_stage_control", "sum"),
        )
    )
    return summary.merge(trajectory, on="variant", how="left", validate="one_to_one").sort_values(
        ["readout_family", "objective_variant"]
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


def fmt(value: float) -> str:
    return f"{value:.2f}"


def write_report(output_dir: Path, comparison: pd.DataFrame, summary: pd.DataFrame) -> None:
    best_idx = int(summary["mean_relative_mse_vs_best_stage_control_pct"].idxmin())
    best = summary.loc[best_idx]
    lbf = summary[summary["readout_family"].eq("lbf_r256")]
    der = summary[summary["readout_family"].eq("der")]
    lines = [
        "# Phase5-A6OD Objective Drift Diagnostic Report",
        "",
        "本文档分析 ETTh2-only A6 objective drift diagnostic。该实验为 diagnostic-only，所有 run 使用 "
        "`official-last` / without early stop。",
        "",
        "## Conclusion",
        "",
        (
            "[Strong Evidence] objective switch 没有修复 A6 的 ETTh2 gap。最佳 variant "
            f"`{best['variant']}` 相对 best stage control 平均 `{fmt(best['mean_relative_mse_vs_best_stage_control_pct'])}%`，"
            f"wins `{int(best['wins_vs_best_stage_control'])}/4`。"
        ),
        "",
        (
            "[Fact] `full` 与 stochastic/continuous prefix 目标没有消除 official-last drift："
            f"最佳 variant 的 last-vs-best validation drift 仍为 `{fmt(best['last_vs_best_val_mse_pct'])}%`。"
        ),
        "",
        "## Variant Summary",
        "",
        "| Variant | Family | Objective | mean MSE | vs best control | wins | last-vs-best val | best epoch |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| `{variant}` | `{family}` | `{objective}` | {mse:.4f} | {gap:+.2f}% | {wins}/4 | {drift:+.2f}% | {epoch} |".format(
                variant=row["variant"],
                family=row["readout_family"],
                objective=row["objective_variant"],
                mse=row["mean_mse"],
                gap=row["mean_relative_mse_vs_best_stage_control_pct"],
                wins=int(row["wins_vs_best_stage_control"]),
                drift=row["last_vs_best_val_mse_pct"],
                epoch=int(row["best_epoch"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "[Strong Evidence] 仅调整 prefix supervision sampling 不足以解释或修复 A6-LBF 的 ETTh2 partial-pass："
                f"LBF variants 的 best-control gap 范围是 `{fmt(float(lbf['mean_relative_mse_vs_best_stage_control_pct'].min()))}%` "
                f"到 `{fmt(float(lbf['mean_relative_mse_vs_best_stage_control_pct'].max()))}%`，"
                f"DER variants 范围是 `{fmt(float(der['mean_relative_mse_vs_best_stage_control_pct'].min()))}%` "
                f"到 `{fmt(float(der['mean_relative_mse_vs_best_stage_control_pct'].max()))}%`。"
            ),
            "",
            (
                "[Decision] A6OD 不通过 repair gate。下一步不应继续 objective-sampling sweep；应回 Step 4/5 "
                "设计 explicit stability path，例如 official-last-compatible regularization、teacher/nested stability "
                "control，或重新评估 best controls 的 regularization advantage。"
            ),
            "",
            "## Statistic Definitions",
            "",
            "`relative_mse_vs_best_stage_control_pct` 来自每个 target horizon 的 final test MSE 与 A6 comparison "
            "中 ETTh2 best stage control MSE 的比值。`last_vs_best_val_mse_pct` 来自该 run 的 `training_log.csv`，"
            "计算 `last_val_mean_mse / best_val_mean_mse - 1`，用于衡量 official-last drift。",
            "",
            "## Artifacts",
            "",
            "- `phase5_timealign_hss_a6od_comparison.csv`",
            "- `phase5_timealign_hss_a6od_summary.csv`",
            "- `phase5_timealign_hss_a6od_analysis_config.json`",
            "- ignored raw metrics/logs under `raw/`",
            "",
        ]
    )
    (output_dir / "phase5_timealign_hss_a6_objective_drift_diagnostic_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase5 A6 objective drift diagnostic.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--a6-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics(args.raw_root)
    trajectory = read_trajectory(args.raw_root)
    comparison = add_references(metrics, args.a6_comparison)
    summary = summarize(comparison, trajectory)

    write_csv(output_dir / "phase5_timealign_hss_a6od_comparison.csv", comparison.to_dict("records"))
    write_csv(output_dir / "phase5_timealign_hss_a6od_summary.csv", summary.to_dict("records"))
    (output_dir / "phase5_timealign_hss_a6od_analysis_config.json").write_text(
        json.dumps(
            {
                "raw_root": str(args.raw_root),
                "a6_comparison": str(args.a6_comparison),
                "output_dir": str(output_dir),
                "protocol": "official-last / without early stop",
                "role": "diagnostic_only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(output_dir, comparison, summary)


if __name__ == "__main__":
    main()

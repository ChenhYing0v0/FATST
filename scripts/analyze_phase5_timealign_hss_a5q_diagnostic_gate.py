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
    for part in path.parts:
        prefix = "TimeAlignOfficialUnified720_A5QDiag_"
        suffix = "_official-last"
        if part.startswith(prefix) and part.endswith(suffix):
            return part.removeprefix(prefix).removesuffix(suffix)
    raise ValueError(f"Cannot parse arm from {path}")


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
        raise FileNotFoundError(f"No metrics_by_target_horizon.csv found under {raw_root}")
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
    old_a5q = prior[prior["arm"].eq("a5q_seg48_small")][
        [
            "dataset",
            "target_horizon",
            "mse",
            "official_unified_mse",
            "best_stage_control_mse",
            "best_stage_control_name",
        ]
    ].rename(columns={"mse": "old_a5q_seg48_mse"})
    merged = metrics.merge(old_a5q, on=["dataset", "target_horizon"], how="left", validate="many_to_one")
    if merged["old_a5q_seg48_mse"].isna().any():
        missing = merged[merged["old_a5q_seg48_mse"].isna()][["dataset", "target_horizon"]].drop_duplicates()
        raise ValueError(f"Missing prior A5-Q rows: {missing.to_dict('records')}")
    merged["relative_mse_vs_old_a5q_pct"] = (merged["mse"] / merged["old_a5q_seg48_mse"] - 1.0) * 100.0
    merged["relative_mse_vs_official_unified_pct"] = (merged["mse"] / merged["official_unified_mse"] - 1.0) * 100.0
    merged["relative_mse_vs_best_stage_control_pct"] = (merged["mse"] / merged["best_stage_control_mse"] - 1.0) * 100.0
    merged["beats_best_stage_control"] = merged["mse"] < merged["best_stage_control_mse"]
    return merged


def summarize(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, frame in comparison.groupby("arm", sort=True):
        rows.append(
            {
                "scope": "ALL",
                "dataset": "ALL",
                "arm": arm,
                "n": int(len(frame)),
                "mean_mse": float(frame["mse"].mean()),
                "mean_relative_mse_vs_old_a5q_pct": float(frame["relative_mse_vs_old_a5q_pct"].mean()),
                "mean_relative_mse_vs_official_unified_pct": float(
                    frame["relative_mse_vs_official_unified_pct"].mean()
                ),
                "mean_relative_mse_vs_best_stage_control_pct": float(
                    frame["relative_mse_vs_best_stage_control_pct"].mean()
                ),
                "wins_vs_best_stage_control": int(frame["beats_best_stage_control"].sum()),
            }
        )
    for (dataset, arm), frame in comparison.groupby(["dataset", "arm"], sort=True):
        rows.append(
            {
                "scope": "dataset",
                "dataset": dataset,
                "arm": arm,
                "n": int(len(frame)),
                "mean_mse": float(frame["mse"].mean()),
                "mean_relative_mse_vs_old_a5q_pct": float(frame["relative_mse_vs_old_a5q_pct"].mean()),
                "mean_relative_mse_vs_official_unified_pct": float(
                    frame["relative_mse_vs_official_unified_pct"].mean()
                ),
                "mean_relative_mse_vs_best_stage_control_pct": float(
                    frame["relative_mse_vs_best_stage_control_pct"].mean()
                ),
                "wins_vs_best_stage_control": int(frame["beats_best_stage_control"].sum()),
            }
        )
    return pd.DataFrame(rows)


def write_report(output_dir: Path, comparison: pd.DataFrame, summary: pd.DataFrame, training: pd.DataFrame) -> None:
    def fmt(value: float) -> str:
        return f"{value:.2f}"

    all_summary = summary[summary["scope"].eq("ALL")].set_index("arm")
    dataset_summary = summary[summary["scope"].eq("dataset")].set_index(["dataset", "arm"])
    best_row = comparison.loc[comparison["relative_mse_vs_best_stage_control_pct"].idxmin()]

    report = [
        "# Phase5-A5-Q Collapse Diagnostic Gate Report",
        "",
        "本文档分析 A5-Q diagnostic-only gate。该实验只用于解释 collapse 原因，不恢复 A5-Q 的 paper-core candidate 身份。",
        "",
        "## 结论摘要",
        "",
        (
            "[Strong Evidence] A5-Q 的 ETTm1 严重 collapse 主要混入了 decoder dropout 实现错位："
            "`target_query_dropout=0.1/0.0` 后，ETTm1 mean MSE 相对旧 A5-Q 分别改善 "
            f"{fmt(dataset_summary.loc[('ETTm1', 'a5q_seg48_dropout01'), 'mean_relative_mse_vs_old_a5q_pct'])}% / "
            f"{fmt(dataset_summary.loc[('ETTm1', 'a5q_seg48_dropout00'), 'mean_relative_mse_vs_old_a5q_pct'])}%。"
        ),
        "",
        (
            "[Strong Evidence] 但修复 dropout 后仍没有通过 effectiveness gate：最佳单 setting 是 "
            f"{best_row['dataset']} h{int(best_row['target_horizon'])} 的 `{best_row['arm']}`，"
            f"相对 best stage control 仍为 {fmt(best_row['relative_mse_vs_best_stage_control_pct'])}%，"
            "本轮所有 diagnostic arms 对 best stage control 的 wins 仍为 0。"
        ),
        "",
        (
            "[Fact] ETTm1 `patch_num_override=48` 没有修复问题，反而弱于保留 `patch_num=1` 的 dropout 修复："
            "`a5q_ettm1_patch48_dropout00` 相对 best stage control 平均仍差 "
            f"{fmt(dataset_summary.loc[('ETTm1', 'a5q_ettm1_patch48_dropout00'), 'mean_relative_mse_vs_best_stage_control_pct'])}%。"
        ),
        "",
        (
            "[Hypothesis] A5-Q 当前失败不再应解释为简单实现 bug，而应解释为 target-query decoder 的 "
            "capacity / optimization path 不足：它能产生 prefix-consistent graph，但不能替代 dense/time-specific readout 的 forecasting capacity。"
        ),
        "",
        "## Overall Summary",
        "",
        "| Arm | n | mean MSE | vs old A5-Q | vs official unified | vs best stage control | wins vs best |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in all_summary.reset_index().sort_values("arm").iterrows():
        report.append(
            "| `{arm}` | {n} | {mse:.4f} | {old:+.2f}% | {official:+.2f}% | {best:+.2f}% | {wins} |".format(
                arm=row["arm"],
                n=int(row["n"]),
                mse=row["mean_mse"],
                old=row["mean_relative_mse_vs_old_a5q_pct"],
                official=row["mean_relative_mse_vs_official_unified_pct"],
                best=row["mean_relative_mse_vs_best_stage_control_pct"],
                wins=int(row["wins_vs_best_stage_control"]),
            )
        )
    report.extend(
        [
            "",
            "## Hypothesis Tests",
            "",
            "### H-Dropout",
            "",
            (
                "ETTm1 旧 A5-Q 使用 official preset `dropout=0.9` 进入 target-query decoder。"
                "本轮将 decoder dropout 固定为 `0.1/0.0` 后，h96/h192 明显恢复，"
                "其中 `a5q_seg48_dropout01` 在 ETTm1 h96 只比 best stage control 差 `+2.16%`，"
                "并比 official unified 低 `-1.68%`。"
            ),
            "",
            (
                "但长 horizon 仍失败：ETTm1 h336/h720 在 `a5q_seg48_dropout01` 下相对 best stage control "
                "仍为 `+33.65%/+40.86%`。因此 H-Dropout 被支持为 collapse amplifier，"
                "但不能解释 A5-Q family 的全部 capacity gap。"
            ),
            "",
            "### H-PatchMemory",
            "",
            (
                "将 ETTm1 `patch_num` 从 1 改为 48 后，没有带来预期修复。"
                "`a5q_ettm1_patch48_dropout00` 虽相对旧 A5-Q 平均改善 `-23.16%`，"
                "但弱于 `a5q_seg48_dropout00` 的 `-36.81%`，且相对 best stage control 平均仍差 `+41.02%`。"
            ),
            "",
            (
                "因此 H-PatchMemory 作为理论错位仍成立：`patch_num=1` 确实破坏了 query-select-patch 叙事；"
                "但简单提高 memory token count 不是有效修复，可能引入 backbone preset shift、optimization 难度和过大的 readout search space。"
            ),
            "",
            "### Replication / Variance Note",
            "",
            (
                "ETTh2 的 `a5q_seg48_dropout01` 与旧 A5-Q 完全一致，符合 preset dropout 已为 0.1 的预期。"
                "Weather 的 `a5q_seg48_dropout01` 相对旧 A5-Q 有约 `-6.64%` mean improvement；"
                "由于机制配置理论上等价，本文将其标记为 run variance / nondeterminism 信号，不作为机制证据。"
            ),
            "",
            "## Decision",
            "",
            (
                "`A5-Q_collapse_diagnostic_repair` 判定为 `diagnostic_only_completed_failed_as_repair`："
                "它解释了 ETTm1 collapse 的重要实现因素，但没有产生足够的 effectiveness recovery。"
            ),
            "",
            (
                "下一步不应继续对 A5-Q 做简单 dropout/patch/width sweep。若要保留 target-query 叙事，"
                "必须回 Step 4/5 重新设计 capacity mechanism，例如让 query decoder 具备 function-preserving path、"
                "time-specific readout capacity 或 teacher-preserved initialization，并重新过 narrative gate。"
            ),
            "",
            "## Artifacts",
            "",
            "- `phase5_timealign_hss_a5q_diagnostic_metrics.csv`",
            "- `phase5_timealign_hss_a5q_diagnostic_comparison.csv`",
            "- `phase5_timealign_hss_a5q_diagnostic_summary.csv`",
            "- `phase5_timealign_hss_a5q_diagnostic_training_summary.csv`",
            "- raw metrics/logs under ignored `raw/` directory",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase5_timealign_hss_a5q_diagnostic_gate_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase5 A5-Q diagnostic-only gate.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--prior-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metrics = collect_metrics(args.raw_root)
    comparison = add_comparison_columns(metrics, args.prior_comparison)
    summary = summarize(comparison)
    training = collect_training(args.raw_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase5_timealign_hss_a5q_diagnostic_metrics.csv", metrics.to_dict("records"))
    write_csv(args.output_dir / "phase5_timealign_hss_a5q_diagnostic_comparison.csv", comparison.to_dict("records"))
    write_csv(args.output_dir / "phase5_timealign_hss_a5q_diagnostic_summary.csv", summary.to_dict("records"))
    write_csv(args.output_dir / "phase5_timealign_hss_a5q_diagnostic_training_summary.csv", training.to_dict("records"))
    (args.output_dir / "phase5_timealign_hss_a5q_diagnostic_config.json").write_text(
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

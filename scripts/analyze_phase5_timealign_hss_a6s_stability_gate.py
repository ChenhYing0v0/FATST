from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


RUN_PREFIX = "TimeAlignOfficialUnified720_A6S_"
RUN_SUFFIX = "_official-last"


def variant_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith(RUN_PREFIX) and part.endswith(RUN_SUFFIX):
            return part.removeprefix(RUN_PREFIX).removesuffix(RUN_SUFFIX)
    raise ValueError(f"Cannot parse A6S variant from {path}")


def dataset_from_path(path: Path) -> str:
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part.startswith(RUN_PREFIX) and part.endswith(RUN_SUFFIX):
            if index + 1 < len(parts):
                return parts[index + 1]
    raise ValueError(f"Cannot parse dataset from {path}")


def read_metrics(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = "official-last/*/*/mixed_h96_h192_h336_h720/seed2021/metrics_by_target_horizon.csv"
    for path in sorted(raw_root.glob(pattern)):
        variant = variant_from_path(path)
        frame = pd.read_csv(path)
        for record in frame.to_dict("records"):
            rows.append(
                {
                    "dataset": record["dataset"],
                    "variant": variant,
                    "target_horizon": int(record["target_horizon"]),
                    "mse": float(record["mse"]),
                    "mae": float(record["mae"]),
                    "source_path": str(path),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No A6S metrics found under {raw_root}")
    return pd.DataFrame(rows)


def read_trajectory(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = "official-last/*/*/mixed_h96_h192_h336_h720/seed2021/training_log.csv"
    for path in sorted(raw_root.glob(pattern)):
        frame = pd.read_csv(path)
        variant = variant_from_path(path)
        dataset = dataset_from_path(path)
        best_idx = int(frame["val_mean_mse"].idxmin())
        best = float(frame.iloc[best_idx]["val_mean_mse"])
        last = float(frame.iloc[-1]["val_mean_mse"])
        row: dict[str, Any] = {
            "dataset": dataset,
            "variant": variant,
            "best_epoch": int(frame.iloc[best_idx]["epoch"]),
            "last_epoch": int(frame.iloc[-1]["epoch"]),
            "last_train_loss": float(frame.iloc[-1]["train_loss"]),
            "first_val_mean_mse": float(frame.iloc[0]["val_mean_mse"]),
            "best_val_mean_mse": best,
            "last_val_mean_mse": last,
            "last_vs_best_val_mse_pct": (last / best - 1.0) * 100.0,
            "ema_decay": float(frame.iloc[-1].get("ema_decay", 0.0)),
            "ema_eval": int(frame.iloc[-1].get("ema_eval", 0)),
            "self_teacher_decay": float(frame.iloc[-1].get("self_teacher_decay", 0.0)),
            "self_teacher_loss_weight": float(frame.iloc[-1].get("self_teacher_loss_weight", 0.0)),
            "self_teacher_warmup_epochs": int(frame.iloc[-1].get("self_teacher_warmup_epochs", 0)),
            "self_teacher_gate_mode": frame.iloc[-1].get("self_teacher_gate_mode", "none"),
            "self_teacher_gate_threshold": float(frame.iloc[-1].get("self_teacher_gate_threshold", 0.0)),
            "self_teacher_gate_temperature": float(frame.iloc[-1].get("self_teacher_gate_temperature", 1.0)),
            "last_train_self_teacher_l1": float(frame.iloc[-1].get("train_self_teacher_l1", 0.0)),
            "last_train_self_teacher_target_l1": float(
                frame.iloc[-1].get("train_self_teacher_target_l1", 0.0)
            ),
            "last_train_self_teacher_advantage_l1": float(
                frame.iloc[-1].get("train_self_teacher_advantage_l1", 0.0)
            ),
            "last_train_self_teacher_gate": float(frame.iloc[-1].get("train_self_teacher_gate", 1.0)),
            "last_train_weighted_self_teacher_l1": float(
                frame.iloc[-1].get("train_weighted_self_teacher_l1", frame.iloc[-1].get("train_self_teacher_l1", 0.0))
            ),
            "basis_operator_smoothness_weight": float(
                frame.iloc[-1].get("basis_operator_smoothness_weight", 0.0)
            ),
            "last_train_basis_operator_smoothness_loss": float(
                frame.iloc[-1].get("train_basis_operator_smoothness_loss", 0.0)
            ),
            "source_path": str(path),
        }
        row["last_weighted_basis_operator_smoothness_loss"] = (
            row["basis_operator_smoothness_weight"]
            * row["last_train_basis_operator_smoothness_loss"]
        )
        row["last_weighted_smoothness_to_train_loss"] = (
            row["last_weighted_basis_operator_smoothness_loss"] / row["last_train_loss"]
            if row["last_train_loss"] > 0.0
            else 0.0
        )
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No A6S training logs found under {raw_root}")
    return pd.DataFrame(rows)


def add_references(metrics: pd.DataFrame, a6_comparison_path: Path) -> pd.DataFrame:
    a6 = pd.read_csv(a6_comparison_path)
    lbf = a6[a6["arm"].eq("a6_lbf_r256")][["dataset", "target_horizon", "mse"]].rename(
        columns={"mse": "a6_lbf_r256_mse"}
    )
    der = a6[a6["arm"].eq("a6_der")][
        ["dataset", "target_horizon", "mse", "best_stage_control_mse", "best_stage_control_name"]
    ].rename(columns={"mse": "a6_der_mse"})
    merged = metrics.merge(lbf, on=["dataset", "target_horizon"], how="left", validate="many_to_one")
    merged = merged.merge(der, on=["dataset", "target_horizon"], how="left", validate="many_to_one")
    required = ["a6_lbf_r256_mse", "a6_der_mse", "best_stage_control_mse"]
    if merged[required].isna().any().any():
        missing = merged[merged[required].isna().any(axis=1)][["dataset", "variant", "target_horizon"]]
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
        comparison.groupby("variant", as_index=False)
        .agg(
            mean_mse=("mse", "mean"),
            mean_relative_mse_vs_a6_lbf_r256_pct=("relative_mse_vs_a6_lbf_r256_pct", "mean"),
            mean_relative_mse_vs_a6_der_pct=("relative_mse_vs_a6_der_pct", "mean"),
            mean_relative_mse_vs_best_stage_control_pct=("relative_mse_vs_best_stage_control_pct", "mean"),
            wins_vs_best_stage_control=("beats_best_stage_control", "sum"),
            horizon_count=("beats_best_stage_control", "count"),
        )
    )
    trajectory_summary = (
        trajectory.groupby("variant", as_index=False)
        .agg(
            best_epoch=("best_epoch", "min"),
            last_epoch=("last_epoch", "max"),
            last_train_loss=("last_train_loss", "mean"),
            first_val_mean_mse=("first_val_mean_mse", "mean"),
            best_val_mean_mse=("best_val_mean_mse", "mean"),
            last_val_mean_mse=("last_val_mean_mse", "mean"),
            last_vs_best_val_mse_pct=("last_vs_best_val_mse_pct", "mean"),
            ema_decay=("ema_decay", "first"),
            ema_eval=("ema_eval", "first"),
            self_teacher_decay=("self_teacher_decay", "first"),
            self_teacher_loss_weight=("self_teacher_loss_weight", "first"),
            self_teacher_warmup_epochs=("self_teacher_warmup_epochs", "first"),
            self_teacher_gate_mode=("self_teacher_gate_mode", "first"),
            self_teacher_gate_threshold=("self_teacher_gate_threshold", "first"),
            self_teacher_gate_temperature=("self_teacher_gate_temperature", "first"),
            last_train_self_teacher_l1=("last_train_self_teacher_l1", "mean"),
            last_train_self_teacher_target_l1=("last_train_self_teacher_target_l1", "mean"),
            last_train_self_teacher_advantage_l1=("last_train_self_teacher_advantage_l1", "mean"),
            last_train_self_teacher_gate=("last_train_self_teacher_gate", "mean"),
            last_train_weighted_self_teacher_l1=("last_train_weighted_self_teacher_l1", "mean"),
            basis_operator_smoothness_weight=("basis_operator_smoothness_weight", "first"),
            last_train_basis_operator_smoothness_loss=("last_train_basis_operator_smoothness_loss", "mean"),
            last_weighted_basis_operator_smoothness_loss=(
                "last_weighted_basis_operator_smoothness_loss",
                "mean",
            ),
            last_weighted_smoothness_to_train_loss=("last_weighted_smoothness_to_train_loss", "mean"),
            source_path=("source_path", "first"),
        )
    )
    return summary.merge(trajectory_summary, on="variant", how="left", validate="one_to_one").sort_values(
        "mean_relative_mse_vs_best_stage_control_pct"
    )


def summarize_by_dataset(comparison: pd.DataFrame, trajectory: pd.DataFrame) -> pd.DataFrame:
    summary = (
        comparison.groupby(["dataset", "variant"], as_index=False)
        .agg(
            mean_mse=("mse", "mean"),
            mean_relative_mse_vs_a6_lbf_r256_pct=("relative_mse_vs_a6_lbf_r256_pct", "mean"),
            mean_relative_mse_vs_a6_der_pct=("relative_mse_vs_a6_der_pct", "mean"),
            mean_relative_mse_vs_best_stage_control_pct=("relative_mse_vs_best_stage_control_pct", "mean"),
            wins_vs_best_stage_control=("beats_best_stage_control", "sum"),
            horizon_count=("beats_best_stage_control", "count"),
        )
    )
    trajectory_summary = (
        trajectory.groupby(["dataset", "variant"], as_index=False)
        .agg(
            best_epoch=("best_epoch", "min"),
            last_epoch=("last_epoch", "max"),
            last_train_loss=("last_train_loss", "mean"),
            first_val_mean_mse=("first_val_mean_mse", "mean"),
            best_val_mean_mse=("best_val_mean_mse", "mean"),
            last_val_mean_mse=("last_val_mean_mse", "mean"),
            last_vs_best_val_mse_pct=("last_vs_best_val_mse_pct", "mean"),
            last_train_self_teacher_l1=("last_train_self_teacher_l1", "mean"),
            last_train_self_teacher_target_l1=("last_train_self_teacher_target_l1", "mean"),
            last_train_self_teacher_advantage_l1=("last_train_self_teacher_advantage_l1", "mean"),
            last_train_self_teacher_gate=("last_train_self_teacher_gate", "mean"),
            last_train_weighted_self_teacher_l1=("last_train_weighted_self_teacher_l1", "mean"),
            source_path=("source_path", "first"),
        )
    )
    return summary.merge(
        trajectory_summary,
        on=["dataset", "variant"],
        how="left",
        validate="one_to_one",
    ).sort_values(["dataset", "mean_relative_mse_vs_best_stage_control_pct"])


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


def write_report(output_dir: Path, summary: pd.DataFrame, dataset_summary: pd.DataFrame) -> None:
    best = summary.iloc[0]
    datasets = ", ".join(sorted(dataset_summary["dataset"].unique()))
    smooth = summary[summary["basis_operator_smoothness_weight"].gt(0.0)]
    max_smooth_ratio = (
        float(smooth["last_weighted_smoothness_to_train_loss"].max()) if not smooth.empty else 0.0
    )
    output_name = output_dir.name.lower()
    if "a8tag" in output_name:
        gate_label = "A8TAG"
    elif "a7dg" in output_name:
        gate_label = "A7DG"
    elif "a6st" in output_name:
        gate_label = "A6ST"
    elif "a6s2" in output_name:
        gate_label = "A6S2"
    else:
        gate_label = "A6S"
    title = (
        "# Phase5-A8TAG Official-Last Teacher-Advantage Gate Report"
        if gate_label == "A8TAG"
        else
        "# Phase5-A7DG Official-Last Selective Self-Teacher Gate Report"
        if gate_label == "A7DG"
        else
        "# Phase5-A6ST Official-Last Self-Teacher Gate Report"
        if gate_label == "A6ST"
        else
        "# Phase5-A6S2 Official-Last Stability Calibration Gate Report"
        if gate_label == "A6S2"
        else "# Phase5-A6S Official-Last Stability Gate Report"
    )
    lines = [
        title,
        "",
        f"本文档分析 {gate_label} stability gate。数据集范围：`{datasets}`。所有 run 使用 `official-last` / without early stop。",
        "",
        "## Conclusion",
        "",
        (
            "[Fact] 最佳 variant 为 "
            f"`{best['variant']}`：相对 best stage control 平均 `{best['mean_relative_mse_vs_best_stage_control_pct']:+.2f}%`，"
            f"wins `{int(best['wins_vs_best_stage_control'])}/{int(best['horizon_count'])}`，last-vs-best validation drift "
            f"`{best['last_vs_best_val_mse_pct']:+.2f}%`。"
        ),
        "",
        (
            "[Strong Evidence] 最佳 variant 相对 A6-LBF-r256 的平均 MSE 变化为 "
            f"`{best['mean_relative_mse_vs_a6_lbf_r256_pct']:+.2f}%`。"
        ),
        "",
    ]
    if gate_label in {"A6ST", "A7DG"}:
        lines.extend(
            [
                (
                    "[Fact] self-teacher 为 train-time detached EMA teacher consistency；最终评估仍使用 raw "
                    "`official-last` student weights，不使用 `ema_eval`。"
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                (
                    "[Fact] 本轮 smoothness regularizer 的最大实际强度为 "
                    f"`weighted_smoothness / train_loss = {max_smooth_ratio:.2e}`。"
                    "该值用于判断 regularizer 是否真的进入优化，而不是只看 flag 是否开启。"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Variant Summary",
            "",
            "| Variant | mean MSE | vs best control | wins | vs A6-LBF-r256 | last-vs-best val | EMA | self-teacher | gate | smooth/train |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            "| `{variant}` | {mse:.4f} | {gap:+.2f}% | {wins}/{count} | {lbf:+.2f}% | {drift:+.2f}% | {ema:.3g} | w={stw:.3g}, d={std:.3g}, wu={warmup} | {gate_mode}:{gate:.2f} | {ratio:.2e} |".format(
                variant=row["variant"],
                mse=row["mean_mse"],
                gap=row["mean_relative_mse_vs_best_stage_control_pct"],
                wins=int(row["wins_vs_best_stage_control"]),
                count=int(row["horizon_count"]),
                lbf=row["mean_relative_mse_vs_a6_lbf_r256_pct"],
                drift=row["last_vs_best_val_mse_pct"],
                ema=row["ema_decay"],
                stw=row["self_teacher_loss_weight"],
                std=row["self_teacher_decay"],
                warmup=int(row["self_teacher_warmup_epochs"]),
                gate_mode=row["self_teacher_gate_mode"],
                gate=row["last_train_self_teacher_gate"],
                ratio=row["last_weighted_smoothness_to_train_loss"],
            )
        )
    lines.extend(
        [
            "",
            "## Dataset Summary",
            "",
            "| Dataset | Variant | mean MSE | vs best control | wins | vs A6-LBF-r256 | last-vs-best val | gate | teacher advantage |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in dataset_summary.iterrows():
        lines.append(
            "| `{dataset}` | `{variant}` | {mse:.4f} | {gap:+.2f}% | {wins}/{count} | {lbf:+.2f}% | {drift:+.2f}% | {gate:.2f} | {adv:+.4f} |".format(
                dataset=row["dataset"],
                variant=row["variant"],
                mse=row["mean_mse"],
                gap=row["mean_relative_mse_vs_best_stage_control_pct"],
                wins=int(row["wins_vs_best_stage_control"]),
                count=int(row["horizon_count"]),
                lbf=row["mean_relative_mse_vs_a6_lbf_r256_pct"],
                drift=row["last_vs_best_val_mse_pct"],
                gate=row["last_train_self_teacher_gate"],
                adv=row["last_train_self_teacher_advantage_l1"],
            )
        )
    lines.extend(["", "## Gate Decision", ""])
    if gate_label == "A8TAG":
        lines.extend(
            [
                "[Decision] A8TAG effectiveness 必须同时看三点：teacher 是否在 supervised prefix 上确实有正 advantage，teacher-advantage gate 是否避免低质量 teacher imitation，以及 metrics 是否超过 A7DG/A6-LBF。",
                "",
                "[Decision] 若 teacher advantage 多数为负或接近零，说明 EMA teacher 不是可靠 target，应停止 self-teacher route。",
                "",
                "[Decision] 若 teacher advantage gate 改善 ETTm1/Weather 但损失 ETTh2 gain，则需要回 Step 4/5 重新建模 stability 与 capacity 的冲突，而不是加回 threshold。",
                "",
            ]
        )
    elif gate_label == "A7DG":
        lines.extend(
            [
                "[Decision] A7DG effectiveness 必须同时看三点：是否保留 ETTh2 positive signal，是否降低 ETTm1/Weather 的 uniform A6ST 负向，以及 `train_self_teacher_gate` 是否按 dataset 产生选择性降权。",
                "",
                "[Decision] 若 A7DG 只优于 uniform A6ST 但仍系统性弱于 A6-LBF 或 best controls，则它只能作为 selective-stability partial evidence，不能直接升级为 paper-core。",
                "",
                "[Decision] 若 gate 强度在 ETTh2 显著高于 ETTm1/Weather，且 metrics 接近 A6-LBF，则下一步应围绕 adaptive/selective stability objective 做更严格 narrative gate，而不是继续人工调 threshold。",
                "",
            ]
        )
    elif gate_label == "A6ST":
        lines.extend(
            [
                "[Decision] 该 gate 的 effectiveness 必须同时检查 raw final checkpoint 是否改善、是否跨 dataset 安全、以及是否只是 ETTh2-specific repair。",
                "",
                "[Decision] 若 ETTm1/Weather 出现系统性负向，即使 ETTh2 改善，也不能把当前 self-teacher setting 升级为 paper-core universal method。",
                "",
                "[Decision] 下一步应回 Step 4/5 重审为什么 stability target 对 ETTh2 有益但对 ETTm1/Weather 负向；不得直接做 full-matrix 扩大实验。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "[Decision] 该 gate 的 effectiveness 必须结合 best-control gap、wins、A6-LBF 相对改善和 regularizer 实际强度判断；不能只看单个 variant 的平均 MSE。",
                "",
                "[Decision] 若改善主要来自 EMA，则它首先是 generic trajectory-averaging control evidence，不能直接升级为 paper-core。",
                "",
                "[Decision] 若 stronger smoothness 独立改善，才支持继续设计 operator-level stability mechanism；若 stronger smoothness 变差，则应暂停该 route。",
                "",
            ]
        )
    lines.extend(
        [
            "## Reader Path",
            "",
            "先读取 `phase5_timealign_hss_a6s_summary.csv` 判断 variant-level gate，再读取 `phase5_timealign_hss_a6s_dataset_summary.csv` 判断 dataset-level 安全性，最后读取 `phase5_timealign_hss_a6s_comparison.csv` 判断 prefix-wise wins/gaps，并回到 stage ledger 写入 11-step decision。",
            "",
            "## Artifacts",
            "",
            "- `phase5_timealign_hss_a6s_comparison.csv`",
            "- `phase5_timealign_hss_a6s_summary.csv`",
            "- `phase5_timealign_hss_a6s_dataset_summary.csv`",
            "- `phase5_timealign_hss_a6s_analysis_config.json`",
            "- ignored raw metrics/logs under `raw/`",
            "",
        ]
    )
    (output_dir / "phase5_timealign_hss_a6s_stability_gate_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase5 A6S stability gate.")
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
    dataset_summary = summarize_by_dataset(comparison, trajectory)

    write_csv(output_dir / "phase5_timealign_hss_a6s_comparison.csv", comparison.to_dict("records"))
    write_csv(output_dir / "phase5_timealign_hss_a6s_summary.csv", summary.to_dict("records"))
    write_csv(
        output_dir / "phase5_timealign_hss_a6s_dataset_summary.csv",
        dataset_summary.to_dict("records"),
    )
    (output_dir / "phase5_timealign_hss_a6s_analysis_config.json").write_text(
        json.dumps(
            {
                "raw_root": str(args.raw_root),
                "a6_comparison": str(args.a6_comparison),
                "output_dir": str(output_dir),
                "protocol": "official-last / without early stop",
                "role": "diagnostic_control_first",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(output_dir, summary, dataset_summary)


if __name__ == "__main__":
    main()

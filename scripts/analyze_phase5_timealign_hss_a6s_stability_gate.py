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


def read_metrics(raw_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = "official-last/*/ETTh2/mixed_h96_h192_h336_h720/seed2021/metrics_by_target_horizon.csv"
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
    pattern = "official-last/*/ETTh2/mixed_h96_h192_h336_h720/seed2021/training_log.csv"
    for path in sorted(raw_root.glob(pattern)):
        frame = pd.read_csv(path)
        variant = variant_from_path(path)
        best_idx = int(frame["val_mean_mse"].idxmin())
        best = float(frame.iloc[best_idx]["val_mean_mse"])
        last = float(frame.iloc[-1]["val_mean_mse"])
        row: dict[str, Any] = {
            "variant": variant,
            "best_epoch": int(frame.iloc[best_idx]["epoch"]),
            "last_epoch": int(frame.iloc[-1]["epoch"]),
            "first_val_mean_mse": float(frame.iloc[0]["val_mean_mse"]),
            "best_val_mean_mse": best,
            "last_val_mean_mse": last,
            "last_vs_best_val_mse_pct": (last / best - 1.0) * 100.0,
            "ema_decay": float(frame.iloc[-1].get("ema_decay", 0.0)),
            "ema_eval": int(frame.iloc[-1].get("ema_eval", 0)),
            "basis_operator_smoothness_weight": float(
                frame.iloc[-1].get("basis_operator_smoothness_weight", 0.0)
            ),
            "last_train_basis_operator_smoothness_loss": float(
                frame.iloc[-1].get("train_basis_operator_smoothness_loss", 0.0)
            ),
            "source_path": str(path),
        }
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No A6S training logs found under {raw_root}")
    return pd.DataFrame(rows)


def add_references(metrics: pd.DataFrame, a6_comparison_path: Path) -> pd.DataFrame:
    a6 = pd.read_csv(a6_comparison_path)
    etth2 = a6[a6["dataset"].eq("ETTh2")]
    lbf = etth2[etth2["arm"].eq("a6_lbf_r256")][["target_horizon", "mse"]].rename(
        columns={"mse": "a6_lbf_r256_mse"}
    )
    der = etth2[etth2["arm"].eq("a6_der")][
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
        comparison.groupby("variant", as_index=False)
        .agg(
            mean_mse=("mse", "mean"),
            mean_relative_mse_vs_a6_lbf_r256_pct=("relative_mse_vs_a6_lbf_r256_pct", "mean"),
            mean_relative_mse_vs_a6_der_pct=("relative_mse_vs_a6_der_pct", "mean"),
            mean_relative_mse_vs_best_stage_control_pct=("relative_mse_vs_best_stage_control_pct", "mean"),
            wins_vs_best_stage_control=("beats_best_stage_control", "sum"),
        )
    )
    return summary.merge(trajectory, on="variant", how="left", validate="one_to_one").sort_values(
        "mean_relative_mse_vs_best_stage_control_pct"
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


def write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    best = summary.iloc[0]
    lines = [
        "# Phase5-A6S Official-Last Stability Gate Report",
        "",
        "本文档分析 A6S ETTh2-only stability gate。所有 run 使用 `official-last` / without early stop。",
        "",
        "## Conclusion",
        "",
        (
            "[Fact] 最佳 variant 为 "
            f"`{best['variant']}`：相对 ETTh2 best stage control 平均 `{fmt(best['mean_relative_mse_vs_best_stage_control_pct'])}%`，"
            f"wins `{int(best['wins_vs_best_stage_control'])}/4`，last-vs-best validation drift "
            f"`{fmt(best['last_vs_best_val_mse_pct'])}%`。"
        ),
        "",
        "## Variant Summary",
        "",
        "| Variant | mean MSE | vs best control | wins | vs A6-LBF-r256 | last-vs-best val | EMA | smooth weight |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| `{variant}` | {mse:.4f} | {gap:+.2f}% | {wins}/4 | {lbf:+.2f}% | {drift:+.2f}% | {ema:.2f} | {smooth:g} |".format(
                variant=row["variant"],
                mse=row["mean_mse"],
                gap=row["mean_relative_mse_vs_best_stage_control_pct"],
                wins=int(row["wins_vs_best_stage_control"]),
                lbf=row["mean_relative_mse_vs_a6_lbf_r256_pct"],
                drift=row["last_vs_best_val_mse_pct"],
                ema=row["ema_decay"],
                smooth=row["basis_operator_smoothness_weight"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- 若 EMA 明显改善但 HeadStability 无效，则 A6 drift 更像 generic weight trajectory variance。",
            "- 若 HeadStability 改善且不依赖 EMA，则 A6-LBF 需要 operator-level stability mechanism。",
            "- 若二者均无效，回 Step 4/5 设计更强 stability path，不继续 rank/objective sweep。",
            "",
            "## Artifacts",
            "",
            "- `phase5_timealign_hss_a6s_comparison.csv`",
            "- `phase5_timealign_hss_a6s_summary.csv`",
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

    write_csv(output_dir / "phase5_timealign_hss_a6s_comparison.csv", comparison.to_dict("records"))
    write_csv(output_dir / "phase5_timealign_hss_a6s_summary.csv", summary.to_dict("records"))
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
    write_report(output_dir, summary)


if __name__ == "__main__":
    main()

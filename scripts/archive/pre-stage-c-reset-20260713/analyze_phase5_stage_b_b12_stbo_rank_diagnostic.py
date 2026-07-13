from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ["ETTh2", "ETTm1", "Weather"]
HORIZONS = [96, 192, 336, 720]
ARMS = ["stbo_shared", "stbo_bank4", "stbo_dct", "stbo_independent"]
CONFIGS = ["l48_r32", "l120_r64", "l144_r128", "l360_r256_capacity_probe"]
SEED = 2021
MIXED_LABEL = "mixed_h96_h192_h336_h720"


def read_csv(path: Path) -> list[dict[str, str]]:
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


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def fmt_pct(value: float) -> str:
    if abs(value) < 0.01:
        return f"{value:+.4f}%"
    return f"{value:+.2f}%"


def stbo_run_dir(raw_root: Path, config: str, arm: str, dataset: str) -> Path:
    return (
        raw_root
        / config
        / "official-last"
        / f"TimeAlignOfficialUnified720_{arm}_official-last"
        / dataset
        / MIXED_LABEL
        / f"seed{SEED}"
    )


def a6_run_dir(a6_root: Path, dataset: str) -> Path:
    return (
        a6_root
        / "TimeAlignOfficialUnified720_a6_clean_official-last"
        / dataset
        / MIXED_LABEL
        / f"seed{SEED}"
    )


def metric_row(path: Path, horizon: int) -> dict[str, float] | None:
    if not path.exists():
        return None
    for row in read_csv(path):
        if int(row["target_horizon"]) == horizon:
            return {"mse": float(row["mse"]), "mae": float(row["mae"])}
    return None


def collect_metrics(raw_root: Path, a6_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        a6_metrics_path = a6_run_dir(a6_root, dataset) / "metrics_by_target_horizon.csv"
        for horizon in HORIZONS:
            a6 = metric_row(a6_metrics_path, horizon)
            if a6 is None:
                rows.append(
                    {
                        "config": "a6_clean",
                        "arm": "a6_clean",
                        "dataset": dataset,
                        "target_horizon": horizon,
                        "status": "missing_a6",
                    }
                )
                continue
            rows.append(
                {
                    "config": "a6_clean",
                    "arm": "a6_clean",
                    "dataset": dataset,
                    "target_horizon": horizon,
                    "status": "ok",
                    "mse": a6["mse"],
                    "mae": a6["mae"],
                }
            )
    for config in CONFIGS:
        for arm in ARMS:
            for dataset in DATASETS:
                metrics_path = stbo_run_dir(raw_root, config, arm, dataset) / "metrics_by_target_horizon.csv"
                for horizon in HORIZONS:
                    metric = metric_row(metrics_path, horizon)
                    if metric is None:
                        rows.append(
                            {
                                "config": config,
                                "arm": arm,
                                "dataset": dataset,
                                "target_horizon": horizon,
                                "status": "missing",
                            }
                        )
                        continue
                    rows.append(
                        {
                            "config": config,
                            "arm": arm,
                            "dataset": dataset,
                            "target_horizon": horizon,
                            "status": "ok",
                            "mse": metric["mse"],
                            "mae": metric["mae"],
                        }
                    )
    return rows


def metric_lookup(metric_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int], dict[str, float]]:
    lookup = {}
    for row in metric_rows:
        if row.get("status") == "ok":
            lookup[(row["config"], row["arm"], row["dataset"], int(row["target_horizon"]))] = {
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
            }
    return lookup


def collect_comparisons(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = metric_lookup(metric_rows)
    rows: list[dict[str, Any]] = []
    comparison_specs = [
        ("a6_clean", "a6_clean"),
        ("stbo_dct", "stbo_dct"),
        ("stbo_independent", "stbo_independent"),
    ]
    for config in CONFIGS:
        for arm in ARMS:
            for baseline_kind, baseline_arm in comparison_specs:
                if arm == baseline_arm and baseline_kind != "a6_clean":
                    continue
                comparison = f"{config}_{arm}_vs_{baseline_kind}"
                for dataset in DATASETS:
                    for horizon in HORIZONS:
                        candidate = lookup.get((config, arm, dataset, horizon))
                        if baseline_kind == "a6_clean":
                            baseline = lookup.get(("a6_clean", "a6_clean", dataset, horizon))
                        else:
                            baseline = lookup.get((config, baseline_arm, dataset, horizon))
                        if candidate is None or baseline is None:
                            rows.append(
                                {
                                    "config": config,
                                    "candidate_arm": arm,
                                    "baseline": baseline_kind,
                                    "comparison": comparison,
                                    "dataset": dataset,
                                    "target_horizon": horizon,
                                    "status": "missing",
                                }
                            )
                            continue
                        rows.append(
                            {
                                "config": config,
                                "candidate_arm": arm,
                                "baseline": baseline_kind,
                                "comparison": comparison,
                                "dataset": dataset,
                                "target_horizon": horizon,
                                "status": "ok",
                                "candidate_mse": candidate["mse"],
                                "baseline_mse": baseline["mse"],
                                "relative_mse_pct": pct(candidate["mse"], baseline["mse"]),
                                "candidate_mae": candidate["mae"],
                                "baseline_mae": baseline["mae"],
                                "relative_mae_pct": pct(candidate["mae"], baseline["mae"]),
                                "candidate_wins_mse": candidate["mse"] < baseline["mse"],
                            }
                        )
    return rows


def summarize_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    summary: list[dict[str, Any]] = []
    for config in CONFIGS:
        for comparison in sorted({row["comparison"] for row in ok_rows if row["config"] == config}):
            comp_rows = [row for row in ok_rows if row["comparison"] == comparison]
            for dataset in DATASETS:
                subset = [row for row in comp_rows if row["dataset"] == dataset]
                if not subset:
                    continue
                summary.append(
                    {
                        "config": config,
                        "comparison": comparison,
                        "dataset": dataset,
                        "settings": len(subset),
                        "mse_wins": sum(1 for row in subset if row["candidate_wins_mse"]),
                        "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in subset),
                        "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in subset),
                    }
                )
            if comp_rows:
                summary.append(
                    {
                        "config": config,
                        "comparison": comparison,
                        "dataset": "ALL",
                        "settings": len(comp_rows),
                        "mse_wins": sum(1 for row in comp_rows if row["candidate_wins_mse"]),
                        "mean_relative_mse_pct": mean(row["relative_mse_pct"] for row in comp_rows),
                        "mean_relative_mae_pct": mean(row["relative_mae_pct"] for row in comp_rows),
                    }
                )
    return summary


def collect_diagnostics(raw_root: Path, a6_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        path = a6_run_dir(a6_root, dataset) / "model_diagnostics.json"
        if path.exists():
            payload = json.loads(path.read_text())
            rows.append(
                {
                    "config": "a6_clean",
                    "arm": "a6_clean",
                    "dataset": dataset,
                    "status": "ok",
                    "readout_mode": payload.get("readout_mode", ""),
                    "total_parameters": payload.get("total_parameters", ""),
                    "trainable_parameters": payload.get("trainable_parameters", ""),
                    "basis_rank": payload.get("basis_rank", ""),
                }
            )
    for config in CONFIGS:
        for arm in ARMS:
            for dataset in DATASETS:
                path = stbo_run_dir(raw_root, config, arm, dataset) / "model_diagnostics.json"
                if not path.exists():
                    rows.append({"config": config, "arm": arm, "dataset": dataset, "status": "missing"})
                    continue
                payload = json.loads(path.read_text())
                rows.append(
                    {
                        "config": config,
                        "arm": arm,
                        "dataset": dataset,
                        "status": "ok",
                        "readout_mode": payload.get("readout_mode", ""),
                        "total_parameters": payload.get("total_parameters", ""),
                        "trainable_parameters": payload.get("trainable_parameters", ""),
                        "stbo_tile_len": payload.get("stbo_tile_len", ""),
                        "stbo_tile_count": payload.get("stbo_tile_count", ""),
                        "stbo_rank": payload.get("stbo_rank", ""),
                        "stbo_coeff_l2": payload.get("stbo_coeff_l2", ""),
                        "stbo_shared_basis_l2": payload.get("stbo_shared_basis_l2", ""),
                        "stbo_bank_count": payload.get("stbo_bank_count", ""),
                        "stbo_basis_bank_l2": payload.get("stbo_basis_bank_l2", ""),
                        "stbo_tile_bank_entropy_mean": payload.get("stbo_tile_bank_entropy_mean", ""),
                        "stbo_tile_basis_l2": payload.get("stbo_tile_basis_l2", ""),
                        "stbo_dct_basis_l2": payload.get("stbo_dct_basis_l2", ""),
                    }
                )
    return rows


def summarize_best(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = metric_lookup(metric_rows)
    rows: list[dict[str, Any]] = []
    arms = ["a6_clean"] + [f"{config}:{arm}" for config in CONFIGS for arm in ARMS]
    for dataset in DATASETS:
        for horizon in HORIZONS:
            candidates: list[tuple[str, float, float]] = []
            a6 = lookup.get(("a6_clean", "a6_clean", dataset, horizon))
            if a6 is not None:
                candidates.append(("a6_clean", a6["mse"], a6["mae"]))
            for config in CONFIGS:
                for arm in ARMS:
                    metric = lookup.get((config, arm, dataset, horizon))
                    if metric is not None:
                        candidates.append((f"{config}:{arm}", metric["mse"], metric["mae"]))
            if not candidates:
                continue
            best = min(candidates, key=lambda item: item[1])
            row: dict[str, Any] = {
                "dataset": dataset,
                "target_horizon": horizon,
                "best_arm": best[0],
                "best_mse": best[1],
            }
            for name, mse, _ in candidates:
                row[name] = mse
            for name in arms:
                row.setdefault(name, "")
            rows.append(row)
    return rows


def find_summary(
    summary_rows: list[dict[str, Any]], config: str, candidate_arm: str, baseline: str
) -> dict[str, Any] | None:
    prefix = f"{config}_{candidate_arm}_vs_{baseline}"
    for row in summary_rows:
        if row["config"] == config and row["comparison"] == prefix and row["dataset"] == "ALL":
            return row
    return None


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                if field.startswith("mean_relative"):
                    value = fmt_pct(value)
                else:
                    value = f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(output_dir: Path, summary_rows: list[dict[str, Any]], diagnostics_rows: list[dict[str, Any]]) -> None:
    config_rows: list[dict[str, Any]] = []
    for config in CONFIGS:
        for arm in ARMS:
            vs_a6 = find_summary(summary_rows, config, arm, "a6_clean")
            if vs_a6 is None:
                continue
            config_rows.append(
                {
                    "config": config,
                    "arm": arm,
                    "vs_a6_mse": float(vs_a6["mean_relative_mse_pct"]),
                    "vs_a6_wins": f"{int(vs_a6['mse_wins'])}/12",
                }
            )
    learned_rows: list[dict[str, Any]] = []
    for config in CONFIGS:
        for arm in ["stbo_shared", "stbo_bank4"]:
            vs_a6 = find_summary(summary_rows, config, arm, "a6_clean")
            vs_dct = find_summary(summary_rows, config, arm, "stbo_dct")
            vs_ind = find_summary(summary_rows, config, arm, "stbo_independent")
            if vs_a6 is None or vs_dct is None or vs_ind is None:
                continue
            entropy_values = []
            if arm == "stbo_bank4":
                for row in diagnostics_rows:
                    if row["config"] == config and row["arm"] == arm and row["status"] == "ok":
                        value = row.get("stbo_tile_bank_entropy_mean", "")
                        if value != "":
                            entropy_values.append(float(value))
            learned_rows.append(
                {
                    "config": config,
                    "arm": arm,
                    "vs_a6_mse": float(vs_a6["mean_relative_mse_pct"]),
                    "vs_a6_wins": f"{int(vs_a6['mse_wins'])}/12",
                    "vs_dct_mse": float(vs_dct["mean_relative_mse_pct"]),
                    "vs_dct_wins": f"{int(vs_dct['mse_wins'])}/12",
                    "vs_ind_mse": float(vs_ind["mean_relative_mse_pct"]),
                    "vs_ind_wins": f"{int(vs_ind['mse_wins'])}/12",
                    "bank_entropy_mean": mean(entropy_values) if entropy_values else "",
                }
            )

    best_learned = min(
        (row for row in learned_rows if row["arm"] in {"stbo_shared", "stbo_bank4"}),
        key=lambda row: float(row["vs_a6_mse"]),
    )
    dct_blocks = all(float(row["vs_dct_mse"]) >= 0.0 or int(str(row["vs_dct_wins"]).split("/")[0]) < 7 for row in learned_rows)
    a6_blocks = all(float(row["vs_a6_mse"]) > 0.0 for row in learned_rows)
    bank_inactive = all(
        row["arm"] != "stbo_bank4"
        or row["bank_entropy_mean"] == ""
        or float(row["bank_entropy_mean"]) > 0.98
        for row in learned_rows
    )

    lines = [
        "# Phase5 StageB B12-STBO Rank Diagnostic Report",
        "",
        "## Scope",
        "",
        "Valid configs: `L48-R32`, `L120-R64`, `L144-R128`, `L360-R256_capacity_probe`.",
        "Invalid config `L96-R64` is excluded because `96` does not divide `720`.",
        "A6 is not rerun; comparisons use the validated clean A6 anchor.",
        "",
        "## Overall STBO vs A6",
        "",
        *markdown_table(config_rows, ["config", "arm", "vs_a6_mse", "vs_a6_wins"]),
        "",
        "## Learned STBO Mechanism Controls",
        "",
        *markdown_table(
            learned_rows,
            [
                "config",
                "arm",
                "vs_a6_mse",
                "vs_a6_wins",
                "vs_dct_mse",
                "vs_dct_wins",
                "vs_ind_mse",
                "vs_ind_wins",
                "bank_entropy_mean",
            ],
        ),
        "",
        "## Gate Reading",
        "",
    ]
    if a6_blocks:
        lines.append(
            "[Decision] `rank_capacity_repair_insufficient`: increasing rank/tile length does not produce a learned shared/bank STBO candidate that matches or beats A6 overall."
        )
    elif dct_blocks:
        lines.append(
            "[Decision] `generic_local_basis_control_explains`: some STBO variants improve, but learned shared/bank still do not beat same-rank DCT."
        )
    else:
        lines.append(
            "[Decision] `requires_followup`: at least one learned STBO variant beats DCT; inspect dataset/horizon split before any method claim."
        )
    lines.extend(
        [
            "",
            f"- Best learned STBO vs A6: `{best_learned['config']}:{best_learned['arm']}` with mean MSE {fmt_pct(float(best_learned['vs_a6_mse']))}, wins {best_learned['vs_a6_wins']}.",
            f"- Learned-vs-DCT block: `{dct_blocks}`.",
            f"- Bank specialization inactive: `{bank_inactive}`.",
            "",
            "## Failure Attribution",
            "",
            "- `hypothesis_false`: not fully proven for all native multi-horizon architectures.",
            "- `readout_or_head_design_wrong`: supported; the tested tiled readout does not preserve A6 performance even under higher local rank.",
            "- `generic_basis_control_explains`: supported if learned shared/bank fail to beat same-rank DCT.",
            "- `capacity_control_explains`: not sufficient if the 256-rank capacity probe still fails A6 or DCT.",
            "- `direction_level_rejection`: no; reject the tested STBO family, not all future architecture search.",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "b12_stbo_rank_diagnostic_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase5 StageB B12-STBO rank diagnostic artifacts.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--a6-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    metric_rows = collect_metrics(args.raw_root, args.a6_root)
    comparison_rows = collect_comparisons(metric_rows)
    summary_rows = summarize_comparisons(comparison_rows)
    diagnostics_rows = collect_diagnostics(args.raw_root, args.a6_root)
    best_rows = summarize_best(metric_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "b12_stbo_rank_metrics.csv", metric_rows)
    write_csv(args.output_dir / "b12_stbo_rank_comparisons.csv", comparison_rows)
    write_csv(args.output_dir / "b12_stbo_rank_summary.csv", summary_rows)
    write_csv(args.output_dir / "b12_stbo_rank_model_diagnostics.csv", diagnostics_rows)
    write_csv(args.output_dir / "b12_stbo_rank_best_by_setting.csv", best_rows)
    write_report(args.output_dir, summary_rows, diagnostics_rows)


if __name__ == "__main__":
    main()

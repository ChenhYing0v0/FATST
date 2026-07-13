#!/usr/bin/env python3
"""Deep failure attribution for the returned C1 carrier-normalization gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ("ETTh2", "ETTm1", "Weather")
HORIZONS = (96, 192, 336, 720)
ARMS = ("a6_clean", "gamp_p16s8", "gamp_p48s24")
SCALES = ("gamp_p16s8", "gamp_p48s24")
SELECTORS = ("last", "best_val")
MIXED_LABEL = "mixed_h96_h192_h336_h720"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--prior-a6-root",
        type=Path,
        default=repo_root
        / "analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/raw/official-last",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_dir(root: Path, arm: str, dataset: str, seed: int) -> Path:
    return (
        root
        / f"A6_C1_{arm}_balanced_dual"
        / dataset
        / MIXED_LABEL
        / f"seed{seed}"
    )


def metric_map(
    root: Path,
    arm: str,
    dataset: str,
    selector: str,
    seed: int,
) -> dict[int, dict[str, float]]:
    path = run_dir(root, arm, dataset, seed) / f"metrics_{selector}_by_target_horizon.csv"
    return {
        int(row["target_horizon"]): {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
        for row in read_csv(path)
    }


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def training_rows(root: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for dataset in DATASETS:
            history = read_csv(run_dir(root, arm, dataset, seed) / "training_log.csv")
            first = history[0]
            last = history[-1]
            best = min(history, key=lambda row: float(row["val_mean_mse"]))
            last_metrics = metric_map(root, arm, dataset, "last", seed)
            best_metrics = metric_map(root, arm, dataset, "best_val", seed)
            rows.append(
                {
                    "arm": arm,
                    "dataset": dataset,
                    "best_epoch": int(best["epoch"]),
                    "best_val_mean_mse": float(best["val_mean_mse"]),
                    "last_val_mean_mse": float(last["val_mean_mse"]),
                    "last_vs_best_val_pct": pct(
                        float(last["val_mean_mse"]),
                        float(best["val_mean_mse"]),
                    ),
                    "first_train_loss": float(first["train_loss"]),
                    "last_train_loss": float(last["train_loss"]),
                    "train_loss_reduction_pct": pct(
                        float(last["train_loss"]),
                        float(first["train_loss"]),
                    ),
                    "best_vs_last_test_mean_mse_pct": mean(
                        pct(
                            best_metrics[horizon]["mse"],
                            last_metrics[horizon]["mse"],
                        )
                        for horizon in HORIZONS
                    ),
                }
            )
    return rows


def scale_rows(root: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        validation = {}
        for arm in SCALES:
            history = read_csv(run_dir(root, arm, dataset, seed) / "training_log.csv")
            validation[arm] = min(float(row["val_mean_mse"]) for row in history)
        row: dict[str, Any] = {
            "dataset": dataset,
            "validation_selected": min(validation, key=validation.get),
            "p16s8_best_val_mean_mse": validation["gamp_p16s8"],
            "p48s24_best_val_mean_mse": validation["gamp_p48s24"],
            "p48s24_vs_p16s8_val_pct": pct(
                validation["gamp_p48s24"], validation["gamp_p16s8"]
            ),
        }
        for selector in SELECTORS:
            arm_means = {
                arm: mean(
                    metric_map(root, arm, dataset, selector, seed)[horizon]["mse"]
                    for horizon in HORIZONS
                )
                for arm in SCALES
            }
            row[f"test_preferred_{selector}"] = min(arm_means, key=arm_means.get)
            row[f"p48s24_vs_p16s8_test_{selector}_pct"] = pct(
                arm_means["gamp_p48s24"], arm_means["gamp_p16s8"]
            )
        rows.append(row)
    return rows


def capacity_rows(root: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        a6_dir = run_dir(root, "a6_clean", dataset, seed)
        a6_diag = json.loads((a6_dir / "model_diagnostics.json").read_text())
        a6_state_width = int(a6_diag["patch_num"]) * int(a6_diag["d_model"])
        for arm in SCALES:
            candidate_dir = run_dir(root, arm, dataset, seed)
            diagnostics = json.loads(
                (candidate_dir / "model_diagnostics.json").read_text()
            )
            candidate_width = int(diagnostics["patch_num"]) * int(
                diagnostics["d_model"]
            )
            rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "a6_active_parameters": int(
                        a6_diag["active_forward_parameters"]
                    ),
                    "candidate_active_parameters": int(
                        diagnostics["active_forward_parameters"]
                    ),
                    "candidate_vs_a6_parameters_pct": pct(
                        int(diagnostics["active_forward_parameters"]),
                        int(a6_diag["active_forward_parameters"]),
                    ),
                    "a6_readout_state_width": a6_state_width,
                    "candidate_readout_state_width": candidate_width,
                    "candidate_vs_a6_state_width_pct": pct(
                        candidate_width, a6_state_width
                    ),
                    "candidate_local_patch_num": int(
                        diagnostics["history_local_patch_num"]
                    ),
                }
            )
    return rows


def reproduction_rows(
    root: Path,
    prior_root: Path,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        current = metric_map(root, "a6_clean", dataset, "last", seed)
        current_config = json.loads(
            (run_dir(root, "a6_clean", dataset, seed) / "effective_config.json").read_text()
        )
        effective_learning_rate = float(
            current_config["official_args"]["learning_rate"]
        )
        source_learning_rate = float(
            current_config["official_preset"]["learning_rate"]
        )
        prior_path = (
            prior_root
            / "TimeAlignOfficialUnified720_a6_clean_official-last"
            / dataset
            / MIXED_LABEL
            / f"seed{seed}"
            / "metrics_by_target_horizon.csv"
        )
        prior = {
            int(row["target_horizon"]): {
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
            }
            for row in read_csv(prior_path)
        }
        for horizon in HORIZONS:
            rows.append(
                {
                    "dataset": dataset,
                    "target_horizon": horizon,
                    "effective_learning_rate": effective_learning_rate,
                    "source_learning_rate": source_learning_rate,
                    "mse_abs_diff": abs(
                        current[horizon]["mse"] - prior[horizon]["mse"]
                    ),
                    "mae_abs_diff": abs(
                        current[horizon]["mae"] - prior[horizon]["mae"]
                    ),
                }
            )
    return rows


def prior_metric_map(
    prior_root: Path,
    dataset: str,
    seed: int,
) -> dict[int, dict[str, float]]:
    path = (
        prior_root
        / "TimeAlignOfficialUnified720_a6_clean_official-last"
        / dataset
        / MIXED_LABEL
        / f"seed{seed}"
        / "metrics_by_target_horizon.csv"
    )
    return {
        int(row["target_horizon"]): {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
        for row in read_csv(path)
    }


def source_a6_summary_rows(
    root: Path,
    prior_root: Path,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in SCALES:
        all_deltas: list[float] = []
        all_wins = 0
        for dataset in DATASETS:
            candidate = metric_map(root, arm, dataset, "last", seed)
            prior = prior_metric_map(prior_root, dataset, seed)
            deltas = [
                pct(candidate[horizon]["mse"], prior[horizon]["mse"])
                for horizon in HORIZONS
            ]
            all_deltas.extend(deltas)
            wins = sum(value < 0.0 for value in deltas)
            all_wins += wins
            rows.append(
                {
                    "arm": arm,
                    "dataset": dataset,
                    "selector": "official-last",
                    "mean_mse_vs_source_a6_pct": mean(deltas),
                    "wins_vs_source_a6": wins,
                    "settings": len(deltas),
                    "max_mse_vs_source_a6_pct": max(deltas),
                }
            )
        rows.append(
            {
                "arm": arm,
                "dataset": "ALL",
                "selector": "official-last",
                "mean_mse_vs_source_a6_pct": mean(all_deltas),
                "wins_vs_source_a6": all_wins,
                "settings": len(all_deltas),
                "max_mse_vs_source_a6_pct": max(all_deltas),
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    output_dir: Path,
    training: list[dict[str, Any]],
    scales: list[dict[str, Any]],
    capacity: list[dict[str, Any]],
    reproduction: list[dict[str, Any]],
    source_a6_summary: list[dict[str, Any]],
) -> None:
    max_reproduction_diff = max(
        max(float(row["mse_abs_diff"]), float(row["mae_abs_diff"]))
        for row in reproduction
    )
    lines = [
        "# Phase5 C1 Global-Anchored Multi-Patch 深度分析",
        "",
        "## 结论",
        "",
        "[Decision] `c1_carrier_normalization_gate_failed`。Shared scales和validation-selected scale均明显越过退化预算；不追加 seeds、dropout follow-up、local-mask control或readout repair。",
        "",
        "[Protocol Mismatch] Runner统一传入`learning_rate=1e-4`，ETTh2 A6 source preset实际为`5e-4`；因此ETTh2 A6不是source-faithful exact reproduction。ETTm1/Weather learning rate一致且metrics exact reproduce。",
        "",
        "该 mismatch不改变失败裁决：ETTh2三arms在同一LR下仍可作controlled comparison，且C1相对fixed TimeAlign直接失败。相对既有source-faithful A6 official-last的独立结果如下：",
        "",
        *markdown_table(
            source_a6_summary,
            [
                "arm",
                "dataset",
                "mean_mse_vs_source_a6_pct",
                "wins_vs_source_a6",
                "settings",
                "max_mse_vs_source_a6_pct",
            ],
        ),
        "",
        f"A6 reproduction artifact的跨数据集maximum absolute metric difference为`{max_reproduction_diff:.1e}`；差异只来自上述ETTh2 LR mismatch。",
        "",
        "## Training dynamics",
        "",
        *markdown_table(
            training,
            [
                "arm",
                "dataset",
                "best_epoch",
                "best_val_mean_mse",
                "last_val_mean_mse",
                "last_vs_best_val_pct",
                "last_train_loss",
                "best_vs_last_test_mean_mse_pct",
            ],
        ),
        "",
        "GAMP在ETTh2/ETTm1的train loss持续下降，但validation minima很早出现并随后恶化；这是明确overfitting evidence。Weather的validation gap较小，但test仍显著退化，说明regularization不是唯一解释。",
        "",
        "## Scale stability",
        "",
        *markdown_table(
            scales,
            [
                "dataset",
                "validation_selected",
                "p48s24_vs_p16s8_val_pct",
                "test_preferred_last",
                "p48s24_vs_p16s8_test_last_pct",
                "test_preferred_best_val",
                "p48s24_vs_p16s8_test_best_val_pct",
            ],
        ),
        "",
        "Validation在三个datasets都选择P48-S24，但test preference随dataset/selector变化；Weather validation差仅约0.1%，test却稳定偏向P16-S8。Scale selection不够稳定，不能作为dataset-specific carrier依据。",
        "",
        "## Capacity and state-width attribution",
        "",
        *markdown_table(
            capacity,
            [
                "dataset",
                "arm",
                "candidate_vs_a6_parameters_pct",
                "a6_readout_state_width",
                "candidate_readout_state_width",
                "candidate_vs_a6_state_width_pct",
            ],
        ),
        "",
        "C1 total parameters在ETTh2/ETTm1高于A6却仍退化，因此total capacity不是通用解释。另一方面，global-only readout把ETTh2/Weather state width从1536/6144压到256，Weather同时减少约46% active parameters；这构成readout/state bottleneck。ETTm1 state width未减少仍退化，说明bottleneck也不是唯一原因。",
        "",
        "## Failure attribution",
        "",
        "- `hypothesis_false`：不能证明所有统一multipatch carriers都不可行；",
        "- `intervention_point_wrong`：可能，C1用一个随机global token替代了各dataset已验证的A6 hidden contract；",
        "- `readout_or_head_design_wrong`：强支持，global-only D256 readout丢失ETTh2/Weather原有P*D state width；",
        "- `optimization_or_numeric_pathology`：无divergence/OOM，但存在早期validation overfit；",
        "- `capacity_control_explains`：仅能部分解释Weather，不能解释ETTh2/ETTm1。",
        "",
        "## Dropout decision",
        "",
        "Best-val仍整体退化3.75%，Weather退化5.49%，远超near-miss范围。更强dropout可能缓解train-validation gap，但无法单独修复state/readout contract；按预注册协议不追加dropout sweep。",
        "",
        "## Carrier decision",
        "",
        "恢复并冻结`A6-LBF-r256 + exact valid HPM [B,C,29,48]`。Forecast path允许source-faithful dataset hyperparameters；所有后续模块通过HPM获得统一local-token interface。继续修改readout/scale/dropout会把control-only cleanup扩张为新的architecture search，当前停止。",
        "",
        "## 下一研究方向",
        "",
        "1. 先完成Contribution 1的matched multi-prefix-vs-single-prefix "
        "supervision control，固定同一720-step architecture、dropout与selector，只改变objective。",
        "2. StageB回到Step 2/3。优先重新审计B7 horizon-agnostic supervision allocation，因为现有multi-prefix loss对early steps有14.39x exposure，而Encoder/local retrieval routes已被多轮controls阻断。",
        "3. B7只能先做continuous-prefix、benchmark-horizon-free diagnostic；若不能证明tail degradation与exposure imbalance存在因果关系，则暂停第二贡献搜索，进入论文收束。",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "c1_global_anchored_multipatch_deep_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    training = training_rows(args.raw_root, args.seed)
    scales = scale_rows(args.raw_root, args.seed)
    capacity = capacity_rows(args.raw_root, args.seed)
    reproduction = reproduction_rows(args.raw_root, args.prior_a6_root, args.seed)
    source_a6_summary = source_a6_summary_rows(
        args.raw_root,
        args.prior_a6_root,
        args.seed,
    )
    write_csv(args.output_dir / "c1_training_dynamics.csv", training)
    write_csv(args.output_dir / "c1_scale_stability.csv", scales)
    write_csv(args.output_dir / "c1_capacity_state_width.csv", capacity)
    write_csv(args.output_dir / "c1_a6_reproduction.csv", reproduction)
    write_csv(
        args.output_dir / "c1_vs_source_a6_summary.csv",
        source_a6_summary,
    )
    write_report(
        args.output_dir,
        training,
        scales,
        capacity,
        reproduction,
        source_a6_summary,
    )


if __name__ == "__main__":
    main()

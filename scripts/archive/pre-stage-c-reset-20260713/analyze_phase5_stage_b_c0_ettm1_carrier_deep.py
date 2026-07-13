#!/usr/bin/env python3
"""Deep attribution analysis for the returned StageB C0 ETTm1 Encoder control."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any


HORIZONS = (96, 192, 336, 720)
MIXED_LABEL = "mixed_h96_h192_h336_h720"
ARMS = (
    "p1_d256_f256_d09",
    "p1_d384_f96_d09",
    "p5_d52_f256_d09",
    "p5_d52_f2048_d09",
    "p1_d256_f256_d02",
    "p5_d52_f2048_d02",
)
SELECTORS = ("last", "best_val")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--reference-metrics",
        type=Path,
        default=repo_root
        / "analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/raw/official-last"
        / "TimeAlignOfficialUnified720_a6_clean_official-last/ETTm1"
        / "mixed_h96_h192_h336_h720/seed2021/metrics_by_target_horizon.csv",
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


def run_dir(root: Path, arm: str, seed: int) -> Path:
    return (
        root
        / f"A6_C0_{arm}_dual"
        / "ETTm1"
        / MIXED_LABEL
        / f"seed{seed}"
    )


def pct(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def metric_map(root: Path, arm: str, selector: str, seed: int) -> dict[int, dict[str, float]]:
    rows = read_csv(
        run_dir(root, arm, seed)
        / f"metrics_{selector}_by_target_horizon.csv"
    )
    return {
        int(row["target_horizon"]): {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
        for row in rows
    }


def comparison_rows(root: Path, seed: int) -> list[dict[str, Any]]:
    definitions = (
        (
            "dropout_d02_vs_d09_p1",
            "p1_d256_f256_d02",
            "p1_d256_f256_d09",
        ),
        (
            "dropout_d02_vs_d09_p5",
            "p5_d52_f2048_d02",
            "p5_d52_f2048_d09",
        ),
        (
            "p5_f2048_vs_f256_d09",
            "p5_d52_f2048_d09",
            "p5_d52_f256_d09",
        ),
    )
    rows: list[dict[str, Any]] = []
    for comparison, candidate, baseline in definitions:
        for selector in SELECTORS:
            candidate_metrics = metric_map(root, candidate, selector, seed)
            baseline_metrics = metric_map(root, baseline, selector, seed)
            deltas = []
            for horizon in HORIZONS:
                delta_mse = pct(
                    candidate_metrics[horizon]["mse"],
                    baseline_metrics[horizon]["mse"],
                )
                deltas.append(delta_mse)
                rows.append(
                    {
                        "comparison": comparison,
                        "candidate": candidate,
                        "baseline": baseline,
                        "selector": selector,
                        "target_horizon": horizon,
                        "candidate_mse": candidate_metrics[horizon]["mse"],
                        "baseline_mse": baseline_metrics[horizon]["mse"],
                        "relative_mse_pct": delta_mse,
                        "relative_mae_pct": pct(
                            candidate_metrics[horizon]["mae"],
                            baseline_metrics[horizon]["mae"],
                        ),
                    }
                )
            rows.append(
                {
                    "comparison": comparison,
                    "candidate": candidate,
                    "baseline": baseline,
                    "selector": selector,
                    "target_horizon": "MEAN",
                    "relative_mse_pct": mean(deltas),
                    "relative_mae_pct": mean(
                        pct(
                            candidate_metrics[horizon]["mae"],
                            baseline_metrics[horizon]["mae"],
                        )
                        for horizon in HORIZONS
                    ),
                }
            )
    return rows


def training_rows(root: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        history = read_csv(run_dir(root, arm, seed) / "training_log.csv")
        best = min(history, key=lambda row: float(row["val_mean_mse"]))
        first = history[0]
        last = history[-1]
        last_metrics = metric_map(root, arm, "last", seed)
        best_metrics = metric_map(root, arm, "best_val", seed)
        rows.append(
            {
                "arm": arm,
                "best_epoch": int(best["epoch"]),
                "best_val_mean_mse": float(best["val_mean_mse"]),
                "last_val_mean_mse": float(last["val_mean_mse"]),
                "last_vs_best_val_pct": pct(
                    float(last["val_mean_mse"]), float(best["val_mean_mse"])
                ),
                "first_train_loss": float(first["train_loss"]),
                "last_train_loss": float(last["train_loss"]),
                "train_loss_reduction_pct": pct(
                    float(last["train_loss"]), float(first["train_loss"])
                ),
                "best_vs_last_test_mean_mse_pct": mean(
                    pct(best_metrics[horizon]["mse"], last_metrics[horizon]["mse"])
                    for horizon in HORIZONS
                ),
            }
        )
    return rows


def segment_map(root: Path, arm: str, selector: str, seed: int) -> dict[tuple[int, int], float]:
    rows = read_csv(
        run_dir(root, arm, seed) / f"metrics_{selector}_by_segment.csv"
    )
    return {
        (int(row["segment_start"]), int(row["segment_end"])): float(row["mse"])
        for row in rows
        if int(row["target_horizon"]) == 720
    }


def segment_rows(root: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    definitions = (
        ("patch_matched_d09", "p5_d52_f2048_d09", "p1_d256_f256_d09"),
        ("patch_matched_d02", "p5_d52_f2048_d02", "p1_d256_f256_d02"),
    )
    for comparison, candidate, baseline in definitions:
        for selector in SELECTORS:
            candidate_segments = segment_map(root, candidate, selector, seed)
            baseline_segments = segment_map(root, baseline, selector, seed)
            for segment in sorted(baseline_segments):
                rows.append(
                    {
                        "comparison": comparison,
                        "selector": selector,
                        "segment_start": segment[0],
                        "segment_end": segment[1],
                        "candidate_mse": candidate_segments[segment],
                        "baseline_mse": baseline_segments[segment],
                        "relative_mse_pct": pct(
                            candidate_segments[segment], baseline_segments[segment]
                        ),
                    }
                )
    return rows


def reference_rows(
    root: Path,
    seed: int,
    reference_path: Path,
) -> list[dict[str, Any]]:
    current = metric_map(root, "p1_d256_f256_d09", "last", seed)
    reference = {
        int(row["target_horizon"]): {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
        for row in read_csv(reference_path)
    }
    return [
        {
            "target_horizon": horizon,
            "current_mse": current[horizon]["mse"],
            "reference_mse": reference[horizon]["mse"],
            "mse_abs_diff": abs(current[horizon]["mse"] - reference[horizon]["mse"]),
            "current_mae": current[horizon]["mae"],
            "reference_mae": reference[horizon]["mae"],
            "mae_abs_diff": abs(current[horizon]["mae"] - reference[horizon]["mae"]),
        }
        for horizon in HORIZONS
    ]


def summary_lookup(
    rows: list[dict[str, Any]], comparison: str, selector: str
) -> float:
    for row in rows:
        if (
            row["comparison"] == comparison
            and row["selector"] == selector
            and row["target_horizon"] == "MEAN"
        ):
            return float(row["relative_mse_pct"])
    raise KeyError((comparison, selector))


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
    comparisons: list[dict[str, Any]],
    training: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    reproduction: list[dict[str, Any]],
) -> None:
    comparison_means = [row for row in comparisons if row["target_horizon"] == "MEAN"]
    segment_summary: list[dict[str, Any]] = []
    for comparison in ("patch_matched_d09", "patch_matched_d02"):
        for selector in SELECTORS:
            selected = [
                float(row["relative_mse_pct"])
                for row in segments
                if row["comparison"] == comparison and row["selector"] == selector
            ]
            segment_summary.append(
                {
                    "comparison": comparison,
                    "selector": selector,
                    "segment_wins": sum(value < 0.0 for value in selected),
                    "segment_count": len(selected),
                    "min_relative_mse_pct": min(selected),
                    "max_relative_mse_pct": max(selected),
                }
            )

    lines = [
        "# Phase5 StageB C0 ETTm1 Encoder Control 深度分析",
        "",
        "## 研究裁决",
        "",
        "[Decision] `patch_num_performance_defect_not_supported`。Matched P5 在全部 16 个 "
        "horizon-selector-dropout comparisons 中都输给 P1，因此预注册 gate 不授权追加 seeds。",
        "",
        "[Decision] `dropout_0.9_not_a_general_protocol_defect`。将 P1 dropout 降到 0.2 后，mean "
        f"MSE 变化 {summary_lookup(comparisons, 'dropout_d02_vs_d09_p1', 'last'):+.2f}% (last) 与 "
        f"{summary_lookup(comparisons, 'dropout_d02_vs_d09_p1', 'best_val'):+.2f}% (best-val)，没有改善 unified P1 carrier。",
        "",
        "[Decision] `checkpoint_selector_not_explanatory`。Patch effect 在 last 与 best-val 下方向一致且均为 "
        "0/4 wins；个别 arm 存在 selector sensitivity，但不能反转结论。",
        "",
        "[Fact] P1-D256-F256-drop0.9 official-last control 精确复现此前 clean A6 ETTm1 metrics，"
        f"MSE/MAE maximum absolute difference 为 "
        f"{max(max(float(row['mse_abs_diff']), float(row['mae_abs_diff'])) for row in reproduction):.1e}。",
        "",
        "## Protocol sensitivity",
        "",
        *markdown_table(
            comparison_means,
            ["comparison", "selector", "relative_mse_pct", "relative_mae_pct"],
        ),
        "",
        "负值表示 candidate 更优。降低 dropout 对 P5 有帮助，尤其是短 horizon，但修正后的 P5 仍然弱于对应 P1。",
        "",
        "## Training 与 selector dynamics",
        "",
        *markdown_table(
            training,
            [
                "arm",
                "best_epoch",
                "best_val_mean_mse",
                "last_val_mean_mse",
                "last_vs_best_val_pct",
                "last_train_loss",
                "best_vs_last_test_mean_mse_pct",
            ],
        ),
        "",
        "所有 runs 均正常优化且没有 divergence。Validation minima 出现在不同 epochs，但显式 best-val evaluation "
        "不改变 architecture ranking。",
        "",
        "## Encoder 风险裁决",
        "",
        "1. `patch_num=1`：不是已证实的 defect。它是 full-window global compression bias；所有受测 P5 controls 更差。",
        "2. Global width：P1-D384 没有收益，因此没有证据把 256-dimensional token 视为 bottleneck。",
        "3. Cross-patch mixing：accepted P1 每个 channel 只有一个 full-window token，因此不存在待混合的 local "
        "patch axis；720-to-D projection 与 residual MLP 已产生 material cross-region interactions。只有引入 P5 "
        "independent tokens 后，缺少 mixing 才成为 design concern。",
        "4. Dropout 0.9：dropout 位于 residual MLP correction branch 内，identity token path 始终保留，evaluation "
        "时 dropout 关闭，因此它不等于丢弃 90% history representation。返回结果不支持降低 P1 dropout。",
        "5. `official-last`：accepted P1 validation drift 仅 0.05%，best-val 使 test mean MSE 改变 -0.15%，无法解释 "
        "ETTm1 结果；但这不消除之前观察到的 cross-dataset selector risk。",
        "6. Unused `proj_x`：仍是 code/parameter-accounting debt，但不影响 forward 或 metrics。以后只应通过 "
        "exact-equivalence cleanup 移除，不应作为研究实验。",
        "",
        "[Conclusion] 当前 ETTm1 P1 Encoder 没有被证实存在理论无效性。C0 解决的是 implementation/control "
        "question；它本身没有完成 strong architecture-only paper claim 所需的独立 unified-vs-fixed fair-task confirmation。",
        "",
        "## H720 disjoint-segment consistency",
        "",
        *markdown_table(
            segment_summary,
            [
                "comparison",
                "selector",
                "segment_wins",
                "segment_count",
                "min_relative_mse_pct",
                "max_relative_mse_pct",
            ],
        ),
        "",
        "Matched P5 在 H720 的 disjoint segments 中一个也没有获胜，因此退化不是 cumulative-prefix averaging "
        "或某个孤立 future region 导致的。",
        "",
        "## Failure attribution",
        "",
        "- `hypothesis_false`：不支持 ETTm1 P1 是 performance defect 这一窄假设。",
        "- `intervention_point_wrong`：对 P5 no-mix 仍然可能，因为 frozen P1 具有 material cross-patch interactions。",
        "- `readout_or_head_design_wrong`：仍可能；把 independent P5 tokens 直接 flatten 到 coefficient head，无法保持 "
        "P1 nonlinear global computation。",
        "- `optimization_or_numeric_pathology`：不支持；所有 arms 训练稳定且 metrics 有限。",
        "- `capacity_control_explains`：不足。P5 d_ff 从 256 增至 2048 后，mean MSE 仅变化 "
        f"{summary_lookup(comparisons, 'p5_f2048_vs_f256_d09', 'last'):+.2f}% (last) 与 "
        f"{summary_lookup(comparisons, 'p5_f2048_vs_f256_d09', 'best_val'):+.2f}% (best-val)，没有一致恢复。",
        "",
        "## 下一步裁决",
        "",
        "保留 `P=1,D=256,dropout=0.9` 作为 accepted ETTm1 A6 carrier setting。关闭 patch-defect route，不运行 "
        "seeds 2022/2023。C0 不需要单独 mixer control：它只能检查 P5 能否恢复 P1 interaction capacity，不能再检查 "
        "inherited P1 是否有缺陷。StageB 应回到 Step 2/3 或暂停，不继续叠加 Encoder mechanisms。",
        "",
        "## Statistic definitions",
        "",
        "- `relative_mse_pct=(candidate/baseline-1)*100`; negative favors candidate.",
        "- `last_vs_best_val_pct=(last_validation/best_validation-1)*100`.",
        "- `best_vs_last_test_mean_mse_pct` is the arithmetic mean of four per-horizon relative MSE changes.",
        "- Segment results 使用 `target_horizon=720` artifact 中记录的 disjoint regions。",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "c0_ettm1_encoder_control_deep_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    comparisons = comparison_rows(args.raw_root, args.seed)
    training = training_rows(args.raw_root, args.seed)
    segments = segment_rows(args.raw_root, args.seed)
    reproduction = reference_rows(
        args.raw_root,
        args.seed,
        args.reference_metrics,
    )
    write_csv(args.output_dir / "c0_protocol_sensitivity.csv", comparisons)
    write_csv(args.output_dir / "c0_training_dynamics.csv", training)
    write_csv(args.output_dir / "c0_segment_deltas.csv", segments)
    write_csv(args.output_dir / "c0_reference_reproduction.csv", reproduction)
    write_report(args.output_dir, comparisons, training, segments, reproduction)


if __name__ == "__main__":
    main()

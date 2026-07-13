from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DATASETS = ("ETTh2", "ETTm1", "Weather")
HORIZONS = (96, 192, 336, 720)
ARMS = ("cpe_p16s8", "cpe_p48s24")
RUN_NAMES = {
    arm: f"A6_CPE_LBF_r256_{arm}_best-val" for arm in ARMS
}
REFERENCE_RUN = "TimeAlignOfficialUnified720_A6LBF_r256_main_official-last"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_map(path: Path) -> dict[int, dict[str, float]]:
    rows = read_csv(path)
    return {
        int(row["target_horizon"]): {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
        for row in rows
    }


def reference_path(root: Path, dataset: str) -> Path:
    return (
        root
        / "official-last"
        / REFERENCE_RUN
        / dataset
        / "mixed_h96_h192_h336_h720"
        / "seed2021"
        / "metrics_by_target_horizon.csv"
    )


def candidate_dirs(root: Path, arm: str, dataset: str) -> list[Path]:
    base = (
        root
        / "best-val"
        / RUN_NAMES[arm]
        / dataset
        / "mixed_h96_h192_h336_h720"
    )
    return sorted(path for path in base.glob("seed*") if path.is_dir())


def seed_from_dir(path: Path) -> int:
    return int(path.name.removeprefix("seed"))


def percent_delta(value: float, reference: float) -> float:
    return (value / reference - 1.0) * 100.0


def collect_comparisons(
    candidate_root: Path,
    reference_root: Path,
) -> list[dict[str, Any]]:
    reference = {
        dataset: metric_map(reference_path(reference_root, dataset))
        for dataset in DATASETS
    }
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for dataset in DATASETS:
            for run_dir in candidate_dirs(candidate_root, arm, dataset):
                metrics_path = run_dir / "metrics_by_target_horizon.csv"
                if not metrics_path.exists():
                    continue
                seed = seed_from_dir(run_dir)
                metrics = metric_map(metrics_path)
                for horizon in HORIZONS:
                    candidate = metrics[horizon]
                    baseline = reference[dataset][horizon]
                    rows.append(
                        {
                            "arm": arm,
                            "dataset": dataset,
                            "seed": seed,
                            "horizon": horizon,
                            "candidate_mse": candidate["mse"],
                            "reference_mse": baseline["mse"],
                            "mse_delta_pct": percent_delta(
                                candidate["mse"], baseline["mse"]
                            ),
                            "mse_win": candidate["mse"] < baseline["mse"],
                            "candidate_mae": candidate["mae"],
                            "reference_mae": baseline["mae"],
                            "mae_delta_pct": percent_delta(
                                candidate["mae"], baseline["mae"]
                            ),
                            "metrics_path": str(metrics_path),
                        }
                    )
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], row["seed"], row["dataset"])].append(row)
        grouped[(row["arm"], row["seed"], "overall")].append(row)
    summaries: list[dict[str, Any]] = []
    for (arm, seed, scope), group in sorted(grouped.items()):
        candidate_mse = mean([row["candidate_mse"] for row in group])
        reference_mse = mean([row["reference_mse"] for row in group])
        candidate_mae = mean([row["candidate_mae"] for row in group])
        reference_mae = mean([row["reference_mae"] for row in group])
        summaries.append(
            {
                "arm": arm,
                "seed": seed,
                "scope": scope,
                "settings": len(group),
                "candidate_mean_mse": candidate_mse,
                "reference_mean_mse": reference_mse,
                "mean_mse_delta_pct": percent_delta(
                    candidate_mse,
                    reference_mse,
                ),
                "mse_wins": sum(bool(row["mse_win"]) for row in group),
                "max_setting_mse_delta_pct": max(
                    row["mse_delta_pct"] for row in group
                ),
                "candidate_mean_mae": candidate_mae,
                "reference_mean_mae": reference_mae,
                "mean_mae_delta_pct": percent_delta(
                    candidate_mae,
                    reference_mae,
                ),
            }
        )
    return summaries


def collect_training(candidate_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for dataset in DATASETS:
            for run_dir in candidate_dirs(candidate_root, arm, dataset):
                diagnostics_path = run_dir / "model_diagnostics.json"
                training_path = run_dir / "training_log.csv"
                if not diagnostics_path.exists() or not training_path.exists():
                    continue
                diagnostics = json.loads(diagnostics_path.read_text())
                training = read_csv(training_path)
                rows.append(
                    {
                        "arm": arm,
                        "dataset": dataset,
                        "seed": seed_from_dir(run_dir),
                        "patch_num": diagnostics["patch_num"],
                        "d_model": diagnostics["d_model"],
                        "total_parameters": diagnostics["total_parameters"],
                        "mean_epoch_seconds": mean(
                            [float(row["epoch_seconds"]) for row in training]
                        ),
                        "best_val_mean_mse": min(
                            float(row["val_mean_mse"]) for row in training
                        ),
                        "epochs": len(training),
                    }
                )
    return rows


def small_gate(
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_key = {
        (row["arm"], row["seed"], row["scope"]): row for row in summaries
    }
    decisions: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        overall = by_key.get((arm, 2021, "overall"))
        datasets = [by_key.get((arm, 2021, dataset)) for dataset in DATASETS]
        arm_rows = [
            row
            for row in comparisons
            if row["arm"] == arm and row["seed"] == 2021
        ]
        complete = overall is not None and all(row is not None for row in datasets)
        finite = complete and all(
            math.isfinite(float(row["candidate_mse"])) for row in arm_rows
        )
        passed = bool(
            complete
            and finite
            and overall["mean_mse_delta_pct"] <= 0.5
            and overall["mse_wins"] >= 6
            and all(row["mean_mse_delta_pct"] <= 1.0 for row in datasets)
            and overall["max_setting_mse_delta_pct"] <= 5.0
        )
        decisions[arm] = {
            "complete": complete,
            "finite": finite,
            "passed": passed,
            "overall_delta_pct": (
                overall["mean_mse_delta_pct"] if overall is not None else None
            ),
            "wins": overall["mse_wins"] if overall is not None else None,
        }
    return decisions


def choose_winner(decisions: dict[str, dict[str, Any]]) -> str | None:
    passed = [arm for arm in ARMS if decisions[arm]["passed"]]
    if not passed:
        return None
    if len(passed) == 1:
        return passed[0]
    deltas = {arm: float(decisions[arm]["overall_delta_pct"]) for arm in passed}
    if abs(deltas["cpe_p16s8"] - deltas["cpe_p48s24"]) <= 0.3:
        return "cpe_p48s24"
    return min(passed, key=deltas.get)


def write_report(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    training: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> None:
    seed_rows = {
        (row["arm"], row["scope"]): row
        for row in summaries
        if row["seed"] == 2021
    }
    winner = choose_winner(decisions)
    lines = [
        "# B14 Prerequisite Contextual Patch Encoder Gate Report",
        "",
        "## Small-Gate Result",
        "",
        "| Arm | Mean MSE delta | MSE wins | ETTh2 | ETTm1 | Weather | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm in ARMS:
        overall = seed_rows.get((arm, "overall"))
        if overall is None:
            lines.append(f"| `{arm}` | missing | missing | missing | missing | missing | pending |")
            continue
        dataset_deltas = [
            seed_rows[(arm, dataset)]["mean_mse_delta_pct"] for dataset in DATASETS
        ]
        lines.append(
            f"| `{arm}` | `{overall['mean_mse_delta_pct']:+.3f}%` | "
            f"`{overall['mse_wins']}/12` | `{dataset_deltas[0]:+.3f}%` | "
            f"`{dataset_deltas[1]:+.3f}%` | `{dataset_deltas[2]:+.3f}%` | "
            f"`{'pass' if decisions[arm]['passed'] else 'fail'}` |"
        )
    lines.extend(["", "## Decision", ""])
    if winner is None:
        lines.extend(
            [
                "[Decision] No arm passes the pre-registered carrier effectiveness gate.",
                "Legacy A6 remains active；rollback to Step 5/6 failure attribution before B14.",
            ]
        )
    else:
        lines.extend(
            [
                f"[Decision] Small-gate winner: `{winner}`.",
                "Only this arm is authorized for seeds 2022/2023 confirmation；legacy A6 remains active until confirmation passes.",
            ]
        )
    lines.extend(
        [
            "",
            "## Capacity And Runtime",
            "",
            "| Arm | Dataset | P | D | Parameters | Mean epoch seconds |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(training, key=lambda item: (item["arm"], item["dataset"], item["seed"])):
        if row["seed"] != 2021:
            continue
        lines.append(
            f"| `{row['arm']}` | {row['dataset']} | {row['patch_num']} | "
            f"{row['d_model']} | {row['total_parameters']} | "
            f"{row['mean_epoch_seconds']:.2f} |"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "contextual_patch_encoder_gate_report.md").write_text(
        "\n".join(lines) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path(
            "analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/raw"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/phase5_stage_b_b14_prerequisite_patchwise_encoder_20260710"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparisons = collect_comparisons(args.candidate_root, args.reference_root)
    if not comparisons:
        raise FileNotFoundError(
            f"No candidate metrics found under {args.candidate_root}"
        )
    summaries = summarize(comparisons)
    training = collect_training(args.candidate_root)
    decisions = small_gate(summaries, comparisons)
    write_csv(args.output_dir / "contextual_patch_encoder_comparison.csv", comparisons)
    write_csv(args.output_dir / "contextual_patch_encoder_summary.csv", summaries)
    write_csv(args.output_dir / "contextual_patch_encoder_training.csv", training)
    (args.output_dir / "contextual_patch_encoder_decision.json").write_text(
        json.dumps(
            {
                "small_gate": decisions,
                "winner": choose_winner(decisions),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_report(args.output_dir, summaries, training, decisions)


if __name__ == "__main__":
    main()

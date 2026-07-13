#!/usr/bin/env python3
"""Audit unified early-stopping candidates against SC0 validation trajectories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


PATIENCE_CANDIDATES = (3, 5, 7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_trajectory(path: Path) -> list[tuple[int, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [(int(row["epoch"]), float(row["val_mean_mse"])) for row in rows]


def simulate(
    trajectory: list[tuple[int, float]],
    patience: int,
    min_delta: float = 0.0,
) -> dict[str, Any]:
    best_epoch = 0
    best_value = float("inf")
    counter = 0
    stop_epoch = trajectory[-1][0]
    stopped_early = False
    for epoch, value in trajectory:
        if value < best_value - min_delta:
            best_epoch = epoch
            best_value = value
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                stop_epoch = epoch
                stopped_early = True
                break
    return {
        "stop_epoch": stop_epoch,
        "selected_epoch": best_epoch,
        "selected_val_mse": best_value,
        "stopped_early": int(stopped_early),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    paths = sorted(args.raw_root.glob("SC0_*/**/seed2021/training_log.csv"))
    if len(paths) != 9:
        raise RuntimeError(f"expected 9 SC0 trajectories, found {len(paths)}")

    for path in paths:
        trajectory = read_trajectory(path)
        full_best_epoch, full_best = min(trajectory, key=lambda item: item[1])
        relative = path.relative_to(args.raw_root)
        arm = relative.parts[0].removeprefix("SC0_").removesuffix("_validation_only")
        dataset = relative.parts[1]
        for patience in PATIENCE_CANDIDATES:
            result = simulate(trajectory, patience)
            rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "patience": patience,
                    "max_epochs": len(trajectory),
                    "stop_epoch": result["stop_epoch"],
                    "selected_epoch": result["selected_epoch"],
                    "full_trajectory_best_epoch": full_best_epoch,
                    "selected_val_mse": result["selected_val_mse"],
                    "full_trajectory_best_val_mse": full_best,
                    "retains_full_trajectory_best": int(
                        abs(float(result["selected_val_mse"]) - full_best) <= 1e-12
                    ),
                    "stopped_early": result["stopped_early"],
                    "epochs_saved": len(trajectory) - int(result["stop_epoch"]),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "sc0_r1_stopping_policy_audit.csv", rows)
    summary = []
    for patience in PATIENCE_CANDIDATES:
        selected = [row for row in rows if row["patience"] == patience]
        summary.append(
            {
                "patience": patience,
                "retained_runs": sum(row["retains_full_trajectory_best"] for row in selected),
                "total_runs": len(selected),
                "stopped_early_runs": sum(row["stopped_early"] for row in selected),
                "total_epochs_saved": sum(row["epochs_saved"] for row in selected),
                "mean_epochs_saved": sum(row["epochs_saved"] for row in selected)
                / len(selected),
            }
        )
    write_csv(args.output_dir / "sc0_r1_stopping_policy_summary.csv", summary)

    chosen = next(
        (row for row in summary if row["retained_runs"] == row["total_runs"]),
        None,
    )
    lines = [
        "# StageC SC0-R1 Unified Stopping Policy Offline Gate",
        "",
        "## Metric Definitions",
        "",
        "- `stop_epoch`: 从原SC0 validation trajectory按连续未改善epoch数模拟得到的停止点。",
        "- `selected_epoch`: stop前最低`val_mean_mse`对应epoch，即restore-best checkpoint。",
        "- `retains_full_trajectory_best`: stop前best是否等于完整20-epoch trajectory的best。",
        "- `epochs_saved`: `20 - stop_epoch`；只表示计算节省，不参与profile选择。",
        "",
        "## Candidate Summary",
        "",
        "| patience | retained runs | early-stopped runs | total epochs saved | mean saved/run |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['patience']} | {row['retained_runs']}/{row['total_runs']} | "
            f"{row['stopped_early_runs']}/{row['total_runs']} | "
            f"{row['total_epochs_saved']} | {row['mean_epochs_saved']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                f"[Strong Evidence] 最小且保留9/9原SC0 best checkpoint的候选是"
                f"`patience={chosen['patience']}`。SC0-R1预注册`max_epochs=20`, "
                "`min_delta=0`, `restore_best=true`。"
                if chosen
                else "[Decision] 没有候选保留9/9 best checkpoint；不得启动SC0-R1。"
            ),
            "",
            "该离线结果只验证stopping rule不会在已有trajectory上截断已知best；它不证明新seed也会"
            "稳定，因此SC0-R1仍须运行三臂全部三个seeds。",
        ]
    )
    (args.output_dir / "sc0_r1_stopping_policy_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"sc0_r1_stopping_policy_done chosen={chosen}")


if __name__ == "__main__":
    main()

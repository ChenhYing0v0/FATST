#!/usr/bin/env python3
"""Analyze validation-only StageC SC0 carrier calibration artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_c_mechanism_control.json"
SELECTORS = ("best_val", "last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=None)
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


def config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir(root: Path, dataset: str, arm: str, seed: int) -> Path:
    return root / f"SC0_{arm}_validation_only" / dataset / "h720_full" / f"seed{seed}"


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def collect(
    raw_root: Path,
    config_path: Path,
    config: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    expected_hash = config_hash(config_path)
    common = config["common"]

    for dataset in config["datasets"]:
        for arm_name, arm in config["arms"].items():
            directory = run_dir(raw_root, dataset, arm_name, seed)
            diagnostics_path = directory / "model_diagnostics.json"
            effective_path = directory / "effective_config.json"
            training_path = directory / "training_log.csv"
            required = (diagnostics_path, effective_path, training_path)
            if not all(path.exists() for path in required):
                diagnostic_rows.append(
                    {
                        "dataset": dataset,
                        "arm": arm_name,
                        "status": "missing",
                        "run_dir": str(directory),
                    }
                )
                continue

            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            effective = json.loads(effective_path.read_text(encoding="utf-8"))
            adapter = effective["adapter"]
            official = effective["official_args"]
            training = read_csv(training_path)
            training_finite = bool(training) and all(
                finite(row.get(field))
                for row in training
                for field in ("train_loss", "val_mean_mse", "lr")
            )
            mean_epoch_seconds = (
                mean(float(row["epoch_seconds"]) for row in training)
                if training and all(finite(row.get("epoch_seconds")) for row in training)
                else math.inf
            )
            config_ok = (
                adapter.get("protocol_class") == config["protocol_class"]
                and adapter.get("protocol_profile") == config["protocol_profile"]
                and adapter.get("profile_hash") == expected_hash
                and adapter.get("final_evaluation_split") == "val"
                and int(official["patch_num"]) == int(arm["patch_num"])
                and int(official["d_model"]) == int(arm["d_model"])
                and int(official["d_ff"]) == int(arm["d_ff"])
                and abs(float(official["dropout"]) - float(common["dropout"])) < 1e-12
                and int(official["layer_norm"]) == int(common["layer_norm"])
            )
            parameter_ok = (
                int(diagnostics["active_forward_parameters"])
                == int(arm["active_forward_parameters"])
                and int(diagnostics["unused_proj_x_parameters"])
                == int(arm["unused_proj_x_parameters"])
            )
            metric_files_present = all(
                (directory / f"metrics_{selector}_by_target_horizon.csv").exists()
                for selector in SELECTORS
            )
            diagnostic_rows.append(
                {
                    "dataset": dataset,
                    "arm": arm_name,
                    "status": (
                        "ok"
                        if config_ok and parameter_ok and training_finite and metric_files_present
                        else "mismatch"
                    ),
                    "config_ok": int(config_ok),
                    "parameter_ok": int(parameter_ok),
                    "training_finite": int(training_finite),
                    "dual_metrics_present": int(metric_files_present),
                    "patch_num": official["patch_num"],
                    "d_model": official["d_model"],
                    "d_ff": official["d_ff"],
                    "active_forward_parameters": diagnostics[
                        "active_forward_parameters"
                    ],
                    "unused_proj_x_parameters": diagnostics[
                        "unused_proj_x_parameters"
                    ],
                    "mean_epoch_seconds": mean_epoch_seconds,
                    "run_dir": str(directory),
                }
            )

            if not metric_files_present:
                continue
            for selector in SELECTORS:
                rows = read_csv(
                    directory / f"metrics_{selector}_by_target_horizon.csv"
                )
                for row in rows:
                    if row.get("evaluation_split") != "val":
                        raise ValueError(
                            f"forbidden non-validation metric in {directory}: {row}"
                        )
                    if row.get("protocol_class") != "mechanism_control":
                        raise ValueError(f"wrong protocol class in {directory}")
                    horizon = int(row["target_horizon"])
                    metric_rows.append(
                        {
                            "dataset": dataset,
                            "arm": arm_name,
                            "selector": selector,
                            "target_horizon": horizon,
                            "mse": float(row["mse"]),
                            "mae": float(row["mae"]),
                            "status": (
                                "ok"
                                if finite(row["mse"]) and finite(row["mae"])
                                else "nonfinite"
                            ),
                        }
                    )
    return metric_rows, diagnostic_rows


def efficiency_by_arm(diagnostics: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in diagnostics:
        if row["status"] == "ok" and finite(row.get("mean_epoch_seconds")):
            values.setdefault(str(row["arm"]), []).append(
                float(row["mean_epoch_seconds"])
            )
    return {arm: mean(times) for arm, times in values.items()}


def selection_rows(
    metrics: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    full_horizon = int(config["common"]["pred_len"])
    efficiency = efficiency_by_arm(diagnostics)
    rows: list[dict[str, Any]] = []
    winners: dict[str, str] = {}

    for selector in SELECTORS:
        selected = [
            row
            for row in metrics
            if row["selector"] == selector
            and row["target_horizon"] == full_horizon
            and row["status"] == "ok"
        ]
        by_dataset: dict[str, dict[str, float]] = {}
        for row in selected:
            by_dataset.setdefault(str(row["dataset"]), {})[str(row["arm"])] = float(
                row["mse"]
            )
        if any(
            set(by_dataset.get(dataset, {})) != set(config["arms"])
            for dataset in config["datasets"]
        ):
            continue

        arm_rows: list[dict[str, Any]] = []
        for arm in config["arms"]:
            regrets = {
                dataset: by_dataset[dataset][arm]
                / min(by_dataset[dataset].values())
                - 1.0
                for dataset in config["datasets"]
            }
            arm_rows.append(
                {
                    "selector": selector,
                    "arm": arm,
                    "macro_regret": mean(regrets.values()),
                    "max_dataset_regret": max(regrets.values()),
                    "per_dataset_gate_pass": int(
                        max(regrets.values())
                        <= float(config["gates"]["max_per_dataset_regret"])
                    ),
                    "mean_epoch_seconds": efficiency.get(arm, math.inf),
                    **{
                        f"{dataset}_val_mse": by_dataset[dataset][arm]
                        for dataset in config["datasets"]
                    },
                    **{
                        f"{dataset}_regret": regrets[dataset]
                        for dataset in config["datasets"]
                    },
                }
            )
        arm_rows.sort(key=lambda row: (row["macro_regret"], row["mean_epoch_seconds"]))
        best = arm_rows[0]
        if len(arm_rows) > 1:
            score_gap = float(arm_rows[1]["macro_regret"]) - float(
                best["macro_regret"]
            )
            if score_gap < float(config["gates"]["tie_macro_score_delta"]):
                tied = [
                    row
                    for row in arm_rows
                    if float(row["macro_regret"]) - float(best["macro_regret"])
                    < float(config["gates"]["tie_macro_score_delta"])
                ]
                best = min(
                    tied,
                    key=lambda row: (
                        float(row["mean_epoch_seconds"]),
                        int(config["arms"][str(row["arm"])]["active_forward_parameters"]),
                    ),
                )
        winners[selector] = str(best["arm"])
        for row in arm_rows:
            row["selected"] = int(row["arm"] == best["arm"])
        rows.extend(arm_rows)
    return rows, winners


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_outputs(
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    seed: int,
    metrics: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    winners: dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "sc0_validation_horizon_metrics.csv", metrics)
    write_csv(output_dir / "sc0_run_diagnostics.csv", diagnostics)
    write_csv(output_dir / "sc0_global_selection.csv", selections)

    expected_runs = len(config["datasets"]) * len(config["arms"])
    complete = (
        len(diagnostics) == expected_runs
        and all(row["status"] == "ok" for row in diagnostics)
        and set(winners) == set(SELECTORS)
    )
    selector_stable = complete and winners["best_val"] == winners["last"]
    primary_selected = next(
        (
            row
            for row in selections
            if row["selector"] == "best_val" and row["selected"] == 1
        ),
        None,
    )
    regret_gate = bool(
        primary_selected and primary_selected["per_dataset_gate_pass"] == 1
    )
    if complete and selector_stable and regret_gate:
        decision = "preliminary_global_profile_selected_needs_seed_confirmation"
    elif complete:
        decision = "common_token_mlp_profile_not_supported"
    else:
        decision = "analysis_incomplete"

    summary = {
        "candidate": "SC0-MCP",
        "seed": seed,
        "profile_hash": config_hash(config_path),
        "expected_runs": expected_runs,
        "complete": complete,
        "selector_stable": selector_stable,
        "regret_gate": regret_gate,
        "winners": winners,
        "selected_arm": winners.get("best_val") if regret_gate else None,
        "decision": decision,
        "test_metrics_used_for_selection": False,
    }
    (output_dir / "sc0_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    selected_rows = [row for row in selections if row.get("selected") == 1]
    report = [
        "# StageC SC0 Standardized Carrier Calibration Report",
        "",
        f"- `current_step`: StageC Step 3 control calibration, seed `{seed}`.",
        "- `role`: validation-only protocol control; not a paper-core method.",
        f"- `profile_hash`: `{config_hash(config_path)}`.",
        f"- `decision`: `{decision}`.",
        "- `test_metrics_used_for_selection`: `false`.",
        "",
        "## Selected Global Profiles",
        "",
        *markdown_table(
            selected_rows,
            [
                "selector",
                "arm",
                "macro_regret",
                "max_dataset_regret",
                "per_dataset_gate_pass",
                "mean_epoch_seconds",
            ],
        ),
        "",
        "## Full Selection Table",
        "",
        *markdown_table(
            selections,
            [
                "selector",
                "arm",
                "macro_regret",
                "max_dataset_regret",
                "per_dataset_gate_pass",
                "selected",
            ],
        ),
        "",
        "## Gate Interpretation",
        "",
        f"- Complete run/config/numeric gate: `{complete}`.",
        f"- Last/best global winner stability: `{selector_stable}`.",
        f"- Best-validation per-dataset regret gate: `{regret_gate}`.",
        "- A preliminary pass only authorizes seeds 2022/2023 for the selected arm.",
        "- A failure rolls StageC back to Step 2/3; it does not authorize dataset-specific presets.",
        "",
        "## Failure Attribution Boundary",
        "",
        "SC0 only tests whether one standardized token-MLP carrier profile is viable. A failure does not reject",
        "unified forecasting, projective decoding, or horizon-measure learning. It rejects this common carrier",
        "family as a sufficiently stable research instrument under the preregistered gate.",
    ]
    (output_dir / "sc0_carrier_calibration_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = args.seed or int(config["common"]["initial_seed"])
    metrics, diagnostics = collect(
        args.raw_root,
        args.config,
        config,
        seed,
    )
    selections, winners = selection_rows(metrics, diagnostics, config)
    write_outputs(
        args.output_dir,
        args.config,
        config,
        seed,
        metrics,
        diagnostics,
        selections,
        winners,
    )
    print(
        f"stage_c_sc0_analysis_done output_dir={args.output_dir} "
        f"winners={winners}"
    )


if __name__ == "__main__":
    main()

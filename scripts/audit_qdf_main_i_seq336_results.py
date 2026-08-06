#!/usr/bin/env python3
"""Audit QDF L336 artifacts and compare the complete Main I scorecard."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Solar", "Exchange")
HORIZONS = (96, 192, 336, 720)
ROLES = ("checkpoint", "learned_qdf_loss", "metrics", "effective_config", "stdout")


def parse_args() -> argparse.Namespace:
    base = (
        ROOT
        / "analysis"
        / "iscf_bsca_paper_experiment_consolidation_20260731"
        / "qdf_main_i_l336_20260806"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs" / "qdf_main_i_seq336_reproduction.json")
    parser.add_argument("--remote-lite", type=Path, default=base / "remote_lite")
    parser.add_argument("--qdf-metrics", type=Path, default=base / "remote_lite" / "audit" / "qdf_main_i_l336_local_metrics.csv")
    parser.add_argument("--artifact-manifest", type=Path, default=base / "remote_lite" / "audit" / "qdf_main_i_l336_artifact_manifest.csv")
    parser.add_argument("--iscf", type=Path, default=ROOT / "analysis" / "iscf_bsca_main_v1_hpo_20260731" / "final_hpo_freeze_20260806" / "selected_main_scorecard_final.csv")
    parser.add_argument("--timealign", type=Path, default=ROOT / "analysis" / "iscf_bsca_paper_experiment_consolidation_20260731" / "timealign_main_i_full_reproduction_20260806" / "timealign_main_i_local_metrics.csv")
    parser.add_argument("--qdf-l96-published", type=Path, default=ROOT / "analysis" / "iscf_bsca_paper_experiment_consolidation_20260731" / "qdf_main_i_20260806" / "qdf_table6_published.csv")
    parser.add_argument("--qdf-l96-solar", type=Path, default=ROOT / "analysis" / "iscf_bsca_paper_experiment_consolidation_20260731" / "qdf_main_i_20260806" / "qdf_solar_local_metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=base)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def key_rows(rows: list[dict[str, str]], mse: str, mae: str) -> dict[tuple[str, int], tuple[float, float]]:
    return {
        (row["dataset"], int(row["horizon"])): (float(row[mse]), float(row[mae]))
        for row in rows
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def pct(candidate: float, reference: float) -> float:
    return 100.0 * (candidate / reference - 1.0)


def validate_matrix(name: str, values: dict[tuple[str, int], tuple[float, float]]) -> None:
    expected = {(dataset, horizon) for dataset in DATASETS for horizon in HORIZONS}
    if set(values) != expected:
        raise ValueError(f"{name} matrix mismatch: missing={sorted(expected - set(values))}, extra={sorted(set(values) - expected)}")
    if not all(math.isfinite(metric) for pair in values.values() for metric in pair):
        raise ValueError(f"{name} contains a non-finite metric")


def audit_artifacts(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    manifest = read_csv(args.artifact_manifest)
    if len(manifest) != 160:
        raise ValueError(f"expected 160 artifact rows, found {len(manifest)}")
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    synced_hash_verified_rows = 0
    for row in manifest:
        grouped[(row["dataset"], int(row["horizon"]))].append(row)
        if len(row["sha256"]) != 64 or int(row["bytes"]) <= 0:
            raise ValueError(f"invalid manifest entry: {row}")
        if row["artifact_role"] in {"metrics", "effective_config", "stdout"}:
            marker = "/qdf_main_i_seq336_20260806/"
            if marker not in row["path"]:
                raise ValueError(f"unexpected remote artifact path: {row['path']}")
            local_path = args.remote_lite / row["path"].split(marker, 1)[1]
            if not local_path.is_file() or sha256(local_path) != row["sha256"]:
                raise ValueError(f"synced artifact hash mismatch: {local_path}")
            synced_hash_verified_rows += 1
    expected_keys = {(dataset, horizon) for dataset in DATASETS for horizon in HORIZONS}
    if set(grouped) != expected_keys:
        raise ValueError("artifact job keys do not match 8x4 matrix")
    for job, rows in grouped.items():
        if Counter(row["artifact_role"] for row in rows) != Counter(ROLES):
            raise ValueError(f"artifact roles mismatch for {job}")

    config_paths = list((args.remote_lite / "runs").rglob("config.yaml"))
    stdout_paths = list((args.remote_lite / "runs").glob("QDF__*/stdout.log"))
    if len(config_paths) != 32 or len(stdout_paths) != 32:
        raise ValueError(f"synced config/stdout count mismatch: {len(config_paths)}/{len(stdout_paths)}")
    audited_configs = 0
    for config_path in config_paths:
        run_name = next(part for part in config_path.parts if part.startswith("QDF__"))
        match = re.fullmatch(r"QDF__(.+)__H(\d+)__seed2023", run_name)
        if match is None:
            raise ValueError(f"unexpected run name: {run_name}")
        dataset, horizon = match.group(1), int(match.group(2))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        dataset_spec = protocol["dataset_contracts"][dataset]
        profile = protocol["profiles"][dataset][str(horizon)]
        expected = {
            "data": dataset_spec["data"], "data_id": dataset,
            "seq_len": 336, "label_len": 48, "pred_len": horizon,
            "enc_in": dataset_spec["channels"], "dec_in": dataset_spec["channels"], "c_out": dataset_spec["channels"],
            "cycle": dataset_spec["cycle"], "dropout": dataset_spec["dropout"],
            "learning_rate": profile[0], "inner_lr": profile[1], "meta_lr": profile[2],
            "warmup_steps": profile[3], "num_tasks": profile[4], "meta_inner_steps": profile[5], "batch_size": profile[6],
            "train_epochs": 30, "patience": 5, "num_workers": 0,
            "max_train_batches": 0, "max_eval_batches": 0,
            "final_evaluation_split": "test", "fix_seed": 2023,
        }
        for field, expected_value in expected.items():
            if config.get(field) != expected_value:
                raise ValueError(f"config mismatch {dataset} H{horizon} {field}: {config.get(field)!r} != {expected_value!r}")
        audited_configs += 1

    failure_pattern = re.compile(r"Traceback|CUDA out of memory|Too many open files|(?<![A-Za-z])nan(?![A-Za-z])", re.IGNORECASE)
    for stdout_path in stdout_paths:
        text = stdout_path.read_text(encoding="utf-8", errors="replace")
        if failure_pattern.search(text):
            raise ValueError(f"failure marker in {stdout_path}")
    queue_text = (args.remote_lite / "queue_run.log").read_text(encoding="utf-8")
    queue_starts = sum(line.startswith("run_start=") for line in queue_text.splitlines())
    queue_completions = sum(line.startswith("run_done=") for line in queue_text.splitlines())
    if queue_starts != 32 or queue_completions != 32 or "qdf_main_i_l336_run_done=" not in queue_text:
        raise ValueError("formal queue log is not a complete 32-start/32-done ledger")

    role_hash_counts = {
        role: len({row["sha256"] for row in manifest if row["artifact_role"] == role})
        for role in ROLES
    }
    if role_hash_counts["checkpoint"] != 32 or role_hash_counts["learned_qdf_loss"] != 32:
        raise ValueError(f"checkpoint/loss hashes are not unique: {role_hash_counts}")
    return {
        "artifact_rows": len(manifest),
        "audited_configs": audited_configs,
        "audited_stdout_logs": len(stdout_paths),
        "synced_hash_verified_rows": synced_hash_verified_rows,
        "queue_starts": queue_starts,
        "queue_completions": queue_completions,
        "role_unique_hash_counts": role_hash_counts,
        "artifact_gate": "pass",
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    artifact_audit = audit_artifacts(args, protocol)

    qdf = key_rows(read_csv(args.qdf_metrics), "mse", "mae")
    iscf = key_rows(read_csv(args.iscf), "test_mse", "test_mae")
    timealign = key_rows(read_csv(args.timealign), "mse", "mae")
    for name, values in (("QDF L336", qdf), ("ISCF-BSCA", iscf), ("TimeAlign", timealign)):
        validate_matrix(name, values)
    qdf_l96 = key_rows(read_csv(args.qdf_l96_published) + read_csv(args.qdf_l96_solar), "mse", "mae")

    cell_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            key = (dataset, horizon)
            q_mse, q_mae = qdf[key]
            i_mse, i_mae = iscf[key]
            t_mse, t_mae = timealign[key]
            old = qdf_l96.get(key)
            cell_rows.append({
                "dataset": dataset, "horizon": horizon,
                "qdf_l336_mse": q_mse, "qdf_l336_mae": q_mae,
                "iscf_mse": i_mse, "iscf_mae": i_mae,
                "timealign_mse": t_mse, "timealign_mae": t_mae,
                "qdf_l96_mse": "" if old is None else old[0], "qdf_l96_mae": "" if old is None else old[1],
                "qdf_vs_iscf_mse_pct": pct(q_mse, i_mse), "qdf_vs_iscf_mae_pct": pct(q_mae, i_mae),
                "qdf_vs_timealign_mse_pct": pct(q_mse, t_mse), "qdf_vs_timealign_mae_pct": pct(q_mae, t_mae),
                "qdf_l336_vs_l96_mse_pct": "" if old is None else pct(q_mse, old[0]),
                "qdf_l336_vs_l96_mae_pct": "" if old is None else pct(q_mae, old[1]),
                "qdf_beats_iscf_mse": q_mse < i_mse, "qdf_beats_iscf_mae": q_mae < i_mae,
                "qdf_beats_timealign_mse": q_mse < t_mse, "qdf_beats_timealign_mae": q_mae < t_mae,
            })
    write_csv(args.output_dir / "qdf_l336_cell_comparison.csv", cell_rows)

    dataset_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        subset = [row for row in cell_rows if row["dataset"] == dataset]
        old_available = all(row["qdf_l96_mse"] != "" for row in subset)
        dataset_rows.append({
            "dataset": dataset,
            "qdf_l336_mean_mse": mean([float(row["qdf_l336_mse"]) for row in subset]),
            "qdf_l336_mean_mae": mean([float(row["qdf_l336_mae"]) for row in subset]),
            "iscf_mean_mse": mean([float(row["iscf_mse"]) for row in subset]),
            "iscf_mean_mae": mean([float(row["iscf_mae"]) for row in subset]),
            "timealign_mean_mse": mean([float(row["timealign_mse"]) for row in subset]),
            "timealign_mean_mae": mean([float(row["timealign_mae"]) for row in subset]),
            "qdf_l96_mean_mse": "" if not old_available else mean([float(row["qdf_l96_mse"]) for row in subset]),
            "qdf_l96_mean_mae": "" if not old_available else mean([float(row["qdf_l96_mae"]) for row in subset]),
            "qdf_vs_iscf_mse_pct": pct(mean([float(row["qdf_l336_mse"]) for row in subset]), mean([float(row["iscf_mse"]) for row in subset])),
            "qdf_vs_iscf_mae_pct": pct(mean([float(row["qdf_l336_mae"]) for row in subset]), mean([float(row["iscf_mae"]) for row in subset])),
            "qdf_vs_timealign_mse_pct": pct(mean([float(row["qdf_l336_mse"]) for row in subset]), mean([float(row["timealign_mse"]) for row in subset])),
            "qdf_vs_timealign_mae_pct": pct(mean([float(row["qdf_l336_mae"]) for row in subset]), mean([float(row["timealign_mae"]) for row in subset])),
            "qdf_l336_vs_l96_mse_pct": "" if not old_available else pct(mean([float(row["qdf_l336_mse"]) for row in subset]), mean([float(row["qdf_l96_mse"]) for row in subset])),
            "qdf_l336_vs_l96_mae_pct": "" if not old_available else pct(mean([float(row["qdf_l336_mae"]) for row in subset]), mean([float(row["qdf_l96_mae"]) for row in subset])),
            "qdf_beats_iscf_cells": sum(bool(row[metric]) for row in subset for metric in ("qdf_beats_iscf_mse", "qdf_beats_iscf_mae")),
            "qdf_beats_timealign_cells": sum(bool(row[metric]) for row in subset for metric in ("qdf_beats_timealign_mse", "qdf_beats_timealign_mae")),
        })
    write_csv(args.output_dir / "qdf_l336_dataset_summary.csv", dataset_rows)

    seven = [row for row in cell_rows if row["dataset"] != "Exchange"]
    qdf_l336_7_mse = mean([float(row["qdf_l336_mse"]) for row in seven])
    qdf_l336_7_mae = mean([float(row["qdf_l336_mae"]) for row in seven])
    qdf_l96_7_mse = mean([float(row["qdf_l96_mse"]) for row in seven])
    qdf_l96_7_mae = mean([float(row["qdf_l96_mae"]) for row in seven])
    iscf_7_mse = mean([float(row["iscf_mse"]) for row in seven])
    iscf_7_mae = mean([float(row["iscf_mae"]) for row in seven])
    timealign_7_mse = mean([float(row["timealign_mse"]) for row in seven])
    timealign_7_mae = mean([float(row["timealign_mae"]) for row in seven])
    summary = {
        "protocol_id": protocol["protocol_id"],
        "matrix_complete": True,
        "artifact_audit": artifact_audit,
        "qdf_l336_macro_8_mse": mean([float(row["qdf_l336_mse"]) for row in cell_rows]),
        "qdf_l336_macro_8_mae": mean([float(row["qdf_l336_mae"]) for row in cell_rows]),
        "qdf_l336_macro_7_mse": qdf_l336_7_mse,
        "qdf_l336_macro_7_mae": qdf_l336_7_mae,
        "qdf_l96_macro_7_mse": qdf_l96_7_mse,
        "qdf_l96_macro_7_mae": qdf_l96_7_mae,
        "iscf_macro_7_mse": iscf_7_mse,
        "iscf_macro_7_mae": iscf_7_mae,
        "timealign_macro_7_mse": timealign_7_mse,
        "timealign_macro_7_mae": timealign_7_mae,
        "qdf_l336_vs_l96_macro_7_mse_pct": pct(qdf_l336_7_mse, qdf_l96_7_mse),
        "qdf_l336_vs_l96_macro_7_mae_pct": pct(qdf_l336_7_mae, qdf_l96_7_mae),
        "qdf_l336_vs_iscf_macro_7_mse_pct": pct(qdf_l336_7_mse, iscf_7_mse),
        "qdf_l336_vs_iscf_macro_7_mae_pct": pct(qdf_l336_7_mae, iscf_7_mae),
        "qdf_l336_vs_timealign_macro_7_mse_pct": pct(qdf_l336_7_mse, timealign_7_mse),
        "qdf_l336_vs_timealign_macro_7_mae_pct": pct(qdf_l336_7_mae, timealign_7_mae),
        "qdf_beats_iscf_cells_8_datasets": sum(bool(row[metric]) for row in cell_rows for metric in ("qdf_beats_iscf_mse", "qdf_beats_iscf_mae")),
        "qdf_beats_timealign_cells_8_datasets": sum(bool(row[metric]) for row in cell_rows for metric in ("qdf_beats_timealign_mse", "qdf_beats_timealign_mae")),
        "qdf_beats_iscf_cells_7_dense_datasets": sum(bool(row[metric]) for row in seven for metric in ("qdf_beats_iscf_mse", "qdf_beats_iscf_mae")),
        "qdf_beats_timealign_cells_7_dense_datasets": sum(bool(row[metric]) for row in seven for metric in ("qdf_beats_timealign_mse", "qdf_beats_timealign_mae")),
        "qdf_beats_iscf_mse_cells_7_dense_datasets": sum(bool(row["qdf_beats_iscf_mse"]) for row in seven),
        "qdf_beats_iscf_mae_cells_7_dense_datasets": sum(bool(row["qdf_beats_iscf_mae"]) for row in seven),
        "qdf_beats_timealign_mse_cells_7_dense_datasets": sum(bool(row["qdf_beats_timealign_mse"]) for row in seven),
        "qdf_beats_timealign_mae_cells_7_dense_datasets": sum(bool(row["qdf_beats_timealign_mae"]) for row in seven),
        "claim_boundary": "QDF is a single-seed horizon-specific native baseline; Solar and Exchange profiles are source-informed; comparisons are effectiveness context, not matched attribution",
    }
    (args.output_dir / "qdf_l336_result_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

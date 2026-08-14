#!/usr/bin/env python3
"""Freeze or verify the 77-object ISCF-BSCA efficiency checkpoint manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DATASETS = ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather", "ECL", "Solar")
HORIZONS = (96, 192, 336, 720)
OUTPUT_ROOT = Path("/home/yingch/exp_outputs/r-2026-fatst")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/iscf_bsca_efficiency_protocol.json"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verify-manifest", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError(f"missing artifact: {path}")
    return path


def row(
    system: str,
    dataset: str,
    horizon: int | str,
    checkpoint: Path,
    training_log: Path,
    effective_config: Path | None,
    service_role: str,
) -> dict[str, Any]:
    require_file(checkpoint)
    require_file(training_log)
    if effective_config is not None:
        require_file(effective_config)
    return {
        "object_id": f"{system}__{dataset}__H{horizon}",
        "system": system,
        "dataset": dataset,
        "horizon": horizon,
        "service_role": service_role,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "training_log_path": str(training_log),
        "effective_config_path": (
            str(effective_config) if effective_config is not None else ""
        ),
    }


def iscf_rows() -> list[dict[str, Any]]:
    root = OUTPUT_ROOT / "iscf_bsca_main_v1_hpo"
    run_dirs = {
        "ETTm1": root / "h2/trials/ETTm1/ETTm1__h2_table5_capacity/seed2021",
        "ETTm2": root / "h4m/trials/ETTm2/ETTm2__h4m_p6_lr5e5/seed2021",
        "ETTh1": root / "h5d/trials/ETTh1/ETTh1__h5d_bs16_lr2p4/seed2021",
        "ETTh2": root / "h2/trials/ETTh2/ETTh2__h2_lr5e4/seed2021",
        "Weather": root
        / "h4n/trials/Weather/Weather__h4n_seq608_p19_lr2e5/seed2021",
        "ECL": root / "h5a/trials/ECL/ECL__h5a_seq336_p1/seed2021",
        "Solar": root
        / "h5a/trials/Solar/Solar__h5a_seq512_p4_lr2p5e4/seed2021",
    }
    return [
        row(
            "ISCF-BSCA",
            dataset,
            "all",
            directory / "checkpoint.pt",
            directory / "training_log.csv",
            directory / "effective_config.json",
            "one_model_native_unified",
        )
        for dataset, directory in run_dirs.items()
    ]


def timealign_rows() -> list[dict[str, Any]]:
    primary = (
        OUTPUT_ROOT
        / "timealign_official_reproduction/main_i_8dataset_20260806/runs"
    )
    reused = (
        OUTPUT_ROOT
        / "timealign_official_reproduction/ettm2_weather_20260804/runs"
    )
    rows = []
    for dataset in DATASETS:
        root = reused if dataset in {"ETTm2", "Weather"} else primary
        for horizon in HORIZONS:
            directory = root / f"TimeAlign__{dataset}__H{horizon}__seed2021"
            rows.append(
                row(
                    "TimeAlign",
                    dataset,
                    horizon,
                    directory / "checkpoint.pt",
                    directory / "training_log.csv",
                    directory / "effective_config.json",
                    "four_horizon_specific_models",
                )
            )
    return rows


def qdf_rows() -> list[dict[str, Any]]:
    artifact_manifest = Path(
        "analysis/iscf_bsca_paper_experiment_consolidation_20260731/"
        "qdf_main_i_l336_20260806/remote_lite/audit/"
        "qdf_main_i_l336_artifact_manifest.csv"
    )
    with artifact_manifest.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    lookup = {
        (record["dataset"], int(record["horizon"]), record["artifact_role"]): record
        for record in records
    }
    rows = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            checkpoint = Path(
                lookup[(dataset, horizon, "checkpoint")]["path"]
            )
            effective = Path(
                lookup[(dataset, horizon, "effective_config")]["path"]
            )
            stdout = Path(lookup[(dataset, horizon, "stdout")]["path"])
            rows.append(
                row(
                    "QDF",
                    dataset,
                    horizon,
                    checkpoint,
                    stdout,
                    effective,
                    "four_horizon_specific_models",
                )
            )
    return rows


def main_ii_rows(system: str) -> list[dict[str, Any]]:
    root = OUTPUT_ROOT / "main_ii_h720_prefix_20260808/training"
    rows = []
    for dataset in DATASETS:
        directory = root / f"{system}__{dataset}"
        checkpoints = list((directory / "checkpoints").glob("**/checkpoint.pth"))
        if len(checkpoints) != 1:
            raise RuntimeError(
                f"expected one {system} {dataset} checkpoint, got {checkpoints}"
            )
        rows.append(
            row(
                f"{system}-H720-prefix",
                dataset,
                720,
                checkpoints[0],
                directory / "run.log",
                None,
                "one_model_full_forecast_then_prefix_views",
            )
        )
    return rows


def freeze(config_path: Path, manifest_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = (
        iscf_rows()
        + timealign_rows()
        + qdf_rows()
        + main_ii_rows("DLinear")
        + main_ii_rows("PatchTST")
    )
    if len(rows) != config["matrix"]["checkpoint_objects"]:
        raise RuntimeError(f"expected 77 checkpoint rows, got {len(rows)}")
    object_ids = {record["object_id"] for record in rows}
    if len(object_ids) != len(rows):
        raise RuntimeError("duplicate efficiency object IDs")
    payload = {
        "manifest_version": 1,
        "protocol_id": config["protocol_id"],
        "protocol_sha256": sha256(config_path),
        "created_at": datetime.now().astimezone().isoformat(),
        "row_count": len(rows),
        "unique_object_ids": len(object_ids),
        "unique_checkpoint_hashes": len(
            {record["checkpoint_sha256"] for record in rows}
        ),
        "test_loader_or_labels_accessed": False,
        "rows": rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "efficiency_manifest_freeze=pass "
        f"rows={len(rows)} unique_hashes={payload['unique_checkpoint_hashes']}"
    )


def verify(config_path: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != sha256(config_path):
        raise RuntimeError("efficiency protocol changed after manifest freeze")
    for record in manifest["rows"]:
        checkpoint = Path(record["checkpoint_path"])
        if sha256(checkpoint) != record["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint mutated: {checkpoint}")
    print(f"efficiency_manifest_verify=pass rows={len(manifest['rows'])}")


def main(args: argparse.Namespace) -> None:
    if args.verify_manifest:
        verify(args.config, args.manifest)
    else:
        freeze(args.config, args.manifest)


if __name__ == "__main__":
    main(parse_args())

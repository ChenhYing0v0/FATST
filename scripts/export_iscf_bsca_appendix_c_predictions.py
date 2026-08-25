#!/usr/bin/env python3
"""Export selected Appendix C trajectories from frozen Main-I/II profiles.

This script is evaluation-only. It reads the frozen selected-profile manifest,
verifies the checkpoint hash and profile identity, evaluates the validation
split without shuffling, and saves two visually faithful channel-level
trajectories per paper-core dataset. It never trains, tunes, or loads an
ablation checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

import train_repo  # noqa: E402  (the official adapter uses local imports)


PAPER_CORE_DATASETS = (
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Weather",
    "ECL",
    "Solar",
)
HORIZONS = (96, 192, 336, 720)
PRED_LEN = 720
# Frozen channel choices for the qualitative appendix. Each channel was
# selected on the validation split by the lowest global visual-fidelity score
# after excluding the lowest-variance 20% of channels for that dataset.
VISUAL_CHANNEL_MAP = {
    "ETTh1": 2,
    "ETTh2": 6,
    "ETTm1": 0,
    "ETTm2": 6,
    "Weather": 18,
    "ECL": 306,
    "Solar": 99,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "analysis/iscf_bsca_main_v1_hpo_20260731/final_hpo_freeze_20260806/"
        / "selected_profile_manifest_final.csv",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/yingch/dataset"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        default=",".join(PAPER_CORE_DATASETS),
        help="Comma-separated paper-core datasets to export.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=-1,
        help="Channel index; -1 uses the frozen validation-audited channel map.",
    )
    parser.add_argument("--min-origin-gap", type=int, default=PRED_LEN)
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=0,
        help="Also save the lowest-scoring validation candidates for audit only.",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        dataset = row["dataset"]
        if dataset not in PAPER_CORE_DATASETS:
            continue
        if dataset in selected:
            raise ValueError(f"duplicate selected profile for {dataset}")
        if row.get("checkpoint_hash_immutable", "").lower() != "true":
            raise ValueError(f"checkpoint hash is not immutable for {dataset}")
        if "ablation" in row.get("training_protocol_id", "").lower():
            raise ValueError(f"ablation protocol selected for {dataset}")
        selected[dataset] = row
    return selected


def validation_origin_base(dataset: str, raw_length: int) -> int:
    """Return raw index of the last history point for validation index zero."""
    if dataset in {"ETTh1", "ETTh2"}:
        train_end = 12 * 30 * 24
    elif dataset in {"ETTm1", "ETTm2"}:
        train_end = 12 * 30 * 24 * 4
    else:
        train_end = int(raw_length * 0.7)
    return train_end - 1


def raw_length(dataset: str, official_args: argparse.Namespace) -> int:
    if dataset in {"ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL"}:
        import pandas as pd

        frame = pd.read_csv(Path(official_args.root_path) / official_args.data_path)
        return len(frame)
    if dataset == "Solar":
        with (Path(official_args.root_path) / official_args.data_path).open(
            encoding="utf-8"
        ) as handle:
            return sum(1 for _ in handle)
    raise ValueError(f"unsupported dataset: {dataset}")


def load_effective_args(
    effective_config: Path,
    dataset: str,
    dataset_root: Path,
    output_dir: Path,
    device: str,
) -> argparse.Namespace:
    with effective_config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    args = SimpleNamespace(**dict(config["adapter"]))
    args.dataset = dataset
    args.dataset_root = dataset_root
    args.output_dir = output_dir
    args.device = device
    args.pred_len = PRED_LEN
    args.mode = "unified"
    args.target_horizons = [PRED_LEN]
    # Older effective_config.json files omit this parser default even though
    # it is part of the frozen ISCF scope contract.
    args.pcsd_scales = list(getattr(args, "pcsd_scales", [1, 48, 144, 360, 720]))
    args.validation_horizons = list(HORIZONS)
    args.evaluation_horizons = list(HORIZONS)
    args.segment_horizons = list(HORIZONS)
    args.final_evaluation_split = "val"
    args.official_test_mode = False
    args.save_predictions = False
    args.allow_archived_research_modes = False
    return args


def verify_profile_and_checkpoint(
    row: dict[str, str],
    dataset: str,
    checkpoint: Path,
    effective_config: Path,
) -> str:
    expected_hash = row["checkpoint_sha256_before_test"]
    actual_hash = sha256_file(checkpoint)
    if actual_hash != expected_hash:
        raise ValueError(
            f"checkpoint hash mismatch for {dataset}: {actual_hash} != {expected_hash}"
        )
    with effective_config.open(encoding="utf-8") as handle:
        adapter = json.load(handle)["adapter"]
    if adapter.get("dataset") != dataset:
        raise ValueError(f"effective config dataset mismatch for {dataset}")
    if adapter.get("hpo_trial_id") != row["trial_id"]:
        raise ValueError(f"HPO trial mismatch for {dataset}")
    if adapter.get("hpo_profile_id") != row["profile_id"]:
        raise ValueError(f"HPO profile mismatch for {dataset}")
    if adapter.get("readout_mode") != "siff-independent-scope-control":
        raise ValueError(f"non-ISCF readout selected for {dataset}")
    if "ablation" in str(adapter.get("protocol_profile", "")).lower():
        raise ValueError(f"ablation profile selected for {dataset}")
    if adapter.get("final_evaluation_split") != "val":
        raise ValueError(f"unexpected training split contract for {dataset}")
    return actual_hash


def select_indices(
    scores: np.ndarray,
    origins: np.ndarray,
    min_origin_gap: int,
) -> tuple[list[int], bool]:
    selected: list[int] = []
    for index in np.argsort(scores, kind="stable"):
        if all(
            abs(int(origins[index]) - int(origins[item])) >= min_origin_gap
            for item in selected
        ):
            selected.append(int(index))
        if len(selected) == 2:
            return selected, True
    for index in np.argsort(scores, kind="stable"):
        if int(index) not in selected:
            selected.append(int(index))
        if len(selected) == 2:
            break
    return selected, False


def _rowwise_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return finite row-wise Pearson correlations for two 2-D arrays."""
    left_centered = left - np.mean(left, axis=1, keepdims=True)
    right_centered = right - np.mean(right, axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(np.square(left_centered), axis=1)
        * np.sum(np.square(right_centered), axis=1)
    )
    correlation = np.zeros_like(numerator, dtype=np.float64)
    valid = denominator > 1e-10
    correlation[valid] = numerator[valid] / denominator[valid]
    return np.clip(correlation, -1.0, 1.0)


def visual_fidelity_scores(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a deterministic shape-aware score for qualitative examples.

    The score is averaged over the four benchmark prefixes. It combines
    train-scale level error, trajectory correlation, first-difference
    correlation and amplitude agreement. Lower values indicate forecasts that
    are both numerically close and visually faithful to the target trajectory.
    """
    component_rows: list[np.ndarray] = []
    for horizon in HORIZONS:
        pred_prefix = prediction[:, :horizon]
        target_prefix = target[:, :horizon]
        rmse = np.sqrt(np.mean(np.square(pred_prefix - target_prefix), axis=1))
        # The model outputs are already standardized with the train-split
        # scaler. Keeping this level error in that common scale prevents a
        # strongly drifting sample from looking artificially good merely
        # because its own target variance is large.
        level_error = rmse

        level_corr = _rowwise_correlation(pred_prefix, target_prefix)
        level_corr_loss = (1.0 - level_corr) / 2.0

        pred_diff = np.diff(pred_prefix, axis=1)
        target_diff = np.diff(target_prefix, axis=1)
        diff_corr = _rowwise_correlation(pred_diff, target_diff)
        diff_corr_loss = (1.0 - diff_corr) / 2.0

        pred_std = np.std(pred_prefix, axis=1)
        target_std = np.std(target_prefix, axis=1)
        amplitude_error = np.abs(pred_std - target_std) / (target_std + 1e-6)
        amplitude_error = np.clip(amplitude_error, 0.0, 2.0)
        component_rows.append(
            np.stack(
                [level_error, level_corr_loss, diff_corr_loss, amplitude_error],
                axis=1,
            )
        )
    components = np.mean(np.stack(component_rows, axis=1), axis=1)
    scores = (
        0.70 * components[:, 0]
        + 0.15 * components[:, 1]
        + 0.10 * components[:, 2]
        + 0.05 * components[:, 3]
    )
    return scores, components


def evaluate_dataset(
    dataset: str,
    row: dict[str, str],
    dataset_root: Path,
    output_dir: Path,
    channel: int,
    min_origin_gap: int,
    candidate_count: int,
    device_name: str,
) -> dict[str, Any]:
    training_dir = Path(row["training_artifact_dir"])
    checkpoint = training_dir / "checkpoint.pt"
    effective_config = training_dir / "effective_config.json"
    if not checkpoint.is_file() or not effective_config.is_file():
        raise FileNotFoundError(f"missing frozen Main-I/II artifacts for {dataset}")

    checkpoint_hash = verify_profile_and_checkpoint(
        row, dataset, checkpoint, effective_config
    )
    args = load_effective_args(
        effective_config, dataset, dataset_root, output_dir, device_name
    )
    preset = train_repo.OFFICIAL_PRESETS[dataset][PRED_LEN]
    official_args = train_repo.build_official_args(args, preset)
    train_repo.set_seed(int(args.seed))
    model = train_repo.TimeAlign.Model(official_args).float().to(official_args.device)
    state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"unexpected checkpoint object for {dataset}")
    model.load_state_dict(state, strict=True)
    model.eval()

    data_set, _unused_loader = train_repo.data_provider(official_args, "val")
    if not 0 <= channel < int(official_args.enc_in):
        raise ValueError(f"channel {channel} is invalid for {dataset}")
    loader = DataLoader(
        data_set,
        batch_size=int(official_args.batch_size),
        shuffle=False,
        num_workers=int(official_args.num_workers),
        drop_last=False,
    )
    scaled_predictions: list[np.ndarray] = []
    scaled_targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            batch_x = batch_x.float().to(official_args.device)
            batch_y = batch_y.float().to(official_args.device)
            outputs, _recon, _alignment = model(
                batch_x,
                batch_y[:, -PRED_LEN:, :],
                is_training=False,
            )
            scaled_predictions.append(
                outputs[:, -PRED_LEN:, channel].detach().cpu().numpy()
            )
            scaled_targets.append(
                batch_y[:, -PRED_LEN:, channel].detach().cpu().numpy()
            )

    prediction_scaled = np.concatenate(scaled_predictions, axis=0)
    target_scaled = np.concatenate(scaled_targets, axis=0)
    if prediction_scaled.shape != target_scaled.shape:
        raise RuntimeError(f"prediction/target shape mismatch for {dataset}")
    horizon_errors = np.stack(
        [
            np.mean(
                np.square(prediction_scaled[:, :h] - target_scaled[:, :h]),
                axis=1,
            )
            for h in HORIZONS
        ],
        axis=1,
    )
    visual_scores, visual_components = visual_fidelity_scores(
        prediction_scaled, target_scaled
    )
    total_raw_length = raw_length(dataset, official_args)
    origin_base = validation_origin_base(dataset, total_raw_length)
    origins = origin_base + np.arange(prediction_scaled.shape[0], dtype=np.int64)
    selected, separated = select_indices(visual_scores, origins, min_origin_gap)
    if len(selected) != 2:
        raise RuntimeError(f"could not select two validation samples for {dataset}")

    scaler = data_set.scaler
    scale = float(scaler.scale_[channel])
    mean = float(scaler.mean_[channel])
    prediction_raw = prediction_scaled[selected] * scale + mean
    target_raw = target_scaled[selected] * scale + mean
    dataset_output = output_dir / dataset
    dataset_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_output / "appendix_c_predictions.npz",
        prediction=prediction_raw.astype(np.float32),
        ground_truth=target_raw.astype(np.float32),
        prediction_scaled=prediction_scaled[selected].astype(np.float32),
        ground_truth_scaled=target_scaled[selected].astype(np.float32),
        horizons=np.asarray(HORIZONS, dtype=np.int64),
        channel=np.asarray(channel, dtype=np.int64),
    )
    if candidate_count < 0:
        raise ValueError("candidate_count must be non-negative")
    if candidate_count:
        candidate_indices = np.argsort(visual_scores, kind="stable")[:candidate_count]
        np.savez_compressed(
            dataset_output / "candidate_pool.npz",
            prediction=(prediction_scaled[candidate_indices] * scale + mean).astype(
                np.float32
            ),
            ground_truth=(
                target_scaled[candidate_indices] * scale + mean
            ).astype(np.float32),
            prediction_scaled=prediction_scaled[candidate_indices].astype(np.float32),
            ground_truth_scaled=target_scaled[candidate_indices].astype(np.float32),
            visual_scores=visual_scores[candidate_indices].astype(np.float32),
            visual_components=visual_components[candidate_indices].astype(np.float32),
            mse_scores=np.mean(horizon_errors, axis=1)[candidate_indices].astype(
                np.float32
            ),
            horizon_errors=horizon_errors[candidate_indices].astype(np.float32),
            validation_window_index=candidate_indices.astype(np.int64),
            raw_forecast_origin=origins[candidate_indices].astype(np.int64),
            horizons=np.asarray(HORIZONS, dtype=np.int64),
            channel=np.asarray(channel, dtype=np.int64),
        )

    selected_rows = []
    for rank, index in enumerate(selected, start=1):
        row_out: dict[str, Any] = {
            "rank": rank,
            "dataset": dataset,
            "split": "val",
            "validation_window_index": int(index),
            "raw_forecast_origin": int(origins[index]),
            "channel": int(channel),
            "selection_score_visual_fidelity": float(visual_scores[index]),
            "visual_level_error": float(visual_components[index, 0]),
            "visual_level_correlation_loss": float(visual_components[index, 1]),
            "visual_difference_correlation_loss": float(visual_components[index, 2]),
            "visual_amplitude_error": float(visual_components[index, 3]),
            "selection_score_scaled_mse": float(np.mean(horizon_errors, axis=1)[index]),
            "selection_separated_by_min_gap": bool(separated),
        }
        for horizon_index, horizon in enumerate(HORIZONS):
            row_out[f"mse_scaled_h{horizon}"] = float(horizon_errors[index, horizon_index])
        selected_rows.append(row_out)
    with (dataset_output / "selection.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)

    metadata = {
        "dataset": dataset,
        "split": "val",
        "prediction_length": PRED_LEN,
        "horizons": list(HORIZONS),
        "channel": channel,
        "selection_rule": (
            f"lowest four-horizon visual-fidelity score on channel {channel}, combining "
            "train-scale level error (0.70), trajectory correlation loss (0.15), "
            "first-difference correlation loss (0.10) and amplitude error (0.05), "
            f"with a minimum raw-origin separation of {min_origin_gap} steps"
        ),
        "profile_id": row["profile_id"],
        "trial_id": row["trial_id"],
        "candidate_version": row["candidate_version"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_role": "frozen Main-I/II selected profile",
        "ablation_checkpoint": False,
        "test_labels_accessed": False,
        "raw_length": total_raw_length,
        "validation_window_count": int(len(data_set)),
        "candidate_count": int(candidate_count),
        "selected": selected_rows,
    }
    with (dataset_output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = sorted(set(datasets) - set(PAPER_CORE_DATASETS))
    if unknown:
        raise ValueError(f"non-paper-core datasets requested: {unknown}")
    manifest = load_manifest(args.manifest)
    missing = sorted(set(datasets) - set(manifest))
    if missing:
        raise ValueError(f"missing frozen selected profiles: {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    export_rows = []
    for dataset in datasets:
        print(f"[Appendix C] exporting {dataset}", flush=True)
        channel = args.channel
        if channel < 0:
            channel = VISUAL_CHANNEL_MAP[dataset]
        export_rows.append(
            evaluate_dataset(
                dataset,
                manifest[dataset],
                args.dataset_root,
                args.output_dir,
                channel,
                args.min_origin_gap,
                args.candidate_count,
                device,
            )
        )
    run_metadata = {
        "purpose": "Appendix C validation-only qualitative trajectories",
        "datasets": datasets,
        "split": "val",
        "prediction_length": PRED_LEN,
        "horizons": list(HORIZONS),
        "device": device,
        "selection_channel": args.channel,
        "selection_channel_map": VISUAL_CHANNEL_MAP,
        "min_origin_gap": args.min_origin_gap,
        "model_source": "selected_profile_manifest_final.csv",
        "ablation_checkpoints_used": False,
        "test_labels_accessed": False,
        "dataset_outputs": export_rows,
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

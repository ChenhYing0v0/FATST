#!/usr/bin/env python3
"""Export aligned split losses and past-only descriptors for D21-EVS."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from data_provider.data_factory import data_provider  # noqa: E402

from evaluate_stage_c_d14a1_checkpoint import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-descriptors", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def history_descriptor(
    history_rows: torch.Tensor,
    descriptor_config: dict[str, object],
) -> tuple[torch.Tensor, list[str]]:
    """Build the frozen inference-visible descriptor from [N, 720] rows."""
    if history_rows.ndim != 2 or history_rows.shape[1] != 720:
        raise ValueError("D21 history rows must have shape [N, 720]")
    epsilon = float(descriptor_config["normalization_epsilon"])
    mean = history_rows.mean(dim=1, keepdim=True)
    std = torch.sqrt(history_rows.var(dim=1, unbiased=False, keepdim=True) + epsilon)
    normalized = (history_rows - mean) / std

    pool_width = int(descriptor_config["mean_pool_width"])
    if 720 % pool_width != 0:
        raise ValueError("mean_pool_width must divide 720")
    pooled = normalized.reshape(
        normalized.shape[0], 720 // pool_width, pool_width
    ).mean(dim=2)
    pooled_names = [f"pool_mean_{index:02d}" for index in range(pooled.shape[1])]

    recent_count = int(descriptor_config["recent_values"])
    recent = normalized[:, -recent_count:]
    recent_names = [f"recent_z_{index:02d}" for index in range(recent_count)]

    coefficient_count = int(
        descriptor_config["fourier_non_dc_coefficients"]
    )
    spectrum = torch.fft.rfft(normalized, dim=1, norm="ortho")[
        :, 1 : coefficient_count + 1
    ]
    spectral = torch.cat([spectrum.real, spectrum.imag], dim=1)
    spectral_names = [
        f"fft_real_{index:02d}" for index in range(1, coefficient_count + 1)
    ] + [
        f"fft_imag_{index:02d}" for index in range(1, coefficient_count + 1)
    ]

    autocorrelations = []
    autocorrelation_names = []
    for value in descriptor_config["autocorrelation_lags"]:
        lag = int(value)
        autocorrelations.append(
            (normalized[:, :-lag] * normalized[:, lag:]).mean(
                dim=1, keepdim=True
            )
        )
        autocorrelation_names.append(f"autocorr_lag_{lag}")
    autocorrelation = torch.cat(autocorrelations, dim=1)

    recent_statistics = []
    recent_statistic_names = []
    for value in descriptor_config["recent_stat_windows"]:
        width = int(value)
        window = normalized[:, -width:]
        recent_statistics.extend(
            [
                window.mean(dim=1, keepdim=True),
                window.std(dim=1, unbiased=False, keepdim=True),
            ]
        )
        recent_statistic_names.extend(
            [f"recent_mean_{width}", f"recent_std_{width}"]
        )
    recent_statistic = torch.cat(recent_statistics, dim=1)

    raw_statistics = torch.cat(
        [
            mean,
            std,
            history_rows[:, -1:],
            normalized[:, -1:],
        ],
        dim=1,
    )
    raw_statistic_names = ["raw_mean", "raw_std", "raw_last", "last_z"]

    features = torch.cat(
        [
            raw_statistics,
            pooled,
            recent,
            spectral,
            autocorrelation,
            recent_statistic,
        ],
        dim=1,
    )
    names = (
        raw_statistic_names
        + pooled_names
        + recent_names
        + spectral_names
        + autocorrelation_names
        + recent_statistic_names
    )
    if features.shape[1] != len(names):
        raise RuntimeError("descriptor feature/name mismatch")
    return features, names


def evaluate(args: argparse.Namespace) -> None:
    design = json.loads(args.design.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, config, official_args = load_model(args.run_dir, device)
    split_data, _ = data_provider(official_args, args.split)
    split_loader = DataLoader(
        split_data,
        batch_size=official_args.batch_size,
        shuffle=False,
        num_workers=official_args.num_workers,
        drop_last=False,
    )

    total_rows = len(split_data) * int(official_args.enc_in)
    requested_probe_rows = min(int(design["probe_rows"]), total_rows)
    probe_indices = np.unique(
        np.linspace(0, total_rows - 1, requested_probe_rows, dtype=np.int64)
    )
    row_bin_mse: list[np.ndarray] = []
    descriptor_parts: list[np.ndarray] = []
    descriptor_names: list[str] | None = None
    row_offset = 0
    all_finite = True

    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in split_loader:
            batch_x = batch_x.float().to(device)
            target = batch_y[:, -720:, :].float().to(device)
            output, _recon, _alignment = model(
                batch_x,
                target,
                is_training=False,
            )
            error_rows = (output - target).permute(0, 2, 1).reshape(-1, 720)
            mse_parts = []
            for entry in design["future_bins"]:
                start, end = int(entry["start"]), int(entry["end"])
                mse_parts.append(
                    error_rows[:, start:end].square().mean(dim=1)
                )
            row_bin_mse.append(torch.stack(mse_parts, dim=1).cpu().numpy())

            batch_rows = batch_x.permute(0, 2, 1).reshape(-1, 720)
            batch_end = row_offset + batch_rows.shape[0]
            if args.save_descriptors:
                selected = probe_indices[
                    (probe_indices >= row_offset) & (probe_indices < batch_end)
                ]
                if selected.size:
                    local_indices = torch.as_tensor(
                        selected - row_offset,
                        dtype=torch.long,
                        device=device,
                    )
                    features, names = history_descriptor(
                        batch_rows.index_select(0, local_indices),
                        design["history_descriptor"],
                    )
                    if descriptor_names is None:
                        descriptor_names = names
                    elif descriptor_names != names:
                        raise RuntimeError("descriptor names changed across batches")
                    descriptor_parts.append(features.cpu().numpy())
            row_offset = batch_end
            all_finite = all_finite and bool(
                torch.isfinite(output).all()
                and torch.isfinite(target).all()
                and torch.isfinite(batch_x).all()
            )

    if row_offset != total_rows:
        raise RuntimeError(f"row count mismatch: {row_offset} != {total_rows}")
    losses = np.concatenate(row_bin_mse).astype(np.float32)
    payload: dict[str, np.ndarray] = {
        "row_bin_mse": losses,
        "probe_indices": probe_indices,
        "bin_names": np.asarray(
            [entry["name"] for entry in design["future_bins"]]
        ),
    }
    if args.save_descriptors:
        if descriptor_names is None or not descriptor_parts:
            raise RuntimeError("descriptor export requested but no rows were saved")
        descriptors = np.concatenate(descriptor_parts).astype(np.float32)
        if descriptors.shape[0] != probe_indices.shape[0]:
            raise RuntimeError("descriptor/probe row mismatch")
        payload["history_features"] = descriptors
        payload["history_feature_names"] = np.asarray(descriptor_names)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    invariant = {
        "diagnostic_id": design["diagnostic_id"],
        "split": args.split,
        "uses_test_split": args.split == "test",
        "test_role": "problem_gate_evaluation" if args.split == "test" else None,
        "checkpoint_hash": _sha256(args.run_dir / "checkpoint.pt"),
        "checkpoint_mutated": False,
        "row_order": "dataset_sequential_batch_channel",
        "total_rows": int(total_rows),
        "probe_rows": int(probe_indices.shape[0]),
        "feature_count": int(
            payload.get("history_features", np.empty((0, 0))).shape[1]
        ),
        "descriptors_saved": args.save_descriptors,
        "all_finite": all_finite and bool(np.isfinite(losses).all()),
    }
    invariant["pass"] = bool(
        invariant["all_finite"]
        and invariant["checkpoint_mutated"] is False
        and (not args.save_descriptors or invariant["feature_count"] > 0)
    )
    invariant_path = args.output.with_name(
        f"{args.output.stem}_invariants.json"
    )
    invariant_path.write_text(
        json.dumps(invariant, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not invariant["pass"]:
        raise RuntimeError(f"D21 evaluator invariant failed: {invariant}")
    print(
        f"d21_evs_export=pass split={args.split} rows={total_rows} "
        f"probe={probe_indices.shape[0]} features={invariant['feature_count']}"
    )


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()

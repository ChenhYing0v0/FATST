#!/usr/bin/env python3
"""Evaluate one D14-A1 checkpoint into row/bin losses and prediction probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from data_provider.data_factory import data_provider  # noqa: E402
from models import TimeAlign  # noqa: E402
from train_repo import model_diagnostics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-rows", type=int, default=1024)
    return parser.parse_args()


def load_model(run_dir: Path, device: torch.device) -> tuple[TimeAlign.Model, dict[str, Any], SimpleNamespace]:
    config = json.loads((run_dir / "effective_config.json").read_text(encoding="utf-8"))
    official = dict(config["official_args"])
    official["device"] = device
    official["use_gpu"] = device.type == "cuda"
    official_args = SimpleNamespace(**official)
    model = TimeAlign.Model(official_args).to(device).float()
    state = torch.load(
        run_dir / "checkpoint.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model, config, official_args


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    design = json.loads(args.design.read_text(encoding="utf-8"))
    model, config, official_args = load_model(args.run_dir, device)
    validation_data, _shuffled_validation_loader = data_provider(
        official_args, "val"
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=official_args.batch_size,
        shuffle=False,
        num_workers=official_args.num_workers,
        drop_last=False,
    )
    bin_mse: list[np.ndarray] = []
    bin_mae: list[np.ndarray] = []
    persistence_bin_mse: list[np.ndarray] = []
    persistence_bin_mae: list[np.ndarray] = []
    probe_predictions: list[np.ndarray] = []
    probe_targets: list[np.ndarray] = []
    probe_count = 0
    all_finite = True
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in validation_loader:
            batch_x = batch_x.float().to(device)
            target = batch_y[:, -720:, :].float().to(device)
            output, _recon, _alignment = model(
                batch_x,
                target,
                is_training=False,
            )
            error = output - target
            persistence = batch_x[:, -1:, :].expand_as(target)
            persistence_error = persistence - target
            rows = error.permute(0, 2, 1).reshape(-1, 720)
            persistence_rows = persistence_error.permute(0, 2, 1).reshape(-1, 720)
            target_rows = target.permute(0, 2, 1).reshape(-1, 720)
            output_rows = output.permute(0, 2, 1).reshape(-1, 720)
            mse_parts = []
            mae_parts = []
            persistence_mse_parts = []
            persistence_mae_parts = []
            for entry in design["future_bins"]:
                start, end = int(entry["start"]), int(entry["end"])
                mse_parts.append(rows[:, start:end].square().mean(dim=1))
                mae_parts.append(rows[:, start:end].abs().mean(dim=1))
                persistence_mse_parts.append(
                    persistence_rows[:, start:end].square().mean(dim=1)
                )
                persistence_mae_parts.append(
                    persistence_rows[:, start:end].abs().mean(dim=1)
                )
            bin_mse.append(torch.stack(mse_parts, dim=1).cpu().numpy())
            bin_mae.append(torch.stack(mae_parts, dim=1).cpu().numpy())
            persistence_bin_mse.append(
                torch.stack(persistence_mse_parts, dim=1).cpu().numpy()
            )
            persistence_bin_mae.append(
                torch.stack(persistence_mae_parts, dim=1).cpu().numpy()
            )
            if probe_count < args.probe_rows:
                count = min(args.probe_rows - probe_count, rows.shape[0])
                probe_predictions.append(output_rows[:count].cpu().numpy())
                probe_targets.append(target_rows[:count].cpu().numpy())
                probe_count += count
            all_finite = all_finite and bool(
                torch.isfinite(output).all() and torch.isfinite(target).all()
            )
    if not bin_mse:
        raise RuntimeError("validation evaluation produced no rows")
    payload = {
        "row_bin_mse": np.concatenate(bin_mse).astype(np.float32),
        "row_bin_mae": np.concatenate(bin_mae).astype(np.float32),
        "persistence_row_bin_mse": np.concatenate(persistence_bin_mse).astype(np.float32),
        "persistence_row_bin_mae": np.concatenate(persistence_bin_mae).astype(np.float32),
        "probe_predictions": np.concatenate(probe_predictions).astype(np.float32),
        "probe_targets": np.concatenate(probe_targets).astype(np.float32),
        "bin_names": np.asarray([entry["name"] for entry in design["future_bins"]]),
    }
    np.savez_compressed(args.run_dir / "validation_diagnostics.npz", **payload)
    diagnostics = model_diagnostics(model)
    adapter = config["adapter"]
    training = config["training_contract"]
    invariant = {
        "diagnostic_id": design["diagnostic_id"],
        "dataset": adapter["dataset"],
        "encoder_mode": adapter["encoder_mode"],
        "readout_mode": adapter["readout_mode"],
        "scale": int(adapter.get("grouped_mlp_scale", 0)),
        "partition": adapter.get("grouped_mlp_partition", "control"),
        "validation_rows": int(payload["row_bin_mse"].shape[0]),
        "probe_rows": int(payload["probe_predictions"].shape[0]),
        "row_order": "dataset_sequential",
        "all_finite": all_finite,
        "uses_test_split": False,
        "final_evaluation_split": adapter["final_evaluation_split"],
        "from_scratch": training["initialization"] == "from_scratch",
        "checkpoint_input": training["checkpoint_input"],
        "frozen_parameter_tensors": diagnostics["frozen_parameter_tensors"],
        "active_forward_parameters": diagnostics["active_forward_parameters"],
        "grouped_mlp_decoder_parameters": diagnostics.get(
            "grouped_mlp_decoder_parameters", 0
        ),
        "grouped_mlp_parameter_relative_gap": diagnostics.get(
            "grouped_mlp_parameter_relative_gap", 0.0
        ),
        "pass": bool(
            all_finite
            and adapter["final_evaluation_split"] == "val"
            and training["initialization"] == "from_scratch"
            and training["checkpoint_input"] is None
            and diagnostics["frozen_parameter_tensors"] == 0
        ),
    }
    (args.run_dir / "trained_invariants.json").write_text(
        json.dumps(invariant, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not invariant["pass"]:
        raise RuntimeError(f"trained invariant failed: {invariant}")
    print(
        f"d14a1_checkpoint=pass dataset={invariant['dataset']} "
        f"rows={invariant['validation_rows']} probe={invariant['probe_rows']}"
    )


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()

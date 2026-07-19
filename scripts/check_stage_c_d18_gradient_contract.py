#!/usr/bin/env python3
"""Verify D18 A6 shape, initialization, projectivity, and prefix-gradient contracts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from models import TimeAlign  # noqa: E402
from train_repo import initialization_contract, model_diagnostics  # noqa: E402


CHANNELS = {
    "ETTh1": 7,
    "ETTh2": 7,
    "ETTm1": 7,
    "ETTm2": 7,
    "Weather": 21,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_d18_soft_projectivity_cost.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_post_ccsf_step24_reset_20260719/"
            "d18_prelaunch/gradient_contract"
        ),
    )
    return parser.parse_args()


def model_config(
    profile: dict[str, Any],
    channels: int,
    target_horizons: list[int],
) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        patch_num=int(profile["patch_num"]),
        d_model=int(profile["d_model"]),
        n_heads=8,
        e_layers=2,
        d_ff=int(profile["d_ff"]),
        dropout=0.1,
        layer_norm=1,
        pos=1,
        enc_in=channels,
        readout_mode="learned-basis-forecast-operator",
        encoder_mode="timealign-token-mlp",
        basis_rank=256,
        target_horizons=target_horizons,
        local_margin=0.5,
        global_margin=0.0,
        loc=1,
        glo=1,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    profiles = json.loads(
        Path(config["profiles"]["path"]).read_text(encoding="utf-8")
    )["dataset_profiles"]
    torch.set_num_threads(1)
    rows = []
    max_prefix_gap = 0.0
    max_tail_gradient = 0.0
    for dataset in config["datasets"]:
        profile = profiles[dataset]
        channels = CHANNELS[dataset]
        initialization_hashes = set()
        parameter_counts = set()
        for horizon in config["matrix"]["own_horizons"]:
            torch.manual_seed(config["seeds"][0])
            model = TimeAlign.Model(
                model_config(profile, channels, [horizon])
            ).float().eval()
            initialization = initialization_contract(model)
            diagnostics = model_diagnostics(model)
            initialization_hashes.add(
                (
                    initialization["encoder_initialization_hash"],
                    initialization["operator_initialization_hash"],
                )
            )
            parameter_counts.add(
                (
                    diagnostics["total_parameters"],
                    diagnostics["active_forward_parameters"],
                )
            )
            generator = torch.Generator().manual_seed(18000 + horizon)
            batch_x = torch.randn(
                2,
                720,
                channels,
                generator=generator,
            )
            target = torch.randn(
                2,
                720,
                channels,
                generator=generator,
            )
            full = model(
                batch_x,
                target,
                is_training=True,
                target_prefix=720,
            )[0]
            prefix = model(
                batch_x,
                target,
                is_training=True,
                target_prefix=horizon,
            )[0]
            prefix_gap = float(
                (prefix - full[:, :horizon]).abs().max().detach()
            )
            max_prefix_gap = max(max_prefix_gap, prefix_gap)
            model.zero_grad(set_to_none=True)
            loss = (prefix - target[:, :horizon]).abs().mean()
            loss.backward()
            basis_gradient = model.learned_temporal_basis.grad
            bias_gradient = model.learned_temporal_bias.grad
            if basis_gradient is None or bias_gradient is None:
                raise RuntimeError("A6 temporal parameters lack gradients")
            prefix_gradient = float(
                basis_gradient[:horizon].abs().sum()
                + bias_gradient[:horizon].abs().sum()
            )
            tail_gradient = 0.0
            if horizon < 720:
                tail_gradient = float(
                    max(
                        basis_gradient[horizon:].abs().max(),
                        bias_gradient[horizon:].abs().max(),
                    )
                )
            max_tail_gradient = max(max_tail_gradient, tail_gradient)
            rows.append(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "full_shape": list(full.shape),
                    "prefix_shape": list(prefix.shape),
                    "prefix_gap": prefix_gap,
                    "prefix_gradient_l1": prefix_gradient,
                    "tail_gradient_max_abs": tail_gradient,
                    "total_parameters": diagnostics["total_parameters"],
                    "active_forward_parameters": diagnostics[
                        "active_forward_parameters"
                    ],
                }
            )
        if len(initialization_hashes) != 1 or len(parameter_counts) != 1:
            raise RuntimeError(
                f"{dataset} horizon arms changed initialization or capacity"
            )
    overall_pass = bool(
        len(rows) == 15
        and max_prefix_gap <= 1e-6
        and max_tail_gradient == 0.0
        and all(row["prefix_gradient_l1"] > 0.0 for row in rows)
        and all(row["prefix_shape"][1] == row["horizon"] for row in rows)
        and all(row["full_shape"][1] == 720 for row in rows)
    )
    report = {
        "candidate_version": config["candidate_version"],
        "checks": len(rows),
        "max_prefix_gap": max_prefix_gap,
        "max_tail_gradient_abs": max_tail_gradient,
        "positive_prefix_gradient_checks": sum(
            row["prefix_gradient_l1"] > 0.0 for row in rows
        ),
        "same_initialization_and_parameter_count_by_dataset": True,
        "overall_pass": overall_pass,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "gradient_contract.csv", rows)
    (args.output_dir / "gradient_contract.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not overall_pass:
        raise RuntimeError(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

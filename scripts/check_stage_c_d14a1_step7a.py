#!/usr/bin/env python3
"""Validate the local tensor and theory contracts of D14-A1."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
for import_root in (str(TIMEALIGN_ROOT), str(REPO_ROOT / "scripts")):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from layers.GroupedMLP import GroupedMLPReadout  # noqa: E402
from models import TimeAlign  # noqa: E402
from train_repo import (  # noqa: E402
    initialization_contract,
    model_diagnostics,
    parse_args as parse_training_args,
)


CHANNELS = {"ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7, "Weather": 21}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=Path("configs/stage_c_d14a1_dual_carrier_grouped_mlp.json"),
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("configs/stage_c_five_dataset_natural_profiles.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage_c_d14a1_dual_carrier_grouped_mlp_20260715/local_gate"),
    )
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_config(
    encoder_mode: str,
    profile: dict[str, Any],
    channels: int,
    scale: int,
    partition: str,
    design: dict[str, Any],
) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode=encoder_mode,
        readout_mode="grouped-mlp",
        e_layers=2,
        patch_num=int(profile["patch_num"]),
        d_model=int(profile["d_model"]),
        d_ff=int(profile["d_ff"]),
        dropout=0.1,
        pos=1,
        layer_norm=1,
        enc_in=channels,
        grouped_mlp_scale=scale,
        grouped_mlp_point_hidden_width=int(design["point_hidden_width"]),
        grouped_mlp_partition=partition,
        grouped_mlp_partition_seed=int(design["partition_seed"]),
    )


def affine_witness(readout: GroupedMLPReadout, seed: int) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn(
        3, 1, readout.readout_dim, generator=generator, dtype=torch.float32
    )
    rank = min(readout.readout_dim, readout.scale)
    with torch.no_grad():
        for parameter in readout.parameters():
            parameter.zero_()
        dimensions = torch.arange(rank)
        readout.input_weight[0, dimensions, 2 * dimensions] = 1.0
        readout.input_weight[0, dimensions, 2 * dimensions + 1] = -1.0
        readout.output_weight[0, 2 * dimensions, dimensions] = 1.0
        readout.output_weight[0, 2 * dimensions + 1, dimensions] = -1.0
    prediction = readout(hidden).permute(0, 2, 1)
    expected = torch.zeros_like(prediction)
    target_indices = readout.group_indices[0, :rank]
    expected[:, 0, target_indices] = hidden[:, 0, :rank]
    return float((prediction - expected).abs().max())


def parameter_and_theory_rows(
    design: dict[str, Any], profiles: dict[str, Any], seed: int
) -> tuple[list[dict[str, Any]], float, float]:
    rows: list[dict[str, Any]] = []
    max_parameter_gap = 0.0
    max_affine_gap = 0.0
    scales = [int(value) for value in design["coupling_scales"]]
    random_scales = [int(value) for value in design["random_partition_scales"]]
    carriers = ["neutral_raw", "a6_natural"]
    for carrier in carriers:
        for dataset, profile in profiles["dataset_profiles"].items():
            readout_dim = (
                720
                if carrier == "neutral_raw"
                else int(profile["patch_num"]) * int(profile["d_model"])
            )
            for partition, active_scales in (
                ("canonical", scales),
                ("random", random_scales),
            ):
                for scale in active_scales:
                    readout = GroupedMLPReadout(
                        readout_dim=readout_dim,
                        scale=scale,
                        point_hidden_width=int(design["point_hidden_width"]),
                        partition=partition,
                        partition_seed=int(design["partition_seed"]),
                    )
                    cover = torch.sort(readout.group_indices.flatten()).values
                    cover_ok = torch.equal(cover, torch.arange(720))
                    labels = readout.target_group_labels()
                    group_counts = torch.bincount(labels, minlength=readout.group_count)
                    topology_ok = bool((group_counts == scale).all())
                    gap = affine_witness(readout, seed + readout_dim + scale)
                    max_parameter_gap = max(
                        max_parameter_gap, readout.parameter_relative_gap
                    )
                    max_affine_gap = max(max_affine_gap, gap)
                    rows.append(
                        {
                            "carrier": carrier,
                            "dataset": dataset,
                            "readout_dim": readout_dim,
                            "partition": partition,
                            "scale": scale,
                            "groups": readout.group_count,
                            "hidden_width": readout.hidden_width,
                            "minimum_affine_width": 2 * min(readout_dim, scale),
                            "decoder_parameters": readout.decoder_parameters,
                            "target_decoder_parameters": readout.target_decoder_parameters,
                            "parameter_relative_gap": readout.parameter_relative_gap,
                            "partition_cover_pass": cover_ok,
                            "sharing_topology_pass": topology_ok,
                            "affine_witness_max_abs": gap,
                        }
                    )
                    del readout
    return rows, max_parameter_gap, max_affine_gap


def model_rows(
    design: dict[str, Any], profiles: dict[str, Any], seed: int
) -> tuple[list[dict[str, Any]], float, bool]:
    rows: list[dict[str, Any]] = []
    max_prefix_gap = 0.0
    all_gradients = True
    scales = [int(value) for value in design["coupling_scales"]]
    representatives = [
        ("neutral_raw", "raw-history-identity", "ETTh2"),
        ("a6_natural", "timealign-token-mlp", "ETTh2"),
        ("a6_natural", "timealign-token-mlp", "ETTh1"),
        ("a6_natural", "timealign-token-mlp", "ETTm2"),
    ]
    for carrier, encoder_mode, dataset in representatives:
        profile = profiles["dataset_profiles"][dataset]
        for scale in scales:
            torch.manual_seed(seed)
            model = TimeAlign.Model(
                build_config(
                    encoder_mode,
                    profile,
                    CHANNELS[dataset],
                    scale,
                    "canonical",
                    design,
                )
            ).float()
            model.eval()
            generator = torch.Generator(device="cpu").manual_seed(
                seed + scale + int(profile["state_width"])
            )
            x = torch.randn(
                2, 720, CHANNELS[dataset], generator=generator
            )
            y = torch.zeros(2, 720, CHANNELS[dataset])
            full, _recon, _align = model(x, y, is_training=False)
            prefix, _recon, _align = model(
                x, y, is_training=False, target_prefix=336
            )
            prefix_gap = float((prefix - full[:, :336]).abs().max())
            max_prefix_gap = max(max_prefix_gap, prefix_gap)
            model.train()
            model.zero_grad(set_to_none=True)
            output, _recon, _align = model(x, y, is_training=False)
            output.square().mean().backward()
            active = [
                parameter
                for name, parameter in model.named_parameters()
                if name.startswith(
                    (
                        "patch_emb_x.",
                        "encoder.",
                        "norm_x.",
                        "grouped_mlp_readout.",
                    )
                )
            ]
            gradient_ok = bool(active) and all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and float(parameter.grad.abs().sum()) > 0.0
                for parameter in active
            )
            all_gradients = all_gradients and gradient_ok
            diagnostics = model_diagnostics(model)
            rows.append(
                {
                    "carrier": carrier,
                    "dataset": dataset,
                    "scale": scale,
                    "output_shape": str(tuple(full.shape)),
                    "prefix_shape": str(tuple(prefix.shape)),
                    "prefix_max_abs": prefix_gap,
                    "gradient_pass": gradient_ok,
                    "frozen_parameter_tensors": diagnostics[
                        "frozen_parameter_tensors"
                    ],
                    "active_forward_parameters": diagnostics[
                        "active_forward_parameters"
                    ],
                    "decoder_parameters": diagnostics[
                        "grouped_mlp_decoder_parameters"
                    ],
                }
            )
            del model, x, y, full, prefix, output
    return rows, max_prefix_gap, all_gradients


def encoder_pairing_rows(
    design: dict[str, Any], profiles: dict[str, Any], seed: int
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    all_paired = True
    for dataset, profile in profiles["dataset_profiles"].items():
        hashes = []
        for scale in [int(value) for value in design["coupling_scales"]]:
            torch.manual_seed(seed)
            model = TimeAlign.Model(
                build_config(
                    "timealign-token-mlp",
                    profile,
                    CHANNELS[dataset],
                    scale,
                    "canonical",
                    design,
                )
            )
            hashes.append(initialization_contract(model)["encoder_initialization_hash"])
            del model
        paired = len(set(hashes)) == 1
        all_paired = all_paired and paired
        rows.append(
            {
                "dataset": dataset,
                "scale_count": len(hashes),
                "encoder_hash_count": len(set(hashes)),
                "encoder_initialization_paired": paired,
                "encoder_initialization_hash": hashes[0],
            }
        )
    return rows, all_paired


def runner_cli_contract() -> bool:
    original_argv = list(sys.argv)
    common = [
        "train_repo.py",
        "--dataset-root",
        "/tmp/d14a1_config_only",
        "--dataset",
        "ETTh2",
        "--mode",
        "unified",
        "--pred-len",
        "720",
        "--target-horizons",
        "720",
        "--run-name",
        "D14A1_CONFIG_ONLY",
        "--output-dir",
        "/tmp/d14a1_config_only/output",
        "--readout-mode",
        "grouped-mlp",
        "--protocol-class",
        "method_screening",
        "--protocol-profile",
        "stage_c_d14a1_config_only",
        "--profile-hash",
        "config_only",
        "--checkpoint-policy",
        "best-val",
        "--enable-early-stopping",
        "--final-evaluation-split",
        "val",
    ]
    cases = [
        [*common, "--encoder-mode", "raw-history-identity"],
        [
            *common,
            "--encoder-mode",
            "timealign-token-mlp",
            "--legacy-patch-num",
            "12",
            "--legacy-d-model",
            "64",
            "--legacy-d-ff",
            "128",
        ],
    ]
    try:
        for case in cases:
            sys.argv = case
            parse_training_args()
    finally:
        sys.argv = original_argv
    return True


def main() -> None:
    args = parse_args()
    design = json.loads(args.design.read_text(encoding="utf-8"))
    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))
    parameter_rows, max_parameter_gap, max_affine_gap = parameter_and_theory_rows(
        design, profiles, args.seed
    )
    forward_rows, max_prefix_gap, gradient_pass = model_rows(
        design, profiles, args.seed
    )
    pairing_rows, pairing_pass = encoder_pairing_rows(
        design, profiles, args.seed
    )
    cli_contract_pass = runner_cli_contract()
    gates = design["local_gates"]
    gate = {
        "diagnostic_id": design["diagnostic_id"],
        "parameter_cases": len(parameter_rows),
        "forward_cases": len(forward_rows),
        "encoder_pairing_cases": len(pairing_rows),
        "max_parameter_relative_gap": max_parameter_gap,
        "max_affine_witness_gap": max_affine_gap,
        "max_prefix_equivalence_gap": max_prefix_gap,
        "parameter_gate_pass": max_parameter_gap
        <= float(gates["max_parameter_relative_gap"]),
        "affine_gate_pass": max_affine_gap
        <= float(gates["max_affine_witness_gap"]),
        "prefix_gate_pass": max_prefix_gap
        <= float(gates["max_prefix_equivalence_gap"]),
        "partition_gate_pass": all(
            row["partition_cover_pass"] and row["sharing_topology_pass"]
            for row in parameter_rows
        ),
        "gradient_gate_pass": gradient_pass,
        "encoder_pairing_pass": pairing_pass,
        "runner_cli_contract_pass": cli_contract_pass,
        "uses_test_split": False,
        "forecast_training_performed": False,
    }
    gate["pass"] = all(
        gate[name]
        for name in (
            "parameter_gate_pass",
            "affine_gate_pass",
            "prefix_gate_pass",
            "partition_gate_pass",
            "gradient_gate_pass",
            "encoder_pairing_pass",
            "runner_cli_contract_pass",
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "parameter_and_theory_cases.csv", parameter_rows)
    write_csv(args.output_dir / "forward_gradient_cases.csv", forward_rows)
    write_csv(args.output_dir / "encoder_pairing_cases.csv", pairing_rows)
    (args.output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# StageC D14-A1 Step7A Local Gate",
        "",
        f"- `decision`: `{'pass_neutral_remote_only' if gate['pass'] else 'fail_return_step4'}`",
        f"- parameter cases: {gate['parameter_cases']}",
        f"- forward/gradient cases: {gate['forward_cases']}",
        f"- max parameter gap: {max_parameter_gap:.8f}",
        f"- max affine witness gap: {max_affine_gap:.8e}",
        f"- max prefix gap: {max_prefix_gap:.8e}",
        f"- encoder pairing: {pairing_pass}",
        f"- runner CLI contract: {cli_contract_pass}",
        "- test=false；forecast training=false；A6 remote仍held。",
        "",
    ]
    (args.output_dir / "local_gate_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    if not gate["pass"]:
        raise RuntimeError(f"D14-A1 local gate failed: {gate}")
    print(
        "stage_c_d14a1_step7a=pass "
        f"parameter_cases={len(parameter_rows)} forward_cases={len(forward_rows)}"
    )


if __name__ == "__main__":
    main()

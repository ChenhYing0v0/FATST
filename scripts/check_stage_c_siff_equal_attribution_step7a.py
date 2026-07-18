#!/usr/bin/env python3
"""Run the local production gate for SIFF_EQUAL attribution Step 7A."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from layers.PCC import (  # noqa: E402
    prefix_measure,
    projective_coupling_credit_loss,
)
from layers.SIFF import SIFFCouplingFieldReadout, siff_parameter_count  # noqa: E402
from models import TimeAlign  # noqa: E402
import train_repo as training_adapter  # noqa: E402
from train_repo import initialization_contract  # noqa: E402


CHANNELS = {"ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7, "Weather": 21}
SIFF_EXPECTED = {
    "siff-coupling-field": (2, "ordered"),
    "siff-constant-control": (2, "constant"),
    "siff-permuted-scale-control": (2, "permuted"),
    "siff-q1-wide-control": (1, "ordered"),
    "siff-independent-scope-control": (5, "independent"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_siff_equal_attribution_v2.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_siff_equal_attribution_step7a_20260718"
        ),
    )
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def manifest_rows(
    config: dict[str, Any],
    profiles: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for dataset in config["datasets"]:
        profile = profiles[dataset]
        for arm in config["arms"]:
            rule = arm["rank_rule"]
            rank = (
                256
                if rule == "fixed_256"
                else config["matched_ranks"][dataset][rule]
            )
            rows.append(
                {
                    "job_index": len(rows) + 1,
                    "dataset": dataset,
                    "arm": arm["id"],
                    "readout_mode": arm["readout_mode"],
                    "objective_mode": arm["objective_mode"],
                    "mode_rank": rank,
                    "profile": profile["profile"],
                    "patch_num": profile["patch_num"],
                    "d_model": profile["d_model"],
                    "d_ff": profile["d_ff"],
                    "seed": config["seeds"][0],
                    "checkpoint_score": config["training"][
                        "validation_checkpoint_score"
                    ],
                    "formal_evaluation_split": "test",
                }
            )
    return rows


def training_argv(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    training = config["training"]
    return [
        "train_repo.py",
        "--dataset-root",
        "/home/yingch/dataset",
        "--dataset",
        row["dataset"],
        "--mode",
        "unified",
        "--seq-len",
        "720",
        "--pred-len",
        "720",
        "--target-horizons",
        "720",
        "--validation-horizons",
        "96,192,336,720",
        "--evaluation-horizons",
        "96,192,336,720",
        "--segment-horizons",
        "96,192,336,720",
        "--evaluation-prefix-mode",
        "full-crop",
        "--e-layers",
        "2",
        "--batch-size",
        str(training["batch_size"]),
        "--epochs",
        str(training["epochs"]),
        "--patience",
        str(training["patience"]),
        "--enable-early-stopping",
        "--seed",
        str(row["seed"]),
        "--run-name",
        f"SIFF_EQ_ATTR_{row['arm']}",
        "--output-dir",
        f"/tmp/{row['arm']}_{row['dataset']}",
        "--device",
        "cuda",
        "--checkpoint-policy",
        "best-val",
        "--protocol-class",
        "method_screening",
        "--protocol-profile",
        "stage_c_siff_equal_attribution_v2",
        "--profile-hash",
        config["profiles"]["sha256"],
        "--legacy-patch-num",
        str(row["patch_num"]),
        "--legacy-d-model",
        str(row["d_model"]),
        "--legacy-d-ff",
        str(row["d_ff"]),
        "--legacy-dropout",
        "0.1",
        "--legacy-layer-norm",
        "1",
        "--learning-rate",
        str(training["learning_rate"]),
        "--readout-mode",
        row["readout_mode"],
        "--basis-rank",
        "256",
        "--pcsd-coordinate-dim",
        "4",
        "--pcsd-mode-rank",
        str(row["mode_rank"]),
        "--pcsd-policy-history-dim",
        "32",
        "--pcsd-policy-hidden-dim",
        "64",
        "--pcsd-policy-mode",
        "direct",
        "--pcsd-fixed-scale",
        "720",
        "--pcsd-partition",
        "canonical",
        "--pcsd-partition-seed",
        "15101",
        "--pcc-objective-mode",
        row["objective_mode"],
        "--pred-loss-mode",
        "full",
        "--final-evaluation-split",
        "val",
        "--no-save-predictions",
    ]


def cli_audit(rows: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    original = sys.argv
    try:
        for row in rows:
            sys.argv = training_argv(row, config)
            parsed = training_adapter.parse_args()
            if not (
                parsed.dataset == row["dataset"]
                and parsed.readout_mode == row["readout_mode"]
                and parsed.pcc_objective_mode == row["objective_mode"]
                and parsed.pcsd_mode_rank == row["mode_rank"]
                and parsed.validation_horizons == [96, 192, 336, 720]
                and parsed.final_evaluation_split == "val"
                and parsed.evaluation_prefix_mode == "full-crop"
            ):
                return False
    finally:
        sys.argv = original
    return True


def model_config(
    row: dict[str, Any],
    *,
    small: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode=row["readout_mode"],
        e_layers=1 if small else 2,
        patch_num=2 if small else int(row["patch_num"]),
        d_model=4 if small else int(row["d_model"]),
        d_ff=8 if small else int(row["d_ff"]),
        dropout=0.0 if small else 0.1,
        pos=1,
        layer_norm=1,
        enc_in=2 if small else CHANNELS[row["dataset"]],
        basis_rank=8 if small else 256,
        pcsd_coordinate_dim=4,
        pcsd_mode_rank=8 if small else int(row["mode_rank"]),
        pcsd_policy_history_dim=4 if small else 32,
        pcsd_policy_hidden_dim=8 if small else 64,
        pcsd_policy_mode="direct",
        pcsd_fixed_scale=720,
        pcsd_partition="canonical",
        pcsd_partition_seed=15101,
        pcsd_group_chunk_size=64,
        pcsd_target_chunk_size=128,
    )


def construction_audit(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    representatives = {
        (row["dataset"], row["readout_mode"], row["mode_rank"]): row
        for row in rows
    }
    results = []
    for row in representatives.values():
        torch.manual_seed(int(row["seed"]))
        model = TimeAlign.Model(model_config(row)).float().eval()
        contract = initialization_contract(model)
        channels = CHANNELS[row["dataset"]]
        x = torch.randn(1, 720, channels)
        y = torch.zeros(1, 720, channels)
        with torch.no_grad():
            full = model(x, y, is_training=False, target_prefix=720)[0]
            prefix = model(x, y, is_training=False, target_prefix=96)[0]
        prefix_gap = float((prefix - full[:, :96]).abs().max())
        readout_pass = True
        if row["readout_mode"] in SIFF_EXPECTED:
            readout = model.pcsd_readout
            readout_pass = (
                (
                    int(readout.scale_components),
                    readout.scale_basis_mode,
                )
                == SIFF_EXPECTED[row["readout_mode"]]
            )
        results.append(
            {
                "dataset": row["dataset"],
                "readout_mode": row["readout_mode"],
                "mode_rank": row["mode_rank"],
                "encoder_hash": contract["encoder_initialization_hash"],
                "output_shape": str(tuple(full.shape)),
                "prefix_gap": prefix_gap,
                "readout_contract_pass": readout_pass,
                "pass": bool(
                    tuple(full.shape) == (1, 720, channels)
                    and prefix_gap == 0.0
                    and readout_pass
                ),
            }
        )
        del model
    return results


def parameter_audit(
    config: dict[str, Any],
    profiles: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for dataset, profile in profiles.items():
        readout_dim = int(profile["patch_num"]) * int(profile["d_model"])
        target = siff_parameter_count(
            readout_dim,
            mode_rank=256,
            scale_components=2,
        )
        for rule, components in (
            ("q1_dataset_matched", 1),
            ("independent_dataset_matched", 5),
        ):
            rank = int(config["matched_ranks"][dataset][rule])
            actual = siff_parameter_count(
                readout_dim,
                mode_rank=rank,
                scale_components=components,
            )
            relative_gap = abs(actual - target) / float(target)
            rows.append(
                {
                    "dataset": dataset,
                    "control": rule,
                    "target_parameters": target,
                    "actual_parameters": actual,
                    "relative_gap": relative_gap,
                    "pass": relative_gap <= 0.005,
                }
            )
    return rows


def gradient_audit(
    manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    representatives = {row["arm"]: row for row in manifest}
    rows = []
    for arm, row in representatives.items():
        torch.manual_seed(2021)
        model = TimeAlign.Model(model_config(row, small=True)).float().train()
        x = torch.randn(1, 720, 2)
        target = torch.randn(1, 720, 2)
        if hasattr(model, "pcsd_readout"):
            output, _recon, _align, details = model(
                x,
                target,
                is_training=True,
                target_prefix=720,
                return_pcsd_training_details=True,
            )
            arms = details["arm_forecasts"].permute(0, 1, 3, 2)
            result = projective_coupling_credit_loss(
                output,
                arms,
                details["policy"],
                target,
                mode=row["objective_mode"],
                progress=1.0,
            )
            loss = result.total_loss
            readout_prefix = "pcsd_readout."
        else:
            output = model(
                x,
                target,
                is_training=True,
                target_prefix=720,
            )[0]
            if row["objective_mode"] == "measure_only":
                measure = prefix_measure(
                    720,
                    device=output.device,
                    dtype=output.dtype,
                )
                loss = (
                    (output - target).abs() * measure.view(1, -1, 1)
                ).sum(dim=1).mean()
            else:
                loss = (output - target).abs().mean()
            readout_prefix = "learned_"
        loss.backward()
        encoder_norm = sum(
            float(parameter.grad.norm())
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and name.startswith(("patch_emb_x.", "encoder."))
        )
        readout_norm = sum(
            float(parameter.grad.norm())
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and name.startswith(readout_prefix)
        )
        rows.append(
            {
                "arm": arm,
                "loss": float(loss.detach()),
                "encoder_gradient_norm": encoder_norm,
                "readout_gradient_norm": readout_norm,
                "pass": bool(
                    math.isfinite(float(loss))
                    and encoder_norm > 0.0
                    and readout_norm > 0.0
                ),
            }
        )
    return rows


def component_audit() -> list[dict[str, Any]]:
    torch.manual_seed(2021)
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    ordered = SIFFCouplingFieldReadout(
        readout_dim=8,
        mode_rank=8,
        scale_components=2,
        scale_basis_mode="ordered",
        policy_history_dim=4,
        policy_hidden_dim=8,
    )
    constant = SIFFCouplingFieldReadout(
        readout_dim=8,
        mode_rank=8,
        scale_components=2,
        scale_basis_mode="constant",
        policy_history_dim=4,
        policy_hidden_dim=8,
    )
    with torch.no_grad():
        for name, parameter in ordered.named_parameters():
            dict(constant.named_parameters())[name].copy_(parameter)
    full, ablated = ordered.component_ablation_forecasts(hidden)
    direct = (
        ordered.arm_forecasts(hidden)
        * ordered.policy_weights(hidden).permute(0, 1, 3, 2)
    ).sum(dim=2)
    consistency_gap = float((full - direct).abs().max())
    contribution = full.unsqueeze(2) - ablated
    nonconstant_rms = float(contribution[:, :, 1].square().mean().sqrt())
    ordered_constant_gap = float(
        (
            ordered(hidden, 720) - constant(hidden, 720)
        ).square().mean().sqrt()
    )
    contribution.square().mean().backward()
    component_gradient = float(ordered.mode_weight.grad[1].norm())
    return [
        {
            "case": "component_ablation_shape",
            "value": str(tuple(ablated.shape)),
            "threshold": "(2,3,2,720)",
            "pass": tuple(ablated.shape) == (2, 3, 2, 720),
        },
        {
            "case": "full_path_consistency",
            "value": consistency_gap,
            "threshold": 0.0,
            "pass": consistency_gap == 0.0,
        },
        {
            "case": "nonconstant_component_active",
            "value": nonconstant_rms,
            "threshold": ">0",
            "pass": nonconstant_rms > 0.0,
        },
        {
            "case": "ordered_constant_contrast",
            "value": ordered_constant_gap,
            "threshold": ">0",
            "pass": ordered_constant_gap > 0.0,
        },
        {
            "case": "nonconstant_component_gradient",
            "value": component_gradient,
            "threshold": ">0",
            "pass": component_gradient > 0.0,
        },
    ]


def smoke_script(script: str, config: Path) -> bool:
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--config",
            str(config),
            "--synthetic-smoke",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def evaluator_smoke() -> bool:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_stage_c_pcsd_cf_checkpoint.py",
            "--synthetic-smoke",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def remote_runner_guard(config: Path) -> bool:
    environment = dict(os.environ)
    environment["CONFIG"] = str(config)
    result = subprocess.run(
        ["bash", "scripts/remote/run_stage_c_siff_equal_attribution_v2.sh"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(
        result.returncode == 3
        and "remote/test launch is not authorized" in result.stderr
    )


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    profile_path = Path(config["profiles"]["path"])
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))[
        "dataset_profiles"
    ]
    manifest = manifest_rows(config, profiles)
    construction = construction_audit(manifest)
    parameters = parameter_audit(config, profiles)
    gradients = gradient_audit(manifest)
    components = component_audit()

    encoder_hashes: dict[str, set[str]] = {}
    for row in construction:
        encoder_hashes.setdefault(row["dataset"], set()).add(
            row["encoder_hash"]
        )
    categories = {
        "profile_hash": file_hash(profile_path) == config["profiles"]["sha256"],
        "matrix_size": len(manifest)
        == config["matrix"]["phase_a_expected_runs"]
        == 50,
        "arm_count": len({row["arm"] for row in manifest}) == 10,
        "cli_contracts": cli_audit(manifest, config),
        "model_construction": all(row["pass"] for row in construction),
        "paired_encoder_initialization": all(
            len(values) == 1 for values in encoder_hashes.values()
        ),
        "parameter_matching": all(row["pass"] for row in parameters),
        "gradient_paths": all(row["pass"] for row in gradients),
        "scale_component_artifact": all(row["pass"] for row in components),
        "checkpoint_evaluator_smoke": evaluator_smoke(),
        "four_layer_analyzer_smoke": smoke_script(
            "scripts/analyze_stage_c_siff_equal_attribution_v2.py",
            args.config,
        ),
        "remote_runner_authorization_guard": remote_runner_guard(args.config),
        "authorization_boundary": bool(
            config["authorization"]["step7a_local_implementation_authorized"]
            and not config["authorization"]["remote_training_authorized"]
            and not config["authorization"]["formal_test_access_authorized"]
        ),
    }
    overall_pass = all(categories.values())
    report = {
        "candidate_version": config["candidate_version"],
        "current_step": "Step7A local implementation gate",
        "categories": categories,
        "categories_passed": sum(categories.values()),
        "categories_total": len(categories),
        "jobs": len(manifest),
        "construction_cases": len(construction),
        "gradient_cases": len(gradients),
        "component_cases": len(components),
        "overall_pass": overall_pass,
        "next_step": (
            "Step7B prelaunch audit"
            if overall_pass
            else "return Step6 implementation repair"
        ),
        "remote_training_authorized": False,
        "formal_test_access_authorized": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "jobs_seed2021.csv", manifest)
    write_csv(args.output_dir / "model_construction.csv", construction)
    write_csv(args.output_dir / "parameter_matching.csv", parameters)
    write_csv(args.output_dir / "gradient_paths.csv", gradients)
    write_csv(args.output_dir / "scale_component_cases.csv", components)
    (args.output_dir / "local_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not overall_pass:
        raise RuntimeError(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

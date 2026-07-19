#!/usr/bin/env python3
"""Run the local implementation gate for SC-D19-IFC Step 7A."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from layers.ImplicitForecast import (  # noqa: E402
    DirectNonlinearMatchedReadout,
    ImplicitFrequencyReadout,
)
from models import TimeAlign  # noqa: E402
import train_repo as training_adapter  # noqa: E402
from train_repo import initialization_contract, model_diagnostics  # noqa: E402


NEW_ARMS = {
    "IF_MEASURE": "implicit-frequency-readout",
    "IF_NOSKIP_MEASURE": "implicit-frequency-noskip-control",
    "DIRECT_NONLINEAR_MATCHED_MEASURE": "implicit-direct-nonlinear-matched",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_d19_if_control_step7a.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_post_ccsf_step24_reset_20260719/"
            "d19_step7a_local"
        ),
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def record(
    rows: list[dict[str, Any]],
    category: str,
    case: str,
    passed: bool,
    value: Any,
    threshold: Any,
) -> None:
    rows.append(
        {
            "category": category,
            "case": case,
            "value": value,
            "threshold": threshold,
            "pass": bool(passed),
        }
    )


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def governance_audit(
    config: dict[str, Any],
    step6: dict[str, Any],
    profiles_path: Path,
) -> list[dict[str, Any]]:
    rows = []
    record(
        rows,
        "governance",
        "candidate_supersession_identity",
        step6["candidate_version"] == "SC-D19-IFC-control-v1.1"
        and step6["supersedes"] == "SC-D19-IFC-control-v1"
        and config["candidate_version"] == step6["candidate_version"],
        step6["candidate_version"],
        "SC-D19-IFC-control-v1.1",
    )
    implicit = step6["implicit_forecaster"]
    record(
        rows,
        "governance",
        "same_history_spectrum_contract",
        step6["data"]["history_length"] == 720
        and implicit["history_spectrum_bins"] == 361
        and implicit["spectrum_bins"] == 361,
        f"{step6['data']['history_length']}/{implicit['history_spectrum_bins']}",
        "720/361",
    )
    actual_arms = {
        arm["id"]: arm["readout_mode"]
        for arm in step6["arms"]
        if arm["training_new"]
    }
    record(
        rows,
        "governance",
        "three_new_arm_mapping",
        actual_arms == NEW_ARMS,
        json.dumps(actual_arms, sort_keys=True),
        json.dumps(NEW_ARMS, sort_keys=True),
    )
    profile_hash = hashlib.sha256(profiles_path.read_bytes()).hexdigest()
    record(
        rows,
        "governance",
        "frozen_profile_hash",
        profile_hash == config["profile_hash"],
        profile_hash,
        config["profile_hash"],
    )
    authorization = config["authorization"]
    step6_authorization = step6["authorization"]
    record(
        rows,
        "governance",
        "authorization_boundary",
        authorization["local_implementation"] is True
        and authorization["remote"] is False
        and authorization["official_test"] is False
        and authorization["paper_method"] is False
        and step6_authorization["remote"] is False
        and step6_authorization["official_test"] is False,
        json.dumps(authorization, sort_keys=True),
        "local-only",
    )
    return rows


def manifest_rows(
    config: dict[str, Any],
    profiles: dict[str, Any],
    step6: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for dataset in config["datasets"]:
        profile = profiles[dataset]
        contract = step6["profile_contracts"][profile["profile"]]
        for arm, readout_mode in NEW_ARMS.items():
            rows.append(
                {
                    "job_index": len(rows) + 1,
                    "dataset": dataset,
                    "arm": arm,
                    "readout_mode": readout_mode,
                    "profile": profile["profile"],
                    "patch_num": profile["patch_num"],
                    "d_model": profile["d_model"],
                    "d_ff": profile["d_ff"],
                    "readout_dim": profile["state_width"],
                    "if_direct_hidden_width": contract[
                        "direct_hidden_width"
                    ],
                    "seed": config["seed"],
                    "objective_mode": config["training"]["objective_mode"],
                }
            )
    return rows


def training_argv(
    row: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
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
        "--learning-rate",
        str(training["learning_rate"]),
        "--seed",
        str(row["seed"]),
        "--run-name",
        f"D19_{row['arm']}",
        "--output-dir",
        f"/tmp/d19_{row['dataset']}_{row['arm']}",
        "--device",
        "cuda",
        "--checkpoint-policy",
        training["checkpoint_policy"],
        "--protocol-class",
        "method_screening",
        "--protocol-profile",
        config["protocol_profile"],
        "--profile-hash",
        config["profile_hash"],
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
        "--readout-mode",
        row["readout_mode"],
        "--if-hidden-width",
        "2048",
        "--if-direct-hidden-width",
        str(row["if_direct_hidden_width"]),
        "--if-head-dropout",
        "0.1",
        "--if-fourier-norm",
        "ortho",
        "--pcc-objective-mode",
        "measure_only",
        "--pred-loss-mode",
        "full",
        "--final-evaluation-split",
        "val",
        "--no-save-predictions",
    ]


def cli_audit(
    manifest: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    original = sys.argv
    try:
        for row in manifest:
            sys.argv = training_argv(row, config)
            parsed = training_adapter.parse_args()
            passed = (
                parsed.dataset == row["dataset"]
                and parsed.readout_mode == row["readout_mode"]
                and parsed.seq_len == 720
                and parsed.pred_len == 720
                and parsed.validation_horizons == [96, 192, 336, 720]
                and parsed.pcc_objective_mode == "measure_only"
                and parsed.if_hidden_width == 2048
                and parsed.if_direct_hidden_width
                == row["if_direct_hidden_width"]
                and parsed.if_head_dropout == 0.1
                and parsed.if_fourier_norm == "ortho"
                and parsed.final_evaluation_split == "val"
            )
            record(
                rows,
                "cli",
                f"{row['dataset']}_{row['arm']}",
                passed,
                row["readout_mode"],
                "frozen D19 CLI",
            )
    finally:
        sys.argv = original
    return rows


def parameter_audit(
    step6: dict[str, Any],
    gates: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for profile, contract in step6["profile_contracts"].items():
        implicit = ImplicitFrequencyReadout(
            readout_dim=int(contract["readout_dim"]),
            hidden_width=2048,
            history_length=720,
            series_length=720,
            dropout=0.1,
        )
        direct = DirectNonlinearMatchedReadout(
            readout_dim=int(contract["readout_dim"]),
            hidden_width=int(contract["direct_hidden_width"]),
            history_length=720,
            series_length=720,
            dropout=0.1,
        )
        implicit_count = parameter_count(implicit)
        direct_count = parameter_count(direct)
        gap = 100.0 * abs(implicit_count - direct_count) / implicit_count
        record(
            rows,
            "parameters",
            f"{profile}_implicit_formula",
            implicit_count == int(contract["if_decoder_parameters"]),
            implicit_count,
            contract["if_decoder_parameters"],
        )
        record(
            rows,
            "parameters",
            f"{profile}_direct_formula",
            direct_count == int(contract["direct_decoder_parameters"]),
            direct_count,
            contract["direct_decoder_parameters"],
        )
        record(
            rows,
            "parameters",
            f"{profile}_relative_gap",
            gap <= gates["parameter_gap_percent_max"],
            gap,
            gates["parameter_gap_percent_max"],
        )
        del implicit, direct
    return rows


def source_invariant_audit() -> list[dict[str, Any]]:
    rows = []
    values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    observed = torch.nn.functional.leaky_relu(
        values,
        negative_slope=0.5,
    ).abs()
    expected = torch.tensor([1.0, 0.5, 0.0, 1.0, 2.0])
    alu_gap = float((observed - expected).abs().max())
    record(
        rows,
        "source_invariant",
        "alu_w0_5_reference",
        alu_gap == 0.0,
        alu_gap,
        0.0,
    )

    phase_sine = torch.tensor(
        [0.0, 1.0, 0.0, -1.0],
        dtype=torch.float64,
    )
    phase_cosine = torch.tensor(
        [1.0, 0.0, -1.0, 0.0],
        dtype=torch.float64,
    )
    observed_phase = torch.atan2(phase_sine, phase_cosine)
    expected_phase = torch.tensor(
        [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0],
        dtype=torch.float64,
    )
    phase_gap = float((observed_phase - expected_phase).abs().max())
    record(
        rows,
        "source_invariant",
        "atan2_quadrant_reference",
        phase_gap <= 1e-12,
        phase_gap,
        1e-12,
    )

    torch.manual_seed(2021)
    history = torch.randn(2, 3, 720, dtype=torch.float64)
    spectrum = torch.fft.rfft(history, dim=-1, norm="ortho")
    reconstructed = torch.fft.irfft(
        spectrum,
        n=720,
        dim=-1,
        norm="ortho",
    )
    reconstruction_gap = float((reconstructed - history).abs().max())
    record(
        rows,
        "source_invariant",
        "orthonormal_fft_roundtrip",
        reconstruction_gap <= 1e-12,
        reconstruction_gap,
        1e-12,
    )
    return rows


def small_model_config(readout_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode=readout_mode,
        e_layers=1,
        patch_num=2,
        d_model=4,
        d_ff=8,
        dropout=0.0,
        pos=1,
        layer_norm=1,
        enc_in=2,
        basis_rank=8,
        target_horizons=[720],
        if_hidden_width=16,
        if_direct_hidden_width=17,
        if_head_dropout=0.0,
        if_fourier_norm="ortho",
    )


def initialization_audit() -> list[dict[str, Any]]:
    rows = []
    contracts = {}
    modes = [
        "learned-basis-forecast-operator",
        "implicit-frequency-readout",
        "implicit-frequency-noskip-control",
        "implicit-direct-nonlinear-matched",
    ]
    for mode in modes:
        torch.manual_seed(2021)
        model = TimeAlign.Model(small_model_config(mode))
        contracts[mode] = initialization_contract(model)
    encoder_hashes = {
        contract["encoder_initialization_hash"]
        for contract in contracts.values()
    }
    record(
        rows,
        "initialization",
        "paired_encoder_hash_all_four_arms",
        len(encoder_hashes) == 1,
        len(encoder_hashes),
        1,
    )
    if_hash = contracts["implicit-frequency-readout"][
        "implicit_frequency_initialization_hash"
    ]
    no_skip_hash = contracts["implicit-frequency-noskip-control"][
        "implicit_frequency_initialization_hash"
    ]
    record(
        rows,
        "initialization",
        "if_no_skip_decoder_hash",
        if_hash == no_skip_hash,
        if_hash == no_skip_hash,
        True,
    )
    return rows


def shape_projectivity_audit(
    gates: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = []
    torch.manual_seed(2021)
    hidden = torch.randn(2, 3, 12)
    history = torch.randn(2, 720, 3)
    torch.manual_seed(2022)
    implicit = ImplicitFrequencyReadout(
        12,
        hidden_width=16,
        dropout=0.0,
        use_input_spectrum=True,
    ).eval()
    torch.manual_seed(2022)
    no_skip = ImplicitFrequencyReadout(
        12,
        hidden_width=16,
        dropout=0.0,
        use_input_spectrum=False,
    ).eval()
    torch.manual_seed(2022)
    direct = DirectNonlinearMatchedReadout(
        12,
        hidden_width=17,
        dropout=0.0,
    ).eval()
    modules = {"IF": implicit, "IF_NOSKIP": no_skip, "DIRECT": direct}
    with torch.no_grad():
        full_outputs = {
            name: module(hidden, history, 720)
            for name, module in modules.items()
        }
        horizons = [
            1,
            2,
            47,
            48,
            95,
            96,
            143,
            144,
            191,
            192,
            287,
            288,
            335,
            336,
            511,
            512,
            600,
            719,
            720,
            360,
        ]
        max_gap = 0.0
        for name, module in modules.items():
            for horizon in horizons:
                prefix = module(hidden, history, horizon)
                gap = float(
                    (prefix - full_outputs[name][:, :horizon]).abs().max()
                )
                max_gap = max(max_gap, gap)
                record(
                    rows,
                    "shape_projectivity",
                    f"{name}_h{horizon}",
                    tuple(prefix.shape) == (2, horizon, 3)
                    and gap <= gates["prefix_max_abs_max"],
                    gap,
                    gates["prefix_max_abs_max"],
                )
        scale = full_outputs["IF"].square().mean().sqrt().clamp_min(1e-12)
        skip_nrmse = float(
            (
                full_outputs["IF"] - full_outputs["IF_NOSKIP"]
            ).square().mean().sqrt()
            / scale
        )
        direct_nrmse = float(
            (full_outputs["IF"] - full_outputs["DIRECT"])
            .square()
            .mean()
            .sqrt()
            / scale
        )
        record(
            rows,
            "prediction_deformation",
            "if_vs_no_skip_nrmse",
            skip_nrmse >= gates["prediction_nrmse_min"],
            skip_nrmse,
            gates["prediction_nrmse_min"],
        )
        record(
            rows,
            "prediction_deformation",
            "if_vs_direct_nrmse",
            direct_nrmse >= gates["prediction_nrmse_min"],
            direct_nrmse,
            gates["prediction_nrmse_min"],
        )
    return rows, {
        "max_prefix_gap": max_gap,
        "if_no_skip_nrmse": skip_nrmse,
        "if_direct_nrmse": direct_nrmse,
    }


def gradient_group(
    model: torch.nn.Module,
    prefixes: tuple[str, ...],
) -> tuple[float, bool]:
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes) and parameter.grad is not None
    ]
    finite = bool(gradients) and all(
        torch.isfinite(gradient).all().item() for gradient in gradients
    )
    norm = math.sqrt(
        sum(float(gradient.detach().square().sum()) for gradient in gradients)
    )
    return norm, finite


def gradient_audit() -> list[dict[str, Any]]:
    rows = []
    torch.manual_seed(2022)
    x = torch.randn(2, 720, 2)
    y = torch.randn(2, 720, 2)
    for mode in (
        "implicit-frequency-readout",
        "implicit-frequency-noskip-control",
        "implicit-direct-nonlinear-matched",
    ):
        torch.manual_seed(2021)
        model = TimeAlign.Model(small_model_config(mode)).train()
        output, _recon, _alignment = model(
            x,
            y,
            is_training=True,
            target_prefix=720,
        )
        loss = output.square().mean()
        loss.backward()
        groups = {
            "encoder": ("patch_emb_x.", "encoder.", "norm_x."),
        }
        if mode in TimeAlign.D19_IMPLICIT_READOUTS:
            groups.update(
                {
                    "amplitude": (
                        "implicit_frequency_readout.amplitude_head.",
                    ),
                    "phase_sine": (
                        "implicit_frequency_readout.phase_sine_head.",
                    ),
                    "phase_cosine": (
                        "implicit_frequency_readout.phase_cosine_head.",
                    ),
                }
            )
        else:
            groups["direct"] = ("implicit_direct_readout.",)
        for group, prefixes in groups.items():
            norm, finite = gradient_group(model, prefixes)
            record(
                rows,
                "gradient",
                f"{mode}_{group}",
                finite and norm > 0.0,
                norm,
                "finite and >0",
            )

    torch.manual_seed(2023)
    readout = ImplicitFrequencyReadout(
        12,
        hidden_width=16,
        dropout=0.0,
    ).double()
    hidden = torch.zeros(2, 3, 12, dtype=torch.float64, requires_grad=True)
    history = (
        torch.full((2, 720, 3), 1e-12, dtype=torch.float64)
        .requires_grad_()
    )
    amplitude, phase, phase_sine, phase_cosine = readout.polar_spectrum(
        hidden,
        history,
    )
    probe = amplitude.mean() + phase.mean()
    probe.backward()
    finite = (
        torch.isfinite(amplitude).all()
        and torch.isfinite(phase).all()
        and torch.isfinite(phase_sine).all()
        and torch.isfinite(phase_cosine).all()
        and torch.isfinite(hidden.grad).all()
        and torch.isfinite(history.grad).all()
    )
    phase_radius_min = float(
        torch.sqrt(phase_sine.square() + phase_cosine.square()).min()
    )
    record(
        rows,
        "numeric",
        "near_zero_history_phase_gradient",
        bool(finite) and phase_radius_min > 0.0,
        phase_radius_min,
        "finite and >0",
    )
    record(
        rows,
        "numeric",
        "amplitude_nonzero",
        float(amplitude.abs().max()) > 0.0,
        float(amplitude.abs().max()),
        ">0",
    )
    return rows


def model_wiring_audit(gates: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    torch.manual_seed(2024)
    x = torch.randn(1, 720, 2)
    y = torch.randn(1, 720, 2)
    for mode in NEW_ARMS.values():
        torch.manual_seed(2021)
        model = TimeAlign.Model(small_model_config(mode)).eval()
        diagnostics = model_diagnostics(model)
        with torch.no_grad():
            full = model(x, y, is_training=False, target_prefix=720)[0]
            prefix = model(x, y, is_training=False, target_prefix=192)[0]
        gap = float((prefix - full[:, :192]).abs().max())
        passed = (
            tuple(full.shape) == (1, 720, 2)
            and tuple(prefix.shape) == (1, 192, 2)
            and torch.isfinite(full).all().item()
            and gap <= gates["prefix_max_abs_max"]
            and (
                "implicit_decoder_parameters" in diagnostics
                if mode in TimeAlign.D19_IMPLICIT_READOUTS
                else "implicit_direct_decoder_parameters" in diagnostics
            )
        )
        record(
            rows,
            "model_wiring",
            mode,
            passed,
            gap,
            gates["prefix_max_abs_max"],
        )
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    step6_path = ROOT / config["step6_contract"]
    profiles_path = ROOT / config["profiles"]
    step6 = json.loads(step6_path.read_text(encoding="utf-8"))
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))[
        "dataset_profiles"
    ]
    gates = config["local_gates"]
    manifest = manifest_rows(config, profiles, step6)
    checks = []
    checks.extend(governance_audit(config, step6, profiles_path))
    checks.extend(cli_audit(manifest, config))
    checks.extend(parameter_audit(step6, gates))
    checks.extend(source_invariant_audit())
    checks.extend(initialization_audit())
    shape_rows, deformation = shape_projectivity_audit(gates)
    checks.extend(shape_rows)
    checks.extend(gradient_audit())
    checks.extend(model_wiring_audit(gates))

    category_summary = {}
    for category in sorted({row["category"] for row in checks}):
        subset = [row for row in checks if row["category"] == category]
        category_summary[category] = {
            "passed": sum(bool(row["pass"]) for row in subset),
            "total": len(subset),
            "overall_pass": all(bool(row["pass"]) for row in subset),
        }
    summary = {
        "candidate_version": config["candidate_version"],
        "manifest_rows": len(manifest),
        "checks_passed": sum(bool(row["pass"]) for row in checks),
        "checks_total": len(checks),
        "overall_pass": all(bool(row["pass"]) for row in checks),
        "categories": category_summary,
        "deformation": deformation,
        "authorization_after_gate": {
            "step7a_local_complete": all(
                bool(row["pass"]) for row in checks
            ),
            "step7b_prelaunch_next": all(
                bool(row["pass"]) for row in checks
            ),
            "remote": False,
            "official_test": False,
            "paper_method": False,
        },
    }
    if len(manifest) != gates["manifest_rows"]:
        raise AssertionError("D19 manifest row count mismatch")
    if sum(row["category"] == "cli" for row in checks) != gates["cli_rows"]:
        raise AssertionError("D19 CLI row count mismatch")
    if (
        sum(row["category"] == "shape_projectivity" for row in checks)
        < gates["shape_cases_min"]
    ):
        raise AssertionError("D19 shape/projectivity matrix is incomplete")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "manifest.csv", manifest)
    write_csv(args.output_dir / "local_gate_cases.csv", checks)
    (args.output_dir / "gate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    if not summary["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

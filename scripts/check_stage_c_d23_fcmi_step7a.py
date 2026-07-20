#!/usr/bin/env python3
"""Run the SC-D23-FCMI Step 7A local production gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from layers.FCMI import (  # noqa: E402
    GENERIC_DUAL_MODE,
)
from models import TimeAlign  # noqa: E402
import train_repo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, name: str, details: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(condition), "details": details}


def cli_arguments(
    dataset_root: Path,
    dataset: str,
    arm: dict[str, Any],
    profile: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    return [
        "train_repo.py",
        "--dataset-root",
        str(dataset_root),
        "--dataset",
        dataset,
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
        "32",
        "--epochs",
        "1",
        "--seed",
        str(config["seed"]),
        "--num-workers",
        "0",
        "--run-name",
        f"D23_{arm['id']}_{dataset}",
        "--output-dir",
        str(dataset_root / "output" / arm["id"] / dataset),
        "--device",
        "cpu",
        "--checkpoint-policy",
        "best-val",
        "--protocol-class",
        "method_screening",
        "--protocol-profile",
        "stage_c_d23_fcmi_v1_step7a",
        "--profile-hash",
        config["profiles"]["sha256"],
        "--legacy-patch-num",
        str(profile["patch_num"]),
        "--legacy-d-model",
        str(profile["d_model"]),
        "--legacy-d-ff",
        str(profile["d_ff"]),
        "--legacy-dropout",
        "0.1",
        "--legacy-layer-norm",
        "1",
        "--learning-rate",
        "0.0001",
        "--readout-mode",
        arm["readout_mode"],
        "--basis-rank",
        "256",
        "--fcmi-n-heads",
        str(config["implementation_contract"]["fcmi_n_heads"]),
        "--fcmi-dropout",
        str(config["implementation_contract"]["fcmi_dropout"]),
        "--fcmi-permutation-seed",
        str(config["implementation_contract"]["fcmi_permutation_seed"]),
        "--pcc-objective-mode",
        "measure_only",
        "--pred-loss-mode",
        "full",
        "--final-evaluation-split",
        "none",
    ]


def parse_production_cli(arguments: list[str]) -> argparse.Namespace:
    previous = sys.argv
    try:
        sys.argv = arguments
        return train_repo.parse_args()
    finally:
        sys.argv = previous


def model_from_cli(
    parsed: argparse.Namespace,
    dataset_root: Path,
) -> TimeAlign.Model:
    preset = train_repo.OFFICIAL_PRESETS[parsed.dataset][720]
    (dataset_root / preset.data_path).touch()
    official_args = train_repo.build_official_args(parsed, preset)
    torch.manual_seed(parsed.seed)
    return TimeAlign.Model(official_args).float()


def gradient_norm(module: torch.nn.Module) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared += parameter.grad.detach().double().square().sum()
    return float(squared.sqrt().item())


def sorted_patch_values(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sort(tensor, dim=1).values


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    profiles = json.loads(
        Path(config["profiles"]["path"]).read_text(encoding="utf-8")
    )["dataset_profiles"]
    arms = {arm["id"]: arm for arm in config["arms"]}
    gates = config["gates"]
    dense_control_threshold = gates[
        "a6_fcmi_parameter_relative_gap_requires_future_dense_control"
    ]
    results: list[dict[str, Any]] = []

    step46_path = Path(config["step46"]["path"])
    profile_path = Path(config["profiles"]["path"])
    results.append(
        check(
            sha256(step46_path) == config["step46"]["sha256"]
            and sha256(profile_path) == config["profiles"]["sha256"],
            "frozen_input_hashes",
            {
                "step46": sha256(step46_path),
                "profiles": sha256(profile_path),
            },
        )
    )

    cli_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    contracts_by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    diagnostics_by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="fatst_d23_step7a_") as temp:
        dataset_root = Path(temp)
        for dataset in config["datasets"]:
            contracts_by_dataset[dataset] = {}
            diagnostics_by_dataset[dataset] = {}
            for arm in config["arms"]:
                parsed = parse_production_cli(
                    cli_arguments(
                        dataset_root,
                        dataset,
                        arm,
                        profiles[dataset],
                        config,
                    )
                )
                model = model_from_cli(parsed, dataset_root)
                contract = train_repo.initialization_contract(model)
                diagnostics = train_repo.model_diagnostics(model)
                contracts_by_dataset[dataset][arm["id"]] = contract
                diagnostics_by_dataset[dataset][arm["id"]] = diagnostics
                cli_rows.append(
                    {
                        "dataset": dataset,
                        "arm": arm["id"],
                        "readout_mode": parsed.readout_mode,
                        "objective": parsed.pcc_objective_mode,
                        "final_split": parsed.final_evaluation_split,
                    }
                )

            diagnostics = diagnostics_by_dataset[dataset]
            dual_ids = [
                "STANDARD_DUAL_MATCHED",
                "GENERIC_DUAL_MATCHED",
                "FCMI",
                "FCMI_ORDER_SHUFFLED",
                "TARGET_SHUFFLED_QUERY",
            ]
            active_counts = {
                diagnostics[arm_id]["active_forward_parameters"]
                for arm_id in dual_ids
            }
            decoder_counts = {
                diagnostics[arm_id]["fcmi_decoder_parameters"]
                for arm_id in dual_ids
            }
            fcmi_active = diagnostics["FCMI"]["active_forward_parameters"]
            a6_active = diagnostics["A6_MEASURE"]["active_forward_parameters"]
            parameter_rows.append(
                {
                    "dataset": dataset,
                    "dual_active_counts": sorted(active_counts),
                    "dual_decoder_counts": sorted(decoder_counts),
                    "standard_active": diagnostics["STANDARD_QUERY"][
                        "active_forward_parameters"
                    ],
                    "fcmi_active": fcmi_active,
                    "a6_active": a6_active,
                    "a6_fcmi_relative_gap": abs(a6_active - fcmi_active)
                    / a6_active,
                    "future_dense_control_required": (
                        abs(a6_active - fcmi_active) / a6_active
                        > dense_control_threshold
                    ),
                }
            )

    results.append(
        check(
            len(cli_rows) == config["expected_counts"]["production_cli_cases"]
            and all(
                row["readout_mode"] == arms[row["arm"]]["readout_mode"]
                and row["objective"] == "measure_only"
                and row["final_split"] == "none"
                for row in cli_rows
            ),
            "production_cli_synthetic_cases",
            cli_rows,
        )
    )

    dual_ids = [
        "STANDARD_DUAL_MATCHED",
        "GENERIC_DUAL_MATCHED",
        "FCMI",
        "FCMI_ORDER_SHUFFLED",
        "TARGET_SHUFFLED_QUERY",
    ]
    initialization_rows = []
    initialization_pass = True
    parameter_pass = True
    for dataset in config["datasets"]:
        contracts = contracts_by_dataset[dataset]
        common_hashes = {
            contracts[arm_id]["fcmi_common_initialization_hash"]
            for arm_id in dual_ids
        }
        encoder_hashes = {
            contracts[arm_id]["encoder_initialization_hash"]
            for arm_id in dual_ids
        }
        branch_gaps = [
            contracts[arm_id]["fcmi_branch_initial_max_abs_gap"]
            for arm_id in dual_ids
        ]
        initialization_rows.append(
            {
                "dataset": dataset,
                "common_hash_count": len(common_hashes),
                "encoder_hash_count": len(encoder_hashes),
                "branch_max_abs_gaps": branch_gaps,
            }
        )
        initialization_pass = initialization_pass and bool(
            len(common_hashes) == 1
            and len(encoder_hashes) == 1
            and max(branch_gaps)
            <= gates["dual_branch_initialization_max_abs"]
        )
        parameter_pass = parameter_pass and bool(
            len(parameter_rows[config["datasets"].index(dataset)][
                "dual_active_counts"
            ])
            == 1
            and len(parameter_rows[config["datasets"].index(dataset)][
                "dual_decoder_counts"
            ])
            == 1
        )
    results.append(
        check(
            initialization_pass,
            "paired_initialization_contract",
            initialization_rows,
        )
    )
    results.append(
        check(
            parameter_pass,
            "dual_parameter_matching_and_a6_gap_audit",
            parameter_rows,
        )
    )

    tensor_dataset = config["expected_counts"]["tensor_profile"]
    tensor_profile = profiles[tensor_dataset]
    with tempfile.TemporaryDirectory(prefix="fatst_d23_tensor_") as temp:
        dataset_root = Path(temp)
        tensor_models: dict[str, TimeAlign.Model] = {}
        for arm_id in dual_ids:
            parsed = parse_production_cli(
                cli_arguments(
                    dataset_root,
                    tensor_dataset,
                    arms[arm_id],
                    tensor_profile,
                    config,
                )
            )
            tensor_models[arm_id] = model_from_cli(parsed, dataset_root)
            tensor_models[arm_id].eval()

        batch = int(config["expected_counts"]["synthetic_batch"])
        channels = int(config["expected_counts"]["synthetic_channels"])
        generator = torch.Generator().manual_seed(20260720)
        history = torch.randn(
            batch,
            720,
            channels,
            generator=generator,
        )
        memories = {
            arm_id: tensor_models[arm_id].encode_history(history)
            for arm_id in dual_ids
        }
        readout_details = {
            arm_id: tensor_models[arm_id].fcmi_readout(
                memories[arm_id],
                target_prefix=720,
                return_details=True,
            )
            for arm_id in dual_ids
        }

    fcmi_output, fcmi_details = readout_details["FCMI"]
    standard_output, _standard_details = readout_details[
        "STANDARD_DUAL_MATCHED"
    ]
    dummy_future = torch.zeros(batch, 720, channels)
    with torch.no_grad():
        production_prefix, _recon, _align = tensor_models["FCMI"](
            history,
            dummy_future,
            is_training=True,
            target_prefix=96,
        )
        production_full, _recon, _align = tensor_models["FCMI"](
            history,
            dummy_future,
            is_training=True,
            target_prefix=720,
        )
    production_prefix_gap = float(
        (
            production_prefix
            - production_full[:, :96, :]
        ).abs().max().item()
    )
    expected_shape = (
        batch,
        channels,
        tensor_profile["patch_num"],
        tensor_profile["d_model"],
    )
    shape_payload = {
        "memory": list(memories["FCMI"].shape),
        "query": list(fcmi_details["query"].shape),
        "context": list(fcmi_details["context"].shape),
        "main": list(fcmi_details["main"].shape),
        "interaction": list(fcmi_details["interaction"].shape),
        "state": list(fcmi_details["state"].shape),
        "output": list(fcmi_output.shape),
        "production_prefix": list(production_prefix.shape),
        "production_full": list(production_full.shape),
        "production_prefix_max_abs": production_prefix_gap,
    }
    shape_pass = bool(
        tuple(memories["FCMI"].shape) == expected_shape
        and tuple(fcmi_details["query"].shape)
        == (batch * channels, 720, tensor_profile["d_model"])
        and tuple(fcmi_details["context"].shape)
        == tuple(fcmi_details["query"].shape)
        and tuple(fcmi_details["main"].shape)
        == (batch * channels, 1, tensor_profile["d_model"])
        and tuple(fcmi_output.shape) == (batch, 720, channels)
        and tuple(production_prefix.shape) == (batch, 96, channels)
        and tuple(production_full.shape) == (batch, 720, channels)
        and production_prefix_gap <= gates["full_model_prefix_max_abs"]
    )
    results.append(check(shape_pass, "tensor_shapes", shape_payload))

    interaction_mean_abs = float(
        fcmi_details["interaction"].mean(dim=1).abs().max().item()
    )
    morph_gap = float((fcmi_output - standard_output).abs().max().item())
    results.append(
        check(
            interaction_mean_abs <= gates["interaction_mean_abs_max"],
            "zero_mean_interaction",
            {"max_abs": interaction_mean_abs},
        )
    )
    results.append(
        check(
            morph_gap <= gates["standard_morph_output_max_abs"],
            "standard_query_exact_initial_morph",
            {"max_abs": morph_gap},
        )
    )

    generic_readout = tensor_models["GENERIC_DUAL_MATCHED"].fcmi_readout
    context = fcmi_details["context"].detach()
    query = fcmi_details["query"].detach()
    perturbation = torch.randn(
        context.shape,
        generator=generator,
        dtype=context.dtype,
    )
    perturbation = perturbation - perturbation.mean(dim=1, keepdim=True)
    generic_state, generic_details = generic_readout.compose_context(
        context,
        query,
        GENERIC_DUAL_MODE,
    )
    perturbed_state, _ = generic_readout.compose_context(
        context + perturbation,
        query,
        GENERIC_DUAL_MODE,
    )
    generic_gap = float(
        (generic_state - perturbed_state).abs().max().item()
    )
    results.append(
        check(
            not bool(generic_details["interaction_used"])
            and generic_gap
            <= gates["generic_zero_mean_perturbation_max_abs"],
            "generic_control_excludes_interaction",
            {
                "interaction_used": generic_details["interaction_used"],
                "zero_mean_perturbation_max_abs": generic_gap,
            },
        )
    )

    gradient_model = tensor_models["FCMI"]
    gradient_model.zero_grad(set_to_none=True)
    gradient_memory = gradient_model.encode_history(history)
    gradient_output = gradient_model.fcmi_readout(
        gradient_memory,
        target_prefix=720,
    )
    gradient_output.square().mean().backward()
    gradient_payload = {
        "main": gradient_norm(
            gradient_model.fcmi_readout.main_projection
        ),
        "interaction": gradient_norm(
            gradient_model.fcmi_readout.interaction_projection
        ),
        "query": gradient_norm(
            gradient_model.fcmi_readout.query_encoder
        ),
        "output": gradient_norm(
            gradient_model.fcmi_readout.output_projection
        ),
    }
    results.append(
        check(
            all(
                torch.isfinite(torch.tensor(value)).item()
                and value >= gates["gradient_norm_min"]
                for value in gradient_payload.values()
            ),
            "main_interaction_query_output_gradients",
            gradient_payload,
        )
    )

    ordered_content = fcmi_details["memory_content"]
    shuffled_output, shuffled_details = readout_details[
        "FCMI_ORDER_SHUFFLED"
    ]
    shuffled_content = shuffled_details["memory_content"]
    marginal_gap = float(
        (
            sorted_patch_values(ordered_content)
            - sorted_patch_values(shuffled_content)
        ).abs().max().item()
    )
    binding_gap = float(
        (
            fcmi_details["attended_memory"]
            - shuffled_details["attended_memory"]
        ).abs().max().item()
    )
    shuffle_output_gap = float(
        (fcmi_output - shuffled_output).abs().max().item()
    )
    results.append(
        check(
            marginal_gap <= gates["order_shuffle_marginal_max_abs"]
            and binding_gap >= gates["order_shuffle_binding_min_abs"]
            and shuffle_output_gap
            >= gates["order_shuffle_output_min_abs"],
            "order_shuffle_value_position_binding",
            {
                "marginal_max_abs": marginal_gap,
                "binding_max_abs": binding_gap,
                "output_max_abs": shuffle_output_gap,
            },
        )
    )

    target_output, _target_details = readout_details[
        "TARGET_SHUFFLED_QUERY"
    ]
    target_output_gap = float(
        (fcmi_output - target_output).abs().max().item()
    )
    results.append(
        check(
            target_output_gap >= gates["target_shuffle_output_min_abs"],
            "target_shuffle_local_sanity",
            {"output_max_abs": target_output_gap},
        )
    )

    all_pass = all(result["pass"] for result in results)
    payload = {
        "candidate_version": config["candidate_version"],
        "decision": (
            "step7a_local_pass_step7b_design_freeze_next"
            if all_pass
            else "step7a_local_fail_return_step5_6"
        ),
        "authorization": config["authorization"],
        "gate_count": len(results),
        "pass_count": sum(result["pass"] for result in results),
        "all_pass": all_pass,
        "results": results,
        "notes": {
            "a6_parameter_gap": (
                "A future formal matrix must add a dense dual matched control "
                "for every profile whose A6-FCMI gap exceeds 1%."
            ),
            "remote_training": "not authorized",
            "official_test": "not authorized",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"SC-D23-FCMI Step7A: {payload['pass_count']}/"
        f"{payload['gate_count']} gates passed; "
        f"decision={payload['decision']}"
    )
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

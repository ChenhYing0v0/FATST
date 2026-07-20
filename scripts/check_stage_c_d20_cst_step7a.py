#!/usr/bin/env python3
"""Run the SC-D20-CST Step 7A production implementation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from models import TimeAlign  # noqa: E402
import train_repo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def check(condition: bool, name: str, details: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(condition), "details": details}


def cli_args(
    dataset: str,
    arm: dict[str, Any],
    profile: dict[str, Any],
    profile_hash: str,
) -> list[str]:
    return [
        "train_repo.py",
        "--dataset-root",
        "/tmp/datasets",
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
        "--enable-early-stopping",
        "--seed",
        "2021",
        "--num-workers",
        "0",
        "--run-name",
        f"D20_{arm['id']}",
        "--output-dir",
        f"/tmp/d20/{arm['id']}/{dataset}",
        "--device",
        "cpu",
        "--checkpoint-policy",
        "best-val",
        "--protocol-class",
        "method_screening",
        "--protocol-profile",
        "stage_c_d20_cst_v1",
        "--profile-hash",
        profile_hash,
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
        "--history-statistic-mode",
        arm["history_statistic_mode"],
        "--history-statistic-dim",
        "64",
        "--history-statistic-random-seed",
        "20260719",
        "--pcc-objective-mode",
        "measure_only",
        "--pred-loss-mode",
        "full",
        "--final-evaluation-split",
        "val",
    ]


def parsed_case(arguments: list[str]) -> argparse.Namespace:
    previous = sys.argv
    try:
        sys.argv = arguments
        return train_repo.parse_args()
    finally:
        sys.argv = previous


def model_for_case(args: argparse.Namespace) -> TimeAlign.Model:
    preset = train_repo.OFFICIAL_PRESETS[args.dataset][720]
    official = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode=args.readout_mode,
        e_layers=2,
        patch_num=args.legacy_patch_num,
        d_model=args.legacy_d_model,
        d_ff=args.legacy_d_ff,
        dropout=args.legacy_dropout,
        pos=1,
        layer_norm=args.legacy_layer_norm,
        enc_in=preset.enc_in,
        target_horizons=[720],
        basis_rank=args.basis_rank,
        history_statistic_mode=args.history_statistic_mode,
        history_statistic_dim=args.history_statistic_dim,
        history_statistic_random_seed=args.history_statistic_random_seed,
    )
    torch.manual_seed(args.seed)
    return TimeAlign.Model(official).float()


def gradient_norm(model: TimeAlign.Model, prefix: str) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for name, parameter in model.named_parameters():
        if name.startswith(prefix) and parameter.grad is not None:
            total += parameter.grad.detach().double().square().sum()
    return float(total.sqrt().item())


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    step6_path = Path(config["step6"]["path"])
    profile_path = Path(config["profiles"]["path"])
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))[
        "dataset_profiles"
    ]
    gates = config["gates"]
    arms = {arm["id"]: arm for arm in config["arms"]}
    results = []
    results.append(
        check(
            sha256(step6_path) == config["step6"]["sha256"]
            and sha256(profile_path) == config["profiles"]["sha256"],
            "frozen_input_hashes",
            {
                "step6": sha256(step6_path),
                "profiles": sha256(profile_path),
            },
        )
    )

    parsed_rows = []
    models_by_dataset: dict[str, dict[str, TimeAlign.Model]] = {}
    cli_pass = True
    for dataset in config["datasets"]:
        models_by_dataset[dataset] = {}
        for arm_id in (
            "A6_MEASURE_RETRAIN",
            "A6_CST_SPEC",
            "A6_CST_RANDOM",
        ):
            arguments = cli_args(
                dataset,
                arms[arm_id],
                profiles[dataset],
                config["profiles"]["sha256"],
            )
            parsed = parsed_case(arguments)
            cli_pass = cli_pass and bool(
                parsed.readout_mode == arms[arm_id]["readout_mode"]
                and parsed.history_statistic_mode
                == arms[arm_id]["history_statistic_mode"]
                and parsed.history_statistic_dim == 64
                and parsed.history_statistic_random_seed == 20260719
            )
            parsed_rows.append(
                {
                    "dataset": dataset,
                    "arm": arm_id,
                    "readout_mode": parsed.readout_mode,
                    "history_statistic_mode": parsed.history_statistic_mode,
                }
            )
            models_by_dataset[dataset][arm_id] = model_for_case(parsed)
    results.append(
        check(
            cli_pass
            and len(parsed_rows) == config["expected_counts"]["cli_cases"],
            "production_cli_cases",
            parsed_rows,
        )
    )

    initialization_rows = []
    parameter_rows = []
    shape_rows = []
    gradient_rows = []
    init_pass = True
    parameter_pass = True
    shape_pass = True
    gradient_pass = True
    projection_pass = True
    max_initial_gap = 0.0
    max_prefix_gap = 0.0
    max_projection_gap = 0.0
    max_spectrum_dc = 0.0
    minimum_gradient_difference = float("inf")

    for dataset, models in models_by_dataset.items():
        contracts = {
            arm_id: train_repo.initialization_contract(model)
            for arm_id, model in models.items()
        }
        diagnostics = {
            arm_id: train_repo.model_diagnostics(model)
            for arm_id, model in models.items()
        }
        common_encoder_hashes = {
            contract["encoder_initialization_hash"]
            for contract in contracts.values()
        }
        common_operator_hashes = {
            contract["operator_base_initialization_hash"]
            for contract in contracts.values()
        }
        projection_hashes = {
            contracts[arm_id]["history_statistic_projection_hash"]
            for arm_id in ("A6_CST_SPEC", "A6_CST_RANDOM")
        }
        zero_norms = [
            contracts[arm_id]["history_statistic_initial_weight_norm"]
            for arm_id in ("A6_CST_SPEC", "A6_CST_RANDOM")
        ]
        row = {
            "dataset": dataset,
            "encoder_hash_count": len(common_encoder_hashes),
            "base_operator_hash_count": len(common_operator_hashes),
            "projection_hash_count": len(projection_hashes),
            "summary_initial_weight_norms": zero_norms,
        }
        initialization_rows.append(row)
        init_pass = init_pass and bool(
            len(common_encoder_hashes) == 1
            and len(common_operator_hashes) == 1
            and len(projection_hashes) == 2
            and zero_norms == [0.0, 0.0]
        )

        a6_parameters = diagnostics["A6_MEASURE_RETRAIN"][
            "active_forward_parameters"
        ]
        spec_parameters = diagnostics["A6_CST_SPEC"][
            "active_forward_parameters"
        ]
        random_parameters = diagnostics["A6_CST_RANDOM"][
            "active_forward_parameters"
        ]
        parameter_row = {
            "dataset": dataset,
            "a6_active_parameters": a6_parameters,
            "spectrum_active_parameters": spec_parameters,
            "random_active_parameters": random_parameters,
            "augmentation": spec_parameters - a6_parameters,
        }
        parameter_rows.append(parameter_row)
        parameter_pass = parameter_pass and bool(
            spec_parameters == random_parameters
            and spec_parameters - a6_parameters
            == gates["augmentation_parameters"]
        )

        for arm_id in ("A6_CST_SPEC", "A6_CST_RANDOM"):
            projection_gap = diagnostics[arm_id][
                "history_statistic_projection_orthogonality_max_abs"
            ]
            max_projection_gap = max(max_projection_gap, projection_gap)
            projection_pass = projection_pass and bool(
                projection_gap
                <= gates["production_projection_orthogonality_max_abs_max"]
            )
        spectrum_dc = diagnostics["A6_CST_SPEC"][
            "history_statistic_projection_dc_leakage_max_abs"
        ]
        max_spectrum_dc = max(max_spectrum_dc, spectrum_dc)
        projection_pass = projection_pass and bool(
            spectrum_dc <= gates["spectrum_dc_leakage_max_abs_max"]
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(7000 + config["datasets"].index(dataset))
        channels = train_repo.OFFICIAL_PRESETS[dataset][720].enc_in
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
        full_outputs = {}
        for model in models.values():
            model.eval()
        with torch.no_grad():
            for arm_id, model in models.items():
                full_outputs[arm_id] = model(
                    batch_x,
                    target,
                    is_training=False,
                    target_prefix=720,
                )[0]
        for arm_id in ("A6_CST_SPEC", "A6_CST_RANDOM"):
            gap = float(
                (
                    full_outputs[arm_id]
                    - full_outputs["A6_MEASURE_RETRAIN"]
                )
                .abs()
                .max()
            )
            max_initial_gap = max(max_initial_gap, gap)
            init_pass = init_pass and gap <= gates["initial_output_max_abs_max"]
        for arm_id, model in models.items():
            for horizon in config["horizons"]:
                with torch.no_grad():
                    prefix = model(
                        batch_x,
                        target,
                        is_training=False,
                        target_prefix=horizon,
                    )[0]
                gap = float(
                    (prefix - full_outputs[arm_id][:, :horizon]).abs().max()
                )
                max_prefix_gap = max(max_prefix_gap, gap)
                finite = bool(torch.isfinite(prefix).all())
                shape_ok = tuple(prefix.shape) == (2, horizon, channels)
                shape_pass = shape_pass and bool(
                    finite
                    and shape_ok
                    and gap <= gates["prefix_max_abs_max"]
                )
                shape_rows.append(
                    {
                        "dataset": dataset,
                        "arm": arm_id,
                        "horizon": horizon,
                        "shape": list(prefix.shape),
                        "prefix_gap": gap,
                        "finite": finite,
                    }
                )

        summary_gradients = {}
        for arm_id in ("A6_CST_SPEC", "A6_CST_RANDOM"):
            model = models[arm_id]
            model.train()
            model.zero_grad(set_to_none=True)
            prediction = model(
                batch_x,
                target,
                is_training=True,
                target_prefix=720,
            )[0]
            loss = (prediction - target).abs().mean()
            loss.backward()
            summary_gradient = gradient_norm(
                model,
                "history_statistic_coeff.",
            )
            base_gradient = gradient_norm(model, "learned_basis_coeff.")
            encoder_gradient = gradient_norm(model, "patch_emb_x.")
            summary_gradients[arm_id] = (
                model.history_statistic_coeff.weight.grad.detach().clone()
            )
            row = {
                "dataset": dataset,
                "arm": arm_id,
                "loss": float(loss.detach()),
                "summary_gradient_norm": summary_gradient,
                "base_gradient_norm": base_gradient,
                "encoder_gradient_norm": encoder_gradient,
            }
            gradient_rows.append(row)
            gradient_pass = gradient_pass and bool(
                torch.isfinite(loss)
                and summary_gradient >= gates["summary_gradient_norm_min"]
                and base_gradient >= gates["base_gradient_norm_min"]
                and encoder_gradient >= gates["encoder_gradient_norm_min"]
            )
        gradient_difference = float(
            (
                summary_gradients["A6_CST_SPEC"]
                - summary_gradients["A6_CST_RANDOM"]
            )
            .norm()
            .item()
        )
        minimum_gradient_difference = min(
            minimum_gradient_difference,
            gradient_difference,
        )
        gradient_pass = gradient_pass and bool(
            gradient_difference
            >= gates["spec_random_summary_gradient_difference_min"]
        )
        initialization_rows[-1][
            "spec_random_summary_gradient_difference"
        ] = gradient_difference

    results.extend(
        [
            check(
                len(models_by_dataset) * len(arms)
                == config["expected_counts"]["model_constructors"],
                "model_constructor_count",
                len(models_by_dataset) * len(arms),
            ),
            check(init_pass, "paired_initialization", initialization_rows),
            check(parameter_pass, "parameter_accounting", parameter_rows),
            check(
                projection_pass,
                "production_projection_contract",
                {
                    "max_orthogonality_gap": max_projection_gap,
                    "max_spectrum_dc_leakage": max_spectrum_dc,
                },
            ),
            check(
                shape_pass
                and len(shape_rows)
                == config["expected_counts"]["shape_prefix_cases"],
                "shape_projectivity_and_finite",
                shape_rows,
            ),
            check(
                gradient_pass
                and len(gradient_rows)
                == config["expected_counts"]["summary_gradient_cases"],
                "gradient_paths_and_semantic_difference",
                gradient_rows,
            ),
            check(
                config["authorization"]["step7a_local"] is True
                and config["authorization"]["step7b_prelaunch"] is False
                and config["authorization"]["remote_training"] is False
                and config["authorization"]["official_test_access"] is False
                and config["authorization"]["paper_method"] is False,
                "authorization_boundary",
                config["authorization"],
            ),
        ]
    )
    summary = {
        "candidate_version": config["candidate_version"],
        "config_sha256": sha256(args.config),
        "checks_passed": sum(item["pass"] for item in results),
        "checks_total": len(results),
        "overall_pass": all(item["pass"] for item in results),
        "counts": {
            "cli_cases": len(parsed_rows),
            "model_constructors": len(models_by_dataset) * len(arms),
            "shape_prefix_cases": len(shape_rows),
            "summary_gradient_cases": len(gradient_rows),
        },
        "max_initial_output_gap": max_initial_gap,
        "max_prefix_gap": max_prefix_gap,
        "max_projection_orthogonality_gap": max_projection_gap,
        "max_spectrum_dc_leakage": max_spectrum_dc,
        "minimum_spec_random_summary_gradient_difference": (
            minimum_gradient_difference
        ),
        "checks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

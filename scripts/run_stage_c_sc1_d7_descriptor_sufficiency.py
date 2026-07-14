#!/usr/bin/env python3
"""Run the SC1-D7 frozen-memory RGNB descriptor-sufficiency diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    from check_stage_c_plgo_step5_theory import (
        active_indices,
        restricted_global_nested_basis,
    )
    from check_stage_c_plgo_step6_design import canonical_atom_descriptors
    from run_stage_c_sc1_d2_diagnostic import (
        DATA_SEED,
        PROBE_SEED,
        SERIES_LENGTH,
        LowRankLinearHead,
        collect_rows,
        count_parameters,
        load_model_and_loaders,
        set_seed,
        split_fit_holdout,
        standardize_features,
        train_head,
        write_csv,
    )
except ModuleNotFoundError:
    from scripts.check_stage_c_plgo_step5_theory import (
        active_indices,
        restricted_global_nested_basis,
    )
    from scripts.check_stage_c_plgo_step6_design import canonical_atom_descriptors
    from scripts.run_stage_c_sc1_d2_diagnostic import (
        DATA_SEED,
        PROBE_SEED,
        SERIES_LENGTH,
        LowRankLinearHead,
        collect_rows,
        count_parameters,
        load_model_and_loaders,
        set_seed,
        split_fit_holdout,
        standardize_features,
        train_head,
        write_csv,
    )


LATENT_WIDTH = 256
DESCRIPTOR_DIM = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-root", type=Path)
    parser.add_argument("--phase-b-root", type=Path)
    parser.add_argument("--phase-c-root", type=Path)
    parser.add_argument("--five-phase-a-root", type=Path)
    parser.add_argument("--five-phase-b-root", type=Path)
    parser.add_argument("--five-phase-c-root", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--d7-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dataset", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-batches", type=int, default=16)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--val-offset-batches", type=int, default=16)
    parser.add_argument("--fit-fraction", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        return args
    required = (
        "phase_a_root",
        "phase_b_root",
        "phase_c_root",
        "contract",
        "d7_config",
        "output_dir",
        "dataset",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


class ProjectiveAtomFunctionalHead(nn.Module):
    """Generate RGNB coefficients from a shared history latent and atom descriptors."""

    def __init__(
        self,
        input_width: int,
        descriptors: torch.Tensor,
        trunk_width: int,
    ) -> None:
        super().__init__()
        if descriptors.shape != (SERIES_LENGTH, DESCRIPTOR_DIM):
            raise ValueError(f"unexpected descriptor shape: {descriptors.shape}")
        self.branch = nn.Linear(input_width, LATENT_WIDTH)
        self.trunk = nn.Sequential(
            nn.Linear(DESCRIPTOR_DIM, trunk_width),
            nn.Tanh(),
            nn.Linear(trunk_width, LATENT_WIDTH),
        )
        self.coefficient_bias = nn.Parameter(torch.zeros(SERIES_LENGTH))
        self.register_buffer("descriptors", descriptors.clone())

    def coefficients(
        self,
        features: torch.Tensor,
        descriptors: torch.Tensor,
        coefficient_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent = self.branch(features)
        atom_features = self.trunk(descriptors)
        coefficients = latent @ atom_features.transpose(0, 1)
        bias = (
            self.coefficient_bias
            if coefficient_indices is None
            else self.coefficient_bias[coefficient_indices]
        )
        return coefficients + bias

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.coefficients(features, self.descriptors)


def descriptor_families(
    atoms: list[Any], permutation_seed: int, random_seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    canonical = canonical_atom_descriptors(atoms).float()
    permutation_generator = torch.Generator(device="cpu").manual_seed(permutation_seed)
    permutation = torch.randperm(SERIES_LENGTH, generator=permutation_generator)
    permuted = canonical[permutation]

    random_generator = torch.Generator(device="cpu").manual_seed(random_seed)
    random_values = torch.randn(
        SERIES_LENGTH,
        DESCRIPTOR_DIM,
        generator=random_generator,
    )
    canonical_mean = canonical.mean(dim=0, keepdim=True)
    canonical_std = canonical.std(dim=0, unbiased=False, keepdim=True)
    random_mean = random_values.mean(dim=0, keepdim=True)
    random_std = random_values.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    random_matched = (random_values - random_mean) / random_std
    random_matched = random_matched * canonical_std + canonical_mean
    moment_gap = max(
        float((random_matched.mean(dim=0) - canonical_mean.squeeze(0)).abs().max()),
        float(
            (
                random_matched.std(dim=0, unbiased=False)
                - canonical_std.squeeze(0)
            )
            .abs()
            .max()
        ),
    )
    return (
        {"geo": canonical, "perm": permuted, "random": random_matched},
        {"random_descriptor_moment_max_abs": moment_gap},
    )


def descriptor_hash(descriptors: torch.Tensor) -> str:
    return hashlib.sha256(descriptors.contiguous().numpy().tobytes()).hexdigest()


def build_arm(
    arm: str,
    input_width: int,
    descriptors: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, int | None, str]:
    set_seed(PROBE_SEED)
    if arm == "free_m0":
        return LowRankLinearHead(input_width).to(device), None, "free"
    descriptor_family, width_code = arm.split("_", maxsplit=1)
    trunk_width = (
        int(config["compact_trunk_width"])
        if width_code == "c256"
        else int(config["matched_trunk_width"])
    )
    head = ProjectiveAtomFunctionalHead(
        input_width,
        descriptors[descriptor_family],
        trunk_width,
    ).to(device)
    return head, trunk_width, descriptor_family


def validate_config(config: dict[str, Any], args: argparse.Namespace) -> None:
    probe = config["probe_contract"]
    expected = {
        "series_length": SERIES_LENGTH,
        "data_seed": DATA_SEED,
        "probe_seed": PROBE_SEED,
        "train_batches": args.train_batches,
        "val_batches": args.val_batches,
        "val_offset_batches": args.val_offset_batches,
        "fit_fraction": args.fit_fraction,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    observed = {key: probe[key] for key in expected}
    if observed != expected:
        raise ValueError(f"runtime arguments do not match D7 config: {observed} != {expected}")
    if int(config["latent_width"]) != LATENT_WIDTH:
        raise ValueError("D7 latent width changed")
    if int(config["descriptor_dim"]) != DESCRIPTOR_DIM:
        raise ValueError("D7 descriptor dimension changed")


def evaluate_rows(
    head: nn.Module,
    basis: torch.Tensor,
    rows: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    horizons: list[int],
) -> dict[str, float | int]:
    sums = {
        horizon: {"se": 0.0, "ae": 0.0, "elements": 0}
        for horizon in horizons
    }
    head.eval()
    with torch.no_grad():
        for start in range(0, rows["features"].shape[0], batch_size):
            end = start + batch_size
            coefficients = head(rows["features"][start:end].to(device))
            prediction = coefficients @ basis
            target = rows["target"][start:end].to(device)
            stdev = rows["stdev"][start:end].to(device)
            for horizon in horizons:
                error = (prediction[:, :horizon] - target[:, :horizon]) * stdev
                sums[horizon]["se"] += float(error.square().sum())
                sums[horizon]["ae"] += float(error.abs().sum())
                sums[horizon]["elements"] += int(error.numel())
    output: dict[str, float | int] = {}
    for horizon in horizons:
        elements = sums[horizon]["elements"]
        if elements == 0:
            raise RuntimeError(f"no evaluation rows for horizon {horizon}")
        output[f"mse_h{horizon}"] = sums[horizon]["se"] / elements
        output[f"mae_h{horizon}"] = sums[horizon]["ae"] / elements
        output[f"elements_h{horizon}"] = elements
    return output


def projectivity_gap(
    head: ProjectiveAtomFunctionalHead,
    features: torch.Tensor,
    basis: torch.Tensor,
    atoms: list[Any],
    horizons: list[int],
) -> tuple[float, float]:
    head.eval()
    max_coefficient_gap = 0.0
    max_prefix_gap = 0.0
    with torch.no_grad():
        full_coefficients = head(features)
        full_prediction = full_coefficients @ basis
        for horizon in horizons:
            active = active_indices(atoms, horizon).to(features.device)
            subset = head.coefficients(
                features,
                head.descriptors[active],
                active,
            )
            max_coefficient_gap = max(
                max_coefficient_gap,
                float((subset - full_coefficients[:, active]).abs().max()),
            )
            prefix = subset @ basis[active, :horizon]
            max_prefix_gap = max(
                max_prefix_gap,
                float((prefix - full_prediction[:, :horizon]).abs().max()),
            )
    return max_coefficient_gap, max_prefix_gap


def run_checkpoint_seed(
    args: argparse.Namespace,
    contract: dict[str, Any],
    config: dict[str, Any],
    checkpoint_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model, train_loader, val_loader, official_args, directory = load_model_and_loaders(
        args, contract, checkpoint_seed
    )
    device = official_args.device
    train_rows = collect_rows(model, train_loader, device, args.train_batches)
    validation = collect_rows(
        model,
        val_loader,
        device,
        args.val_batches,
        start_batch=args.val_offset_batches,
    )
    fit, holdout = split_fit_holdout(train_rows, args.fit_fraction)
    fit, holdout, validation = standardize_features(fit, holdout, validation)
    input_width = int(fit["features"].shape[1])
    if input_width != int(official_args.patch_num * official_args.d_model):
        raise RuntimeError("frozen memory width mismatch")

    synthesis, atoms, _prototypes, _gap = restricted_global_nested_basis(
        SERIES_LENGTH, int(config["global_rank"])
    )
    basis = synthesis.transpose(0, 1).float().to(device)
    orthogonality_gap = float(
        (basis @ basis.transpose(0, 1) - torch.eye(SERIES_LENGTH, device=device))
        .abs()
        .max()
    )
    descriptors, descriptor_stats = descriptor_families(
        atoms,
        int(config["permutation_seed"]),
        int(config["random_descriptor_seed"]),
    )
    horizons = [int(value) for value in config["horizons"]]
    metrics: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    max_coefficient_gap = 0.0
    max_prefix_gap = 0.0
    arms = [str(value) for value in config["arms"]]
    for arm_index, arm in enumerate(arms, start=1):
        print(
            f"d7_arm_start dataset={args.dataset} checkpoint_seed={checkpoint_seed} "
            f"arm={arm} position={arm_index}/{len(arms)}",
            flush=True,
        )
        head, trunk_width, descriptor_family = build_arm(
            arm, input_width, descriptors, config, device
        )
        trained, history, best_epoch, _last_fit, _best_holdout = train_head(
            args,
            arm,
            head,
            "coeff",
            basis,
            fit,
            holdout,
            device,
            checkpoint_seed,
            log_prefix="d7",
        )
        fit_metrics = evaluate_rows(
            trained, basis, fit, device, args.batch_size, [SERIES_LENGTH]
        )
        holdout_metrics = evaluate_rows(
            trained, basis, holdout, device, args.batch_size, [SERIES_LENGTH]
        )
        validation_metrics = evaluate_rows(
            trained, basis, validation, device, args.batch_size, horizons
        )
        if isinstance(trained, ProjectiveAtomFunctionalHead):
            coefficient_gap, prefix_gap = projectivity_gap(
                trained,
                validation["features"][:8].to(device),
                basis,
                atoms,
                horizons,
            )
            max_coefficient_gap = max(max_coefficient_gap, coefficient_gap)
            max_prefix_gap = max(max_prefix_gap, prefix_gap)
        metrics.append(
            {
                "dataset": args.dataset,
                "checkpoint_seed": checkpoint_seed,
                "arm": arm,
                "descriptor_family": descriptor_family,
                "trunk_width": "" if trunk_width is None else trunk_width,
                "parameters": count_parameters(trained),
                "best_epoch": best_epoch,
                "fit_mse_eval_h720": fit_metrics["mse_h720"],
                "holdout_mse_eval_h720": holdout_metrics["mse_h720"],
                **{
                    f"val_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
        )
        histories.extend(history)
        print(
            f"d7_arm_done dataset={args.dataset} checkpoint_seed={checkpoint_seed} "
            f"arm={arm} mse_h720={validation_metrics['mse_h720']:.8f}",
            flush=True,
        )
        del trained
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metadata = {
        "dataset": args.dataset,
        "checkpoint_seed": checkpoint_seed,
        "checkpoint_dir": str(directory),
        "profile": contract["dataset_profiles"][args.dataset]["profile"],
        "memory_shape": [
            "B",
            official_args.enc_in,
            official_args.patch_num,
            official_args.d_model,
        ],
        "state_width": input_width,
        "fit_rows": int(fit["features"].shape[0]),
        "holdout_rows": int(holdout["features"].shape[0]),
        "validation_rows": int(validation["features"].shape[0]),
        "validation_batch_offset": args.val_offset_batches,
        "basis_orthogonality_max_abs": orthogonality_gap,
        "coefficient_subset_max_abs": max_coefficient_gap,
        "prefix_reconstruction_max_abs": max_prefix_gap,
        "descriptor_hashes": {
            name: descriptor_hash(value) for name, value in descriptors.items()
        },
        **descriptor_stats,
        "uses_test_split": False,
        "forecast_model_updated": False,
        "official_validation_used_for_early_stopping": False,
        "training_objective_horizon": SERIES_LENGTH,
        "evaluation_horizons": horizons,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "device": str(device),
    }
    return metrics, histories, metadata


def synthetic_smoke() -> None:
    config = {
        "compact_trunk_width": 256,
        "matched_trunk_width": 694,
        "permutation_seed": 7101,
        "random_descriptor_seed": 7102,
    }
    synthesis, atoms, _prototypes, _gap = restricted_global_nested_basis(
        SERIES_LENGTH, 16
    )
    basis = synthesis.transpose(0, 1).float()
    descriptors, stats = descriptor_families(atoms, 7101, 7102)
    if stats["random_descriptor_moment_max_abs"] > 1e-5:
        raise RuntimeError("random descriptor moment matching failed")
    expected_parameters = {
        "free_m0": 381904,
        "geo_c256": 265680,
        "perm_c256": 265680,
        "random_c256": 265680,
        "geo_m694": 381750,
        "perm_m694": 381750,
        "random_m694": 381750,
    }
    features = torch.randn(4, 768)
    for arm, expected in expected_parameters.items():
        head, _width, _family = build_arm(
            arm, 768, descriptors, config, torch.device("cpu")
        )
        coefficients = head(features)
        if coefficients.shape != (4, SERIES_LENGTH):
            raise RuntimeError(f"D7 coefficient shape failed for {arm}")
        if count_parameters(head) != expected:
            raise RuntimeError(
                f"D7 parameter count failed for {arm}: "
                f"{count_parameters(head)} != {expected}"
            )
        if arm == "geo_c256":
            target = torch.randn(4, SERIES_LENGTH)
            loss = ((coefficients @ basis) - target).square().mean()
            loss.backward()
            if not all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                for parameter in head.parameters()
            ):
                raise RuntimeError("D7 PAF gradient path failed")
        if isinstance(head, ProjectiveAtomFunctionalHead):
            coefficient_gap, prefix_gap = projectivity_gap(
                head, features, basis, atoms, [48, 96, 144, 720]
            )
            if max(coefficient_gap, prefix_gap) > 1e-5:
                raise RuntimeError(f"D7 projectivity failed for {arm}")
    if not torch.allclose(
        basis @ basis.transpose(0, 1),
        torch.eye(SERIES_LENGTH),
        atol=2e-5,
    ):
        raise RuntimeError("D7 RGNB orthogonality failed")
    print("stage_c_sc1_d7_worker_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    config = json.loads(args.d7_config.read_text(encoding="utf-8"))
    validate_config(config, args)
    contract_hash = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(args.d7_config.read_bytes()).hexdigest()
    metrics: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for checkpoint_seed in config["checkpoint_seeds"]:
        seed_metrics, seed_histories, seed_metadata = run_checkpoint_seed(
            args, contract, config, int(checkpoint_seed)
        )
        metrics.extend(seed_metrics)
        histories.extend(seed_histories)
        seed_metadata["contract_hash"] = contract_hash
        seed_metadata["d7_config_hash"] = config_hash
        metadata.append(seed_metadata)
    output_dir = args.output_dir / args.dataset
    write_csv(output_dir / "d7_probe_metrics.csv", metrics)
    write_csv(output_dir / "d7_training_history.csv", histories)
    (output_dir / "d7_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"stage_c_sc1_d7_done dataset={args.dataset} fits={len(metrics)} "
        f"output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

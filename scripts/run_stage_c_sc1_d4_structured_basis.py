#!/usr/bin/env python3
"""Run SC1-D4 structured-basis frozen-memory diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import torch

try:
    from run_stage_c_sc1_d2_diagnostic import (
        DATA_SEED,
        GROUP_HIDDEN,
        PROBE_SEED,
        SCALE_GROUP_SIZES,
        SERIES_LENGTH,
        GroupedNonlinearHead,
        balanced_interval_basis,
        collect_rows,
        count_parameters,
        load_model_and_loaders,
        random_groups,
        random_orthogonal_basis,
        set_seed,
        split_fit_holdout,
        standardize_features,
        train_head,
        write_csv,
    )
except ModuleNotFoundError:
    from scripts.run_stage_c_sc1_d2_diagnostic import (
        DATA_SEED,
        GROUP_HIDDEN,
        PROBE_SEED,
        SCALE_GROUP_SIZES,
        SERIES_LENGTH,
        GroupedNonlinearHead,
        balanced_interval_basis,
        collect_rows,
        count_parameters,
        load_model_and_loaders,
        random_groups,
        random_orthogonal_basis,
        set_seed,
        split_fit_holdout,
        standardize_features,
        train_head,
        write_csv,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-root", type=Path)
    parser.add_argument("--phase-b-root", type=Path)
    parser.add_argument("--phase-c-root", type=Path)
    parser.add_argument("--five-phase-a-root", type=Path)
    parser.add_argument("--five-phase-b-root", type=Path)
    parser.add_argument("--five-phase-c-root", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--d4-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dataset", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-batches", type=int, default=16)
    parser.add_argument("--val-batches", type=int, default=8)
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
        "d4_config",
        "output_dir",
        "dataset",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def dct2_basis(length: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float64) + 0.5
    frequencies = torch.arange(length, device=device, dtype=torch.float64).unsqueeze(1)
    basis = torch.cos(math.pi * frequencies * positions / length)
    basis[0] *= math.sqrt(1.0 / length)
    basis[1:] *= math.sqrt(2.0 / length)
    return basis.float()


def random_interval_tree_basis(length: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scaling = torch.full((length,), 1.0 / math.sqrt(length), dtype=torch.float32)
    atoms = [scaling]
    intervals = [(0, length)]
    while intervals:
        next_intervals: list[tuple[int, int]] = []
        for start, end in intervals:
            width = end - start
            if width <= 1:
                continue
            if width <= 3:
                left = int(torch.randint(1, width, (1,), generator=generator).item())
            else:
                low = max(1, math.ceil(0.25 * width))
                high = min(width - 1, math.floor(0.75 * width))
                left = int(torch.randint(low, high + 1, (1,), generator=generator).item())
            middle = start + left
            right = width - left
            atom = torch.zeros(length, dtype=torch.float32)
            atom[start:middle] = math.sqrt(right / (left * width))
            atom[middle:end] = -math.sqrt(left / (right * width))
            atoms.append(atom)
            next_intervals.extend([(start, middle), (middle, end)])
        intervals = next_intervals
    if len(atoms) != length:
        raise RuntimeError(f"random interval basis incomplete: {len(atoms)} != {length}")
    return torch.stack(atoms)


def pca_basis(covariance: torch.Tensor) -> torch.Tensor:
    _eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
    return eigenvectors.flip(1).transpose(0, 1).float()


def block_diagonal_basis(
    length: int,
    block_size: int,
    block_builder: Any,
    device: torch.device,
) -> torch.Tensor:
    if block_size <= 0 or block_size > length:
        raise ValueError(f"invalid block size: {block_size}")
    basis = torch.zeros((length, length), device=device)
    for start in range(0, length, block_size):
        end = min(start + block_size, length)
        basis[start:end, start:end] = block_builder(start, end).to(device)
    return basis


def block_dct2_basis(length: int, block_size: int, device: torch.device) -> torch.Tensor:
    return block_diagonal_basis(
        length,
        block_size,
        lambda start, end: dct2_basis(end - start, device),
        device,
    )


def block_pca_basis(
    covariance: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    length = int(covariance.shape[0])
    return block_diagonal_basis(
        length,
        block_size,
        lambda start, end: pca_basis(covariance[start:end, start:end]),
        device,
    )


def parse_block_family(family: str, prefix: str) -> int | None:
    marker = f"{prefix}_b"
    if not family.startswith(marker):
        return None
    return int(family.removeprefix(marker))


def build_basis(
    family: str,
    structure_seed: int,
    balanced: torch.Tensor,
    covariance: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, int | None]:
    if family == "balanced_interval":
        return balanced.to(device), None
    if family == "identity":
        return torch.eye(SERIES_LENGTH, device=device), None
    if family == "dct2":
        return dct2_basis(SERIES_LENGTH, device), None
    if family == "pca_fit":
        return pca_basis(covariance).to(device), None
    block_size = parse_block_family(family, "block_dct2")
    if block_size is not None:
        return block_dct2_basis(SERIES_LENGTH, block_size, device), None
    block_size = parse_block_family(family, "block_pca_fit")
    if block_size is not None:
        return block_pca_basis(covariance, block_size, device), None
    if family == "permuted_interval":
        seed = structure_seed + 10000
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(SERIES_LENGTH, generator=generator)
        return balanced[:, permutation].to(device), seed
    if family == "random_interval_tree":
        seed = structure_seed + 20000
        return random_interval_tree_basis(SERIES_LENGTH, seed).to(device), seed
    if family == "random_orthogonal":
        return random_orthogonal_basis(SERIES_LENGTH, structure_seed).to(device), structure_seed
    raise ValueError(f"unknown basis family: {family}")


def target_covariance(target: torch.Tensor, device: torch.device) -> torch.Tensor:
    values = target.to(device)
    values = values - values.mean(dim=0, keepdim=True)
    return values.transpose(0, 1) @ values / max(values.shape[0] - 1, 1)


def basis_geometry(
    basis: torch.Tensor,
    covariance: torch.Tensor,
    horizons: list[int],
) -> dict[str, Any]:
    coefficient_covariance = basis.double() @ covariance.double() @ basis.double().transpose(0, 1)
    diagonal = torch.diagonal(coefficient_covariance).clamp_min(0.0)
    off_diagonal = coefficient_covariance - torch.diag(diagonal)
    total_norm = float(torch.linalg.matrix_norm(coefficient_covariance).item())
    sorted_variance = torch.sort(diagonal, descending=True).values
    total_variance = max(float(sorted_variance.sum().item()), 1e-12)
    support = basis.detach().abs().cpu() > 1e-7
    result: dict[str, Any] = {
        "covariance_offdiag_ratio": float(torch.linalg.matrix_norm(off_diagonal).item())
        / max(total_norm, 1e-12),
        "mean_atom_support_fraction": float(support.float().mean().item()),
    }
    for count in (16, 64, 144, 256):
        result[f"variance_capture_top{count}"] = float(sorted_variance[:count].sum().item()) / total_variance
    for horizon in horizons:
        result[f"active_atoms_h{horizon}"] = int(support[:, :horizon].any(dim=1).sum().item())
    return result


def evaluate_horizons(
    head: GroupedNonlinearHead,
    basis: torch.Tensor,
    validation: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    horizons: list[int],
) -> dict[str, Any]:
    sums = {h: {"se": 0.0, "ae": 0.0, "elements": 0} for h in horizons}
    head.eval()
    with torch.no_grad():
        for start in range(0, validation["features"].shape[0], batch_size):
            end = start + batch_size
            coefficients = head(validation["features"][start:end].to(device))
            prediction = coefficients @ basis
            target = validation["target"][start:end].to(device)
            stdev = validation["stdev"][start:end].to(device)
            for horizon in horizons:
                error = (prediction[:, :horizon] - target[:, :horizon]) * stdev
                sums[horizon]["se"] += float(error.square().sum().item())
                sums[horizon]["ae"] += float(error.abs().sum().item())
                sums[horizon]["elements"] += int(error.numel())
    output: dict[str, Any] = {}
    for horizon in horizons:
        elements = sums[horizon]["elements"]
        if elements == 0:
            raise RuntimeError(f"no validation elements for horizon {horizon}")
        output[f"val_mse_eval_h{horizon}"] = sums[horizon]["se"] / elements
        output[f"val_mae_eval_h{horizon}"] = sums[horizon]["ae"] / elements
        output[f"val_elements_h{horizon}"] = elements
    return output


def validate_config(config: dict[str, Any], args: argparse.Namespace) -> None:
    probe = config["probe_contract"]
    expected = {
        "series_length": SERIES_LENGTH,
        "group_sizes": list(SCALE_GROUP_SIZES),
        "group_hidden": GROUP_HIDDEN,
        "data_seed": DATA_SEED,
        "probe_seed": PROBE_SEED,
        "train_batches": args.train_batches,
        "val_batches": args.val_batches,
        "fit_fraction": args.fit_fraction,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    if {key: probe[key] for key in expected} != expected:
        raise ValueError("runtime arguments do not match D4 preregistration")


def run_checkpoint_seed(
    args: argparse.Namespace,
    contract: dict[str, Any],
    config: dict[str, Any],
    checkpoint_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model, train_loader, val_loader, official_args, directory = load_model_and_loaders(
        args, contract, checkpoint_seed
    )
    device = official_args.device
    train_rows = collect_rows(model, train_loader, device, args.train_batches)
    validation = collect_rows(model, val_loader, device, args.val_batches)
    fit, holdout = split_fit_holdout(train_rows, args.fit_fraction)
    fit, holdout, validation = standardize_features(fit, holdout, validation)
    input_width = int(fit["features"].shape[1])
    if input_width != int(official_args.patch_num * official_args.d_model):
        raise RuntimeError("frozen memory width mismatch")
    balanced, sizes = balanced_interval_basis(SERIES_LENGTH)
    if tuple(sizes) != SCALE_GROUP_SIZES:
        raise RuntimeError("balanced interval group sizes changed")
    covariance = target_covariance(fit["target"], device)
    horizons = [int(value) for value in config["horizons"]]
    metrics: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    geometries: list[dict[str, Any]] = []
    families = [str(value) for value in config["basis_families"]]
    structure_seeds = [int(value) for value in config["structure_seeds"]]
    total_arms = len(families) * len(structure_seeds)
    arm_index = 0
    max_orthogonality_gap = 0.0
    basis_cache: dict[
        tuple[str, int | None], tuple[torch.Tensor, int | None, float, dict[str, Any]]
    ] = {}
    for family in families:
        for structure_seed in structure_seeds:
            arm_index += 1
            arm = f"{family}_g{structure_seed}"
            cache_seed = (
                structure_seed
                if family in {"permuted_interval", "random_interval_tree", "random_orthogonal"}
                else None
            )
            cache_key = (family, cache_seed)
            if cache_key not in basis_cache:
                cached_basis, cached_seed = build_basis(
                    family, structure_seed, balanced, covariance, device
                )
                cached_gap = float(
                    (
                        cached_basis @ cached_basis.transpose(0, 1)
                        - torch.eye(SERIES_LENGTH, device=device)
                    )
                    .abs()
                    .max()
                    .item()
                )
                cached_geometry = basis_geometry(
                    cached_basis, covariance, horizons
                )
                basis_cache[cache_key] = (
                    cached_basis,
                    cached_seed,
                    cached_gap,
                    cached_geometry,
                )
            basis, basis_seed, gap, geometry = basis_cache[cache_key]
            max_orthogonality_gap = max(max_orthogonality_gap, gap)
            geometries.append(
                {
                    "dataset": args.dataset,
                    "checkpoint_seed": checkpoint_seed,
                    "family": family,
                    "structure_seed": structure_seed,
                    "basis_seed": "" if basis_seed is None else basis_seed,
                    "orthogonality_max_abs": gap,
                    **geometry,
                }
            )
            print(
                f"d4_arm_start dataset={args.dataset} checkpoint_seed={checkpoint_seed} "
                f"arm={arm} position={arm_index}/{total_arms}",
                flush=True,
            )
            set_seed(PROBE_SEED)
            head = GroupedNonlinearHead(
                input_width, random_groups(SCALE_GROUP_SIZES, structure_seed)
            ).to(device)
            trained, history, best_epoch, final_fit, best_holdout = train_head(
                args,
                arm,
                head,
                "coeff",
                basis,
                fit,
                holdout,
                device,
                checkpoint_seed,
            )
            result = evaluate_horizons(
                trained, basis, validation, device, args.batch_size, horizons
            )
            metrics.append(
                {
                    "dataset": args.dataset,
                    "checkpoint_seed": checkpoint_seed,
                    "arm": arm,
                    "family": family,
                    "structure_seed": structure_seed,
                    "basis_seed": "" if basis_seed is None else basis_seed,
                    "parameters": count_parameters(trained),
                    "best_epoch": best_epoch,
                    "final_fit_mse_eval": final_fit,
                    "best_holdout_mse_eval": best_holdout,
                    **result,
                }
            )
            histories.extend(history)
            print(
                f"d4_arm_done dataset={args.dataset} checkpoint_seed={checkpoint_seed} "
                f"arm={arm} mse_h720={result['val_mse_eval_h720']:.8f}",
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
        "memory_shape": ["B", official_args.enc_in, official_args.patch_num, official_args.d_model],
        "state_width": input_width,
        "fit_rows": int(fit["features"].shape[0]),
        "holdout_rows": int(holdout["features"].shape[0]),
        "validation_rows": int(validation["features"].shape[0]),
        "basis_orthogonality_max_abs": max_orthogonality_gap,
        "uses_test_split": False,
        "forecast_model_updated": False,
        "official_validation_used_for_early_stopping": False,
        "pca_uses_fit_targets_only": True,
        "training_objective_horizon": 720,
        "evaluation_horizons": horizons,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "device": str(device),
    }
    return metrics, histories, geometries, metadata


def synthetic_smoke() -> None:
    device = torch.device("cpu")
    balanced, _sizes = balanced_interval_basis(SERIES_LENGTH)
    target = torch.randn(64, SERIES_LENGTH)
    covariance = target_covariance(target, device)
    horizons = [48, 96, 720]
    families = (
        "balanced_interval",
        "identity",
        "dct2",
        "pca_fit",
        "permuted_interval",
        "random_interval_tree",
        "random_orthogonal",
    )
    for family in families:
        basis, _seed = build_basis(family, 3101, balanced, covariance, device)
        gap = (basis @ basis.transpose(0, 1) - torch.eye(SERIES_LENGTH)).abs().max()
        if float(gap.item()) > 2e-5:
            raise RuntimeError(f"orthogonality failed for {family}: {gap}")
        geometry = basis_geometry(basis, covariance, horizons)
        if not 0.0 <= geometry["covariance_offdiag_ratio"] <= 1.0:
            raise RuntimeError(f"invalid covariance ratio for {family}")
    balanced_geometry = basis_geometry(balanced, covariance, horizons)
    dct_geometry = basis_geometry(dct2_basis(SERIES_LENGTH, device), covariance, horizons)
    if balanced_geometry["active_atoms_h48"] >= SERIES_LENGTH:
        raise RuntimeError("balanced basis lost prefix-local support")
    if dct_geometry["active_atoms_h48"] != SERIES_LENGTH:
        raise RuntimeError("DCT unexpectedly became support local")
    for block_size in (16, 48, 96, 144):
        for family in (f"block_dct2_b{block_size}", f"block_pca_fit_b{block_size}"):
            basis, _seed = build_basis(family, 3101, balanced, covariance, device)
            gap = (basis @ basis.transpose(0, 1) - torch.eye(SERIES_LENGTH)).abs().max()
            if float(gap.item()) > 2e-5:
                raise RuntimeError(f"orthogonality failed for {family}: {gap}")
            geometry = basis_geometry(basis, covariance, horizons)
            expected_active = math.ceil(48 / block_size) * block_size
            expected_active = min(expected_active, SERIES_LENGTH)
            if geometry["active_atoms_h48"] != expected_active:
                raise RuntimeError(f"unexpected H48 active set for {family}")
    print("stage_c_sc1_d4_worker_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    config = json.loads(args.d4_config.read_text(encoding="utf-8"))
    validate_config(config, args)
    metrics: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    geometries: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    contract_hash = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(args.d4_config.read_bytes()).hexdigest()
    for checkpoint_seed in config["checkpoint_seeds"]:
        seed_metrics, seed_histories, seed_geometries, seed_metadata = run_checkpoint_seed(
            args, contract, config, int(checkpoint_seed)
        )
        metrics.extend(seed_metrics)
        histories.extend(seed_histories)
        geometries.extend(seed_geometries)
        seed_metadata["contract_hash"] = contract_hash
        seed_metadata["d4_config_hash"] = config_hash
        metadata.append(seed_metadata)
    output_dir = args.output_dir / args.dataset
    write_csv(output_dir / "d4_probe_metrics.csv", metrics)
    write_csv(output_dir / "d4_training_history.csv", histories)
    write_csv(output_dir / "d4_basis_geometry.csv", geometries)
    (output_dir / "d4_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"stage_c_sc1_d4_done dataset={args.dataset} fits={len(metrics)} "
        f"output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

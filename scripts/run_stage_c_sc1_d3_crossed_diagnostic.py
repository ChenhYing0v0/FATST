#!/usr/bin/env python3
"""Fit the missing random-basis x random-group cell for SC1-D3."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch

try:
    from run_stage_c_sc1_d2_diagnostic import (
        DATA_SEED,
        GROUP_HIDDEN,
        PROBE_SEED,
        RANDOM_CONTROL_SEEDS,
        SCALE_GROUP_SIZES,
        SERIES_LENGTH,
        GroupedNonlinearHead,
        balanced_interval_basis,
        collect_rows,
        count_parameters,
        evaluate_head,
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
        RANDOM_CONTROL_SEEDS,
        SCALE_GROUP_SIZES,
        SERIES_LENGTH,
        GroupedNonlinearHead,
        balanced_interval_basis,
        collect_rows,
        count_parameters,
        evaluate_head,
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
    parser.add_argument("--d3-config", type=Path)
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
        "d3_config",
        "output_dir",
        "dataset",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


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
    observed = {key: probe[key] for key in expected}
    if observed != expected:
        raise ValueError(f"D3 runtime does not match preregistered probe contract: {observed}")
    if tuple(config["structure_seeds"]) != RANDOM_CONTROL_SEEDS:
        raise ValueError("D3 structure seeds do not match D2 controls")
    if args.dataset not in config["datasets"]:
        raise ValueError(f"dataset not preregistered for D3: {args.dataset}")


def build_crossed_arm(
    input_width: int,
    structure_seed: int,
    device: torch.device,
) -> tuple[GroupedNonlinearHead, torch.Tensor]:
    set_seed(PROBE_SEED)
    groups = random_groups(SCALE_GROUP_SIZES, structure_seed)
    basis = random_orthogonal_basis(SERIES_LENGTH, structure_seed)
    return GroupedNonlinearHead(input_width, groups).to(device), basis.to(device)


def run_checkpoint_seed(
    args: argparse.Namespace,
    contract: dict[str, Any],
    structure_seeds: list[int],
    checkpoint_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model, train_loader, val_loader, official_args, directory = load_model_and_loaders(
        args, contract, checkpoint_seed
    )
    device = official_args.device
    train_rows = collect_rows(model, train_loader, device, args.train_batches)
    validation = collect_rows(model, val_loader, device, args.val_batches)
    fit, holdout = split_fit_holdout(train_rows, args.fit_fraction)
    fit, holdout, validation = standardize_features(fit, holdout, validation)
    input_width = int(fit["features"].shape[1])
    expected_width = int(official_args.patch_num * official_args.d_model)
    if input_width != expected_width:
        raise RuntimeError(
            f"memory width mismatch: observed={input_width}, expected={expected_width}"
        )
    true_basis, observed_sizes = balanced_interval_basis(SERIES_LENGTH)
    if tuple(observed_sizes) != SCALE_GROUP_SIZES:
        raise RuntimeError(f"unexpected scale groups: {observed_sizes}")
    true_gap = float(
        (true_basis @ true_basis.transpose(0, 1) - torch.eye(SERIES_LENGTH))
        .abs()
        .max()
        .item()
    )
    metrics: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    random_gaps: dict[str, float] = {}
    for arm_index, structure_seed in enumerate(structure_seeds, start=1):
        arm = f"random_basis_random_group_s{structure_seed}"
        print(
            f"d3_arm_start dataset={args.dataset} checkpoint_seed={checkpoint_seed} "
            f"arm={arm} position={arm_index}/{len(structure_seeds)}",
            flush=True,
        )
        head, basis = build_crossed_arm(input_width, structure_seed, device)
        identity = torch.eye(SERIES_LENGTH, device=device)
        random_gaps[str(structure_seed)] = float(
            (basis @ basis.transpose(0, 1) - identity).abs().max().item()
        )
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
        result = evaluate_head(
            trained, "coeff", basis, validation, device, args.batch_size
        )
        metrics.append(
            {
                "dataset": args.dataset,
                "checkpoint_seed": checkpoint_seed,
                "arm": arm,
                "family": "random_basis_random_group",
                "structure_seed": structure_seed,
                "basis_seed": structure_seed,
                "group_seed": structure_seed,
                "output_space": "coeff",
                "parameters": count_parameters(trained),
                "best_epoch": best_epoch,
                "final_fit_mse_eval": final_fit,
                "best_holdout_mse_eval": best_holdout,
                **result,
            }
        )
        histories.extend(history)
        print(
            f"d3_arm_done dataset={args.dataset} checkpoint_seed={checkpoint_seed} "
            f"arm={arm} val_mse_eval={result['val_mse_eval']:.8f}",
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
        "data_seed": DATA_SEED,
        "probe_seed": PROBE_SEED,
        "structure_seeds": structure_seeds,
        "basis_orthogonality_max_abs": true_gap,
        "random_basis_orthogonality_max_abs": random_gaps,
        "uses_test_split": False,
        "forecast_model_updated": False,
        "official_validation_used_for_early_stopping": False,
        "diagnostic_role": "crossed_missing_cell_only",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dataset_root": official_args.root_path,
        "dataset_file": official_args.data_path,
        "device": str(device),
    }
    return metrics, histories, metadata


def synthetic_smoke() -> None:
    for width in (768, 1536, 3072):
        features = torch.randn(4, width)
        for seed in RANDOM_CONTROL_SEEDS:
            head, basis = build_crossed_arm(width, seed, torch.device("cpu"))
            prediction = head(features)
            if prediction.shape != (4, SERIES_LENGTH):
                raise RuntimeError("crossed head shape mismatch")
            if (prediction @ basis).shape != (4, SERIES_LENGTH):
                raise RuntimeError("crossed reconstruction shape mismatch")
            gap = (basis @ basis.transpose(0, 1) - torch.eye(SERIES_LENGTH)).abs().max()
            if float(gap.item()) > 1e-5:
                raise RuntimeError("random basis orthogonality changed")
    print("stage_c_sc1_d3_worker_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    config = json.loads(args.d3_config.read_text(encoding="utf-8"))
    validate_config(config, args)
    contract_hash = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(args.d3_config.read_bytes()).hexdigest()
    metrics: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    structure_seeds = [int(seed) for seed in config["structure_seeds"]]
    for checkpoint_seed in config["checkpoint_seeds"]:
        seed = int(checkpoint_seed)
        seed_metrics, seed_histories, seed_metadata = run_checkpoint_seed(
            args, contract, structure_seeds, seed
        )
        metrics.extend(seed_metrics)
        histories.extend(seed_histories)
        seed_metadata["contract_hash"] = contract_hash
        seed_metadata["d3_config_hash"] = config_hash
        metadata.append(seed_metadata)
    output_dir = args.output_dir / args.dataset
    write_csv(output_dir / "d3_probe_metrics.csv", metrics)
    write_csv(output_dir / "d3_training_history.csv", histories)
    (output_dir / "d3_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"stage_c_sc1_d3_done dataset={args.dataset} fits={len(metrics)} "
        f"output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train frozen-memory SC1-D2 diagnostic heads without updating the forecast model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    from run_stage_c_d1_offline_diagnostic import (
        Model,
        data_provider,
        load_state,
        run_dir,
        set_seed,
    )
except ModuleNotFoundError:
    from scripts.run_stage_c_d1_offline_diagnostic import (
        Model,
        data_provider,
        load_state,
        run_dir,
        set_seed,
    )


SERIES_LENGTH = 720
SCALE_GROUP_SIZES = (1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 208)
GROUP_HIDDEN = 32
DENSE_UNIT_HIDDEN = GROUP_HIDDEN * len(SCALE_GROUP_SIZES)
DATA_SEED = 20260713
PROBE_SEED = 20260714
RANDOM_CONTROL_SEEDS = (3101, 3102, 3103)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-root", type=Path)
    parser.add_argument("--phase-b-root", type=Path)
    parser.add_argument("--phase-c-root", type=Path)
    parser.add_argument("--five-phase-a-root", type=Path)
    parser.add_argument("--five-phase-b-root", type=Path)
    parser.add_argument("--five-phase-c-root", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dataset", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--diagnostic-suite", choices=["core3", "formal5"], default="core3"
    )
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
    required = {
        "phase_a_root": args.phase_a_root,
        "phase_b_root": args.phase_b_root,
        "phase_c_root": args.phase_c_root,
        "contract": args.contract,
        "output_dir": args.output_dir,
        "dataset": args.dataset,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    if not 0.5 <= args.fit_fraction < 1.0:
        parser.error("--fit-fraction must be in [0.5, 1.0)")
    for name in ("train_batches", "val_batches", "batch_size", "max_epochs", "patience"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def balanced_interval_basis(length: int) -> tuple[torch.Tensor, list[int]]:
    """Return breadth-first balanced interval atoms as rows of an orthogonal matrix."""
    if length <= 0:
        raise ValueError("length must be positive")
    scaling = torch.full((length,), 1.0 / math.sqrt(float(length)), dtype=torch.float32)
    atoms = [scaling]
    depth_counts = [1]
    intervals = [(0, length)]
    while intervals:
        next_intervals: list[tuple[int, int]] = []
        depth_atoms: list[torch.Tensor] = []
        for start, end in intervals:
            if end - start <= 1:
                continue
            middle = start + (end - start) // 2
            left = middle - start
            right = end - middle
            total = left + right
            atom = torch.zeros(length, dtype=torch.float32)
            atom[start:middle] = math.sqrt(right / (left * total))
            atom[middle:end] = -math.sqrt(left / (right * total))
            depth_atoms.append(atom)
            next_intervals.extend([(start, middle), (middle, end)])
        if depth_atoms:
            atoms.extend(depth_atoms)
            depth_counts.append(len(depth_atoms))
        intervals = next_intervals
    basis = torch.stack(atoms)
    return basis, depth_counts


def random_orthogonal_basis(length: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(length, length, generator=generator)
    basis, _ = torch.linalg.qr(matrix)
    return basis.transpose(0, 1).contiguous()


def contiguous_groups(sizes: tuple[int, ...]) -> list[torch.Tensor]:
    groups = []
    start = 0
    for size in sizes:
        groups.append(torch.arange(start, start + size, dtype=torch.long))
        start += size
    if start != SERIES_LENGTH:
        raise ValueError("group sizes do not cover the full future domain")
    return groups


def random_groups(sizes: tuple[int, ...], seed: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(SERIES_LENGTH, generator=generator)
    groups = []
    start = 0
    for size in sizes:
        groups.append(permutation[start : start + size])
        start += size
    return groups


class LowRankLinearHead(nn.Module):
    def __init__(self, input_width: int) -> None:
        super().__init__()
        self.down = nn.Linear(input_width, 256)
        self.up = nn.Linear(256, SERIES_LENGTH)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(features))


class DenseNonlinearHead(nn.Module):
    def __init__(self, input_width: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, SERIES_LENGTH),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class GroupedNonlinearHead(nn.Module):
    def __init__(self, input_width: int, groups: list[torch.Tensor]) -> None:
        super().__init__()
        self.groups = [group.clone() for group in groups]
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_width, GROUP_HIDDEN),
                    nn.GELU(),
                    nn.Linear(GROUP_HIDDEN, int(group.numel())),
                )
                for group in groups
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        coefficients = features.new_empty((features.shape[0], SERIES_LENGTH))
        for group, block in zip(self.groups, self.blocks, strict=True):
            coefficients[:, group.to(features.device)] = block(features)
        return coefficients


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def parameter_matched_dense_hidden(input_width: int) -> int:
    grouped_without_output_bias = GROUP_HIDDEN * (
        len(SCALE_GROUP_SIZES) * (input_width + 1) + SERIES_LENGTH
    )
    dense_per_hidden = input_width + SERIES_LENGTH + 1
    return max(1, round(grouped_without_output_bias / dense_per_hidden))


def build_arm(
    arm: str,
    device: torch.device,
    true_basis: torch.Tensor,
    input_width: int,
) -> tuple[nn.Module, str, torch.Tensor | None, int | None]:
    set_seed(PROBE_SEED)
    if arm == "rank256_linear":
        return LowRankLinearHead(input_width).to(device), "time", None, None
    if arm == "full_affine":
        return nn.Linear(input_width, SERIES_LENGTH).to(device), "time", None, None
    if arm in {"dense_nonlinear_param_matched", "dense_nonlinear_param_h197"}:
        hidden = parameter_matched_dense_hidden(input_width)
        return DenseNonlinearHead(input_width, hidden).to(device), "time", None, None
    if arm == "dense_nonlinear_units_h352":
        return DenseNonlinearHead(input_width, DENSE_UNIT_HIDDEN).to(device), "time", None, None
    if arm == "true_scale_grouped":
        return (
            GroupedNonlinearHead(input_width, contiguous_groups(SCALE_GROUP_SIZES)).to(device),
            "coeff",
            true_basis.to(device),
            None,
        )
    family, seed_text = arm.rsplit("_s", maxsplit=1)
    structure_seed = int(seed_text)
    if family == "random_group":
        groups = random_groups(SCALE_GROUP_SIZES, structure_seed)
        basis = true_basis
    elif family == "random_basis":
        groups = contiguous_groups(SCALE_GROUP_SIZES)
        basis = random_orthogonal_basis(SERIES_LENGTH, structure_seed)
    else:
        raise ValueError(f"unknown arm: {arm}")
    return (
        GroupedNonlinearHead(input_width, groups).to(device),
        "coeff",
        basis.to(device),
        structure_seed,
    )


def arm_names() -> list[str]:
    names = [
        "rank256_linear",
        "full_affine",
        "dense_nonlinear_param_matched",
        "dense_nonlinear_units_h352",
        "true_scale_grouped",
    ]
    for seed in RANDOM_CONTROL_SEEDS:
        names.append(f"random_group_s{seed}")
    for seed in RANDOM_CONTROL_SEEDS:
        names.append(f"random_basis_s{seed}")
    return names


def checkpoint_run_dir(
    args: argparse.Namespace,
    profile: dict[str, Any],
    checkpoint_seed: int,
) -> Path:
    if profile.get("artifact_family", "legacy_r2") != "five_extension":
        return run_dir(args, profile, checkpoint_seed)
    name = profile["profile"]
    if checkpoint_seed == 2021 and name.endswith("_medium"):
        run_name = (
            f"SC0FIVE_R2A_r2a_p{profile['patch_num']}_"
            f"d{profile['d_model']}_ff{profile['d_ff']}"
        )
        root = args.five_phase_a_root
    elif checkpoint_seed == 2021:
        run_name = f"SC0FIVE_R2B_{name}"
        root = args.five_phase_b_root
    else:
        run_name = f"SC0FIVE_R2C_{name}"
        root = args.five_phase_c_root
    if root is None:
        raise ValueError(f"missing five-dataset artifact root for {args.dataset}")
    return root / run_name / args.dataset / "h720_full" / f"seed{checkpoint_seed}"


def load_model_and_loaders(
    args: argparse.Namespace,
    contract: dict[str, Any],
    checkpoint_seed: int,
) -> tuple[Model, Any, Any, SimpleNamespace, Path]:
    profile = contract["dataset_profiles"][args.dataset]
    directory = checkpoint_run_dir(args, profile, checkpoint_seed)
    effective = json.loads((directory / "effective_config.json").read_text(encoding="utf-8"))
    payload = effective["official_args"]
    expected = {
        "patch_num": int(profile["patch_num"]),
        "d_model": int(profile["d_model"]),
        "d_ff": int(profile["d_ff"]),
    }
    observed = {key: int(payload[key]) for key in expected}
    if observed != expected:
        raise ValueError(f"frozen contract mismatch at {directory}: {observed} != {expected}")
    payload["device"] = torch.device(args.device)
    payload["use_gpu"] = args.device.startswith("cuda")
    official_args = SimpleNamespace(**payload)
    set_seed(DATA_SEED)
    _train_data, train_loader = data_provider(official_args, "train")
    set_seed(DATA_SEED + 1)
    _val_data, val_loader = data_provider(official_args, "val")
    model = Model(official_args).float().to(official_args.device)
    model.load_state_dict(load_state(directory / "checkpoint.pt"), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, train_loader, val_loader, official_args, directory


def collect_rows(
    model: Model,
    loader: Any,
    device: torch.device,
    max_batches: int,
    start_batch: int = 0,
) -> dict[str, torch.Tensor]:
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    means: list[torch.Tensor] = []
    stdevs: list[torch.Tensor] = []
    sample_ids: list[torch.Tensor] = []
    sample_cursor = 0
    with torch.no_grad():
        for batch_index, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader):
            if batch_index < start_batch:
                continue
            if batch_index >= start_batch + max_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            mean = batch_x.mean(dim=1, keepdim=True).detach()
            stdev = torch.sqrt(batch_x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            normalized_history = (batch_x - mean) / stdev
            memory = model._encode_normalized_history(normalized_history)
            target = batch_y[:, -SERIES_LENGTH:, :]
            normalized_target = (target - mean) / stdev
            batch, channels, _patches, _width = memory.shape
            features.append(memory.reshape(batch * channels, -1).cpu())
            targets.append(
                normalized_target.permute(0, 2, 1).reshape(batch * channels, SERIES_LENGTH).cpu()
            )
            means.append(mean.permute(0, 2, 1).reshape(batch * channels, 1).cpu())
            stdevs.append(stdev.permute(0, 2, 1).reshape(batch * channels, 1).cpu())
            ids = torch.arange(sample_cursor, sample_cursor + batch).repeat_interleave(channels)
            sample_ids.append(ids)
            sample_cursor += batch
    if not features:
        raise RuntimeError("no rows collected from split")
    return {
        "features": torch.cat(features),
        "target": torch.cat(targets),
        "mean": torch.cat(means),
        "stdev": torch.cat(stdevs),
        "sample_id": torch.cat(sample_ids),
    }


def split_fit_holdout(
    rows: dict[str, torch.Tensor],
    fit_fraction: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    unique_ids = torch.unique(rows["sample_id"])
    generator = torch.Generator(device="cpu").manual_seed(DATA_SEED + 2)
    shuffled = unique_ids[torch.randperm(unique_ids.numel(), generator=generator)]
    fit_count = max(1, int(math.floor(shuffled.numel() * fit_fraction)))
    fit_ids = shuffled[:fit_count]
    fit_mask = torch.isin(rows["sample_id"], fit_ids)
    if fit_mask.all() or not fit_mask.any():
        raise RuntimeError("invalid fit/holdout split")
    keys = ("features", "target", "mean", "stdev", "sample_id")
    fit = {key: rows[key][fit_mask] for key in keys}
    holdout = {key: rows[key][~fit_mask] for key in keys}
    return fit, holdout


def standardize_features(
    fit: dict[str, torch.Tensor],
    holdout: dict[str, torch.Tensor],
    validation: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    mean = fit["features"].mean(dim=0, keepdim=True)
    stdev = fit["features"].std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    outputs = []
    for rows in (fit, holdout, validation):
        current = dict(rows)
        current["features"] = (rows["features"] - mean) / stdev
        outputs.append(current)
    return outputs[0], outputs[1], outputs[2]


def target_in_space(
    target: torch.Tensor,
    output_space: str,
    basis: torch.Tensor | None,
) -> torch.Tensor:
    if output_space == "time":
        return target
    if basis is None:
        raise ValueError("coefficient output requires a basis")
    return target.to(basis.device) @ basis.transpose(0, 1)


def weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    stdev: torch.Tensor,
) -> torch.Tensor:
    return ((prediction - target) * stdev).square().mean()


def train_head(
    args: argparse.Namespace,
    arm: str,
    head: nn.Module,
    output_space: str,
    basis: torch.Tensor | None,
    fit: dict[str, torch.Tensor],
    holdout: dict[str, torch.Tensor],
    device: torch.device,
    checkpoint_seed: int,
    log_prefix: str = "d2",
) -> tuple[nn.Module, list[dict[str, Any]], int, float, float]:
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    fit_target = target_in_space(fit["target"], output_space, basis).cpu()
    holdout_target = target_in_space(holdout["target"], output_space, basis).to(device)
    best_state = deepcopy(head.state_dict())
    best_holdout = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.max_epochs + 1):
        head.train()
        generator = torch.Generator(device="cpu").manual_seed(PROBE_SEED + epoch)
        order = torch.randperm(fit["features"].shape[0], generator=generator)
        fit_loss_sum = 0.0
        fit_elements = 0
        for start in range(0, order.numel(), args.batch_size):
            indices = order[start : start + args.batch_size]
            features = fit["features"][indices].to(device)
            target = fit_target[indices].to(device)
            stdev = fit["stdev"][indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = head(features)
            loss = weighted_mse(prediction, target, stdev)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite fit loss for {arm}")
            loss.backward()
            optimizer.step()
            elements = int(target.numel())
            fit_loss_sum += float(loss.item()) * elements
            fit_elements += elements
        fit_loss = fit_loss_sum / fit_elements
        head.eval()
        with torch.no_grad():
            holdout_prediction = head(holdout["features"].to(device))
            holdout_loss = float(
                weighted_mse(
                    holdout_prediction,
                    holdout_target,
                    holdout["stdev"].to(device),
                ).item()
            )
        history.append(
            {
                "dataset": args.dataset,
                "checkpoint_seed": checkpoint_seed,
                "arm": arm,
                "epoch": epoch,
                "fit_mse_eval": fit_loss,
                "holdout_mse_eval": holdout_loss,
            }
        )
        if holdout_loss < best_holdout * (1.0 - 1e-4):
            best_holdout = holdout_loss
            best_epoch = epoch
            best_state = deepcopy(head.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{log_prefix}_epoch dataset={args.dataset} checkpoint_seed={checkpoint_seed} "
                f"arm={arm} epoch={epoch}/{args.max_epochs} "
                f"holdout_mse={holdout_loss:.8f}",
                flush=True,
            )
        if stale_epochs >= args.patience:
            break
    head.load_state_dict(best_state)
    return head, history, best_epoch, history[-1]["fit_mse_eval"], best_holdout


def evaluate_head(
    head: nn.Module,
    output_space: str,
    basis: torch.Tensor | None,
    validation: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    sums = {"norm_se": 0.0, "norm_ae": 0.0, "eval_se": 0.0, "eval_ae": 0.0}
    elements = 0
    head.eval()
    with torch.no_grad():
        for start in range(0, validation["features"].shape[0], batch_size):
            end = start + batch_size
            prediction = head(validation["features"][start:end].to(device))
            if output_space == "coeff":
                if basis is None:
                    raise ValueError("coefficient output requires a basis")
                prediction = prediction @ basis
            target = validation["target"][start:end].to(device)
            mean = validation["mean"][start:end].to(device)
            stdev = validation["stdev"][start:end].to(device)
            error_norm = prediction - target
            prediction_eval = prediction * stdev + mean
            target_eval = target * stdev + mean
            error_eval = prediction_eval - target_eval
            sums["norm_se"] += float(error_norm.square().sum().item())
            sums["norm_ae"] += float(error_norm.abs().sum().item())
            sums["eval_se"] += float(error_eval.square().sum().item())
            sums["eval_ae"] += float(error_eval.abs().sum().item())
            elements += int(target.numel())
    if elements == 0:
        raise RuntimeError("validation evaluation produced no elements")
    return {
        "val_mse_norm": sums["norm_se"] / elements,
        "val_mae_norm": sums["norm_ae"] / elements,
        "val_mse_eval": sums["eval_se"] / elements,
        "val_mae_eval": sums["eval_ae"] / elements,
        "val_elements": elements,
    }


def run_checkpoint_seed(
    args: argparse.Namespace,
    contract: dict[str, Any],
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
    orthogonality_gap = float(
        (true_basis @ true_basis.transpose(0, 1) - torch.eye(SERIES_LENGTH)).abs().max().item()
    )
    parseval_probe = validation["target"][:8]
    coefficients = parseval_probe @ true_basis.transpose(0, 1)
    parseval_gap = float(
        (parseval_probe.square().sum() - coefficients.square().sum()).abs().item()
        / max(float(parseval_probe.square().sum().item()), 1e-12)
    )
    metric_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(arm_names(), start=1):
        print(
            f"d2_arm_start dataset={args.dataset} checkpoint_seed={checkpoint_seed} "
            f"arm={arm} position={arm_index}/{len(arm_names())}",
            flush=True,
        )
        head, output_space, basis, structure_seed = build_arm(
            arm, device, true_basis, input_width
        )
        parameters = count_parameters(head)
        trained, history, best_epoch, final_fit, best_holdout = train_head(
            args,
            arm,
            head,
            output_space,
            basis,
            fit,
            holdout,
            device,
            checkpoint_seed,
        )
        metrics = evaluate_head(
            trained,
            output_space,
            basis,
            validation,
            device,
            args.batch_size,
        )
        family = (
            "random_group"
            if arm.startswith("random_group")
            else "random_basis"
            if arm.startswith("random_basis")
            else arm
        )
        metric_rows.append(
            {
                "dataset": args.dataset,
                "checkpoint_seed": checkpoint_seed,
                "arm": arm,
                "family": family,
                "structure_seed": "" if structure_seed is None else structure_seed,
                "output_space": output_space,
                "parameters": parameters,
                "best_epoch": best_epoch,
                "final_fit_mse_eval": final_fit,
                "best_holdout_mse_eval": best_holdout,
                **metrics,
            }
        )
        history_rows.extend(history)
        print(
            f"d2_arm_done dataset={args.dataset} checkpoint_seed={checkpoint_seed} "
            f"arm={arm} val_mse_eval={metrics['val_mse_eval']:.8f}",
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
        "train_batches": args.train_batches,
        "val_batches": args.val_batches,
        "data_seed": DATA_SEED,
        "probe_seed": PROBE_SEED,
        "random_control_seeds": list(RANDOM_CONTROL_SEEDS),
        "scale_group_sizes": list(SCALE_GROUP_SIZES),
        "group_hidden": GROUP_HIDDEN,
        "dense_param_hidden": parameter_matched_dense_hidden(input_width),
        "dense_unit_hidden": DENSE_UNIT_HIDDEN,
        "basis_orthogonality_max_abs": orthogonality_gap,
        "parseval_relative_gap": parseval_gap,
        "uses_test_split": False,
        "forecast_model_updated": False,
        "official_validation_used_for_early_stopping": False,
        "diagnostic_role": (
            "formal5_problem_gate"
            if args.diagnostic_suite == "formal5"
            else "core3_precheck_nonblocking"
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dataset_root": official_args.root_path,
        "dataset_file": official_args.data_path,
        "device": str(device),
    }
    return metric_rows, history_rows, metadata


def synthetic_smoke() -> None:
    basis, sizes = balanced_interval_basis(SERIES_LENGTH)
    if tuple(sizes) != SCALE_GROUP_SIZES:
        raise RuntimeError("balanced interval group sizes changed")
    identity = basis @ basis.transpose(0, 1)
    if not torch.allclose(identity, torch.eye(SERIES_LENGTH), atol=1e-5):
        raise RuntimeError("balanced interval basis is not orthogonal")
    target = torch.randn(4, SERIES_LENGTH)
    coefficient_target = target @ basis.transpose(0, 1)
    reconstruction = coefficient_target @ basis
    if not torch.allclose(target, reconstruction, atol=1e-4):
        raise RuntimeError("coefficient reconstruction failed")
    expected_768 = {
        "rank256_linear": 381904,
        "full_affine": 553680,
        "dense_nonlinear_param_matched": 294053,
        "dense_nonlinear_units_h352": 524848,
        "true_scale_grouped": 294448,
    }
    for input_width in (768, 1536, 3072):
        features = torch.randn(4, input_width)
        for arm in expected_768:
            head, output_space, _basis, _seed = build_arm(
                arm, torch.device("cpu"), basis, input_width
            )
            if head(features).shape != (4, SERIES_LENGTH):
                raise RuntimeError(f"shape check failed for width={input_width} {arm}")
            if input_width == 768 and count_parameters(head) != expected_768[arm]:
                raise RuntimeError(f"parameter count changed for {arm}")
            if output_space not in {"time", "coeff"}:
                raise RuntimeError(f"invalid output space for {arm}")
        true_head, _space, _basis, _seed = build_arm(
            "true_scale_grouped", torch.device("cpu"), basis, input_width
        )
        matched_head, _space, _basis, _seed = build_arm(
            "dense_nonlinear_param_matched", torch.device("cpu"), basis, input_width
        )
        relative_gap = abs(count_parameters(true_head) - count_parameters(matched_head))
        relative_gap /= count_parameters(true_head)
        if relative_gap > 0.005:
            raise RuntimeError(f"parameter matching failed for width={input_width}")
        random_head, _space, _basis, _seed = build_arm(
            "random_group_s3101", torch.device("cpu"), basis, input_width
        )
        if random_head(features).shape != (4, SERIES_LENGTH):
            raise RuntimeError("random group shape check failed")
    print("stage_c_sc1_d2_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    contract_hash = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    metrics: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for checkpoint_seed in contract["global_fields"]["default_seeds"]:
        seed = int(checkpoint_seed)
        print(f"d2_seed_start dataset={args.dataset} checkpoint_seed={seed}", flush=True)
        seed_metrics, seed_history, seed_metadata = run_checkpoint_seed(args, contract, seed)
        metrics.extend(seed_metrics)
        history.extend(seed_history)
        seed_metadata["contract_hash"] = contract_hash
        metadata.append(seed_metadata)
        print(f"d2_seed_done dataset={args.dataset} checkpoint_seed={seed}", flush=True)
    output_dir = args.output_dir / args.dataset
    write_csv(output_dir / "d2_probe_metrics.csv", metrics)
    write_csv(output_dir / "d2_training_history.csv", history)
    (output_dir / "d2_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"stage_c_sc1_d2_done dataset={args.dataset} seeds={len(metadata)} "
        f"output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

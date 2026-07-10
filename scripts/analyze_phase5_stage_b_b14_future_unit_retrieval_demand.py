from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from data_provider.data_factory import data_provider  # noqa: E402
from models.TimeAlign import Model  # noqa: E402
from train_repo import OFFICIAL_PRESETS, build_official_args  # noqa: E402


PRED_LEN = 720
HORIZONS = (96, 192, 336, 720)
PATCH_LEN = 48
PATCH_STRIDE = 24
UNIT_SIZES = (180, 240)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def adapter_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        seq_len=PRED_LEN,
        label_len=48,
        pred_len=PRED_LEN,
        e_layers=2,
        num_workers=0,
        epochs=10,
        batch_size=args.batch_size,
        patience=3,
        use_amp=False,
        seed=args.seed,
        device=args.device,
        readout_mode="learned-basis-forecast-operator",
        encoder_mode="hierarchical-patch-memory",
        history_patch_len=PATCH_LEN,
        history_patch_stride=PATCH_STRIDE,
        history_d_model=128,
        history_n_heads=16,
        history_d_ff=256,
        history_e_layers=3,
        history_dropout=0.2,
        history_attn_dropout=0.0,
        history_res_attention=True,
        learning_rate=None,
        w_align=None,
        w_recon=0.0,
        target_horizons=list(HORIZONS),
        basis_rank=256,
        stage_token_dim=32,
        stage_field_rank=32,
        stage_gate_init=-5.0,
        basis_field_window_len=96,
        basis_field_stride=48,
        basis_field_rank=32,
        basis_field_tau=1.0,
        basis_field_gate_init=-5.0,
        stbo_tile_len=48,
        stbo_rank=16,
        stbo_bank_count=4,
        stbo_basis_init_std=16**-0.5,
    )


def build_args(args: argparse.Namespace) -> argparse.Namespace:
    preset = OFFICIAL_PRESETS[args.dataset][PRED_LEN]
    result = build_official_args(adapter_args(args), preset)
    result.batch_size = args.batch_size
    result.num_workers = 0
    return result


def load_model(config: argparse.Namespace, checkpoint: Path) -> Model:
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = Model(config)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.eval()


def patch_starts() -> list[int]:
    return list(range(0, PRED_LEN - PATCH_LEN + 1, PATCH_STRIDE))


def patch_coverage(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coverage = torch.zeros(PRED_LEN, device=device, dtype=dtype)
    for start in patch_starts():
        coverage[start : start + PATCH_LEN] += 1
    if bool(torch.any(coverage == 0)):
        raise RuntimeError("valid patch supports do not cover the full history")
    return coverage


def aggregate_patch_profile(profile: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Map a non-negative position profile to patches without overlap duplication."""
    if profile.shape != (PRED_LEN,):
        raise ValueError(f"expected [{PRED_LEN}] profile, got {list(profile.shape)}")
    coverage = patch_coverage(profile.device, profile.dtype)
    allocated = profile / coverage
    patch = torch.stack(
        [allocated[start : start + PATCH_LEN].sum() for start in patch_starts()]
    )
    mass_error = float((patch.sum() - profile.sum()).abs().detach().cpu())
    total = patch.sum()
    if not torch.isfinite(total) or float(total.detach().cpu()) <= 1e-12:
        raise RuntimeError("zero or non-finite patch profile")
    return patch / total, mass_error


def reconstruct_history(memory: torch.Tensor) -> torch.Tensor:
    reconstruction = memory.new_zeros(memory.shape[0], memory.shape[1], PRED_LEN)
    coverage = patch_coverage(memory.device, memory.dtype)
    for patch_idx, start in enumerate(patch_starts()):
        reconstruction[..., start : start + PATCH_LEN] += memory[..., patch_idx, :]
    return reconstruction / coverage.view(1, 1, -1)


def dct_basis(
    length: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if rank <= 0 or rank > length:
        raise ValueError(f"DCT rank must be in [1, {length}]")
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    frequencies = torch.arange(rank, device=device, dtype=dtype).unsqueeze(0)
    basis = torch.cos(math.pi / length * (positions + 0.5) * frequencies)
    basis[:, 0] *= math.sqrt(1.0 / length)
    if rank > 1:
        basis[:, 1:] *= math.sqrt(2.0 / length)
    return basis


def patch_descriptors(memory: torch.Tensor, rank: int) -> torch.Tensor:
    basis = dct_basis(
        PATCH_LEN,
        rank,
        memory.device,
        memory.dtype,
    )
    descriptors = torch.einsum("bcpk,kq->bcpq", memory, basis)
    return descriptors.reshape(-1, memory.shape[2], rank)


def label_dependence_profile(
    history_descriptors: torch.Tensor,
    target_unit: torch.Tensor,
    rank: int,
    shuffle_draws: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    """Return a linear-CKA patch profile and its shuffle-control gap."""
    target_basis = dct_basis(
        target_unit.shape[1],
        rank,
        target_unit.device,
        target_unit.dtype,
    )
    target_descriptors = torch.einsum(
        "buc,uq->bcq",
        target_unit,
        target_basis,
    ).reshape(-1, rank)
    history = history_descriptors - history_descriptors.mean(dim=0, keepdim=True)
    target = target_descriptors - target_descriptors.mean(dim=0, keepdim=True)
    cross = torch.einsum("npq,nr->pqr", history, target)
    history_gram = torch.einsum("npq,npr->pqr", history, history)
    target_gram = target.T @ target
    numerator = cross.square().sum(dim=(1, 2))
    denominator = torch.sqrt(history_gram.square().sum(dim=(1, 2)))
    denominator = denominator * torch.sqrt(target_gram.square().sum())
    cka = numerator / denominator.clamp_min(torch.finfo(history.dtype).eps)

    shuffled = torch.zeros_like(cka)
    for _draw in range(shuffle_draws):
        permutation = torch.randperm(
            target.shape[0],
            device=target.device,
            generator=generator,
        )
        shuffled_cross = torch.einsum(
            "npq,nr->pqr",
            history,
            target[permutation],
        )
        shuffled += (
            shuffled_cross.square().sum(dim=(1, 2))
            / denominator.clamp_min(torch.finfo(history.dtype).eps)
        )
    shuffled = shuffled / shuffle_draws
    total = cka.sum()
    if not torch.isfinite(total) or float(total.detach().cpu()) <= 1e-12:
        raise RuntimeError("label-patch dependence profile is zero or non-finite")
    shuffle_gap = float((cka.mean() - shuffled.mean()).detach().cpu())
    return cka / total, shuffle_gap


def normalized_forward(
    model: Model,
    batch_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = model.normalization_x(batch_x, "norm").detach().requires_grad_(True)
    memory = model._encode_normalized_history(normalized)
    hidden = memory.flatten(start_dim=-2)
    coeff = model.learned_basis_coeff(hidden)
    prediction_norm = torch.einsum(
        "hk,bck->bch",
        model.learned_temporal_basis,
        coeff,
    ) + model.learned_temporal_bias.view(1, 1, -1)
    prediction = model.normalization_x(prediction_norm.permute(0, 2, 1), "denorm")
    return prediction, coeff, normalized


def audit_patch_evidence(
    model: Model,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_idx: int,
    dataset: str,
) -> dict[str, Any]:
    with torch.no_grad():
        normalized = model.normalization_x(batch_x, "norm")
        memory = model.encode_retrieval_memory(batch_x)
        manual = torch.stack(
            [
                normalized[:, start : start + PATCH_LEN, :].permute(0, 2, 1)
                for start in patch_starts()
            ],
            dim=2,
        )
        reconstruction = reconstruct_history(memory)
        direct_output = model(
            batch_x,
            batch_y,
            is_training=False,
            target_prefix=PRED_LEN,
        )[0]
        rebuilt_output, _coeff, _normalized = normalized_forward(model, batch_x)
    return {
        "dataset": dataset,
        "batch_idx": batch_idx,
        "patch_count": len(patch_starts()),
        "patch_len": PATCH_LEN,
        "patch_stride": PATCH_STRIDE,
        "first_patch_start": patch_starts()[0],
        "last_patch_start": patch_starts()[-1],
        "min_position_coverage": int(
            patch_coverage(batch_x.device, batch_x.dtype).min().item()
        ),
        "max_position_coverage": int(
            patch_coverage(batch_x.device, batch_x.dtype).max().item()
        ),
        "manual_patch_max_abs_diff": float((memory - manual).abs().max().cpu()),
        "reconstruction_max_abs_diff": float(
            (reconstruction - normalized.permute(0, 2, 1)).abs().max().cpu()
        ),
        "forecast_max_abs_diff": float((direct_output - rebuilt_output).abs().max().cpu()),
    }


def normalize_position_profile(profile: torch.Tensor) -> torch.Tensor:
    total = profile.sum()
    if not torch.isfinite(total) or float(total.detach().cpu()) <= 1e-12:
        raise RuntimeError("zero or non-finite position profile")
    return profile / total


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    value = F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item()
    return float(value)


def jensen_shannon(left: torch.Tensor, right: torch.Tensor) -> float:
    eps = torch.finfo(left.dtype).eps
    left = left.clamp_min(eps)
    right = right.clamp_min(eps)
    middle = 0.5 * (left + right)
    value = 0.5 * torch.sum(left * torch.log(left / middle))
    value += 0.5 * torch.sum(right * torch.log(right / middle))
    return float(value.detach().cpu())


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = rankdata(x[mask])
    y = rankdata(y[mask])
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else float("nan")


def profile_entropy(profile: torch.Tensor) -> float:
    eps = torch.finfo(profile.dtype).eps
    value = -torch.sum(profile.clamp_min(eps) * torch.log(profile.clamp_min(eps)))
    return float(value.cpu()) / math.log(profile.numel())


def profile_centroid(profile: torch.Tensor) -> float:
    centers = torch.tensor(
        [start + 0.5 * (PATCH_LEN - 1) for start in patch_starts()],
        device=profile.device,
        dtype=profile.dtype,
    )
    return float(torch.sum(profile * centers).cpu()) / (PRED_LEN - 1)


def unit_ranges(unit_size: int) -> list[tuple[int, int, int]]:
    if PRED_LEN % unit_size != 0:
        raise ValueError(f"unit size must divide {PRED_LEN}: {unit_size}")
    return [
        (unit_idx, start, start + unit_size)
        for unit_idx, start in enumerate(range(0, PRED_LEN, unit_size))
    ]


def profile_rows(
    dataset: str,
    batch_idx: int,
    unit_size: int,
    unit_idx: int,
    profile_type: str,
    profile: torch.Tensor,
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "batch_idx": batch_idx,
            "unit_size": unit_size,
            "unit_idx": unit_idx,
            "profile_type": profile_type,
            "patch_idx": patch_idx,
            "patch_start": start,
            "patch_end_exclusive": start + PATCH_LEN,
            "profile_mass": float(profile[patch_idx].cpu()),
        }
        for patch_idx, start in enumerate(patch_starts())
    ]


def pair_kind(distance: int, unit_count: int) -> str:
    if distance == 1:
        return "adjacent"
    if distance >= math.ceil(unit_count / 2):
        return "far"
    return "middle"


def analyze_setting(
    args: argparse.Namespace,
    dataset: str,
    batch_idx: int,
    unit_size: int,
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_normalized: torch.Tensor,
    history_descriptors: torch.Tensor,
    coeff: torch.Tensor,
    normalized: torch.Tensor,
    generator: torch.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], float]:
    demands: list[torch.Tensor] = []
    sensitivities: list[torch.Tensor] = []
    label_profiles: list[torch.Tensor] = []
    label_shuffle_gaps: list[float] = []
    coeff_grads: list[torch.Tensor] = []
    losses: list[float] = []
    profiles: list[dict[str, Any]] = []
    max_mass_error = 0.0
    ranges = unit_ranges(unit_size)
    random_patterns = [
        torch.empty(
            prediction.shape[0],
            unit_size,
            prediction.shape[2],
            device=prediction.device,
            dtype=prediction.dtype,
        ).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
        for _ in range(args.hutchinson_draws)
    ]

    for unit_idx, start, end in ranges:
        label_profile, label_shuffle_gap = label_dependence_profile(
            history_descriptors,
            target_normalized[:, start:end, :],
            args.descriptor_rank,
            args.cka_shuffle_draws,
            generator,
        )
        label_profiles.append(label_profile.detach().cpu())
        label_shuffle_gaps.append(label_shuffle_gap)
        loss = F.mse_loss(prediction[:, start:end, :], target[:, start:end, :])
        demand_grad, coeff_grad = torch.autograd.grad(
            loss,
            (normalized, coeff),
            retain_graph=True,
        )
        demand_position = normalize_position_profile(
            demand_grad.abs().mean(dim=(0, 2))
        )
        demand_patch, mass_error = aggregate_patch_profile(demand_position)
        max_mass_error = max(max_mass_error, mass_error)
        demands.append(demand_patch.detach().cpu())
        coeff_grads.append(coeff_grad.detach().cpu())
        losses.append(float(loss.detach().cpu()))

        sensitivity_square = torch.zeros(
            PRED_LEN,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        for pattern in random_patterns:
            scalar = torch.sum(prediction[:, start:end, :] * pattern)
            sensitivity_grad = torch.autograd.grad(
                scalar,
                normalized,
                retain_graph=True,
            )[0]
            sensitivity_square += sensitivity_grad.square().mean(dim=(0, 2))
        sensitivity_position = normalize_position_profile(
            torch.sqrt(sensitivity_square / args.hutchinson_draws)
        )
        sensitivity_patch, mass_error = aggregate_patch_profile(sensitivity_position)
        max_mass_error = max(max_mass_error, mass_error)
        sensitivities.append(sensitivity_patch.detach().cpu())

        profiles.extend(
            profile_rows(
                dataset,
                batch_idx,
                unit_size,
                unit_idx,
                "error_conditioned_demand",
                demands[-1],
            )
        )
        profiles.extend(
            profile_rows(
                dataset,
                batch_idx,
                unit_size,
                unit_idx,
                "label_patch_dependence",
                label_profiles[-1],
            )
        )
        profiles.extend(
            profile_rows(
                dataset,
                batch_idx,
                unit_size,
                unit_idx,
                "target_independent_sensitivity",
                sensitivities[-1],
            )
        )

    pair_rows: list[dict[str, Any]] = []
    demand_cosines: list[float] = []
    sensitivity_cosines: list[float] = []
    demand_js_values: list[float] = []
    sensitivity_js_values: list[float] = []
    label_cosines: list[float] = []
    label_js_values: list[float] = []
    coeff_cosines: list[float] = []
    pair_distances: list[float] = []
    for left in range(len(ranges)):
        for right in range(left + 1, len(ranges)):
            distance = right - left
            demand_cos = cosine(demands[left], demands[right])
            sensitivity_cos = cosine(sensitivities[left], sensitivities[right])
            demand_js = jensen_shannon(demands[left], demands[right])
            sensitivity_js = jensen_shannon(sensitivities[left], sensitivities[right])
            label_cos = cosine(label_profiles[left], label_profiles[right])
            label_js = jensen_shannon(label_profiles[left], label_profiles[right])
            coeff_cos = cosine(coeff_grads[left], coeff_grads[right])
            pair_rows.append(
                {
                    "dataset": dataset,
                    "batch_idx": batch_idx,
                    "unit_size": unit_size,
                    "unit_count": len(ranges),
                    "left_unit": left,
                    "right_unit": right,
                    "unit_distance": distance,
                    "pair_kind": pair_kind(distance, len(ranges)),
                    "demand_cosine": demand_cos,
                    "sensitivity_cosine": sensitivity_cos,
                    "delta_cosine": sensitivity_cos - demand_cos,
                    "demand_js": demand_js,
                    "sensitivity_js": sensitivity_js,
                    "delta_js": demand_js - sensitivity_js,
                    "label_dependence_cosine": label_cos,
                    "label_dependence_js": label_js,
                    "delta_label_cosine": sensitivity_cos - label_cos,
                    "delta_label_js": label_js - sensitivity_js,
                    "coeff_gradient_cosine": coeff_cos,
                }
            )
            demand_cosines.append(demand_cos)
            sensitivity_cosines.append(sensitivity_cos)
            demand_js_values.append(demand_js)
            sensitivity_js_values.append(sensitivity_js)
            label_cosines.append(label_cos)
            label_js_values.append(label_js)
            coeff_cosines.append(coeff_cos)
            pair_distances.append(float(distance))

    unit_alignment = [
        cosine(demand, sensitivity)
        for demand, sensitivity in zip(demands, sensitivities)
    ]
    label_sensitivity_alignment = [
        cosine(label, sensitivity)
        for label, sensitivity in zip(label_profiles, sensitivities)
    ]
    demand_entropies = [profile_entropy(profile) for profile in demands]
    sensitivity_entropies = [profile_entropy(profile) for profile in sensitivities]
    demand_centroids = [profile_centroid(profile) for profile in demands]
    sensitivity_centroids = [profile_centroid(profile) for profile in sensitivities]
    batch_row = {
        "dataset": dataset,
        "batch_idx": batch_idx,
        "unit_size": unit_size,
        "unit_count": len(ranges),
        "loss_mean": float(np.mean(losses)),
        "mean_demand_cosine": float(np.mean(demand_cosines)),
        "mean_sensitivity_cosine": float(np.mean(sensitivity_cosines)),
        "delta_cosine": float(np.mean(sensitivity_cosines) - np.mean(demand_cosines)),
        "mean_demand_js": float(np.mean(demand_js_values)),
        "mean_sensitivity_js": float(np.mean(sensitivity_js_values)),
        "delta_js": float(np.mean(demand_js_values) - np.mean(sensitivity_js_values)),
        "mean_label_dependence_cosine": float(np.mean(label_cosines)),
        "mean_label_dependence_js": float(np.mean(label_js_values)),
        "mean_label_shuffle_gap": float(np.mean(label_shuffle_gaps)),
        "delta_label_cosine": float(
            np.mean(sensitivity_cosines) - np.mean(label_cosines)
        ),
        "delta_label_js": float(
            np.mean(label_js_values) - np.mean(sensitivity_js_values)
        ),
        "pair_matrix_spearman": spearman(demand_cosines, sensitivity_cosines),
        "demand_distance_spearman": spearman(pair_distances, demand_cosines),
        "sensitivity_distance_spearman": spearman(pair_distances, sensitivity_cosines),
        "mean_unit_demand_sensitivity_cosine": float(np.mean(unit_alignment)),
        "mean_unit_label_sensitivity_cosine": float(
            np.mean(label_sensitivity_alignment)
        ),
        "mean_coeff_gradient_cosine": float(np.mean(coeff_cosines)),
        "mean_demand_entropy": float(np.mean(demand_entropies)),
        "mean_sensitivity_entropy": float(np.mean(sensitivity_entropies)),
        "mean_demand_centroid": float(np.mean(demand_centroids)),
        "mean_sensitivity_centroid": float(np.mean(sensitivity_centroids)),
        "max_patch_mass_conservation_error": max_mass_error,
    }
    return batch_row, pair_rows, profiles, max_mass_error


def bootstrap_setting(
    dataset: str,
    unit_size: int,
    rows: list[dict[str, Any]],
    iterations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    output_rows: list[dict[str, Any]] = []
    by_metric: dict[str, dict[str, float]] = {}
    for metric in (
        "delta_cosine",
        "delta_js",
        "delta_label_cosine",
        "delta_label_js",
        "mean_label_shuffle_gap",
        "mean_sensitivity_cosine",
    ):
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            stats = {key: float("nan") for key in ("mean", "p05", "p50", "p95")}
        else:
            estimates = np.empty(iterations, dtype=np.float64)
            for idx in range(iterations):
                estimates[idx] = rng.choice(values, size=len(values), replace=True).mean()
            stats = {
                "mean": float(values.mean()),
                "p05": float(np.quantile(estimates, 0.05)),
                "p50": float(np.quantile(estimates, 0.50)),
                "p95": float(np.quantile(estimates, 0.95)),
            }
        by_metric[metric] = stats
        output_rows.append(
            {
                "dataset": dataset,
                "unit_size": unit_size,
                "metric": metric,
                "batches": len(rows),
                "bootstrap_iterations": iterations,
                **stats,
            }
        )
    return output_rows, by_metric


def summarize(
    dataset: str,
    unit_size: int,
    rows: list[dict[str, Any]],
    bootstrap: dict[str, dict[str, float]],
    audit_pass: bool,
) -> dict[str, Any]:
    finite = all(
        np.isfinite(float(row[field]))
        for row in rows
        for field in (
            "delta_cosine",
            "delta_js",
            "delta_label_cosine",
            "delta_label_js",
            "mean_label_shuffle_gap",
            "mean_sensitivity_cosine",
        )
    )
    support = bool(
        audit_pass
        and finite
        and bootstrap["delta_label_cosine"]["p05"] > 0.05
        and bootstrap["delta_label_js"]["p05"] > 0.01
        and bootstrap["mean_label_shuffle_gap"]["p05"] > 0.0
        and bootstrap["mean_sensitivity_cosine"]["mean"] >= 0.80
    )
    return {
        "dataset": dataset,
        "unit_size": unit_size,
        "unit_count": PRED_LEN // unit_size,
        "batches": len(rows),
        "audit_pass": audit_pass,
        "finite_profiles": finite,
        "delta_cosine_mean": bootstrap["delta_cosine"]["mean"],
        "delta_cosine_p05": bootstrap["delta_cosine"]["p05"],
        "delta_js_mean": bootstrap["delta_js"]["mean"],
        "delta_js_p05": bootstrap["delta_js"]["p05"],
        "delta_label_cosine_mean": bootstrap["delta_label_cosine"]["mean"],
        "delta_label_cosine_p05": bootstrap["delta_label_cosine"]["p05"],
        "delta_label_js_mean": bootstrap["delta_label_js"]["mean"],
        "delta_label_js_p05": bootstrap["delta_label_js"]["p05"],
        "mean_label_shuffle_gap": bootstrap["mean_label_shuffle_gap"]["mean"],
        "label_shuffle_gap_p05": bootstrap["mean_label_shuffle_gap"]["p05"],
        "mean_sensitivity_cosine": bootstrap["mean_sensitivity_cosine"]["mean"],
        "mean_unit_demand_sensitivity_cosine": float(
            np.mean([row["mean_unit_demand_sensitivity_cosine"] for row in rows])
        ),
        "mean_unit_label_sensitivity_cosine": float(
            np.mean([row["mean_unit_label_sensitivity_cosine"] for row in rows])
        ),
        "mean_coeff_gradient_cosine": float(
            np.mean([row["mean_coeff_gradient_cosine"] for row in rows])
        ),
        "max_patch_mass_conservation_error": float(
            max(row["max_patch_mass_conservation_error"] for row in rows)
        ),
        "label_patch_mismatch_support": support,
        "retrieval_demand_mismatch_support": support,
    }


def audit_pass(rows: list[dict[str, Any]]) -> bool:
    return all(
        row["patch_count"] == len(patch_starts())
        and row["first_patch_start"] == 0
        and row["last_patch_start"] == PRED_LEN - PATCH_LEN
        and row["min_position_coverage"] == 1
        and row["max_position_coverage"] == 2
        and row["manual_patch_max_abs_diff"] == 0.0
        and row["reconstruction_max_abs_diff"] == 0.0
        and row["forecast_max_abs_diff"] <= 1e-6
        for row in rows
    )


def write_report(path: Path, summary: list[dict[str, Any]], decision: str) -> None:
    lines = [
        f"# B14-FURD Step 3 Diagnostic: {summary[0]['dataset']}",
        "",
        f"[Decision] `{decision}`。",
        "",
        "## Evidence Contract",
        "",
        "旁路使用 29 个 valid `K48-S24` normalized-history patches；不包含右侧 "
        "replication padding。",
        "raw-position attribution按 position coverage分配到 overlapping patches，因此 "
        "total evidence mass保持不变。",
        "旁路只定义 evidence axis，不进入 A6 forecast path。",
        "",
        "## Gate Results",
        "",
        "| Dataset | U | batches | label dCos p05 | label dJS p05 | "
        "CKA-shuffle p05 | sensitivity cos | support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        lines.append(
            "| {dataset} | {unit_size} | {batches} | {delta_label_cosine_p05:.4f} | "
            "{delta_label_js_p05:.4f} | {label_shuffle_gap_p05:.4f} | "
            "{mean_sensitivity_cosine:.4f} | {support} |".format(
                **row,
                support=("yes" if row["label_patch_mismatch_support"] else "no"),
            )
        )
    lines.extend(
        [
            "",
            "单 setting gate要求 `p05(delta_label_cosine)>0.05`、"
            "`p05(delta_label_js)>0.01`、",
            "`p05(mean_label_shuffle_gap)>0`、",
            "`mean sensitivity cosine>=0.80`。本报告仅是 Step 3 problem evidence，"
            "不验证 trainable retrieval。",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    if (
        args.hutchinson_draws <= 0
        or args.cka_shuffle_draws <= 0
        or args.descriptor_rank <= 0
        or args.max_batches <= 0
    ):
        raise ValueError("draws, descriptor rank, and max-batches must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = build_args(args)
    checkpoint = args.reference_dir / "checkpoint.pt"
    model = load_model(config, checkpoint).float().to(torch.device(args.device))
    _data, loader = data_provider(config, args.split)

    batch_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    profile_output_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    generator = torch.Generator(device=torch.device(args.device))
    generator.manual_seed(args.seed + 7919)

    for batch_idx, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader):
        if batch_idx >= args.max_batches:
            break
        batch_x = batch_x.float().to(args.device)
        batch_y = batch_y.float().to(args.device)
        target = batch_y[:, -PRED_LEN:, :]
        evidence_rows.append(
            audit_patch_evidence(
                model,
                batch_x,
                target,
                batch_idx,
                args.dataset,
            )
        )
        prediction, coeff, normalized = normalized_forward(model, batch_x)
        history_descriptors = patch_descriptors(
            model.retrieval_memory(normalized.detach().permute(0, 2, 1)),
            args.descriptor_rank,
        )
        target_normalized = (
            target - model.normalization_x.mean
        ) / model.normalization_x.stdev
        evidence_position = normalize_position_profile(
            normalized.detach().abs().mean(dim=(0, 2))
        )
        evidence_patch, evidence_mass_error = aggregate_patch_profile(evidence_position)
        profile_output_rows.extend(
            profile_rows(
                args.dataset,
                batch_idx,
                0,
                -1,
                "normalized_history_energy",
                evidence_patch.cpu(),
            )
        )
        for unit_size in args.unit_sizes:
            batch_row, setting_pairs, setting_profiles, mass_error = analyze_setting(
                args,
                args.dataset,
                batch_idx,
                unit_size,
                prediction,
                target,
                target_normalized,
                history_descriptors,
                coeff,
                normalized,
                generator,
            )
            batch_row["evidence_mass_conservation_error"] = evidence_mass_error
            batch_row["max_patch_mass_conservation_error"] = max(mass_error, evidence_mass_error)
            batch_rows.append(batch_row)
            pair_rows.extend(setting_pairs)
            profile_output_rows.extend(setting_profiles)

    if not batch_rows:
        raise RuntimeError(f"no batches analyzed for {args.dataset}")
    evidence_ok = audit_pass(evidence_rows)
    bootstrap_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for size_idx, unit_size in enumerate(args.unit_sizes):
        setting_rows = [row for row in batch_rows if row["unit_size"] == unit_size]
        rows, stats = bootstrap_setting(
            args.dataset,
            unit_size,
            setting_rows,
            args.bootstrap_iterations,
            args.seed + 1009 * size_idx,
        )
        bootstrap_rows.extend(rows)
        summary_rows.append(
            summarize(args.dataset, unit_size, setting_rows, stats, evidence_ok)
        )

    if not evidence_ok:
        decision = "diagnostic_invalid_for_direction_rejection"
    elif all(row["label_patch_mismatch_support"] for row in summary_rows):
        decision = "dataset_supports_label_patch_mismatch"
    elif any(row["label_patch_mismatch_support"] for row in summary_rows):
        decision = "unit_size_specific_label_patch_mismatch"
    elif sum(row["mean_sensitivity_cosine"] < 0.80 for row in summary_rows) >= 1:
        decision = "current_a6_sensitivity_already_unit_specific"
    else:
        decision = "label_patch_mismatch_not_supported"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "b14_future_unit_retrieval_batches.csv", batch_rows)
    write_csv(args.output_dir / "b14_future_unit_retrieval_pairs.csv", pair_rows)
    write_csv(args.output_dir / "b14_future_unit_retrieval_profiles.csv", profile_output_rows)
    write_csv(args.output_dir / "b14_future_unit_retrieval_summary.csv", summary_rows)
    write_csv(args.output_dir / "b14_future_unit_retrieval_bootstrap.csv", bootstrap_rows)
    write_csv(args.output_dir / "b14_history_patch_evidence_audit.csv", evidence_rows)
    (args.output_dir / "b14_future_unit_retrieval_decision.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "decision": decision,
                "evidence_contract_pass": evidence_ok,
                "patch_starts": patch_starts(),
                "patch_len": PATCH_LEN,
                "patch_stride": PATCH_STRIDE,
                "hutchinson_draws": args.hutchinson_draws,
                "descriptor_rank": args.descriptor_rank,
                "cka_shuffle_draws": args.cka_shuffle_draws,
                "max_batches": args.max_batches,
                "checkpoint": str(checkpoint),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_report(
        args.output_dir / "b14_future_unit_retrieval_report.md",
        summary_rows,
        decision,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(OFFICIAL_PRESETS), required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--unit-sizes", nargs="+", type=int, default=list(UNIT_SIZES))
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hutchinson-draws", type=int, default=4)
    parser.add_argument("--descriptor-rank", type=int, default=8)
    parser.add_argument("--cka-shuffle-draws", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

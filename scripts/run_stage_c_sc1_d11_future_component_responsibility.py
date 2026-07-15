#!/usr/bin/env python3
"""Run the SC1-D11 future-component gradient-responsibility diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

try:
    from run_stage_c_d1_offline_diagnostic import (
        Model,
        data_provider,
        load_state,
        run_dir,
        set_seed,
    )
    from run_stage_c_sc1_d2_diagnostic import checkpoint_run_dir
except ModuleNotFoundError:
    from scripts.run_stage_c_d1_offline_diagnostic import (
        Model,
        data_provider,
        load_state,
        run_dir,
        set_seed,
    )
    from scripts.run_stage_c_sc1_d2_diagnostic import checkpoint_run_dir

from layers.PLGO import restricted_global_nested_basis


SERIES_LENGTH = 720
DATA_SEED = 20260715


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-root", type=Path)
    parser.add_argument("--phase-b-root", type=Path)
    parser.add_argument("--phase-c-root", type=Path)
    parser.add_argument("--five-phase-a-root", type=Path)
    parser.add_argument("--five-phase-b-root", type=Path)
    parser.add_argument("--five-phase-c-root", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dataset", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke:
        return args
    required = {
        "phase_a_root": args.phase_a_root,
        "phase_b_root": args.phase_b_root,
        "phase_c_root": args.phase_c_root,
        "contract": args.contract,
        "design": args.design,
        "output_dir": args.output_dir,
        "dataset": args.dataset,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prefix_weights(prefixes: list[int], device: torch.device) -> torch.Tensor:
    weights = torch.zeros(SERIES_LENGTH, dtype=torch.float64, device=device)
    for horizon in prefixes:
        if horizon <= 0 or horizon > SERIES_LENGTH:
            raise ValueError(f"invalid prefix: {horizon}")
        weights[:horizon] += 1.0 / (float(len(prefixes)) * float(horizon))
    weights = weights / weights.sum()
    weights = weights.float()
    weights[0] += 1.0 - weights.sum()
    if abs(float(weights.sum().item()) - 1.0) > 1e-7:
        raise RuntimeError("prefix weights do not sum to one")
    return weights


def dct_basis(length: int) -> torch.Tensor:
    steps = torch.arange(length, dtype=torch.float64).unsqueeze(1) + 0.5
    frequencies = torch.arange(length, dtype=torch.float64).unsqueeze(0)
    basis = torch.cos(torch.pi * steps * frequencies / float(length))
    basis[:, 0] *= math.sqrt(1.0 / float(length))
    basis[:, 1:] *= math.sqrt(2.0 / float(length))
    return basis


def random_basis(length: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(length, length, dtype=torch.float64, generator=generator)
    basis, _upper = torch.linalg.qr(matrix)
    return basis


def basis_families(design: dict[str, Any]) -> dict[str, torch.Tensor]:
    rgnb, _atoms = restricted_global_nested_basis(SERIES_LENGTH, 16)
    families = {"rgnb": rgnb, "dct": dct_basis(SERIES_LENGTH)}
    for name in design["basis_families"]:
        if name.startswith("random_s"):
            families[name] = random_basis(SERIES_LENGTH, int(name.removeprefix("random_s")))
    if set(families) != set(design["basis_families"]):
        raise RuntimeError("basis family contract mismatch")
    return families


def contiguous_groups(group_sizes: list[int]) -> list[slice]:
    groups = []
    start = 0
    for size in group_sizes:
        groups.append(slice(start, start + size))
        start += size
    if start != SERIES_LENGTH:
        raise ValueError("group sizes do not cover the future domain")
    return groups


def parameter_groups(
    model: Model,
) -> tuple[list[tuple[str, torch.nn.Parameter]], dict[str, list[int]]]:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    groups = {"encoder": [], "coeff": [], "basis": [], "all": list(range(len(named)))}
    for index, (name, _parameter) in enumerate(named):
        if name.startswith("patch_emb_x") or name.startswith("encoder."):
            groups["encoder"].append(index)
        if name.startswith("learned_basis_coeff"):
            groups["coeff"].append(index)
        if name.startswith("learned_temporal_basis") or name.startswith(
            "learned_temporal_bias"
        ):
            groups["basis"].append(index)
    for name, indices in groups.items():
        if not indices:
            raise RuntimeError(f"empty parameter group: {name}")
    return named, groups


def flatten_parameter_gradient(
    gradients: tuple[torch.Tensor | None, ...],
    named: list[tuple[str, torch.nn.Parameter]],
    indices: list[int],
) -> torch.Tensor:
    parts = []
    for index in indices:
        gradient = gradients[index]
        parameter = named[index][1]
        parts.append(
            torch.zeros_like(parameter).flatten()
            if gradient is None
            else gradient.detach().flatten()
        )
    return torch.cat(parts)


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.flatten()
    second = second.flatten()
    if float(first.norm().item()) == 0.0 or float(second.norm().item()) == 0.0:
        return float("nan")
    return float(F.cosine_similarity(first, second, dim=0).item())


def gradient_summary(short: torch.Tensor, long: torch.Tensor) -> dict[str, Any]:
    short = short.flatten()
    long = long.flatten()
    short_norm = float(short.norm().item())
    long_norm = float(long.norm().item())
    dot = float(torch.dot(short, long).item())
    return {
        "short_norm": short_norm,
        "long_norm": long_norm,
        "norm_ratio": max(short_norm, long_norm) / max(min(short_norm, long_norm), 1e-12),
        "cosine": cosine(short, long),
        "dot": dot,
        "negative": dot < 0.0,
    }


def manual_forward(
    model: Model, batch_x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = model.normalization_x(batch_x, "norm")
    memory = model._encode_normalized_history(normalized)
    hidden = memory.flatten(start_dim=-2)
    coeff = model.learned_basis_coeff(hidden)
    basis = model.learned_temporal_basis.to(dtype=hidden.dtype)
    bias = model.learned_temporal_bias.to(dtype=hidden.dtype)
    normalized_output = torch.einsum("hk,bck->bch", basis, coeff)
    normalized_output = normalized_output + bias.view(1, 1, -1)
    output = model.normalization_x(normalized_output.permute(0, 2, 1), "denorm")
    return output, coeff, hidden


def loss_value(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    error = prediction - target
    if loss_name == "mse":
        values = error.square()
    elif loss_name == "l1":
        values = error.abs()
    else:
        raise ValueError(f"unsupported loss: {loss_name}")
    return (values * weights.view(1, -1, 1)).sum(dim=1).mean()


def project_gradient(
    output_gradient: torch.Tensor,
    basis: torch.Tensor,
    group: slice,
) -> torch.Tensor:
    columns = basis[:, group]
    coefficients = torch.einsum("tg,btc->bgc", columns, output_gradient)
    return torch.einsum("tg,bgc->btc", columns, coefficients)


def js_divergence(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first / first.sum().clamp_min(1e-12)
    second = second / second.sum().clamp_min(1e-12)
    middle = 0.5 * (first + second)
    first_kl = (first * (first.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum()
    second_kl = (second * (second.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum()
    return float((0.5 * (first_kl + second_kl)).item())


def component_summary(
    short_parts: list[torch.Tensor], long_parts: list[torch.Tensor]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    short_norms = torch.tensor([float(value.norm().item()) for value in short_parts])
    long_norms = torch.tensor([float(value.norm().item()) for value in long_parts])
    short_sum = torch.stack(short_parts).sum(dim=0)
    long_sum = torch.stack(long_parts).sum(dim=0)

    def pair_stats(parts: list[torch.Tensor]) -> tuple[float, float]:
        cosines = []
        negatives = []
        for left in range(len(parts)):
            for right in range(left + 1, len(parts)):
                dot = float(torch.dot(parts[left].flatten(), parts[right].flatten()).item())
                cosines.append(cosine(parts[left], parts[right]))
                negatives.append(float(dot < 0.0))
        return sum(cosines) / len(cosines), sum(negatives) / len(negatives)

    short_pair_cosine, short_negative_fraction = pair_stats(short_parts)
    long_pair_cosine, long_negative_fraction = pair_stats(long_parts)
    same_cosines = [cosine(short, long) for short, long in zip(short_parts, long_parts)]
    same_negatives = [
        float(torch.dot(short.flatten(), long.flatten()).item() < 0.0)
        for short, long in zip(short_parts, long_parts)
    ]
    short_efficiency = float(short_sum.norm().item()) / max(float(short_norms.sum().item()), 1e-12)
    long_efficiency = float(long_sum.norm().item()) / max(float(long_norms.sum().item()), 1e-12)
    group_rows = []
    for index, (short_norm, long_norm) in enumerate(zip(short_norms, long_norms)):
        group_rows.append(
            {
                "group_index": index,
                "short_norm": float(short_norm.item()),
                "long_norm": float(long_norm.item()),
                "short_share": float((short_norm / short_norms.sum().clamp_min(1e-12)).item()),
                "long_share": float((long_norm / long_norms.sum().clamp_min(1e-12)).item()),
                "same_component_cosine": same_cosines[index],
                "same_component_negative": bool(same_negatives[index]),
            }
        )
    return (
        {
            "responsibility_js": js_divergence(short_norms, long_norms),
            "short_pair_cosine": short_pair_cosine,
            "long_pair_cosine": long_pair_cosine,
            "short_negative_pair_fraction": short_negative_fraction,
            "long_negative_pair_fraction": long_negative_fraction,
            "same_component_cosine": sum(same_cosines) / len(same_cosines),
            "same_component_negative_fraction": sum(same_negatives) / len(same_negatives),
            "short_alignment_efficiency": short_efficiency,
            "long_alignment_efficiency": long_efficiency,
            "short_cancellation": 1.0 - short_efficiency,
            "long_cancellation": 1.0 - long_efficiency,
        },
        group_rows,
    )


def load_model_and_loaders(
    args: argparse.Namespace,
    contract: dict[str, Any],
    seed: int,
) -> tuple[Model, Any, Any, SimpleNamespace, Path]:
    profile = contract["dataset_profiles"][args.dataset]
    directory = checkpoint_run_dir(args, profile, seed)
    effective = json.loads((directory / "effective_config.json").read_text(encoding="utf-8"))
    payload = effective["official_args"]
    expected = {
        "patch_num": int(profile["patch_num"]),
        "d_model": int(profile["d_model"]),
        "d_ff": int(profile["d_ff"]),
    }
    observed = {key: int(payload[key]) for key in expected}
    if observed != expected:
        raise ValueError(f"frozen profile mismatch: {observed} != {expected}")
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
    return model, train_loader, val_loader, official_args, directory


def audit_loader(
    dataset: str,
    seed: int,
    split: str,
    model: Model,
    loader: Any,
    device: torch.device,
    max_batches: int,
    design: dict[str, Any],
    bases_cpu: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], float, float]:
    short_weights = prefix_weights(design["short_prefixes"], device)
    long_weights = prefix_weights(design["long_prefixes"], device)
    groups = contiguous_groups([int(value) for value in design["group_sizes"]])
    named, parameter_indices = parameter_groups(model)
    parameters = [parameter for _name, parameter in named]
    total_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    reachability_rows: list[dict[str, Any]] = []
    max_forward_gap = 0.0
    max_additivity_gap = 0.0

    for batch_index, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch_x = batch_x.float().to(device)
        target = batch_y[:, -SERIES_LENGTH:, :].float().to(device)
        with torch.no_grad():
            official, _recon, _alignment = model(
                batch_x, target, is_training=False, target_prefix=SERIES_LENGTH
            )
        prediction, coeff, _hidden = manual_forward(model, batch_x)
        max_forward_gap = max(
            max_forward_gap, float((prediction - official).abs().max().item())
        )

        learned_q, _upper = torch.linalg.qr(
            model.learned_temporal_basis.detach().double(), mode="reduced"
        )
        residual_bct = (prediction - target).detach().permute(0, 2, 1).double()
        learned_coeff = residual_bct @ learned_q
        span_energy = float(learned_coeff.square().sum().item())
        total_energy = float(residual_bct.square().sum().item())
        reachability_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "split": split,
                "batch_index": batch_index,
                "learned_span_error_energy_share": span_energy / max(total_energy, 1e-12),
                "learned_complement_error_energy_share": 1.0 - span_energy / max(total_energy, 1e-12),
            }
        )

        for loss_name in design["losses"]:
            loss_by_regime = {
                "short": loss_value(prediction, target, short_weights, loss_name),
                "long": loss_value(prediction, target, long_weights, loss_name),
            }
            output_gradients = {
                regime: torch.autograd.grad(loss, prediction, retain_graph=True)[0].detach()
                for regime, loss in loss_by_regime.items()
            }
            total_coeff_gradients = {
                regime: torch.autograd.grad(
                    prediction,
                    coeff,
                    grad_outputs=output_gradient,
                    retain_graph=True,
                )[0].detach()
                for regime, output_gradient in output_gradients.items()
            }
            summary = gradient_summary(
                total_coeff_gradients["short"], total_coeff_gradients["long"]
            )
            total_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "split": split,
                    "batch_index": batch_index,
                    "loss": loss_name,
                    "target": "coeff_tensor",
                    **summary,
                }
            )
            parameter_gradients: dict[str, dict[str, torch.Tensor]] = {
                "short": {},
                "long": {},
            }
            for regime, loss in loss_by_regime.items():
                gradients = torch.autograd.grad(
                    loss, parameters, retain_graph=True, allow_unused=True
                )
                for target_name in design["parameter_groups"]:
                    parameter_gradients[regime][target_name] = flatten_parameter_gradient(
                        gradients, named, parameter_indices[target_name]
                    )
            for target_name in design["parameter_groups"]:
                total_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "split": split,
                        "batch_index": batch_index,
                        "loss": loss_name,
                        "target": f"{target_name}_params",
                        **gradient_summary(
                            parameter_gradients["short"][target_name],
                            parameter_gradients["long"][target_name],
                        ),
                    }
                )

            for basis_name, basis_cpu in bases_cpu.items():
                basis = basis_cpu.to(device=device, dtype=prediction.dtype)
                parts: dict[str, list[torch.Tensor]] = {"short": [], "long": []}
                for regime in ("short", "long"):
                    for group in groups:
                        component_output = project_gradient(
                            output_gradients[regime], basis, group
                        )
                        component_coeff = torch.autograd.grad(
                            prediction,
                            coeff,
                            grad_outputs=component_output,
                            retain_graph=True,
                        )[0].detach()
                        parts[regime].append(component_coeff)
                    reconstructed = torch.stack(parts[regime]).sum(dim=0)
                    denominator = max(
                        float(total_coeff_gradients[regime].norm().item()), 1e-12
                    )
                    relative_gap = float(
                        (reconstructed - total_coeff_gradients[regime]).norm().item()
                    ) / denominator
                    max_additivity_gap = max(max_additivity_gap, relative_gap)
                component, per_group = component_summary(parts["short"], parts["long"])
                component_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "split": split,
                        "batch_index": batch_index,
                        "loss": loss_name,
                        "basis": basis_name,
                        "short_additivity_relative_gap": float(
                            (
                                torch.stack(parts["short"]).sum(dim=0)
                                - total_coeff_gradients["short"]
                            ).norm().item()
                        )
                        / max(float(total_coeff_gradients["short"].norm().item()), 1e-12),
                        "long_additivity_relative_gap": float(
                            (
                                torch.stack(parts["long"]).sum(dim=0)
                                - total_coeff_gradients["long"]
                            ).norm().item()
                        )
                        / max(float(total_coeff_gradients["long"].norm().item()), 1e-12),
                        **component,
                    }
                )
                for row in per_group:
                    group_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "split": split,
                            "batch_index": batch_index,
                            "loss": loss_name,
                            "basis": basis_name,
                            **row,
                        }
                    )
    if not total_rows:
        raise RuntimeError(f"no D11 batches produced for {dataset}/{seed}/{split}")
    return (
        total_rows,
        component_rows,
        group_rows,
        reachability_rows,
        max_forward_gap,
        max_additivity_gap,
    )


def synthetic_smoke() -> None:
    design = {
        "basis_families": [
            "rgnb",
            "dct",
            "random_s20260715",
            "random_s20260716",
            "random_s20260717",
        ],
        "group_sizes": [16, 16, 32, 64, 128, 256, 208],
    }
    bases = basis_families(design)
    identity = torch.eye(SERIES_LENGTH, dtype=torch.float64)
    for name, basis in bases.items():
        gap = float((basis.T @ basis - identity).abs().max().item())
        if gap > 1e-8:
            raise RuntimeError(f"orthogonality failed for {name}: {gap}")
    short = prefix_weights([48, 96, 144], torch.device("cpu"))
    long = prefix_weights([336, 512, 720], torch.device("cpu"))
    if max(abs(float(short.sum()) - 1.0), abs(float(long.sum()) - 1.0)) > 1e-7:
        raise RuntimeError("weight invariant failed")
    output = torch.randn(2, SERIES_LENGTH, 3, requires_grad=True)
    coeff = torch.randn(2, 3, 11, requires_grad=True)
    mixing = torch.randn(11, SERIES_LENGTH)
    output = torch.einsum("bck,kt->btc", coeff, mixing)
    target = torch.randn_like(output)
    loss = loss_value(output, target, short, "mse")
    output_gradient = torch.autograd.grad(loss, output, retain_graph=True)[0]
    total = torch.autograd.grad(
        output, coeff, grad_outputs=output_gradient, retain_graph=True
    )[0]
    groups = contiguous_groups(design["group_sizes"])
    parts = []
    for group in groups:
        projected = project_gradient(output_gradient, bases["rgnb"].float(), group)
        parts.append(
            torch.autograd.grad(output, coeff, grad_outputs=projected, retain_graph=True)[0]
        )
    relative_gap = float((torch.stack(parts).sum(dim=0) - total).norm().item()) / max(
        float(total.norm().item()), 1e-12
    )
    if relative_gap > 1e-5:
        raise RuntimeError(f"gradient additivity failed: {relative_gap}")
    print("stage_c_sc1_d11_worker_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    design = json.loads(args.design.read_text(encoding="utf-8"))
    if args.dataset not in design["datasets"]:
        raise ValueError(f"dataset not authorized: {args.dataset}")
    bases = basis_families(design)
    identity = torch.eye(SERIES_LENGTH, dtype=torch.float64)
    orthogonality = {
        name: float((basis.T @ basis - identity).abs().max().item())
        for name, basis in bases.items()
    }
    if max(orthogonality.values()) > float(design["gates"]["orthogonality_max_abs"]):
        raise RuntimeError("basis orthogonality invariant failed")

    total_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    reachability_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for seed in design["checkpoint_seeds"]:
        model, train_loader, val_loader, official_args, directory = load_model_and_loaders(
            args, contract, int(seed)
        )
        seed_forward_gap = 0.0
        seed_additivity_gap = 0.0
        for split, loader, batches in (
            ("train", train_loader, int(design["train_batches"])),
            ("validation", val_loader, int(design["validation_batches"])),
        ):
            outputs = audit_loader(
                args.dataset,
                int(seed),
                split,
                model,
                loader,
                official_args.device,
                batches,
                design,
                bases,
            )
            total, component, groups, reachability, forward_gap, additivity_gap = outputs
            total_rows.extend(total)
            component_rows.extend(component)
            group_rows.extend(groups)
            reachability_rows.extend(reachability)
            seed_forward_gap = max(seed_forward_gap, forward_gap)
            seed_additivity_gap = max(seed_additivity_gap, additivity_gap)
        metadata_rows.append(
            {
                "dataset": args.dataset,
                "seed": int(seed),
                "profile": contract["dataset_profiles"][args.dataset]["profile"],
                "run_dir": str(directory),
                "contract_hash": file_hash(args.contract),
                "design_hash": file_hash(args.design),
                "forward_reconstruction_max_abs": seed_forward_gap,
                "gradient_additivity_relative_max": seed_additivity_gap,
                "orthogonality_max_abs": max(orthogonality.values()),
                "uses_test_split": False,
                "trains_forecast_model": False,
                "updates_forecast_model": False,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "device": str(official_args.device),
            }
        )
    output_dir = args.output_dir / args.dataset
    write_csv(output_dir / "total_gradient_metrics.csv", total_rows)
    write_csv(output_dir / "component_metrics.csv", component_rows)
    write_csv(output_dir / "component_group_metrics.csv", group_rows)
    write_csv(output_dir / "reachability_metrics.csv", reachability_rows)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"stage_c_sc1_d11_worker_done dataset={args.dataset} "
        f"total_rows={len(total_rows)} component_rows={len(component_rows)}"
    )


if __name__ == "__main__":
    main()

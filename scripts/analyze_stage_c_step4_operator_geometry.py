#!/usr/bin/env python3
"""Audit A6/PMFO readout geometry for the StageC Step 4 redesign."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import torch


DATASETS = ("ETTh2", "ETTm1", "Weather")
BLOCK_SIZES = (90, 30, 10, 5)
SERIES_LENGTH = 720
DEFAULT_CHECKPOINT_ROOT = Path("/tmp/fatst-stagec-step4-audit/checkpoints")
DEFAULT_OUTPUT_DIR = Path("analysis/stage_c_step4_source_informed_redesign_20260713")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_path(root: Path, arm: str, dataset: str) -> Path:
    return root / arm / dataset / "h720_full" / "seed2021" / "checkpoint.pt"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def effective_rank(singular_values: torch.Tensor) -> float:
    energy = singular_values.square()
    probability = energy / energy.sum().clamp_min(torch.finfo(energy.dtype).eps)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def energy_rank(singular_values: torch.Tensor, threshold: float) -> int:
    cumulative = singular_values.square().cumsum(0)
    cumulative = cumulative / cumulative[-1].clamp_min(torch.finfo(cumulative.dtype).eps)
    return int(torch.searchsorted(cumulative, torch.tensor(threshold)).item()) + 1


def pairwise_cosine(rows: torch.Tensor) -> float:
    normalized = torch.nn.functional.normalize(rows, dim=-1)
    values = [
        float(torch.dot(normalized[left], normalized[right]))
        for left, right in combinations(range(rows.shape[0]), 2)
    ]
    return sum(values) / len(values)


def normalized_entropy(probability: torch.Tensor) -> torch.Tensor:
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum(dim=-1)
    return entropy / math.log(probability.shape[-1])


def parameter_count(state: dict[str, torch.Tensor], prefix: str) -> int:
    return sum(value.numel() for key, value in state.items() if key.startswith(prefix))


def operator_metrics(operator: torch.Tensor) -> dict[str, float | int]:
    singular_values = torch.linalg.svdvals(operator.double())
    return {
        "operator_effective_rank": effective_rank(singular_values),
        "operator_energy_rank_95": energy_rank(singular_values, 0.95),
        "operator_energy_rank_99": energy_rank(singular_values, 0.99),
        "operator_stable_rank": float(
            singular_values.square().sum() / singular_values[0].square()
        ),
    }


def boundary_rows(dataset: str, operator: torch.Tensor) -> list[dict[str, Any]]:
    jumps = torch.linalg.vector_norm(operator[1:] - operator[:-1], dim=-1)
    all_mean = jumps.mean()
    rows = []
    for block_size in BLOCK_SIZES:
        indices = torch.arange(block_size - 1, SERIES_LENGTH - 1, block_size)
        boundary = jumps[indices]
        rows.append(
            {
                "dataset": dataset,
                "block_size": block_size,
                "boundary_count": int(indices.numel()),
                "boundary_jump_mean": float(boundary.mean()),
                "all_jump_mean": float(all_mean),
                "boundary_to_all_jump_ratio": float(boundary.mean() / all_mean),
                "boundary_top10_fraction": float(
                    (boundary >= torch.quantile(jumps, 0.90)).double().mean()
                ),
            }
        )
    return rows


def local_rank_rows(dataset: str, operator: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    for block_size in BLOCK_SIZES[:-1]:
        captures = {rank: [] for rank in (1, 2, 4, 8, 16)} if block_size >= 16 else {
            rank: [] for rank in (1, 2, 4, 8)
        }
        for start in range(0, SERIES_LENGTH, block_size):
            singular_values = torch.linalg.svdvals(
                operator[start : start + block_size].double()
            )
            energy = singular_values.square()
            for rank in captures:
                captures[rank].append(float(energy[:rank].sum() / energy.sum()))
        for rank, values in captures.items():
            rows.append(
                {
                    "dataset": dataset,
                    "block_size": block_size,
                    "rank": rank,
                    "mean_energy_capture": sum(values) / len(values),
                    "min_energy_capture": min(values),
                    "max_energy_capture": max(values),
                }
            )
    return rows


def history_interface_metrics(
    dataset: str,
    a6_state: dict[str, torch.Tensor],
    pmfo_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    a6_weight = a6_state["learned_basis_coeff.weight"].double()
    seed_weight = pmfo_state["pmfo_readout.seed.weight"].double()
    readout_dim = a6_weight.shape[1]
    patch_num = 24 if dataset == "ETTm1" else 12
    if readout_dim % patch_num:
        raise ValueError(f"readout_dim={readout_dim} is not divisible by patch_num={patch_num}")
    d_model = readout_dim // patch_num

    a6_patch_energy = a6_weight.reshape(256, patch_num, d_model).square().sum(-1)
    a6_patch_probability = a6_patch_energy / a6_patch_energy.sum(-1, keepdim=True)

    seed_nodes = seed_weight.reshape(8, 32, patch_num, d_model)
    node_patch_energy = seed_nodes.square().sum(dim=(1, 3))
    node_patch_probability = node_patch_energy / node_patch_energy.sum(-1, keepdim=True)
    node_flat = seed_nodes.flatten(start_dim=1)
    return {
        "dataset": dataset,
        "patch_num": patch_num,
        "d_model": d_model,
        "a6_coeff_patch_entropy_mean": float(
            normalized_entropy(a6_patch_probability).mean()
        ),
        "pmfo_node_patch_entropy_mean": float(
            normalized_entropy(node_patch_probability).mean()
        ),
        "pmfo_node_patch_max_share_mean": float(
            node_patch_probability.max(dim=-1).values.mean()
        ),
        "pmfo_node_patch_profile_cosine_mean": pairwise_cosine(node_patch_probability),
        "pmfo_seed_node_weight_cosine_mean": pairwise_cosine(node_flat),
    }


def analyze_dataset(root: Path, dataset: str) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    a6_state = load_state(checkpoint_path(root, "a6", dataset))
    pmfo_state = load_state(checkpoint_path(root, "pmfo_rct", dataset))

    basis = a6_state["learned_temporal_basis"].double()
    coefficient_weight = a6_state["learned_basis_coeff.weight"].double()
    operator = basis @ coefficient_weight
    rank = min(basis.shape[1], basis.shape[0], coefficient_weight.shape[1])
    affine_rank_manifold_dimension = rank * (
        operator.shape[0] + operator.shape[1] - rank
    ) + operator.shape[0]
    a6_readout_parameters = (
        a6_state["learned_temporal_basis"].numel()
        + a6_state["learned_temporal_bias"].numel()
        + a6_state["learned_basis_coeff.weight"].numel()
        + a6_state["learned_basis_coeff.bias"].numel()
    )
    pmfo_readout_parameters = parameter_count(pmfo_state, "pmfo_readout.")

    summary = {
        "dataset": dataset,
        "a6_readout_parameters": a6_readout_parameters,
        "pmfo_readout_parameters": pmfo_readout_parameters,
        "pmfo_to_a6_parameter_ratio": pmfo_readout_parameters / a6_readout_parameters,
        "rank256_affine_manifold_dimension": affine_rank_manifold_dimension,
        "pmfo_dimension_gap": pmfo_readout_parameters - affine_rank_manifold_dimension,
        "pmfo_parameter_dimension_sufficient_for_a6_containment": (
            pmfo_readout_parameters >= affine_rank_manifold_dimension
        ),
        **operator_metrics(operator),
    }
    return (
        summary,
        boundary_rows(dataset, operator),
        local_rank_rows(dataset, operator),
        history_interface_metrics(dataset, a6_state, pmfo_state),
    )


def main() -> None:
    args = parse_args()
    summaries: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    local_ranks: list[dict[str, Any]] = []
    interfaces: list[dict[str, Any]] = []
    for dataset in DATASETS:
        summary, boundary, local_rank, interface = analyze_dataset(
            args.checkpoint_root,
            dataset,
        )
        summaries.append(summary)
        boundaries.extend(boundary)
        local_ranks.extend(local_rank)
        interfaces.append(interface)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "operator_summary.csv", summaries)
    write_csv(args.output_dir / "partition_boundary_audit.csv", boundaries)
    write_csv(args.output_dir / "local_rank_capture.csv", local_ranks)
    write_csv(args.output_dir / "history_interface_audit.csv", interfaces)
    payload = {
        "datasets": list(DATASETS),
        "checkpoint_root": str(args.checkpoint_root),
        "a6_containment_dimension_gate": all(
            row["pmfo_parameter_dimension_sufficient_for_a6_containment"]
            for row in summaries
        ),
        "coarse_boundary_positive_datasets": {
            str(block_size): sum(
                row["boundary_to_all_jump_ratio"] > 1.0
                for row in boundaries
                if row["block_size"] == block_size
            )
            for block_size in (90, 30)
        },
        "pmfo_global_history_profile_datasets": sum(
            row["pmfo_node_patch_entropy_mean"] >= 0.90
            and row["pmfo_node_patch_profile_cosine_mean"] >= 0.90
            for row in interfaces
        ),
    }
    (args.output_dir / "operator_geometry_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "stage_c_step4_operator_geometry=complete "
        f"datasets={len(DATASETS)} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()

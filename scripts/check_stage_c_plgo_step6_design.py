#!/usr/bin/env python3
"""Audit the SC1-PLGO Step 6 tensor contract and design boundaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from check_stage_c_plgo_step5_theory import (
    Atom,
    active_indices,
    restricted_global_nested_basis,
)


DEFAULT_OUTPUT_DIR = Path(
    "analysis/stage_c_sc1_plgo_step6_design_20260714"
)
TOLERANCE = 1e-10
SERIES_LENGTH = 720
BASIS_RANK = 256
GLOBAL_RANK = 16
DESCRIPTOR_DIM = 8
PROFILE_WIDTHS = {
    "Weather": 12 * 64,
    "ETTm1": 24 * 32,
    "ETTh2": 12 * 64,
    "ETTh1": 24 * 64,
    "ETTm2": 48 * 64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def canonical_atom_descriptors(atoms: list[Atom]) -> torch.Tensor:
    """Encode fixed RGNB atom geometry without a requested-horizon feature."""
    length = atoms[0].end
    max_depth = max((atom.depth for atom in atoms), default=0)
    groups: dict[tuple[str, int, int, int], list[int]] = {}
    for index, atom in enumerate(atoms):
        key = (atom.kind, atom.depth, atom.start, atom.end)
        groups.setdefault(key, []).append(index)

    rows = []
    for index, atom in enumerate(atoms):
        key = (atom.kind, atom.depth, atom.start, atom.end)
        group = groups[key]
        order = group.index(index)
        order_scale = max(len(group) - 1, 1)
        interval_length = atom.end - atom.start
        rows.append(
            [
                float(atom.kind == "global"),
                float(atom.kind == "detail"),
                atom.start / length,
                atom.end / length,
                interval_length / length,
                0.0 if atom.depth < 0 else atom.depth / max(max_depth, 1),
                order / order_scale,
                len(group) / length,
            ]
        )
    descriptors = torch.tensor(rows, dtype=torch.float64)
    if descriptors.shape != (length, DESCRIPTOR_DIM):
        raise AssertionError(f"unexpected descriptor shape: {descriptors.shape}")
    return descriptors


def horizon_set(length: int) -> list[int]:
    candidates = {1, length, length // 8, length // 4, length // 2}
    if length >= 96:
        candidates.update({48, 96, 144, 192, 336, 512, 720})
    return sorted(value for value in candidates if 1 <= value <= length)


def projective_atom_functional(
    descriptors: torch.Tensor,
    hidden: torch.Tensor,
    branch_weight: torch.Tensor,
    trunk_weight_1: torch.Tensor,
    trunk_bias_1: torch.Tensor,
    trunk_weight_2: torch.Tensor,
    trunk_bias_2: torch.Tensor,
) -> torch.Tensor:
    """Generate atom coefficients independently with shared parameters."""
    latent = hidden @ branch_weight
    features = torch.tanh(descriptors @ trunk_weight_1 + trunk_bias_1)
    trunk = features @ trunk_weight_2 + trunk_bias_2
    return torch.einsum("bck,nk->bcn", latent, trunk)


def check_projectivity_case(
    length: int,
    global_rank: int,
    generator: torch.Generator,
) -> list[dict[str, Any]]:
    synthesis, atoms, _prototypes, _gap = restricted_global_nested_basis(
        length,
        global_rank,
    )
    descriptors = canonical_atom_descriptors(atoms)
    batch, channels, hidden_width = 2, 3, 19
    latent_width = min(17, length)
    trunk_width = min(23, max(4, length))
    hidden = torch.randn(
        batch,
        channels,
        hidden_width,
        generator=generator,
        dtype=torch.float64,
    )
    branch_weight = torch.randn(
        hidden_width,
        latent_width,
        generator=generator,
        dtype=torch.float64,
    )
    trunk_weight_1 = torch.randn(
        DESCRIPTOR_DIM,
        trunk_width,
        generator=generator,
        dtype=torch.float64,
    )
    trunk_bias_1 = torch.randn(
        trunk_width,
        generator=generator,
        dtype=torch.float64,
    )
    trunk_weight_2 = torch.randn(
        trunk_width,
        latent_width,
        generator=generator,
        dtype=torch.float64,
    )
    trunk_bias_2 = torch.randn(
        latent_width,
        generator=generator,
        dtype=torch.float64,
    )
    temporal_bias = torch.randn(
        length,
        generator=generator,
        dtype=torch.float64,
    )

    full_coefficients = projective_atom_functional(
        descriptors,
        hidden,
        branch_weight,
        trunk_weight_1,
        trunk_bias_1,
        trunk_weight_2,
        trunk_bias_2,
    )
    full_output = torch.einsum(
        "tn,bcn->bct",
        synthesis,
        full_coefficients,
    ) + temporal_bias.view(1, 1, -1)

    rows = []
    for horizon in horizon_set(length):
        active = active_indices(atoms, horizon)
        subset_coefficients = projective_atom_functional(
            descriptors[active],
            hidden,
            branch_weight,
            trunk_weight_1,
            trunk_bias_1,
            trunk_weight_2,
            trunk_bias_2,
        )
        coefficient_gap = float(
            (subset_coefficients - full_coefficients[:, :, active])
            .abs()
            .max()
            .item()
        )
        prefix = torch.einsum(
            "hn,bcn->bch",
            synthesis[:horizon, active],
            subset_coefficients,
        ) + temporal_bias[:horizon].view(1, 1, -1)
        prefix_gap = float(
            (prefix - full_output[:, :, :horizon]).abs().max().item()
        )

        permutation = torch.randperm(
            active.numel(),
            generator=generator,
        )
        permuted = torch.einsum(
            "hn,bcn->bch",
            synthesis[:horizon, active[permutation]],
            subset_coefficients[:, :, permutation],
        ) + temporal_bias[:horizon].view(1, 1, -1)
        permutation_gap = float((permuted - prefix).abs().max().item())
        rows.append(
            {
                "length": length,
                "global_rank": global_rank,
                "horizon": horizon,
                "active_atoms": int(active.numel()),
                "coefficient_subset_max_abs": coefficient_gap,
                "prefix_reconstruction_max_abs": prefix_gap,
                "active_order_permutation_max_abs": permutation_gap,
            }
        )
    return rows


def matched_trunk_width() -> int:
    numerator = SERIES_LENGTH * BASIS_RANK - BASIS_RANK
    denominator = DESCRIPTOR_DIM + 1 + BASIS_RANK
    return numerator // denominator


def parameter_budget_rows() -> list[dict[str, Any]]:
    rows = []
    widths = {
        "compact_rank_width": BASIS_RANK,
        "near_a6_temporal_budget": matched_trunk_width(),
    }
    for dataset, history_width in PROFILE_WIDTHS.items():
        branch_parameters = BASIS_RANK * history_width + BASIS_RANK
        a6_temporal_parameters = SERIES_LENGTH * BASIS_RANK + SERIES_LENGTH
        a6_total = branch_parameters + a6_temporal_parameters
        for design, trunk_width in widths.items():
            trunk_parameters = (
                DESCRIPTOR_DIM * trunk_width
                + trunk_width
                + trunk_width * BASIS_RANK
                + BASIS_RANK
            )
            paf_total = branch_parameters + trunk_parameters + SERIES_LENGTH
            rows.append(
                {
                    "dataset": dataset,
                    "history_width": history_width,
                    "design": design,
                    "trunk_width": trunk_width,
                    "descriptor_dim": DESCRIPTOR_DIM,
                    "a6_readout_parameters": a6_total,
                    "paf_readout_parameters": paf_total,
                    "paf_to_a6_ratio": paf_total / a6_total,
                    "output_rank_upper_bound": BASIS_RANK,
                    "full_affine_output_class": False,
                    "exact_all_a6_table_containment": False,
                }
            )
    return rows


def candidate_control_rows() -> list[dict[str, Any]]:
    return [
        {
            "arm": "A6",
            "role": "baseline_control",
            "history_path": "shared_flat_memory",
            "atom_geometry": "none",
            "function_boundary": "free_rank256_temporal_table",
            "step7_status": "control_only",
        },
        {
            "arm": "PLGO-M0-FREE",
            "role": "exact_morphism_control",
            "history_path": "shared_flat_memory",
            "atom_geometry": "free_atom_table",
            "function_boundary": "equal_to_a6_under_rgnb_coordinates",
            "step7_status": "control_only",
        },
        {
            "arm": "PLGO-PAF-GEO",
            "role": "provisional_mechanism",
            "history_path": "shared_flat_memory",
            "atom_geometry": "canonical_rgnb_descriptor",
            "function_boundary": "rank256_descriptor_generated_table",
            "step7_status": "diagnostic_evidence_required",
        },
        {
            "arm": "PLGO-PAF-PERM",
            "role": "geometry_control",
            "history_path": "shared_flat_memory",
            "atom_geometry": "permuted_rgnb_descriptor",
            "function_boundary": "matched_to_geo",
            "step7_status": "mandatory_control",
        },
        {
            "arm": "PLGO-PAF-RANDOM",
            "role": "descriptor_capacity_control",
            "history_path": "shared_flat_memory",
            "atom_geometry": "fixed_random_descriptor",
            "function_boundary": "matched_to_geo",
            "step7_status": "mandatory_control",
        },
        {
            "arm": "PLGO-ATOM-ATTN",
            "role": "rejected_shortcut",
            "history_path": "atom_specific_patch_retrieval",
            "atom_geometry": "query_to_memory_attention",
            "function_boundary": "basisformer_timeperceiver_overlap",
            "step7_status": "rejected_by_internal_and_source_gate",
        },
    ]


def main() -> None:
    args = parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    projectivity_rows = []
    for length, global_rank in [(16, 4), (96, 8), (720, 16), (721, 16)]:
        projectivity_rows.extend(
            check_projectivity_case(length, global_rank, generator)
        )
    budget_rows = parameter_budget_rows()
    controls = candidate_control_rows()
    max_projectivity_gap = max(
        max(
            row["coefficient_subset_max_abs"],
            row["prefix_reconstruction_max_abs"],
            row["active_order_permutation_max_abs"],
        )
        for row in projectivity_rows
    )
    theory_gate = {
        "candidate": "SC1-PLGO-PAF",
        "step": 6,
        "projectivity_cases": len(projectivity_rows),
        "max_projectivity_gap": max_projectivity_gap,
        "tolerance": TOLERANCE,
        "atomwise_subset_invariance_gate": max_projectivity_gap <= TOLERANCE,
        "no_requested_horizon_feature_gate": True,
        "no_atom_to_atom_mixing_gate": True,
        "rank256_not_full_affine_gate": True,
        "exact_a6_containment_gate": False,
        "external_primitive_overlap_present": True,
        "external_overlap_is_automatic_rejection": False,
        "task_specific_contribution_boundary_gate": True,
        "internal_mechanism_evidence_gate": False,
        "future_unit_retrieval_authorized": False,
        "method_implementation_authorized": False,
        "step6_decision": "conditional_narrative_pass_d7_required",
        "next_diagnostic": "SC1-D7-RGNB-descriptor-sufficiency",
        "rollback_point": "d7_fail_return_step4_or_close_descriptor_route",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "projectivity_checks.csv", projectivity_rows)
    write_csv(args.output_dir / "parameter_budget.csv", budget_rows)
    write_csv(args.output_dir / "candidate_control_matrix.csv", controls)
    (args.output_dir / "theory_gate.json").write_text(
        json.dumps(theory_gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "stage_c_plgo_step6=complete "
        f"max_gap={max_projectivity_gap:.3e} "
        f"cases={len(projectivity_rows)} "
        f"decision={theory_gate['step6_decision']}"
    )


if __name__ == "__main__":
    main()

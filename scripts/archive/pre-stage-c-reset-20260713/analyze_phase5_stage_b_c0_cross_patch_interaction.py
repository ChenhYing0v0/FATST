#!/usr/bin/env python3
"""Measure frozen A6 cross-patch non-additivity before testing a token mixer."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch


HORIZONS = (96, 192, 336, 720)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=repo_root
        / "analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/raw/official-last"
        / "TimeAlignOfficialUnified720_a6_clean_official-last/ETTm1"
        / "mixed_h96_h192_h336_h720/seed2021",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root.parent / "datasets/ETT-small",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "analysis/phase5_stage_b_c0_cross_patch_interaction_20260710",
    )
    parser.add_argument("--patch-num", type=int, default=5)
    parser.add_argument("--attenuations", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=32)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def normalized_coeff(model: torch.nn.Module, normalized: torch.Tensor) -> torch.Tensor:
    memory = model._encode_normalized_history(normalized)
    hidden = memory.flatten(start_dim=-2)
    return model.learned_basis_coeff(hidden)


def attenuate_patches(
    normalized: torch.Tensor,
    patch_indices: tuple[int, ...],
    patch_len: int,
    attenuation: float,
) -> torch.Tensor:
    modified = normalized.clone()
    scale = 1.0 - attenuation
    for patch_idx in patch_indices:
        start = patch_idx * patch_len
        end = start + patch_len
        modified[:, start:end, :] *= scale
    return modified


def projected_norm(
    coefficient_delta: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    projected = torch.einsum("hk,bck->bch", basis, coefficient_delta)
    return projected.square().mean(dim=(1, 2)).sqrt()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def main() -> None:
    args = parse_args()
    if args.patch_num <= 1 or 720 % args.patch_num != 0:
        raise ValueError("patch_num must divide 720 and be greater than one")
    if any(value <= 0.0 or value > 1.0 for value in args.attenuations):
        raise ValueError("attenuations must be in (0, 1]")

    sys.path.insert(0, str(args.repo_root / "baselines/timealign_official"))
    from data_provider.data_factory import data_provider
    from models import TimeAlign

    config = read_json(args.run_dir / "effective_config.json")
    official_args = Namespace(**config["official_args"])
    official_args.root_path = str(args.dataset_root)
    official_args.device = torch.device("cpu")
    official_args.use_gpu = False
    official_args.gpu_type = "cpu"
    official_args.batch_size = args.batch_size
    official_args.num_workers = 0
    _test_data, test_loader = data_provider(official_args, "test")

    model = TimeAlign.Model(official_args).float()
    state = torch.load(
        args.run_dir / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    patch_len = official_args.seq_len // args.patch_num
    pair_indices = list(combinations(range(args.patch_num), 2))
    values: dict[tuple[float, int, int, int], list[float]] = {}
    main_effect_values: dict[tuple[float, int, int, int], list[float]] = {}

    with torch.no_grad():
        for batch_idx, (batch_x, _batch_y, _x_mark, _y_mark) in enumerate(test_loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            normalized = model.normalization_x(batch_x.float(), "norm")
            full_coeff = normalized_coeff(model, normalized)
            for attenuation in args.attenuations:
                single_coeff: dict[int, torch.Tensor] = {}
                for patch_idx in range(args.patch_num):
                    modified = attenuate_patches(
                        normalized,
                        (patch_idx,),
                        patch_len,
                        attenuation,
                    )
                    single_coeff[patch_idx] = normalized_coeff(model, modified)

                for first, second in pair_indices:
                    pair_input = attenuate_patches(
                        normalized,
                        (first, second),
                        patch_len,
                        attenuation,
                    )
                    pair_coeff = normalized_coeff(model, pair_input)
                    interaction = (
                        full_coeff
                        - single_coeff[first]
                        - single_coeff[second]
                        + pair_coeff
                    )
                    first_effect = full_coeff - single_coeff[first]
                    second_effect = full_coeff - single_coeff[second]
                    for horizon in HORIZONS:
                        basis = model.learned_temporal_basis[:horizon]
                        interaction_norm = projected_norm(interaction, basis)
                        main_norm = 0.5 * (
                            projected_norm(first_effect, basis)
                            + projected_norm(second_effect, basis)
                        )
                        ratio = interaction_norm / (main_norm + 1e-12)
                        key = (attenuation, first, second, horizon)
                        values.setdefault(key, []).extend(ratio.cpu().tolist())
                        main_effect_values.setdefault(key, []).extend(
                            main_norm.cpu().tolist()
                        )

    rows: list[dict[str, Any]] = []
    for key in sorted(values):
        attenuation, first, second, horizon = key
        ratio_summary = summarize(values[key])
        main_summary = summarize(main_effect_values[key])
        rows.append(
            {
                "dataset": "ETTm1",
                "checkpoint_policy": "official-last",
                "patch_num": args.patch_num,
                "patch_len": patch_len,
                "attenuation": attenuation,
                "first_patch": first,
                "second_patch": second,
                "target_horizon": horizon,
                "interaction_to_main_mean": ratio_summary["mean"],
                "interaction_to_main_median": ratio_summary["median"],
                "interaction_to_main_q25": ratio_summary["q25"],
                "interaction_to_main_q75": ratio_summary["q75"],
                "main_effect_rms_mean": main_summary["mean"],
                "examples": len(values[key]),
            }
        )

    write_csv(args.output_dir / "cross_patch_interaction_pairs.csv", rows)
    aggregate_rows: list[dict[str, Any]] = []
    for attenuation in args.attenuations:
        for horizon in HORIZONS:
            selected = [
                row
                for row in rows
                if row["attenuation"] == attenuation
                and row["target_horizon"] == horizon
            ]
            medians = [row["interaction_to_main_median"] for row in selected]
            aggregate_rows.append(
                {
                    "dataset": "ETTm1",
                    "attenuation": attenuation,
                    "target_horizon": horizon,
                    "pair_median_mean": float(np.mean(medians)),
                    "pair_median_min": float(np.min(medians)),
                    "pair_median_max": float(np.max(medians)),
                    "pairs_ge_0_05": sum(value >= 0.05 for value in medians),
                    "pair_count": len(medians),
                    "material_interaction": int(
                        float(np.mean(medians)) >= 0.05
                        and sum(value >= 0.05 for value in medians)
                        >= int(np.ceil(0.75 * len(medians)))
                    ),
                }
            )
    write_csv(args.output_dir / "cross_patch_interaction_summary.csv", aggregate_rows)


if __name__ == "__main__":
    main()

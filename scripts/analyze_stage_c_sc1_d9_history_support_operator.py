#!/usr/bin/env python3
"""Audit exact A6 history-scale to future-support operator geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path
from typing import Any

import torch

try:
    from check_stage_c_plgo_step5_theory import restricted_global_nested_basis
except ModuleNotFoundError:
    from scripts.check_stage_c_plgo_step5_theory import (
        restricted_global_nested_basis,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-a-root", type=Path)
    parser.add_argument("--legacy-b-root", type=Path)
    parser.add_argument("--legacy-c-root", type=Path)
    parser.add_argument("--five-a-root", type=Path)
    parser.add_argument("--five-b-root", type=Path)
    parser.add_argument("--five-c-root", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    if not args.synthetic_smoke:
        required = (
            "legacy_a_root",
            "legacy_b_root",
            "legacy_c_root",
            "five_a_root",
            "five_b_root",
            "five_c_root",
            "contract",
            "design",
            "output_dir",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def dct_rows(length: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float64).unsqueeze(0) + 0.5
    frequencies = torch.arange(length, dtype=torch.float64).unsqueeze(1)
    basis = torch.cos(math.pi * frequencies * positions / length)
    basis[0] *= math.sqrt(1.0 / length)
    if length > 1:
        basis[1:] *= math.sqrt(2.0 / length)
    return basis


def rank_values(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float64)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float64)
    return ranks


def pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double() - left.double().mean()
    right = right.double() - right.double().mean()
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def spearman(values: torch.Tensor) -> float:
    levels = torch.arange(values.numel(), dtype=torch.float64)
    return pearson(rank_values(levels), rank_values(values.double()))


def atom_group_labels(atoms: list[Any]) -> tuple[torch.Tensor, list[str]]:
    names = ["global_root"] + [f"detail_depth_{depth}" for depth in range(6)]
    mapping = {name: index for index, name in enumerate(names)}
    labels = []
    for atom in atoms:
        name = "global_root" if atom.kind == "global" else f"detail_depth_{atom.depth}"
        labels.append(mapping[name])
    return torch.tensor(labels, dtype=torch.long), names


def energy_matrix(
    operator: torch.Tensor,
    synthesis: torch.Tensor,
    history_basis: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    future_coordinates = synthesis.transpose(0, 1) @ operator.flatten(start_dim=1)
    future_coordinates = future_coordinates.reshape(
        synthesis.shape[1], operator.shape[1], operator.shape[2]
    )
    scale_coordinates = torch.einsum(
        "apd,kp->akd", future_coordinates, history_basis
    )
    energy = scale_coordinates.square().sum(dim=-1)
    original_energy = operator.square().sum()
    transformed_energy = energy.sum()
    gap = float(
        (transformed_energy - original_energy).abs()
        / original_energy.clamp_min(torch.finfo(torch.float64).tiny)
    )
    return energy, gap


def group_statistics(
    energy: torch.Tensor,
    labels: torch.Tensor,
    group_names: list[str],
) -> tuple[list[float], float, float]:
    normalized = energy / energy.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    frequencies = torch.linspace(0.0, 1.0, energy.shape[1], dtype=torch.float64)
    atom_centroids = normalized @ frequencies
    centroids = [
        float(atom_centroids[labels == index].mean())
        for index in range(len(group_names))
    ]
    centroid_tensor = torch.tensor(centroids, dtype=torch.float64)
    return centroids, spearman(centroid_tensor), centroids[-1] - centroids[0]


def random_orthogonal_rows(length: int, generator: torch.Generator) -> torch.Tensor:
    matrix = torch.randn(length, length, generator=generator, dtype=torch.float64)
    q, _upper = torch.linalg.qr(matrix)
    return q.transpose(0, 1).contiguous()


def analyze_operator(
    operator: torch.Tensor,
    synthesis: torch.Tensor,
    labels: torch.Tensor,
    group_names: list[str],
    permutation_count: int,
    random_basis_count: int,
    random_seed: int,
) -> dict[str, Any]:
    patch_num = operator.shape[1]
    canonical_energy, parseval_gap = energy_matrix(
        operator, synthesis, dct_rows(patch_num)
    )
    centroids, scale_rho, contrast = group_statistics(
        canonical_energy, labels, group_names
    )
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    permutation_rhos = []
    for _index in range(permutation_count):
        permuted_labels = labels[torch.randperm(labels.numel(), generator=generator)]
        _centroids, rho, _contrast = group_statistics(
            canonical_energy, permuted_labels, group_names
        )
        permutation_rhos.append(rho)
    random_basis_rhos = []
    random_basis_gaps = []
    for _index in range(random_basis_count):
        random_energy, gap = energy_matrix(
            operator,
            synthesis,
            random_orthogonal_rows(patch_num, generator),
        )
        _centroids, rho, _contrast = group_statistics(
            random_energy, labels, group_names
        )
        random_basis_rhos.append(rho)
        random_basis_gaps.append(gap)
    permutation_p = (1 + sum(value >= scale_rho for value in permutation_rhos)) / (
        permutation_count + 1
    )
    random_percentile = sum(value < scale_rho for value in random_basis_rhos) / len(
        random_basis_rhos
    )
    return {
        "centroids": centroids,
        "scale_rho": scale_rho,
        "fine_global_contrast": contrast,
        "atom_label_permutation_p": permutation_p,
        "random_history_basis_percentile": random_percentile,
        "parseval_relative_gap": max(parseval_gap, max(random_basis_gaps)),
        "permutation_rhos": permutation_rhos,
        "random_basis_rhos": random_basis_rhos,
    }


def checkpoint_dir(
    args: argparse.Namespace,
    profile: dict[str, Any],
    dataset: str,
    seed: int,
) -> Path:
    name = str(profile["profile"])
    if profile.get("artifact_family", "legacy_r2") == "five_extension":
        if seed == 2021 and name.endswith("_medium"):
            root = args.five_a_root
            run_name = (
                f"SC0FIVE_R2A_r2a_p{profile['patch_num']}_"
                f"d{profile['d_model']}_ff{profile['d_ff']}"
            )
        elif seed == 2021:
            root = args.five_b_root
            run_name = f"SC0FIVE_R2B_{name}"
        else:
            root = args.five_c_root
            run_name = f"SC0FIVE_R2C_{name}"
    elif seed == 2021 and name.endswith("_medium"):
        root = args.legacy_a_root
        run_name = (
            f"SC0DAP_R2A_r2a_p{profile['patch_num']}_"
            f"d{profile['d_model']}_ff{profile['d_ff']}"
        )
    elif seed == 2021:
        root = args.legacy_b_root
        run_name = f"SC0DAP_R2B_{name}"
    else:
        root = args.legacy_c_root
        run_name = f"SC0DAP_R2C_{name}"
    return root / run_name / dataset / "h720_full" / f"seed{seed}"


def load_operator(
    directory: Path,
    profile: dict[str, Any],
    series_length: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    effective = json.loads(
        (directory / "effective_config.json").read_text(encoding="utf-8")
    )
    official = effective["official_args"]
    expected = {
        "patch_num": int(profile["patch_num"]),
        "d_model": int(profile["d_model"]),
        "d_ff": int(profile["d_ff"]),
        "pred_len": series_length,
        "basis_rank": 256,
    }
    observed = {key: int(official[key]) for key in expected}
    if observed != expected:
        raise ValueError(f"checkpoint contract mismatch: {observed} != {expected}")
    if official["readout_mode"] != "learned-basis-forecast-operator":
        raise ValueError("D9 requires the frozen A6 learned-basis readout")
    state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    temporal_basis = state["learned_temporal_basis"].double()
    coefficient_weight = state["learned_basis_coeff.weight"].double()
    operator = temporal_basis @ coefficient_weight
    operator = operator.reshape(
        series_length, int(profile["patch_num"]), int(profile["d_model"])
    )
    return operator, effective


def aggregate_dataset_controls(
    rows: list[dict[str, Any]],
    control_values: dict[tuple[str, int], dict[str, list[float]]],
    datasets: list[str],
) -> list[dict[str, Any]]:
    results = []
    for dataset in datasets:
        selected = [row for row in rows if row["dataset"] == dataset]
        seed_count = len(selected)
        permutation_count = len(control_values[(dataset, int(selected[0]["seed"]))]["permutation"])
        random_count = len(control_values[(dataset, int(selected[0]["seed"]))]["random_basis"])
        permutation_means = [
            sum(control_values[(dataset, int(row["seed"]))]["permutation"][index] for row in selected)
            / seed_count
            for index in range(permutation_count)
        ]
        random_means = [
            sum(control_values[(dataset, int(row["seed"]))]["random_basis"][index] for row in selected)
            / seed_count
            for index in range(random_count)
        ]
        rho = sum(float(row["scale_rho"]) for row in selected) / seed_count
        contrast = sum(float(row["fine_global_contrast"]) for row in selected) / seed_count
        results.append(
            {
                "dataset": dataset,
                "checkpoint_count": seed_count,
                "mean_scale_rho": rho,
                "mean_fine_global_contrast": contrast,
                "aggregate_atom_label_permutation_p": (
                    1 + sum(value >= rho for value in permutation_means)
                )
                / (permutation_count + 1),
                "aggregate_random_history_basis_percentile": sum(
                    value < rho for value in random_means
                )
                / random_count,
                "max_parseval_relative_gap": max(
                    float(row["parseval_relative_gap"]) for row in selected
                ),
            }
        )
    return results


def gate_decision(
    unit_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, Any]:
    macro_rho = sum(float(row["mean_scale_rho"]) for row in dataset_rows) / len(
        dataset_rows
    )
    positive_datasets = sum(
        float(row["mean_scale_rho"]) > 0.0
        and float(row["mean_fine_global_contrast"])
        >= float(gates["dataset_positive_contrast_min"])
        for row in dataset_rows
    )
    positive_units = sum(float(row["scale_rho"]) > 0.0 for row in unit_rows)
    permutation_datasets = sum(
        float(row["aggregate_atom_label_permutation_p"])
        <= float(gates["dataset_permutation_p_max"])
        for row in dataset_rows
    )
    random_basis_datasets = sum(
        float(row["aggregate_random_history_basis_percentile"])
        >= float(gates["dataset_random_basis_percentile_min"])
        for row in dataset_rows
    )
    parseval_max = max(float(row["parseval_relative_gap"]) for row in unit_rows)
    checks = {
        "macro_scale_rho": macro_rho >= float(gates["macro_scale_rho_min"]),
        "dataset_positive_contrast": positive_datasets
        >= int(gates["dataset_positive_required"]),
        "unit_sign_consistency": positive_units >= int(gates["unit_positive_required"]),
        "atom_label_permutation": permutation_datasets
        >= int(gates["dataset_permutation_required"]),
        "random_history_basis": random_basis_datasets
        >= int(gates["dataset_random_basis_required"]),
        "parseval_invariant": parseval_max
        <= float(gates["parseval_relative_gap_max"]),
    }
    passed = all(checks.values())
    return {
        "diagnostic_id": "SC1-D9-A",
        "all_primary_gates_pass": passed,
        "decision": "pass_to_d9b" if passed else "operator_scale_hypothesis_not_supported",
        "method_implementation_authorized": False,
        "sc2_authorized": False,
        "macro_scale_rho": macro_rho,
        "positive_datasets": positive_datasets,
        "positive_units": positive_units,
        "permutation_datasets": permutation_datasets,
        "random_basis_datasets": random_basis_datasets,
        "max_parseval_relative_gap": parseval_max,
        "checks": checks,
    }


def write_report(
    path: Path,
    gate: dict[str, Any],
    dataset_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# SC1-D9-A Exact Operator Audit Result",
        "",
        "## Decision",
        "",
        f"- `decision`: `{gate['decision']}`；",
        f"- `all_primary_gates_pass`: `{str(gate['all_primary_gates_pass']).lower()}`；",
        f"- five-dataset macro `scale_rho`: `{gate['macro_scale_rho']:.6f}`；",
        f"- positive dataset/unit counts: `{gate['positive_datasets']}/5`, "
        f"`{gate['positive_units']}/15`；",
        f"- permutation/random-basis dataset gates: `{gate['permutation_datasets']}/5`, "
        f"`{gate['random_basis_datasets']}/5`；",
        f"- max Parseval relative gap: `{gate['max_parseval_relative_gap']:.3e}`。",
        "",
        "## Dataset Aggregates",
        "",
        "| Dataset | scale rho | fine-global contrast | permutation p | random-basis percentile |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in dataset_rows:
        lines.append(
            f"| {row['dataset']} | {row['mean_scale_rho']:.6f} | "
            f"{row['mean_fine_global_contrast']:.6f} | "
            f"{row['aggregate_atom_label_permutation_p']:.6f} | "
            f"{row['aggregate_random_history_basis_percentile']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "该实验不读取任何data split、不训练head，也不替换frozen component。它只判断A6 exact "
            "memory-to-future operator是否支持history-scale × future-support存在性假设。通过只授权D9-B "
            "sample-dependent input-Jacobian confirmation；失败则回Step2/3。无论结果如何，都不授权新model、test或SC2。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def synthetic_smoke() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260715)
    synthesis, atoms, _prototypes, _gap = restricted_global_nested_basis(720, 16)
    labels, names = atom_group_labels(atoms)
    patch_num, feature_dim = 12, 8
    left = torch.randn(720, 32, generator=generator, dtype=torch.float64)
    right = torch.randn(32, patch_num * feature_dim, generator=generator, dtype=torch.float64)
    operator = (left @ right).reshape(720, patch_num, feature_dim)
    result = analyze_operator(
        operator,
        synthesis,
        labels,
        names,
        permutation_count=8,
        random_basis_count=4,
        random_seed=20260715,
    )
    if result["parseval_relative_gap"] > 1e-10:
        raise RuntimeError("synthetic Parseval invariant failed")
    print("stage_c_sc1_d9_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke()
        return
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    design = json.loads(args.design.read_text(encoding="utf-8"))
    datasets = [str(value) for value in design["datasets"]]
    seeds = [int(value) for value in design["checkpoint_seeds"]]
    if datasets != list(contract["dataset_profiles"]):
        raise ValueError("D9 dataset order does not match the frozen contract")
    synthesis, atoms, _prototypes, _gap = restricted_global_nested_basis(
        int(design["series_length"]), int(design["future_global_rank"])
    )
    labels, group_names = atom_group_labels(atoms)
    unit_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    control_values: dict[tuple[str, int], dict[str, list[float]]] = {}
    for dataset_index, dataset in enumerate(datasets):
        profile = contract["dataset_profiles"][dataset]
        for seed in seeds:
            directory = checkpoint_dir(args, profile, dataset, seed)
            operator, _effective = load_operator(
                directory, profile, int(design["series_length"])
            )
            result = analyze_operator(
                operator,
                synthesis,
                labels,
                group_names,
                int(design["atom_label_permutations"]),
                int(design["random_history_bases"]),
                int(design["random_seed"]) + dataset_index * 10000 + seed,
            )
            unit_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "profile": profile["profile"],
                    "checkpoint_dir": str(directory),
                    "patch_num": int(profile["patch_num"]),
                    "d_model": int(profile["d_model"]),
                    "scale_rho": result["scale_rho"],
                    "fine_global_contrast": result["fine_global_contrast"],
                    "atom_label_permutation_p": result["atom_label_permutation_p"],
                    "random_history_basis_percentile": result[
                        "random_history_basis_percentile"
                    ],
                    "parseval_relative_gap": result["parseval_relative_gap"],
                }
            )
            for group_index, (name, centroid) in enumerate(
                zip(group_names, result["centroids"], strict=True)
            ):
                group_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "group_index": group_index,
                        "future_group": name,
                        "atom_count": int((labels == group_index).sum()),
                        "history_frequency_centroid": centroid,
                    }
                )
            control_values[(dataset, seed)] = {
                "permutation": result["permutation_rhos"],
                "random_basis": result["random_basis_rhos"],
            }
            for family in ("permutation", "random_basis"):
                for index, value in enumerate(control_values[(dataset, seed)][family]):
                    control_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "control_family": family,
                            "control_index": index,
                            "scale_rho": value,
                        }
                    )
            print(
                f"d9_unit_done dataset={dataset} seed={seed} "
                f"rho={result['scale_rho']:.6f} contrast={result['fine_global_contrast']:.6f}",
                flush=True,
            )
    dataset_rows = aggregate_dataset_controls(
        unit_rows, control_values, datasets
    )
    gate = gate_decision(unit_rows, dataset_rows, design["gates"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "unit_metrics.csv", unit_rows)
    write_csv(args.output_dir / "group_profiles.csv", group_rows)
    write_csv(args.output_dir / "control_distributions.csv", control_rows)
    write_csv(args.output_dir / "dataset_metrics.csv", dataset_rows)
    (args.output_dir / "gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cpu",
                "reads_data_samples": False,
                "uses_test_split": False,
                "forecast_model_updated": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "research_interpretation.md", gate, dataset_rows)
    print(f"stage_c_sc1_d9_done decision={gate['decision']}")


if __name__ == "__main__":
    main()

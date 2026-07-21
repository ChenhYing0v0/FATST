#!/usr/bin/env python3
"""Audit label-free local response relations in frozen ISCF-v0 readouts."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evaluate_stage_c_pcsd_cf_checkpoint import (  # noqa: E402
    load_model,
    sequential_loader,
)


SCOPE_NAMES = ("scale1", "scale48", "scale144", "scale360", "scale720")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/stage_c_iscf_v0_scope_response_d1.json"),
    )
    parser.add_argument(
        "--carrier-config",
        type=Path,
        default=Path("configs/stage_c_iscf_v0_carrier.json"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rankdata(np.asarray(left, dtype=np.float64))
    right_rank = rankdata(np.asarray(right, dtype=np.float64))
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = math.sqrt(
        float(left_rank @ left_rank) * float(right_rank @ right_rank)
    )
    if denominator <= 1e-12:
        return 0.0
    return float((left_rank @ right_rank) / denominator)


def normalize_scope_responses(bank: np.ndarray) -> np.ndarray:
    """Return centered, unit-RMS scope vectors with shape [S,F]."""
    if bank.ndim != 4 or bank.shape[2] != len(SCOPE_NAMES):
        raise ValueError(f"expected response bank [D,N,5,T], got {bank.shape}")
    vectors = bank.transpose(2, 0, 1, 3).reshape(bank.shape[2], -1).astype(np.float64)
    vectors -= vectors.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(np.square(vectors), axis=1, keepdims=True))
    if np.any(rms <= 1e-12):
        raise ValueError("degenerate scope response RMS")
    return vectors / rms


def relation_metrics(
    vectors: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    common = float(np.mean(np.square(vectors.mean(axis=0))))
    cosines = []
    distances = []
    for left, right in combinations(range(vectors.shape[0]), 2):
        cosine = float(np.mean(vectors[left] * vectors[right]))
        cosines.append(cosine)
        distances.append(
            float(np.sqrt(np.mean(np.square(vectors[left] - vectors[right]))))
        )
    return common, 1.0 - common, np.asarray(cosines), np.asarray(distances)


def direction_null_common(
    bank: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = []
    for _ in range(repetitions):
        shuffled = np.empty_like(bank)
        for scope_index in range(bank.shape[2]):
            permutation = rng.permutation(bank.shape[0])
            shuffled[:, :, scope_index] = bank[permutation, :, scope_index]
        vectors = normalize_scope_responses(shuffled)
        values.append(relation_metrics(vectors)[0])
    return np.asarray(values, dtype=np.float64)


def response_bank(
    readout: torch.nn.Module,
    hidden: torch.Tensor,
    directions: torch.Tensor,
    relative_epsilon: float,
    chunk_rows: int,
) -> np.ndarray:
    """Compute central directional responses as [D,N,S,T]."""
    row_scale = hidden.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    deltas = (
        relative_epsilon
        * row_scale.unsqueeze(0)
        * directions[:, None, :]
    )
    centers = hidden.unsqueeze(0).expand(directions.shape[0], -1, -1)
    plus = (centers + deltas).reshape(-1, 1, hidden.shape[-1])
    minus = (centers - deltas).reshape(-1, 1, hidden.shape[-1])
    denominator = (
        2.0
        * relative_epsilon
        * row_scale.unsqueeze(0).expand(directions.shape[0], -1, -1)
    ).reshape(-1, 1, 1, 1)
    chunks = []
    with torch.no_grad():
        for start in range(0, plus.shape[0], chunk_rows):
            end = min(start + chunk_rows, plus.shape[0])
            plus_arms = readout.arm_forecasts(plus[start:end])
            minus_arms = readout.arm_forecasts(minus[start:end])
            chunks.append(
                ((plus_arms - minus_arms) / denominator[start:end])
                .squeeze(1)
                .cpu()
            )
    responses = torch.cat(chunks, dim=0).reshape(
        directions.shape[0],
        hidden.shape[0],
        len(SCOPE_NAMES),
        -1,
    )
    if not bool(torch.isfinite(responses).all()):
        raise ValueError("non-finite directional responses")
    return responses.numpy()


def collect_hidden_rows(
    model: torch.nn.Module,
    official_args: Any,
    rows: int,
    device: torch.device,
) -> torch.Tensor:
    collected = []
    count = 0
    loader = sequential_loader(official_args, "val")
    with torch.no_grad():
        for batch_x, _batch_y, _batch_x_mark, _batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            memory = model.encode_history(batch_x)
            hidden = memory.flatten(start_dim=-2).reshape(
                -1,
                memory.shape[-2] * memory.shape[-1],
            )
            take = min(rows - count, hidden.shape[0])
            collected.append(hidden[:take].detach())
            count += take
            if count >= rows:
                break
    if count != rows:
        raise RuntimeError(f"validation split supplied {count}/{rows} hidden rows")
    hidden = torch.cat(collected, dim=0)
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError("non-finite hidden rows")
    return hidden


def reset_readout(readout: torch.nn.Module, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    control = copy.deepcopy(readout)
    control.reset_parameters()
    control.eval()
    return control


def run_directory(
    run_dir: Path,
    dataset: str,
    seed: int,
    protocol: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    probe = protocol["probe"]
    diagnostic_seed = int(probe["seed"]) + 1009 * seed + sum(map(ord, dataset))
    rng = np.random.default_rng(diagnostic_seed)
    torch.manual_seed(diagnostic_seed)
    model, _config, official_args = load_model(run_dir, device)
    readout = model.pcsd_readout
    if getattr(readout, "scale_basis_mode", None) != "independent":
        raise ValueError(f"{run_dir} is not an independent-scope readout")
    hidden = collect_hidden_rows(
        model,
        official_args,
        int(probe["hidden_rows"]),
        device,
    )
    direction_values = rng.choice(
        (-1.0, 1.0),
        size=(int(probe["rademacher_directions"]), hidden.shape[-1]),
    )
    directions = torch.from_numpy(direction_values).to(
        device=device,
        dtype=hidden.dtype,
    )
    observed_bank = response_bank(
        readout,
        hidden,
        directions,
        float(probe["relative_epsilon"]),
        int(probe["response_chunk_rows"]),
    )
    vectors = normalize_scope_responses(observed_bank)
    common, private, cosines, distances = relation_metrics(vectors)
    direction_null = direction_null_common(
        observed_bank,
        int(probe["direction_null_repetitions"]),
        rng,
    )

    random_common = []
    for control_index in range(int(probe["random_init_controls"])):
        control = reset_readout(readout, diagnostic_seed + 100_003 + control_index)
        control_bank = response_bank(
            control,
            hidden,
            directions,
            float(probe["relative_epsilon"]),
            int(probe["response_chunk_rows"]),
        )
        random_common.append(
            relation_metrics(normalize_scope_responses(control_bank))[0]
        )
        del control

    pair_rows = []
    scope_pairs = combinations(range(len(SCOPE_NAMES)), 2)
    for pair_index, (left, right) in enumerate(scope_pairs):
        pair_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "left_scope": SCOPE_NAMES[left],
                "right_scope": SCOPE_NAMES[right],
                "response_cosine": float(cosines[pair_index]),
                "response_distance": float(distances[pair_index]),
            }
        )
    random_array = np.asarray(random_common, dtype=np.float64)
    row = {
        "dataset": dataset,
        "seed": seed,
        "hidden_rows": int(hidden.shape[0]),
        "directions": int(directions.shape[0]),
        "synchronized_common_energy": common,
        "private_response_energy": private,
        "direction_null_common_p95": float(np.quantile(direction_null, 0.95)),
        "above_direction_null": int(common > np.quantile(direction_null, 0.95)),
        "random_init_common_median": float(np.median(random_array)),
        "random_init_common_p95": float(np.quantile(random_array, 0.95)),
        "above_random_init": int(common > np.quantile(random_array, 0.95)),
        "mean_pairwise_response_cosine": float(np.mean(cosines)),
        "median_pairwise_response_distance": float(np.median(distances)),
        "hidden_rms": float(hidden.square().mean().sqrt().cpu()),
        "all_finite": True,
    }
    del model
    return row, pair_rows, distances


def summarize(
    run_rows: list[dict[str, Any]],
    topology: dict[tuple[str, int], np.ndarray],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    topology_rows = []
    stable_datasets = 0
    threshold = float(protocol["gates"]["topology_seed_rho_min"])
    for dataset in protocol["matrix"]["datasets"]:
        seeds = sorted(seed for name, seed in topology if name == dataset)
        correlations = [
            spearman(topology[(dataset, left)], topology[(dataset, right)])
            for left, right in combinations(seeds, 2)
        ]
        median_rho = float(np.median(correlations)) if correlations else float("nan")
        stable = bool(len(seeds) >= 3 and median_rho >= threshold)
        stable_datasets += int(stable)
        topology_rows.append(
            {
                "dataset": dataset,
                "seed_count": len(seeds),
                "pair_count": len(correlations),
                "median_cross_seed_topology_spearman": median_rho,
                "stable": int(stable),
            }
        )

    gates = protocol["gates"]
    direction_count = sum(int(row["above_direction_null"]) for row in run_rows)
    random_count = sum(int(row["above_random_init"]) for row in run_rows)
    median_private = float(
        np.median([row["private_response_energy"] for row in run_rows])
    )
    median_distance = float(
        np.median([row["median_pairwise_response_distance"] for row in run_rows])
    )
    gate_values = {
        "synchronized_above_direction_null": direction_count
        >= int(gates["synchronized_above_direction_null_run_count_min"]),
        "learned_above_random_init": random_count
        >= int(gates["learned_above_random_init_run_count_min"]),
        "scope_specific_noncollapse": median_private
        >= float(gates["private_response_energy_median_min"])
        and median_distance >= float(gates["pairwise_response_distance_median_min"]),
        "cross_seed_topology_stability": stable_datasets
        >= int(gates["topology_dataset_count_min"]),
    }
    if all(gate_values.values()):
        decision = protocol["decision_map"]["all_gates_pass"]
    elif gate_values["synchronized_above_direction_null"] and not gate_values[
        "learned_above_random_init"
    ]:
        decision = protocol["decision_map"]["direction_pass_random_init_fail"]
    elif gate_values["synchronized_above_direction_null"] and gate_values[
        "learned_above_random_init"
    ] and not gate_values["cross_seed_topology_stability"]:
        decision = protocol["decision_map"]["relation_pass_topology_fail"]
    else:
        decision = protocol["decision_map"]["otherwise"]
    summary = {
        "diagnostic_id": protocol["diagnostic_id"],
        "run_count": len(run_rows),
        "all_arrays_finite": all(bool(row["all_finite"]) for row in run_rows),
        "synchronized_above_direction_null_run_count": direction_count,
        "learned_above_random_init_run_count": random_count,
        "median_synchronized_common_energy": float(
            np.median([row["synchronized_common_energy"] for row in run_rows])
        ),
        "median_private_response_energy": median_private,
        "median_pairwise_response_distance": median_distance,
        "stable_topology_dataset_count": stable_datasets,
        "gates": gate_values,
        "decision": decision,
    }
    return summary, topology_rows


def run_synthetic_smoke() -> None:
    rng = np.random.default_rng(20260721)
    directions, rows, scopes, horizon = 16, 8, 5, 24
    common = rng.normal(size=(directions, rows, 1, horizon))
    private = rng.normal(scale=0.25, size=(directions, rows, scopes, horizon))
    bank = common + private
    vectors = normalize_scope_responses(bank)
    common_energy, private_energy, _cosines, distances = relation_metrics(vectors)
    null = direction_null_common(bank, 64, rng)
    if not common_energy > np.quantile(null, 0.95):
        raise AssertionError("synthetic synchronized relation did not beat null")
    if not private_energy > 0.0 or not np.all(distances > 0.0):
        raise AssertionError("synthetic private response collapsed")
    print("iscf_v0_scope_response_synthetic_smoke=pass")


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        run_synthetic_smoke()
        return
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    carrier = json.loads(args.carrier_config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(protocol["output"]["local_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or list(protocol["matrix"]["datasets"])
    seeds = args.seeds or list(protocol["matrix"]["seeds"])
    device = torch.device(args.device)
    run_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    topology: dict[tuple[str, int], np.ndarray] = {}
    for dataset in datasets:
        for seed in seeds:
            root_key = "seed2021" if seed == 2021 else "seed2022_2023"
            run_dir = (
                Path(carrier["artifact_roots"][root_key])
                / dataset
                / "h720_full"
                / f"seed{seed}"
            )
            row, pairs, distances = run_directory(
                run_dir,
                dataset,
                seed,
                protocol,
                device,
            )
            run_rows.append(row)
            pair_rows.extend(pairs)
            topology[(dataset, seed)] = distances
            write_csv(output_dir / "run_scope_response.csv", run_rows)
            write_csv(output_dir / "pairwise_scope_response.csv", pair_rows)
            print(
                f"iscf_scope_response dataset={dataset} seed={seed} "
                f"common={row['synchronized_common_energy']:.6f} "
                f"direction={row['above_direction_null']} "
                f"random={row['above_random_init']}",
                flush=True,
            )
    summary, topology_rows = summarize(run_rows, topology, protocol)
    write_csv(output_dir / "seed_topology_stability.csv", topology_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"iscf_v0_scope_response=pass runs={len(run_rows)} "
        f"decision={summary['decision']}"
    )


if __name__ == "__main__":
    main()

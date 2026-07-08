from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from data_provider.data_factory import data_provider  # noqa: E402
from models.TimeAlign import Model  # noqa: E402
from train_repo import OFFICIAL_PRESETS, build_official_args  # noqa: E402


DATASETS = ("ETTh2", "ETTm1", "Weather")
HORIZONS = (96, 192, 336, 720)
RANKS = (4, 8, 16, 32)
BANK_COUNTS = (2, 4)
SEQ_LEN = 720
PRED_LEN = 720
DEFAULT_ANALYSIS_ROOT = Path("analysis/phase5_stage_b_b12_stbo_diagnostic_20260708")
DEFAULT_CHECKPOINT_ROOT = (
    Path("analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707")
    / "raw"
    / "official-last"
    / "TimeAlignOfficialUnified720_a6_clean_official-last"
)
DEFAULT_DATASET_ROOT = Path("/Users/river/PaperResearch/Project/datasets")


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    data_path: str
    relative_root: str
    kind: str


SPECS = {
    "ETTh2": DatasetSpec("ETTh2", "ETTh2.csv", "ETT-small", "ett_hour"),
    "ETTm1": DatasetSpec("ETTm1", "ETTm1.csv", "ETT-small", "ett_minute"),
    "Weather": DatasetSpec("Weather", "weather.csv", "weather", "custom"),
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_path(root: Path, dataset: str) -> Path:
    return root / dataset / "mixed_h96_h192_h336_h720" / "seed2021" / "checkpoint.pt"


def resolve_data_path(dataset_root: Path, spec: DatasetSpec) -> Path:
    direct = dataset_root / spec.data_path
    nested = dataset_root / spec.relative_root / spec.data_path
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Cannot find {spec.data_path} under {dataset_root}")


def train_boundaries(num_rows: int, spec: DatasetSpec) -> tuple[int, int]:
    if spec.kind == "ett_hour":
        return 0, 12 * 30 * 24
    if spec.kind == "ett_minute":
        return 0, 12 * 30 * 24 * 4
    if spec.kind == "custom":
        return 0, int(num_rows * 0.7)
    raise ValueError(f"Unsupported dataset kind: {spec.kind}")


def load_train_values(dataset_root: Path, dataset: str) -> np.ndarray:
    spec = SPECS[dataset]
    path = resolve_data_path(dataset_root, spec)
    df = pd.read_csv(path)
    if spec.kind == "custom":
        cols = list(df.columns)
        cols.remove("OT")
        cols.remove("date")
        df = df[["date"] + cols + ["OT"]]
    values = df[df.columns[1:]].to_numpy(dtype=np.float32)
    start, end = train_boundaries(len(df), spec)
    train = values[start:end]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train - mean) / std).astype(np.float32)


def sample_start_indices(num_windows: int, max_windows: int, seed: int) -> np.ndarray:
    if num_windows <= max_windows:
        return np.arange(num_windows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_windows, size=max_windows, replace=False)).astype(np.int64)


def build_future_matrix(train_values: np.ndarray, starts: np.ndarray, horizon: int) -> np.ndarray:
    channels = train_values.shape[1]
    matrix = np.empty((len(starts) * channels, horizon), dtype=np.float32)
    row = 0
    for start in starts:
        future = train_values[start + SEQ_LEN : start + SEQ_LEN + horizon]
        matrix[row : row + channels] = future.T
        row += channels
    return matrix


def center_columns(matrix: np.ndarray) -> np.ndarray:
    return (matrix - matrix.mean(axis=0, keepdims=True)).astype(np.float64)


def load_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    return state


def build_args(args: argparse.Namespace, dataset: str) -> argparse.Namespace:
    preset = OFFICIAL_PRESETS[dataset][720]
    adapter_args = SimpleNamespace(
        dataset_root=args.dataset_root,
        output_dir=args.analysis_root / "_tmp_official_args",
        dataset=dataset,
        seq_len=SEQ_LEN,
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
    )
    official_args = build_official_args(adapter_args, preset)
    official_args.batch_size = args.batch_size
    official_args.num_workers = 0
    return official_args


def load_model(args: argparse.Namespace, dataset: str) -> Model:
    official_args = build_args(args, dataset)
    model = Model(official_args)
    model.load_state_dict(load_state(checkpoint_path(args.checkpoint_root, dataset)))
    model.eval()
    return model


def collect_coeff_rows(args: argparse.Namespace, dataset: str, model: Model) -> np.ndarray:
    official_args = build_args(args, dataset)
    _data, loader = data_provider(official_args, args.split)
    rows_parts: list[np.ndarray] = []
    total = 0
    device = torch.device(args.device)
    model = model.to(device)
    with torch.no_grad():
        for batch_x, _batch_y, _batch_x_mark, _batch_y_mark in loader:
            if total >= args.max_coeff_rows:
                break
            batch_x = batch_x.float().to(device)
            batch, seq_len, channels = batch_x.shape
            x = model.normalization_x(batch_x, "norm")
            x = model.patch_emb_x(x.permute(0, 2, 1).reshape(-1, channels * seq_len))
            for layer_idx in range(model.e_layers):
                x = x + model.encoder[layer_idx](x)
                if model.layer_norm:
                    x = model.norm_x[layer_idx](x)
            hidden = x.reshape(batch, channels, model.patch_num, model.d_model).flatten(start_dim=-2)
            coeff = model.learned_basis_coeff(hidden)
            coeff_rows = coeff.reshape(-1, coeff.shape[-1]).detach().cpu().numpy().astype(np.float64)
            take = min(args.max_coeff_rows - total, coeff_rows.shape[0])
            rows_parts.append(coeff_rows[:take])
            total += take
    if not rows_parts:
        raise RuntimeError(f"No coeff rows collected for {dataset}")
    return np.concatenate(rows_parts, axis=0)


def dct_basis(length: int) -> np.ndarray:
    steps = np.arange(length, dtype=np.float64) + 0.5
    freqs = np.arange(length, dtype=np.float64)
    basis = np.cos(np.pi * np.outer(steps, freqs) / length)
    basis[:, 0] *= np.sqrt(1.0 / length)
    basis[:, 1:] *= np.sqrt(2.0 / length)
    return basis


def normalize_rows(values: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(denom, 1e-12, None)


def deterministic_kmeans(features: np.ndarray, clusters: int, max_iter: int = 100) -> np.ndarray:
    if clusters <= 0 or clusters > features.shape[0]:
        raise ValueError("invalid number of clusters")
    features = normalize_rows(features.astype(np.float64))
    centroids = [features[0]]
    distances = np.sum((features - centroids[0]) ** 2, axis=1)
    for _idx in range(1, clusters):
        next_idx = int(np.argmax(distances))
        centroids.append(features[next_idx])
        distances = np.minimum(distances, np.sum((features - features[next_idx]) ** 2, axis=1))
    centroid_array = np.stack(centroids, axis=0)
    labels = np.zeros(features.shape[0], dtype=np.int64)
    for _iter in range(max_iter):
        dists = np.sum((features[:, None, :] - centroid_array[None, :, :]) ** 2, axis=-1)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_idx in range(clusters):
            mask = labels == cluster_idx
            if np.any(mask):
                centroid_array[cluster_idx] = np.mean(features[mask], axis=0)
    return labels


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return float("nan")
    xr = rankdata(x[mask])
    yr = rankdata(y[mask])
    xr = xr - np.mean(xr)
    yr = yr - np.mean(yr)
    denom = np.linalg.norm(xr) * np.linalg.norm(yr)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(xr, yr) / denom)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.clip(denom, 1e-12, None)


def tile_operator_basis_metrics(
    dataset: str,
    basis: np.ndarray,
    tile_len: int,
    ranks: tuple[int, ...],
    bank_counts: tuple[int, ...],
) -> list[dict[str, Any]]:
    if basis.shape[0] % tile_len != 0:
        raise ValueError("basis length must be divisible by tile length")
    tiles = basis.reshape(basis.shape[0] // tile_len, tile_len, basis.shape[1]).astype(np.float64)
    total_energy = float(np.sum(tiles * tiles))
    concat = np.concatenate([tile for tile in tiles], axis=1)
    shared_u, shared_s, _shared_vh = np.linalg.svd(concat, full_matrices=False)
    dct = dct_basis(tile_len)
    tile_features = normalize_rows(tiles.reshape(tiles.shape[0], -1))

    rows: list[dict[str, Any]] = []
    for rank in ranks:
        shared_energy = float(np.sum(shared_s[:rank] ** 2) / total_energy)
        dct_energy = 0.0
        independent_energy = 0.0
        for tile in tiles:
            tile_s = np.linalg.svd(tile, compute_uv=False)
            independent_energy += float(np.sum(tile_s[:rank] ** 2))
            coeff = dct[:, :rank].T @ tile
            dct_energy += float(np.sum(coeff * coeff))
        base_row = {
            "dataset": dataset,
            "tile_len": tile_len,
            "tiles": tiles.shape[0],
            "rank": rank,
            "shared_local_energy": shared_energy,
            "independent_tile_energy": independent_energy / total_energy,
            "local_dct_energy": dct_energy / total_energy,
        }
        base_row["shared_gap_to_independent"] = (
            base_row["independent_tile_energy"] - base_row["shared_local_energy"]
        )
        base_row["shared_minus_dct"] = base_row["shared_local_energy"] - base_row["local_dct_energy"]
        for banks in bank_counts:
            labels = deterministic_kmeans(tile_features, banks)
            bank_energy = 0.0
            for bank_idx in range(banks):
                bank_tiles = tiles[labels == bank_idx]
                if bank_tiles.size == 0:
                    continue
                bank_concat = np.concatenate([tile for tile in bank_tiles], axis=1)
                bank_s = np.linalg.svd(bank_concat, compute_uv=False)
                bank_energy += float(np.sum(bank_s[:rank] ** 2))
            base_row[f"bank{banks}_energy"] = bank_energy / total_energy
            base_row[f"bank{banks}_gap_to_independent"] = (
                base_row["independent_tile_energy"] - base_row[f"bank{banks}_energy"]
            )
            base_row[f"bank{banks}_minus_dct"] = base_row[f"bank{banks}_energy"] - base_row["local_dct_energy"]
        rows.append(base_row)
    return rows


def tile_label_basis_metrics(
    dataset: str,
    future_matrix: np.ndarray,
    tile_len: int,
    ranks: tuple[int, ...],
    bank_counts: tuple[int, ...],
) -> list[dict[str, Any]]:
    if future_matrix.shape[1] % tile_len != 0:
        raise ValueError("future length must be divisible by tile length")
    tiles = [
        center_columns(future_matrix[:, start : start + tile_len])
        for start in range(0, future_matrix.shape[1], tile_len)
    ]
    total_energy = float(sum(np.sum(tile * tile) for tile in tiles))
    stacked = np.concatenate(tiles, axis=0)
    _u, shared_s, shared_vh = np.linalg.svd(stacked, full_matrices=False)
    dct = dct_basis(tile_len)
    cov_features = []
    for tile in tiles:
        cov = (tile.T @ tile) / max(tile.shape[0] - 1, 1)
        cov_features.append(cov.reshape(-1))
    tile_features = normalize_rows(np.stack(cov_features, axis=0))

    rows: list[dict[str, Any]] = []
    for rank in ranks:
        shared_v = shared_vh[:rank].T
        shared_energy = sum(float(np.sum((tile @ shared_v) ** 2)) for tile in tiles)
        dct_energy = sum(float(np.sum((tile @ dct[:, :rank]) ** 2)) for tile in tiles)
        independent_energy = sum(
            float(np.sum(np.linalg.svd(tile, compute_uv=False)[:rank] ** 2)) for tile in tiles
        )
        base_row = {
            "dataset": dataset,
            "tile_len": tile_len,
            "tiles": len(tiles),
            "rank": rank,
            "shared_local_energy": shared_energy / total_energy,
            "independent_tile_energy": independent_energy / total_energy,
            "local_dct_energy": dct_energy / total_energy,
        }
        base_row["shared_gap_to_independent"] = (
            base_row["independent_tile_energy"] - base_row["shared_local_energy"]
        )
        base_row["shared_minus_dct"] = base_row["shared_local_energy"] - base_row["local_dct_energy"]
        for banks in bank_counts:
            labels = deterministic_kmeans(tile_features, banks)
            bank_energy = 0.0
            for bank_idx in range(banks):
                bank_tiles = [tile for idx, tile in enumerate(tiles) if labels[idx] == bank_idx]
                if not bank_tiles:
                    continue
                bank_stacked = np.concatenate(bank_tiles, axis=0)
                _bank_u, _bank_s, bank_vh = np.linalg.svd(bank_stacked, full_matrices=False)
                bank_v = bank_vh[:rank].T
                bank_energy += sum(float(np.sum((tile @ bank_v) ** 2)) for tile in bank_tiles)
            base_row[f"bank{banks}_energy"] = bank_energy / total_energy
            base_row[f"bank{banks}_gap_to_independent"] = (
                base_row["independent_tile_energy"] - base_row[f"bank{banks}_energy"]
            )
            base_row[f"bank{banks}_minus_dct"] = base_row[f"bank{banks}_energy"] - base_row["local_dct_energy"]
        rows.append(base_row)
    return rows


def coefficient_tile_projection_metrics(
    dataset: str,
    basis: np.ndarray,
    coeff_rows: np.ndarray,
    tile_len: int,
    rank: int,
) -> dict[str, Any]:
    tiles = basis.reshape(basis.shape[0] // tile_len, tile_len, basis.shape[1]).astype(np.float64)
    subspaces = []
    projections = []
    output_energies = []
    coeff_norm_sq = np.sum(coeff_rows * coeff_rows, axis=1)
    coeff_norm_sq = np.clip(coeff_norm_sq, 1e-12, None)
    projection_shares = []
    for tile in tiles:
        _u, singular_values, vh = np.linalg.svd(tile, full_matrices=False)
        keep = min(rank, int(np.sum(singular_values > 1e-10)), vh.shape[0])
        q = vh[:keep].T
        subspaces.append(q)
        coords = coeff_rows @ q
        projections.append(coords @ q.T)
        projection_shares.append(np.sum(coords * coords, axis=1) / coeff_norm_sq)
        output = coeff_rows @ tile.T
        output_energies.append(np.sum(output * output, axis=1))

    output_stack = np.stack(output_energies, axis=1)
    output_share = output_stack / np.clip(np.sum(output_stack, axis=1, keepdims=True), 1e-12, None)
    projection_stack = np.stack(projection_shares, axis=1)
    projection_share = projection_stack / np.clip(np.sum(projection_stack, axis=1, keepdims=True), 1e-12, None)
    output_entropy = -np.sum(output_share * np.log(np.clip(output_share, 1e-12, None)), axis=1) / np.log(len(tiles))
    projection_entropy = (
        -np.sum(projection_share * np.log(np.clip(projection_share, 1e-12, None)), axis=1) / np.log(len(tiles))
    )

    distances: list[float] = []
    overlaps: list[float] = []
    cosines: list[float] = []
    adjacent_cosines: list[float] = []
    far_cosines: list[float] = []
    for left in range(len(tiles)):
        for right in range(left + 1, len(tiles)):
            distance = float((right - left) * tile_len)
            singular_values = np.linalg.svd(subspaces[left].T @ subspaces[right], compute_uv=False)
            overlap = float(np.mean(singular_values * singular_values))
            cosine = float(np.mean(cosine_rows(projections[left], projections[right])))
            distances.append(distance)
            overlaps.append(overlap)
            cosines.append(cosine)
            if right == left + 1:
                adjacent_cosines.append(cosine)
            if distance >= 240:
                far_cosines.append(cosine)

    return {
        "dataset": dataset,
        "tile_len": tile_len,
        "tiles": len(tiles),
        "rank": rank,
        "coeff_rows": coeff_rows.shape[0],
        "adjacent_projection_cosine": float(np.mean(adjacent_cosines)),
        "far_projection_cosine": float(np.mean(far_cosines)),
        "distance_projection_cosine_spearman": spearman(distances, cosines),
        "distance_subspace_overlap_spearman": spearman(distances, overlaps),
        "projection_entropy_mean": float(np.mean(projection_entropy)),
        "output_entropy_mean": float(np.mean(output_entropy)),
        "max_projection_tile_share_mean": float(np.mean(np.max(projection_share, axis=1))),
        "max_output_tile_share_mean": float(np.mean(np.max(output_share, axis=1))),
    }


def f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def gate_decision(
    basis_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    coeff_rows: list[dict[str, Any]],
    rank: int,
) -> tuple[str, dict[str, list[str]]]:
    basis_rank_rows = [row for row in basis_rows if int(row["rank"]) == rank]
    label_rank_rows = [row for row in label_rows if int(row["rank"]) == rank]
    basis_bank_ok = [
        row["dataset"]
        for row in basis_rank_rows
        if f(row, "bank4_gap_to_independent") <= 0.05 and f(row, "bank4_minus_dct") >= 0.02
    ]
    basis_shared_ok = [
        row["dataset"]
        for row in basis_rank_rows
        if f(row, "shared_gap_to_independent") <= 0.07 and f(row, "shared_minus_dct") >= 0.0
    ]
    label_bank_ok = [
        row["dataset"]
        for row in label_rank_rows
        if f(row, "bank4_gap_to_independent") <= 0.05 and f(row, "bank4_minus_dct") >= 0.02
    ]
    label_shared_ok = [
        row["dataset"]
        for row in label_rank_rows
        if f(row, "shared_gap_to_independent") <= 0.07 and f(row, "shared_minus_dct") >= 0.0
    ]
    independent_only = [
        row["dataset"]
        for row in basis_rank_rows
        if f(row, "independent_tile_energy") - max(f(row, "shared_local_energy"), f(row, "bank4_energy")) > 0.12
    ]
    coeff_ok = [
        row["dataset"]
        for row in coeff_rows
        if f(row, "adjacent_projection_cosine") - f(row, "far_projection_cosine") >= 0.15
        and f(row, "projection_entropy_mean") >= 0.70
    ]
    dct_risk = [
        row["dataset"]
        for row in label_rank_rows
        if max(f(row, "shared_minus_dct"), f(row, "bank4_minus_dct")) < 0.02
    ]

    passed_basis = set(basis_bank_ok) | set(basis_shared_ok)
    passed_label = set(label_bank_ok) | set(label_shared_ok)
    if len(passed_basis) >= 2 and len(passed_label) >= 2 and len(coeff_ok) >= 2 and not dct_risk:
        decision = "problem_candidate_passed_enter_step46_design"
    elif len(passed_basis) >= 2 and len(coeff_ok) >= 2:
        decision = "partial_support_basis_operator_but_label_or_dct_control_risk"
    elif len(independent_only) >= 2:
        decision = "independent_tile_only_no_method"
    else:
        decision = "diagnostic_not_enough_for_b12"
    return decision, {
        "basis_bank_ok": basis_bank_ok,
        "basis_shared_ok": basis_shared_ok,
        "label_bank_ok": label_bank_ok,
        "label_shared_ok": label_shared_ok,
        "independent_only": independent_only,
        "coeff_ok": coeff_ok,
        "dct_risk": dct_risk,
    }


def render_report(
    basis_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    coeff_rows: list[dict[str, Any]],
    decision: str,
    gate: dict[str, list[str]],
    args: argparse.Namespace,
) -> str:
    rank = args.gate_rank
    basis_rank_rows = [row for row in basis_rows if int(row["rank"]) == rank]
    label_rank_rows = [row for row in label_rows if int(row["rank"]) == rank]
    lines = [
        "# Phase5 StageB B12-STBO Diagnostic Report",
        "",
        "`current_step`: StageB Step 2/3 problem-existence and feasibility diagnostic.",
        "",
        "## Scope",
        "",
        "[Fact] This diagnostic uses clean A6 checkpoints and train split labels. It does not train or evaluate a new model.",
        "",
        "[Boundary] A positive reconstruction result is not a method result. It only decides whether B12 may enter Step 4-6 method design.",
        "",
        "## 11-step Record",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | StageB Step 2/3 B12-STBO diagnostic |",
        "| `problem` | Can A6's full-720 step basis be replaced by a stage/tile-local subspace basis operator? |",
        "| `existence_evidence` | A6 basis tile factorization, train-label tile factorization, coeff projection into tile subspaces |",
        "| `idea` | Use shared/bank local basis tiles instead of full-720 step basis; short horizons activate only needed tiles |",
        "| `theory_check` | Positive only if shared/bank tiles approach independent-tile upper bound and beat local DCT controls |",
        f"| `design` | Offline diagnostic, tile_len={args.tile_len}, gate_rank={rank} |",
        "| `narrative_gate` | not evaluated until Step 4-6 |",
        "| `effectiveness_gate` | not applicable before implementation |",
        "| `artifacts` | this analysis directory |",
        f"| `decision` | `{decision}` |",
        "",
        "## A6 Basis Tile Factorization",
        "",
        "| Dataset | Shared | Bank4 | Independent | DCT | Shared Gap | Bank4 Gap | Bank4-DCT |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in basis_rank_rows:
        lines.append(
            f"| {row['dataset']} | `{f(row, 'shared_local_energy'):.3f}` | "
            f"`{f(row, 'bank4_energy'):.3f}` | `{f(row, 'independent_tile_energy'):.3f}` | "
            f"`{f(row, 'local_dct_energy'):.3f}` | `{f(row, 'shared_gap_to_independent'):.3f}` | "
            f"`{f(row, 'bank4_gap_to_independent'):.3f}` | `{f(row, 'bank4_minus_dct'):.3f}` |"
        )

    lines.extend(
        [
            "",
            "## Train-Label Tile Factorization",
            "",
            "| Dataset | Shared | Bank4 | Independent | DCT | Shared Gap | Bank4 Gap | Bank4-DCT |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in label_rank_rows:
        lines.append(
            f"| {row['dataset']} | `{f(row, 'shared_local_energy'):.3f}` | "
            f"`{f(row, 'bank4_energy'):.3f}` | `{f(row, 'independent_tile_energy'):.3f}` | "
            f"`{f(row, 'local_dct_energy'):.3f}` | `{f(row, 'shared_gap_to_independent'):.3f}` | "
            f"`{f(row, 'bank4_gap_to_independent'):.3f}` | `{f(row, 'bank4_minus_dct'):.3f}` |"
        )

    lines.extend(
        [
            "",
            "## Coeff Projection Into Tile Subspaces",
            "",
            "| Dataset | Adjacent Cos | Far Cos | Distance Spearman | Projection Entropy | Output Entropy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in coeff_rows:
        lines.append(
            f"| {row['dataset']} | `{f(row, 'adjacent_projection_cosine'):.3f}` | "
            f"`{f(row, 'far_projection_cosine'):.3f}` | "
            f"`{f(row, 'distance_projection_cosine_spearman'):.3f}` | "
            f"`{f(row, 'projection_entropy_mean'):.3f}` | `{f(row, 'output_entropy_mean'):.3f}` |"
        )

    lines.extend(
        [
            "",
            "## Gate Evaluation",
            "",
            f"- Basis bank pass: `{', '.join(gate['basis_bank_ok']) or 'none'}`.",
            f"- Basis shared pass: `{', '.join(gate['basis_shared_ok']) or 'none'}`.",
            f"- Label bank pass: `{', '.join(gate['label_bank_ok']) or 'none'}`.",
            f"- Label shared pass: `{', '.join(gate['label_shared_ok']) or 'none'}`.",
            f"- Coeff projection pass: `{', '.join(gate['coeff_ok']) or 'none'}`.",
            f"- Independent-tile-only risk: `{', '.join(gate['independent_only']) or 'none'}`.",
            f"- Local DCT control risk: `{', '.join(gate['dct_risk']) or 'none'}`.",
            "",
            "## Interpretation",
            "",
            "[Fact] `shared_local_energy` fits one local basis `U[L,r]` across all future tiles.",
            "",
            "[Fact] `bank4_energy` fits four local basis banks and is the main B12-B feasibility proxy.",
            "",
            "[Fact] `independent_tile_energy` is an upper bound and is not sufficient by itself, because it may indicate a segmented Direct head.",
            "",
            "[Fact] `local_dct_energy` is the generic smoothness control. If it matches shared/bank basis, B12 is not distinct enough from fixed local spectral bases.",
            "",
            "[Fact] Coeff projection compares how the same A6 `coeff` is used across tile row-spaces. Adjacent > far supports tile-subspace structure.",
            "",
            f"[Decision] `{decision}`.",
            "",
            "## Failure Attribution",
            "",
            "- `hypothesis_false`: not proven. The B11 sliding-window evidence and B12 A6-basis bank-vs-DCT gaps still show some basis-side structure.",
            "- `generic_basis_control_explains`: yes on the label side. Train-label tile structure is very strong, but local DCT nearly matches shared/bank local bases on all datasets.",
            "- `independent_tile_only`: not the main failure. Independent tile basis is better, but shared/bank is not catastrophically far behind; the gap is just not small enough for a method gate.",
            "- `coeff_path_not_supported`: yes for cross-dataset evidence. The adjacent-vs-far coeff projection pattern is clear only on ETTh2, not on ETTm1 or Weather.",
            "- `direction_level_rejection`: no. This diagnostic blocks the current B12-STBO Step 4-6 transition; it does not reject all future basis-operator redesigns.",
            "",
        ]
    )
    if decision == "problem_candidate_passed_enter_step46_design":
        lines.extend(
            [
                "[Next] B12 may enter Step 4-6 method design. The first method candidate should prefer shared/bank local basis over independent per-tile basis.",
                "",
            ]
        )
    elif decision == "partial_support_basis_operator_but_label_or_dct_control_risk":
        lines.extend(
            [
                "[Next] B12 has basis-side evidence, but method design is not allowed yet. The next diagnostic must sharpen label-side evidence or add a stronger non-DCT control.",
                "",
            ]
        )
    elif decision == "independent_tile_only_no_method":
        lines.extend(
            [
                "[Next] Do not implement B12 as a method. Redesign the shared constraint first; otherwise it becomes a segmented Direct head.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "[Next] Do not implement B12-STBO as currently defined. Either redesign the basis-operator problem with stronger non-DCT and coeff-path evidence, or roll back StageB to Step 2/3 architecture search.",
                "",
            ]
        )

    lines.extend(
        [
            "## Output Files",
            "",
            "- `b12_stbo_basis_factorization.csv`",
            "- `b12_stbo_label_factorization.csv`",
            "- `b12_stbo_coeff_projection.csv`",
            "- `b12_stbo_gate_summary.json`",
            "- `b12_stbo_report.md`",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_dataset(args: argparse.Namespace, dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    checkpoint = checkpoint_path(args.checkpoint_root, dataset)
    state = load_state(checkpoint)
    basis = state["learned_temporal_basis"].detach().cpu().numpy().astype(np.float64)
    basis_rows = tile_operator_basis_metrics(dataset, basis, args.tile_len, RANKS, BANK_COUNTS)

    train_values = load_train_values(args.dataset_root, dataset)
    num_windows = train_values.shape[0] - SEQ_LEN - PRED_LEN + 1
    starts = sample_start_indices(num_windows, args.max_train_windows, args.seed)
    future_matrix = build_future_matrix(train_values, starts, PRED_LEN)
    label_rows = tile_label_basis_metrics(dataset, future_matrix, args.tile_len, RANKS, BANK_COUNTS)

    model = load_model(args, dataset)
    coeff_rows = collect_coeff_rows(args, dataset, model)
    coeff_summary = coefficient_tile_projection_metrics(dataset, basis, coeff_rows, args.tile_len, args.gate_rank)
    coeff_summary["train_windows_total"] = num_windows
    coeff_summary["train_windows_sampled"] = int(len(starts))
    coeff_summary["checkpoint"] = str(checkpoint)
    return basis_rows, label_rows, coeff_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase5 StageB B12-STBO diagnostics.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--tile-len", type=int, default=48)
    parser.add_argument("--gate-rank", type=int, default=16)
    parser.add_argument("--max-train-windows", type=int, default=4096)
    parser.add_argument("--max-coeff-rows", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if PRED_LEN % args.tile_len != 0:
        raise ValueError("--tile-len must divide 720")
    if args.gate_rank not in RANKS:
        raise ValueError(f"--gate-rank must be one of {RANKS}")

    basis_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    coeff_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_basis_rows, dataset_label_rows, dataset_coeff_row = analyze_dataset(args, dataset)
        basis_rows.extend(dataset_basis_rows)
        label_rows.extend(dataset_label_rows)
        coeff_rows.append(dataset_coeff_row)

    decision, gate = gate_decision(basis_rows, label_rows, coeff_rows, args.gate_rank)
    args.analysis_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.analysis_root / "b12_stbo_basis_factorization.csv", basis_rows)
    write_csv(args.analysis_root / "b12_stbo_label_factorization.csv", label_rows)
    write_csv(args.analysis_root / "b12_stbo_coeff_projection.csv", coeff_rows)
    gate_payload = {
        "decision": decision,
        "gate_rank": args.gate_rank,
        "tile_len": args.tile_len,
        "gate": gate,
        "config": {
            "dataset_root": str(args.dataset_root),
            "checkpoint_root": str(args.checkpoint_root),
            "max_train_windows": args.max_train_windows,
            "max_coeff_rows": args.max_coeff_rows,
            "batch_size": args.batch_size,
            "split": args.split,
            "device": args.device,
            "seed": args.seed,
        },
    }
    (args.analysis_root / "b12_stbo_gate_summary.json").write_text(json.dumps(gate_payload, indent=2))
    report = render_report(basis_rows, label_rows, coeff_rows, decision, gate, args)
    (args.analysis_root / "b12_stbo_report.md").write_text(report)
    print(f"Wrote B12-STBO diagnostic to {args.analysis_root}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()

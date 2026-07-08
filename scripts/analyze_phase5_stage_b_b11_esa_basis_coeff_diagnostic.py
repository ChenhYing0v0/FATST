from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
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
SEGMENTS = (
    ("early_0_96", 0, 96),
    ("mid_96_192", 96, 192),
    ("late_192_336", 192, 336),
    ("tail_336_720", 336, 720),
)
DEFAULT_ANALYSIS_ROOT = Path("analysis/phase5_stage_b_b11_esa_basis_coeff_diagnostic_20260708")
DEFAULT_CHECKPOINT_ROOT = (
    Path("analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707")
    / "raw"
    / "official-last"
    / "TimeAlignOfficialUnified720_a6_clean_official-last"
)
DEFAULT_DATASET_ROOT = Path("/Users/river/PaperResearch/Project/datasets")


@dataclass(frozen=True)
class DatasetResult:
    dataset: str
    summary_rows: list[dict[str, Any]]
    cluster_rows: list[dict[str, Any]]
    usage_rows: list[dict[str, Any]]
    pair_rows: list[dict[str, Any]]
    sliding_rows: list[dict[str, Any]]
    sliding_pair_rows: list[dict[str, Any]]


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


def build_args(args: argparse.Namespace, dataset: str) -> argparse.Namespace:
    preset = OFFICIAL_PRESETS[dataset][720]
    adapter_args = SimpleNamespace(
        dataset_root=args.dataset_root,
        output_dir=args.analysis_root / "_tmp_official_args",
        dataset=dataset,
        seq_len=720,
        label_len=48,
        pred_len=720,
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
        stbo_tile_len=48,
        stbo_rank=16,
        stbo_bank_count=4,
        stbo_basis_init_std=16 ** -0.5,
    )
    official_args = build_official_args(adapter_args, preset)
    official_args.batch_size = args.batch_size
    official_args.num_workers = 0
    return official_args


def load_model(official_args: argparse.Namespace, checkpoint: Path) -> Model:
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = Model(official_args)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def normalize_rows(values: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(denom, 1e-12, None)


def deterministic_kmeans(features: np.ndarray, clusters: int, max_iter: int = 100) -> np.ndarray:
    if clusters <= 0 or clusters > features.shape[0]:
        raise ValueError("invalid number of clusters")
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


def stage_labels() -> np.ndarray:
    labels = np.empty(720, dtype=np.int64)
    for segment_idx, (_name, start, end) in enumerate(SEGMENTS):
        labels[start:end] = segment_idx
    return labels


def entropy(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    _values, counts = np.unique(labels, return_counts=True)
    probs = counts.astype(np.float64) / labels.size
    return float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))


def normalized_mutual_info(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("label arrays must have the same shape")
    h_left = entropy(left)
    h_right = entropy(right)
    if h_left <= 0.0 or h_right <= 0.0:
        return 0.0
    mi = 0.0
    n = left.size
    for left_value in np.unique(left):
        left_mask = left == left_value
        p_left = float(np.sum(left_mask)) / n
        for right_value in np.unique(right):
            right_mask = right == right_value
            p_joint = float(np.sum(left_mask & right_mask)) / n
            if p_joint <= 0.0:
                continue
            p_right = float(np.sum(right_mask)) / n
            mi += p_joint * np.log(p_joint / (p_left * p_right))
    return float(mi / np.sqrt(h_left * h_right))


def run_count(indices: np.ndarray) -> int:
    if indices.size == 0:
        return 0
    return int(1 + np.sum(np.diff(indices) > 1))


def cluster_subspace(basis: np.ndarray, labels: np.ndarray, cluster_idx: int, rank: int) -> np.ndarray:
    rows = basis[labels == cluster_idx]
    if rows.size == 0:
        return np.zeros((basis.shape[1], 0), dtype=np.float64)
    _u, singular_values, vh = np.linalg.svd(rows, full_matrices=False)
    keep = min(rank, int(np.sum(singular_values > 1e-10)), vh.shape[0])
    if keep <= 0:
        return np.zeros((basis.shape[1], 0), dtype=np.float64)
    return vh[:keep].T


def canonical_overlap(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape[1] == 0 or right.shape[1] == 0:
        return 0.0
    singular_values = np.linalg.svd(left.T @ right, compute_uv=False)
    return float(np.mean(singular_values * singular_values))


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


def collect_coeff_rows(
    args: argparse.Namespace,
    model: Model,
    official_args: argparse.Namespace,
) -> np.ndarray:
    _data, loader = data_provider(official_args, args.split)
    parts: list[np.ndarray] = []
    total = 0
    device = torch.device(args.device)
    with torch.no_grad():
        for batch_x, _batch_y, _batch_x_mark, _batch_y_mark in loader:
            if total >= args.max_rows:
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
            rows = coeff.reshape(-1, coeff.shape[-1]).detach().cpu().numpy().astype(np.float64)
            take = min(args.max_rows - total, rows.shape[0])
            parts.append(rows[:take])
            total += take
    if not parts:
        raise RuntimeError("No coeff rows collected")
    return np.concatenate(parts, axis=0)


def analyze_basis_clusters(
    dataset: str,
    basis: np.ndarray,
    labels: np.ndarray,
    clusters: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage = stage_labels()
    cluster_rows: list[dict[str, Any]] = []
    locality_values: list[float] = []
    stage_purity_values: list[float] = []
    for cluster_idx in range(clusters):
        indices = np.where(labels == cluster_idx)[0]
        if indices.size == 0:
            continue
        cluster_stage = stage[indices]
        _stage_values, stage_counts = np.unique(cluster_stage, return_counts=True)
        max_stage_share = float(np.max(stage_counts) / indices.size)
        stage_purity_values.append(max_stage_share)
        runs = run_count(indices)
        span = int(indices[-1] - indices[0] + 1)
        locality = float(indices.size / max(span, 1))
        locality_values.append(locality)
        row_norms = np.linalg.norm(basis[indices], axis=1)
        row: dict[str, Any] = {
            "dataset": dataset,
            "clusters": clusters,
            "cluster": cluster_idx,
            "size": int(indices.size),
            "start": int(indices[0]),
            "end": int(indices[-1] + 1),
            "span": span,
            "runs": runs,
            "locality": locality,
            "max_stage_share": max_stage_share,
            "row_norm_mean": float(np.mean(row_norms)),
            "row_norm_std": float(np.std(row_norms)),
        }
        for segment_idx, (name, _start, _end) in enumerate(SEGMENTS):
            row[f"share_{name}"] = float(np.mean(cluster_stage == segment_idx))
        cluster_rows.append(row)
    labels_nmi = normalized_mutual_info(labels, stage)
    transitions = int(np.sum(labels[1:] != labels[:-1]))
    summary = {
        "dataset": dataset,
        "clusters": clusters,
        "cluster_stage_nmi": labels_nmi,
        "cluster_transitions": transitions,
        "mean_cluster_locality": float(np.mean(locality_values)) if locality_values else float("nan"),
        "mean_cluster_stage_purity": float(np.mean(stage_purity_values)) if stage_purity_values else float("nan"),
    }
    return cluster_rows, summary


def analyze_coeff_usage(
    dataset: str,
    basis: np.ndarray,
    labels: np.ndarray,
    coeff_rows: np.ndarray,
    clusters: int,
    rank: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    subspaces = {
        cluster_idx: cluster_subspace(basis, labels, cluster_idx, rank)
        for cluster_idx in range(clusters)
    }
    coeff_norm_sq = np.sum(coeff_rows * coeff_rows, axis=1)
    coeff_norm_sq = np.clip(coeff_norm_sq, 1e-12, None)
    projections: dict[int, np.ndarray] = {}
    projection_shares: dict[int, np.ndarray] = {}
    output_energies: dict[int, np.ndarray] = {}
    for cluster_idx in range(clusters):
        q = subspaces[cluster_idx]
        coords = coeff_rows @ q
        projected = coords @ q.T
        projections[cluster_idx] = projected
        projection_shares[cluster_idx] = np.sum(coords * coords, axis=1) / coeff_norm_sq
        rows = basis[labels == cluster_idx]
        output = coeff_rows @ rows.T
        output_energies[cluster_idx] = np.sum(output * output, axis=1)

    output_stack = np.stack([output_energies[idx] for idx in range(clusters)], axis=1)
    output_share = output_stack / np.clip(np.sum(output_stack, axis=1, keepdims=True), 1e-12, None)
    proj_stack = np.stack([projection_shares[idx] for idx in range(clusters)], axis=1)
    proj_share_norm = proj_stack / np.clip(np.sum(proj_stack, axis=1, keepdims=True), 1e-12, None)

    usage_rows: list[dict[str, Any]] = []
    for cluster_idx in range(clusters):
        usage_rows.append(
            {
                "dataset": dataset,
                "clusters": clusters,
                "rank": rank,
                "cluster": cluster_idx,
                "projection_share_mean": float(np.mean(projection_shares[cluster_idx])),
                "projection_share_std": float(np.std(projection_shares[cluster_idx])),
                "projection_share_normalized_mean": float(np.mean(proj_share_norm[:, cluster_idx])),
                "output_energy_share_mean": float(np.mean(output_share[:, cluster_idx])),
                "output_energy_share_std": float(np.std(output_share[:, cluster_idx])),
            }
        )

    pair_rows: list[dict[str, Any]] = []
    pair_cosines: list[float] = []
    pair_overlaps: list[float] = []
    for left in range(clusters):
        for right in range(left + 1, clusters):
            overlap = canonical_overlap(subspaces[left], subspaces[right])
            cosine = float(np.mean(cosine_rows(projections[left], projections[right])))
            pair_overlaps.append(overlap)
            pair_cosines.append(cosine)
            pair_rows.append(
                {
                    "dataset": dataset,
                    "clusters": clusters,
                    "rank": rank,
                    "left_cluster": left,
                    "right_cluster": right,
                    "subspace_overlap": overlap,
                    "projection_cosine": cosine,
                }
            )

    output_entropy = -np.sum(output_share * np.log(np.clip(output_share, 1e-12, None)), axis=1) / np.log(clusters)
    proj_entropy = -np.sum(proj_share_norm * np.log(np.clip(proj_share_norm, 1e-12, None)), axis=1) / np.log(clusters)
    summary = {
        "projection_pair_cosine_mean": float(np.mean(pair_cosines)) if pair_cosines else float("nan"),
        "subspace_pair_overlap_mean": float(np.mean(pair_overlaps)) if pair_overlaps else float("nan"),
        "projection_entropy_mean": float(np.mean(proj_entropy)),
        "output_energy_entropy_mean": float(np.mean(output_entropy)),
        "max_output_cluster_share_mean": float(np.mean(np.max(output_share, axis=1))),
        "max_projection_cluster_share_mean": float(np.mean(np.max(proj_share_norm, axis=1))),
    }
    return usage_rows, pair_rows, summary


def window_subspace(basis: np.ndarray, start: int, end: int, rank: int) -> np.ndarray:
    rows = basis[start:end]
    _u, singular_values, vh = np.linalg.svd(rows, full_matrices=False)
    keep = min(rank, int(np.sum(singular_values > 1e-10)), vh.shape[0])
    if keep <= 0:
        return np.zeros((basis.shape[1], 0), dtype=np.float64)
    return vh[:keep].T


def analyze_sliding_windows(
    dataset: str,
    basis: np.ndarray,
    coeff_rows: np.ndarray,
    rank: int,
    window_len: int,
    stride: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    windows: list[tuple[int, int, float]] = []
    for start in range(0, basis.shape[0] - window_len + 1, stride):
        end = start + window_len
        windows.append((start, end, 0.5 * (start + end)))
    if windows[-1][1] != basis.shape[0]:
        start = basis.shape[0] - window_len
        end = basis.shape[0]
        center = 0.5 * (start + end)
        if windows[-1][0] != start:
            windows.append((start, end, center))

    subspaces = [window_subspace(basis, start, end, rank) for start, end, _center in windows]
    projections = []
    output_energies = []
    coeff_norm_sq = np.sum(coeff_rows * coeff_rows, axis=1)
    coeff_norm_sq = np.clip(coeff_norm_sq, 1e-12, None)
    for (start, end, _center), q in zip(windows, subspaces):
        coords = coeff_rows @ q
        projections.append(coords @ q.T)
        rows = basis[start:end]
        output = coeff_rows @ rows.T
        output_energies.append(np.sum(output * output, axis=1))
    output_stack = np.stack(output_energies, axis=1)
    output_share = output_stack / np.clip(np.sum(output_stack, axis=1, keepdims=True), 1e-12, None)
    output_entropy = -np.sum(output_share * np.log(np.clip(output_share, 1e-12, None)), axis=1) / np.log(len(windows))

    pair_rows: list[dict[str, Any]] = []
    distances: list[float] = []
    overlaps: list[float] = []
    projection_cosines: list[float] = []
    adjacent_overlaps: list[float] = []
    adjacent_cosines: list[float] = []
    far_overlaps: list[float] = []
    far_cosines: list[float] = []
    for left in range(len(windows)):
        for right in range(left + 1, len(windows)):
            left_start, left_end, left_center = windows[left]
            right_start, right_end, right_center = windows[right]
            distance = abs(right_center - left_center)
            overlap = canonical_overlap(subspaces[left], subspaces[right])
            cosine = float(np.mean(cosine_rows(projections[left], projections[right])))
            distances.append(distance)
            overlaps.append(overlap)
            projection_cosines.append(cosine)
            if right == left + 1:
                adjacent_overlaps.append(overlap)
                adjacent_cosines.append(cosine)
            if distance >= 240:
                far_overlaps.append(overlap)
                far_cosines.append(cosine)
            pair_rows.append(
                {
                    "dataset": dataset,
                    "window_len": window_len,
                    "stride": stride,
                    "rank": rank,
                    "left_window": left,
                    "right_window": right,
                    "left_start": left_start,
                    "left_end": left_end,
                    "right_start": right_start,
                    "right_end": right_end,
                    "center_distance": distance,
                    "subspace_overlap": overlap,
                    "projection_cosine": cosine,
                }
            )

    summary = [
        {
            "dataset": dataset,
            "window_len": window_len,
            "stride": stride,
            "rank": rank,
            "windows": len(windows),
            "adjacent_subspace_overlap_mean": float(np.mean(adjacent_overlaps)),
            "far_subspace_overlap_mean": float(np.mean(far_overlaps)),
            "distance_subspace_overlap_spearman": spearman(distances, overlaps),
            "adjacent_projection_cosine_mean": float(np.mean(adjacent_cosines)),
            "far_projection_cosine_mean": float(np.mean(far_cosines)),
            "distance_projection_cosine_spearman": spearman(distances, projection_cosines),
            "output_energy_entropy_mean": float(np.mean(output_entropy)),
            "max_output_window_share_mean": float(np.mean(np.max(output_share, axis=1))),
        }
    ]
    return summary, pair_rows


def analyze_dataset(args: argparse.Namespace, dataset: str) -> DatasetResult:
    official_args = build_args(args, dataset)
    model = load_model(official_args, checkpoint_path(args.checkpoint_root, dataset))
    device = torch.device(args.device)
    model.to(device)
    basis = model.learned_temporal_basis.detach().cpu().numpy().astype(np.float64)
    coeff_rows = collect_coeff_rows(args, model, official_args)
    features = normalize_rows(basis)

    all_summary_rows: list[dict[str, Any]] = []
    all_cluster_rows: list[dict[str, Any]] = []
    all_usage_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    all_sliding_rows: list[dict[str, Any]] = []
    all_sliding_pair_rows: list[dict[str, Any]] = []

    for clusters in args.clusters:
        labels = deterministic_kmeans(features, clusters, args.kmeans_iters)
        cluster_rows, basis_summary = analyze_basis_clusters(dataset, basis, labels, clusters)
        usage_rows, pair_rows, usage_summary = analyze_coeff_usage(
            dataset,
            basis,
            labels,
            coeff_rows,
            clusters,
            args.subspace_rank,
        )
        summary = {
            **basis_summary,
            "subspace_rank": args.subspace_rank,
            "coeff_rows": int(coeff_rows.shape[0]),
            **usage_summary,
        }
        if clusters == 4:
            if summary["cluster_stage_nmi"] >= 0.50 and summary["projection_pair_cosine_mean"] <= 0.50:
                decision = "supports_emergent_subspace_utilization"
            elif summary["projection_pair_cosine_mean"] <= 0.50:
                decision = "supports_directional_usage_but_basis_clusters_not_temporally_clear"
            else:
                decision = "weak_emergent_subspace_evidence"
            summary["decision_hint"] = decision
        else:
            summary["decision_hint"] = "auxiliary_resolution"
        all_summary_rows.append(summary)
        all_cluster_rows.extend(cluster_rows)
        all_usage_rows.extend(usage_rows)
        all_pair_rows.extend(pair_rows)

    sliding_rows, sliding_pair_rows = analyze_sliding_windows(
        dataset,
        basis,
        coeff_rows,
        args.subspace_rank,
        args.window_len,
        args.window_stride,
    )
    all_sliding_rows.extend(sliding_rows)
    all_sliding_pair_rows.extend(sliding_pair_rows)

    return DatasetResult(
        dataset=dataset,
        summary_rows=all_summary_rows,
        cluster_rows=all_cluster_rows,
        usage_rows=all_usage_rows,
        pair_rows=all_pair_rows,
        sliding_rows=all_sliding_rows,
        sliding_pair_rows=all_sliding_pair_rows,
    )


def fmt_value(value: Any) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt_value(row[field]) for field in fields) + " |")
    return "\n".join(lines)


def decision_lines(summary_rows: list[dict[str, Any]], sliding_rows: list[dict[str, Any]]) -> list[str]:
    rows4 = [row for row in summary_rows if row["clusters"] == 4]
    nmi = np.asarray([row["cluster_stage_nmi"] for row in rows4], dtype=np.float64)
    proj_cos = np.asarray([row["projection_pair_cosine_mean"] for row in rows4], dtype=np.float64)
    output_entropy = np.asarray([row["output_energy_entropy_mean"] for row in rows4], dtype=np.float64)
    adjacent_overlap = np.asarray([row["adjacent_subspace_overlap_mean"] for row in sliding_rows], dtype=np.float64)
    far_overlap = np.asarray([row["far_subspace_overlap_mean"] for row in sliding_rows], dtype=np.float64)
    distance_overlap_corr = np.asarray(
        [row["distance_subspace_overlap_spearman"] for row in sliding_rows],
        dtype=np.float64,
    )
    lines = [
        "## Decision",
        "",
        f"[Fact] K=4 basis-row clustering 的平均 stage NMI 为 `{np.mean(nmi):.4f}`。",
        f"[Fact] K=4 emergent subspace 中 coeff projection pair cosine 平均为 `{np.mean(proj_cos):.4f}`。",
        f"[Fact] K=4 output energy entropy 平均为 `{np.mean(output_entropy):.4f}`。",
        f"[Fact] Sliding-window adjacent/far subspace overlap 为 `{np.mean(adjacent_overlap):.4f}` / `{np.mean(far_overlap):.4f}`。",
        f"[Fact] Sliding-window distance vs overlap Spearman 为 `{np.mean(distance_overlap_corr):.4f}`。",
        "",
    ]
    if (
        float(np.mean(adjacent_overlap)) > float(np.mean(far_overlap)) * 1.25
        and float(np.mean(distance_overlap_corr)) <= -0.50
        and float(np.mean(proj_cos)) <= 0.50
    ):
        lines.extend(
            [
                "[Decision] 诊断支持 `B11-ESA` 进入 Step 4-6：相比硬聚类，sliding-window subspace",
                "显示 A6 basis 沿未来时间轴存在连续变化的 geometry，且真实 coeff 在这些 subspaces 上的",
                "投影方向不是简单同向复制。",
                "",
                "[Next] 下一步应设计连续 basis-conditioned subspace aggregation，而不是显式 stage encoding。",
                "method gate 必须包含 no-basis / shuffled-basis / constant-slot controls。",
            ]
        )
    elif float(np.mean(proj_cos)) <= 0.50:
        lines.extend(
            [
                "[Decision] 诊断只部分支持 B11：coeff 使用方向存在差异，但 basis-row clustering 与 future",
                "regions 的关系不够清晰。下一步应先改进 subspace discovery，而不是实现方法。",
            ]
        )
    else:
        lines.extend(
            [
                "[Decision] 当前诊断不支持 B11 method design：emergent basis clusters 或 coeff usage direction",
                "不足以支撑新的架构贡献。",
            ]
        )
    return lines


def write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
    sliding_rows: list[dict[str, Any]],
) -> None:
    summary_fields = [
        "dataset",
        "clusters",
        "subspace_rank",
        "coeff_rows",
        "cluster_stage_nmi",
        "cluster_transitions",
        "mean_cluster_locality",
        "mean_cluster_stage_purity",
        "subspace_pair_overlap_mean",
        "projection_pair_cosine_mean",
        "projection_entropy_mean",
        "output_energy_entropy_mean",
        "max_projection_cluster_share_mean",
        "max_output_cluster_share_mean",
        "decision_hint",
    ]
    cluster_fields = [
        "dataset",
        "clusters",
        "cluster",
        "size",
        "start",
        "end",
        "span",
        "runs",
        "locality",
        "max_stage_share",
        "share_early_0_96",
        "share_mid_96_192",
        "share_late_192_336",
        "share_tail_336_720",
    ]
    sliding_fields = [
        "dataset",
        "window_len",
        "stride",
        "rank",
        "windows",
        "adjacent_subspace_overlap_mean",
        "far_subspace_overlap_mean",
        "distance_subspace_overlap_spearman",
        "adjacent_projection_cosine_mean",
        "far_projection_cosine_mean",
        "distance_projection_cosine_spearman",
        "output_energy_entropy_mean",
        "max_output_window_share_mean",
    ]
    lines = [
        "# Phase5 StageB B11-ESA Basis/Coeff Diagnostic",
        "",
        "## 阶段记录",
        "",
        "| 字段 | 内容 |",
        "| --- | --- |",
        "| `candidate_id` | `B11-ESA` |",
        "| `current_step` | StageB Step 2/3：emergent basis-subspace problem diagnostic |",
        "| `problem` | A6 不应依赖显式 stage encoding；需要诊断 basis 是否自发形成 subspaces，以及 coeff 是否沿这些 subspaces 差异化使用 |",
        "| `scope` | Frozen clean A6 checkpoint；不训练模型；不做 method claim |",
        "",
        "## Diagnostic Design",
        "",
        "本诊断不输入 stage token，也不按 horizon 强行分段。它对 `learned_temporal_basis[720,256]`",
        "的 row directions 做 deterministic KMeans，得到 emergent basis row clusters；随后把真实 forward",
        "中的 `coeff[B,C,256]` 投影到每个 cluster 的 row subspace，检查使用方向是否同向。",
        "",
        "`cluster_stage_nmi` 只作为事后解释：它衡量自发 cluster 与 benchmark future regions 的关系，",
        "不是训练或聚类输入。",
        "",
        "## Summary",
        "",
        markdown_table(summary_rows, summary_fields),
        "",
        "## Cluster Detail",
        "",
        markdown_table(cluster_rows, cluster_fields),
        "",
        "## Sliding Window Subspace",
        "",
        markdown_table(sliding_rows, sliding_fields),
        "",
    ]
    lines.extend(decision_lines(summary_rows, sliding_rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one int")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B11 emergent subspace aggregation diagnostic.")
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--clusters", type=parse_int_list, default=parse_int_list("4,8"))
    parser.add_argument("--subspace-rank", type=int, default=16)
    parser.add_argument("--window-len", type=int, default=96)
    parser.add_argument("--window-stride", type=int, default=48)
    parser.add_argument("--max-rows", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--kmeans-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    summary_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    sliding_rows: list[dict[str, Any]] = []
    sliding_pair_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        result = analyze_dataset(args, dataset)
        summary_rows.extend(result.summary_rows)
        cluster_rows.extend(result.cluster_rows)
        usage_rows.extend(result.usage_rows)
        pair_rows.extend(result.pair_rows)
        sliding_rows.extend(result.sliding_rows)
        sliding_pair_rows.extend(result.sliding_pair_rows)

    write_csv(args.analysis_root / "b11_esa_basis_coeff_summary.csv", summary_rows)
    write_csv(args.analysis_root / "b11_esa_basis_clusters.csv", cluster_rows)
    write_csv(args.analysis_root / "b11_esa_coeff_usage.csv", usage_rows)
    write_csv(args.analysis_root / "b11_esa_subspace_pairs.csv", pair_rows)
    write_csv(args.analysis_root / "b11_esa_sliding_windows.csv", sliding_rows)
    write_csv(args.analysis_root / "b11_esa_sliding_pairs.csv", sliding_pair_rows)
    write_report(args.analysis_root / "b11_esa_basis_coeff_report.md", summary_rows, cluster_rows, sliding_rows)


if __name__ == "__main__":
    main()

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
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from data_provider.data_factory import data_provider  # noqa: E402
from models.TimeAlign import Model  # noqa: E402
from train_repo import OFFICIAL_PRESETS, build_official_args  # noqa: E402


DATASETS = ("ETTh2", "ETTm1", "Weather")
HORIZONS = (96, 192, 336, 720)
PRED_LEN = 720
DEFAULT_UNIT_SIZES = (120, 144, 180, 240, 360)
DEFAULT_MAIN_UNIT_SIZES = (120, 144, 180, 240)
DEFAULT_ANALYSIS_ROOT = Path(
    "analysis/phase5_stage_b_b13_future_unit_granularity_20260710"
)
DEFAULT_CHECKPOINT_ROOT = (
    Path("analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707")
    / "raw"
    / "TimeAlignOfficialUnified720_A6LBF_r256_main_official-last"
)
DEFAULT_DATASET_ROOT = Path("/Users/river/PaperResearch/Project/datasets")


@dataclass(frozen=True)
class DatasetResult:
    dataset: str
    batch_rows: list[dict[str, Any]]
    gradient_pair_rows: list[dict[str, Any]]
    basis_pair_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    bootstrap_rows: list[dict[str, Any]]


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
    preset = OFFICIAL_PRESETS[dataset][PRED_LEN]
    adapter_args = SimpleNamespace(
        dataset_root=args.dataset_root,
        output_dir=args.analysis_root / "_tmp_official_args",
        dataset=dataset,
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


def forward_full_with_coeff(
    model: Model,
    batch_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, channels = batch_x.shape
    x = model.normalization_x(batch_x, "norm")
    x = model.patch_emb_x(x.permute(0, 2, 1).reshape(-1, channels * seq_len))
    for layer_idx in range(model.e_layers):
        x = x + model.encoder[layer_idx](x)
        if model.layer_norm:
            x = model.norm_x[layer_idx](x)
    hidden = x.reshape(batch, channels, model.patch_num, model.d_model).flatten(start_dim=-2)
    coeff = model.learned_basis_coeff(hidden)
    basis = model.learned_temporal_basis.to(dtype=hidden.dtype)
    bias = model.learned_temporal_bias.to(dtype=hidden.dtype)
    output = torch.einsum("hk,bck->bch", basis, coeff) + bias.view(1, 1, -1)
    output = model.normalization_x(output.permute(0, 2, 1), "denorm")
    return output, coeff


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    denom = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)
    if float(denom.detach().cpu()) <= 1e-12:
        return float("nan")
    value = torch.dot(left_flat, right_flat) / denom
    return float(value.detach().cpu())


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
    x_rank = rankdata(x[mask])
    y_rank = rankdata(y[mask])
    x_rank -= np.mean(x_rank)
    y_rank -= np.mean(y_rank)
    denom = np.linalg.norm(x_rank) * np.linalg.norm(y_rank)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x_rank, y_rank) / denom)


def unit_ranges(unit_size: int) -> list[tuple[int, int, int]]:
    return [
        (unit_idx, start, start + unit_size)
        for unit_idx, start in enumerate(range(0, PRED_LEN, unit_size))
    ]


def pair_kind(distance: int, unit_count: int) -> str:
    if distance == 1:
        return "adjacent"
    if distance >= int(np.ceil(unit_count / 2.0)):
        return "far"
    return "middle"


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def summarize_batch(
    dataset: str,
    unit_size: int,
    batch_idx: int,
    losses: list[float],
    grads: list[torch.Tensor],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    unit_count = len(grads)
    pair_rows: list[dict[str, Any]] = []
    pair_cosines: list[float] = []
    adjacent_cosines: list[float] = []
    far_cosines: list[float] = []

    for left in range(unit_count):
        for right in range(left + 1, unit_count):
            distance = right - left
            kind = pair_kind(distance, unit_count)
            value = cosine(grads[left], grads[right])
            pair_rows.append(
                {
                    "dataset": dataset,
                    "unit_size": unit_size,
                    "unit_count": unit_count,
                    "batch_idx": batch_idx,
                    "left_unit": left,
                    "right_unit": right,
                    "unit_distance": distance,
                    "pair_kind": kind,
                    "gradient_cosine": value,
                }
            )
            if np.isfinite(value):
                pair_cosines.append(value)
                if kind == "adjacent":
                    adjacent_cosines.append(value)
                elif kind == "far":
                    far_cosines.append(value)

    grad_norms = [float(torch.linalg.vector_norm(grad).cpu()) for grad in grads]
    grad_sum = torch.stack(grads, dim=0).sum(dim=0)
    denominator = unit_count * sum(norm * norm for norm in grad_norms)
    alignment_efficiency = (
        float(torch.linalg.vector_norm(grad_sum).cpu()) ** 2 / denominator
        if denominator > 1e-12
        else float("nan")
    )
    first_last = cosine(grads[0], grads[-1])
    row = {
        "dataset": dataset,
        "unit_size": unit_size,
        "unit_count": unit_count,
        "batch_idx": batch_idx,
        "loss_mean": finite_mean(losses),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "mean_pairwise_cosine": finite_mean(pair_cosines),
        "min_pairwise_cosine": float(np.min(pair_cosines)) if pair_cosines else float("nan"),
        "adjacent_cosine": finite_mean(adjacent_cosines),
        "far_cosine": finite_mean(far_cosines),
        "adjacent_minus_far_cosine": (
            finite_mean(adjacent_cosines) - finite_mean(far_cosines)
            if adjacent_cosines and far_cosines
            else float("nan")
        ),
        "first_last_cosine": first_last,
        "negative_pair_rate": (
            float(np.mean(np.asarray(pair_cosines) < 0.0)) if pair_cosines else float("nan")
        ),
        "max_min_grad_norm_ratio": max(grad_norms) / max(min(grad_norms), 1e-12),
        "shared_alignment_efficiency": alignment_efficiency,
    }
    return row, pair_rows


def analyze_gradient_batches(
    args: argparse.Namespace,
    dataset: str,
    model: Model,
    official_args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _data, loader = data_provider(official_args, args.split)
    criterion = nn.MSELoss()
    device = torch.device(args.device)
    batch_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for batch_idx, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader):
        if batch_idx >= args.max_batches:
            break
        model.zero_grad(set_to_none=True)
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        output, coeff = forward_full_with_coeff(model, batch_x)
        target = batch_y[:, -PRED_LEN:, :]

        for unit_size in args.unit_sizes:
            losses: list[float] = []
            grads: list[torch.Tensor] = []
            for _unit_idx, start, end in unit_ranges(unit_size):
                loss = criterion(output[:, start:end, :], target[:, start:end, :])
                grad = torch.autograd.grad(loss, coeff, retain_graph=True)[0]
                losses.append(float(loss.detach().cpu()))
                grads.append(grad.detach())
            batch_row, batch_pair_rows = summarize_batch(
                dataset,
                unit_size,
                batch_idx,
                losses,
                grads,
            )
            batch_rows.append(batch_row)
            pair_rows.extend(batch_pair_rows)

    if not batch_rows:
        raise RuntimeError(f"No batches analyzed for {dataset}")
    return batch_rows, pair_rows


def row_subspace(rows: np.ndarray, rank: int) -> np.ndarray:
    _u, _s, vh = np.linalg.svd(rows, full_matrices=False)
    effective_rank = min(rank, vh.shape[0])
    return vh[:effective_rank].T


def subspace_overlap(left: np.ndarray, right: np.ndarray) -> float:
    effective_rank = min(left.shape[1], right.shape[1])
    cross = left.T @ right
    return float(np.sum(cross * cross) / max(effective_rank, 1))


def analyze_basis_geometry(
    dataset: str,
    basis: np.ndarray,
    unit_sizes: list[int],
    subspace_rank: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit_size in unit_sizes:
        ranges = unit_ranges(unit_size)
        unit_count = len(ranges)
        subspaces = [
            row_subspace(basis[start:end], subspace_rank)
            for _unit_idx, start, end in ranges
        ]
        for left in range(unit_count):
            for right in range(left + 1, unit_count):
                distance = right - left
                rows.append(
                    {
                        "dataset": dataset,
                        "unit_size": unit_size,
                        "unit_count": unit_count,
                        "subspace_rank": subspace_rank,
                        "left_unit": left,
                        "right_unit": right,
                        "unit_distance": distance,
                        "pair_kind": pair_kind(distance, unit_count),
                        "basis_subspace_overlap": subspace_overlap(
                            subspaces[left],
                            subspaces[right],
                        ),
                    }
                )
    return rows


def bootstrap_metric(
    values: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"),) * 4
    estimates = np.empty(iterations, dtype=np.float64)
    for idx in range(iterations):
        sample = rng.choice(values, size=values.size, replace=True)
        estimates[idx] = float(np.mean(sample))
    return (
        float(np.mean(values)),
        float(np.quantile(estimates, 0.05)),
        float(np.quantile(estimates, 0.50)),
        float(np.quantile(estimates, 0.95)),
    )


def bootstrap_rows_for_setting(
    dataset: str,
    unit_size: int,
    rows: list[dict[str, Any]],
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    metrics = (
        "mean_pairwise_cosine",
        "first_last_cosine",
        "adjacent_minus_far_cosine",
        "shared_alignment_efficiency",
    )
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        mean, p05, p50, p95 = bootstrap_metric(values, iterations, rng)
        output.append(
            {
                "dataset": dataset,
                "unit_size": unit_size,
                "metric": metric,
                "batches": len(rows),
                "bootstrap_iterations": iterations,
                "mean": mean,
                "p05": p05,
                "p50": p50,
                "p95": p95,
            }
        )
    return output


def summarize_setting(
    dataset: str,
    unit_size: int,
    batch_rows: list[dict[str, Any]],
    gradient_pair_rows: list[dict[str, Any]],
    basis_pair_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    main_unit_sizes: set[int],
) -> dict[str, Any]:
    setting_batches = [row for row in batch_rows if row["unit_size"] == unit_size]
    setting_gradient_pairs = [
        row for row in gradient_pair_rows if row["unit_size"] == unit_size
    ]
    setting_basis_pairs = [row for row in basis_pair_rows if row["unit_size"] == unit_size]
    bootstrap_by_metric = {
        row["metric"]: row for row in bootstrap_rows if row["unit_size"] == unit_size
    }

    def batch_mean(field: str) -> float:
        return finite_mean([float(row[field]) for row in setting_batches])

    def basis_mean(kind: str) -> float:
        return finite_mean(
            [
                float(row["basis_subspace_overlap"])
                for row in setting_basis_pairs
                if row["pair_kind"] == kind
            ]
        )

    gradient_pair_means: dict[tuple[int, int], float] = {}
    for row in setting_gradient_pairs:
        key = (int(row["left_unit"]), int(row["right_unit"]))
        gradient_pair_means.setdefault(key, 0.0)
    for key in gradient_pair_means:
        gradient_pair_means[key] = finite_mean(
            [
                float(row["gradient_cosine"])
                for row in setting_gradient_pairs
                if (int(row["left_unit"]), int(row["right_unit"])) == key
            ]
        )

    basis_by_pair = {
        (int(row["left_unit"]), int(row["right_unit"])): float(
            row["basis_subspace_overlap"]
        )
        for row in setting_basis_pairs
    }
    shared_pairs = sorted(set(gradient_pair_means) & set(basis_by_pair))
    gradient_basis_spearman = spearman(
        [gradient_pair_means[key] for key in shared_pairs],
        [basis_by_pair[key] for key in shared_pairs],
    )

    distances = [float(row["unit_distance"]) for row in setting_basis_pairs]
    overlaps = [float(row["basis_subspace_overlap"]) for row in setting_basis_pairs]
    mean_bootstrap = bootstrap_by_metric["mean_pairwise_cosine"]
    first_last_bootstrap = bootstrap_by_metric["first_last_cosine"]
    robust_support = (
        unit_size in main_unit_sizes
        and float(mean_bootstrap["p95"]) < 0.50
        and float(first_last_bootstrap["p95"]) < 0.35
    )
    return {
        "dataset": dataset,
        "unit_size": unit_size,
        "unit_count": PRED_LEN // unit_size,
        "role": "main" if unit_size in main_unit_sizes else "coarse_control",
        "batches": len(setting_batches),
        "mean_pairwise_cosine": batch_mean("mean_pairwise_cosine"),
        "min_pairwise_cosine": batch_mean("min_pairwise_cosine"),
        "adjacent_cosine": batch_mean("adjacent_cosine"),
        "far_cosine": batch_mean("far_cosine"),
        "adjacent_minus_far_cosine": batch_mean("adjacent_minus_far_cosine"),
        "first_last_cosine": batch_mean("first_last_cosine"),
        "negative_pair_rate": batch_mean("negative_pair_rate"),
        "max_min_grad_norm_ratio": batch_mean("max_min_grad_norm_ratio"),
        "shared_alignment_efficiency": batch_mean("shared_alignment_efficiency"),
        "bootstrap_mean_pairwise_p95": float(mean_bootstrap["p95"]),
        "bootstrap_first_last_p95": float(first_last_bootstrap["p95"]),
        "basis_adjacent_overlap": basis_mean("adjacent"),
        "basis_far_overlap": basis_mean("far"),
        "basis_distance_overlap_spearman": spearman(distances, overlaps),
        "gradient_basis_pair_spearman": gradient_basis_spearman,
        "robust_support": robust_support,
    }


def analyze_dataset(args: argparse.Namespace, dataset: str) -> DatasetResult:
    official_args = build_args(args, dataset)
    model = load_model(official_args, checkpoint_path(args.checkpoint_root, dataset))
    model.to(torch.device(args.device))
    batch_rows, gradient_pair_rows = analyze_gradient_batches(
        args,
        dataset,
        model,
        official_args,
    )
    basis = model.learned_temporal_basis.detach().cpu().numpy().astype(np.float64)
    basis_pair_rows = analyze_basis_geometry(
        dataset,
        basis,
        args.unit_sizes,
        args.subspace_rank,
    )

    bootstrap_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for size_idx, unit_size in enumerate(args.unit_sizes):
        setting_batches = [row for row in batch_rows if row["unit_size"] == unit_size]
        setting_bootstrap = bootstrap_rows_for_setting(
            dataset,
            unit_size,
            setting_batches,
            args.bootstrap_iterations,
            args.seed + size_idx * 1009 + DATASETS.index(dataset) * 10007,
        )
        bootstrap_rows.extend(setting_bootstrap)
        summary_rows.append(
            summarize_setting(
                dataset,
                unit_size,
                batch_rows,
                gradient_pair_rows,
                basis_pair_rows,
                setting_bootstrap,
                set(args.main_unit_sizes),
            )
        )

    return DatasetResult(
        dataset=dataset,
        batch_rows=batch_rows,
        gradient_pair_rows=gradient_pair_rows,
        basis_pair_rows=basis_pair_rows,
        summary_rows=summary_rows,
        bootstrap_rows=bootstrap_rows,
    )


def fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt_value(row[field]) for field in fields) + " |")
    return "\n".join(lines)


def gate_decision(
    summary_rows: list[dict[str, Any]],
    main_unit_sizes: set[int],
) -> tuple[str, dict[str, int]]:
    support_counts: dict[str, int] = {}
    for dataset in DATASETS:
        support_counts[dataset] = sum(
            1
            for row in summary_rows
            if row["dataset"] == dataset
            and row["unit_size"] in main_unit_sizes
            and row["robust_support"]
        )
    robust_datasets = sum(count >= 3 for count in support_counts.values())
    any_support = sum(support_counts.values())
    if robust_datasets >= 2:
        return "partial_pass_large_unit_granularity_robust", support_counts
    if any_support:
        return "granularity_or_dataset_specific", support_counts
    return "large_unit_problem_not_supported", support_counts


def write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    main_unit_sizes: set[int],
) -> None:
    decision, support_counts = gate_decision(summary_rows, main_unit_sizes)
    geometry_confounded = [
        row
        for row in summary_rows
        if row["unit_size"] in main_unit_sizes
        and np.isfinite(row["gradient_basis_pair_spearman"])
        and row["gradient_basis_pair_spearman"] >= 0.75
    ]
    lines = [
        "# Phase5 StageB B13-FUCO-A Future-Unit Granularity Diagnostic",
        "",
        "## 阶段记录",
        "",
        "| 字段 | 内容 |",
        "| --- | --- |",
        "| `candidate_id` | `B13-FUCO` |",
        "| `diagnostic_id` | `B13-FUCO-A` |",
        "| `current_step` | Step 2/3：large-unit granularity stability |",
        "| `problem` | A6 global coefficient 是否在较大的 benchmark-independent future units 上仍承受稳定的方向压力 |",
        f"| `decision` | `{decision}` |",
        "",
        "## Scope",
        "",
        "- main unit sizes: `120/144/180/240`;",
        "- coarse control: `360`;",
        "- datasets: `ETTh2/ETTm1/Weather`;",
        "- clean A6 checkpoint, train split, checkpoint-local gradients;",
        "- rank-32 A6 basis subspace geometry control;",
        "- no model training and no residual fitting.",
        "",
        "## Summary",
        "",
        markdown_table(
            summary_rows,
            [
                "dataset",
                "unit_size",
                "unit_count",
                "role",
                "mean_pairwise_cosine",
                "first_last_cosine",
                "adjacent_cosine",
                "far_cosine",
                "shared_alignment_efficiency",
                "bootstrap_mean_pairwise_p95",
                "bootstrap_first_last_p95",
                "basis_adjacent_overlap",
                "basis_far_overlap",
                "gradient_basis_pair_spearman",
                "robust_support",
            ],
        ),
        "",
        "## Gate Reading",
        "",
    ]
    for dataset in DATASETS:
        lines.append(
            f"- {dataset}: `{support_counts[dataset]}/4` main unit sizes pass the pre-registered robust-support gate."
        )
    lines.extend(["", f"[Decision] `{decision}`.", ""])

    if decision == "partial_pass_large_unit_granularity_robust":
        lines.extend(
            [
                "[Strong Evidence] Shared-coefficient gradient pressure survives larger, benchmark-independent unit sizes on at least two datasets. The B9 signal is therefore not limited to canonical horizon boundaries or small units.",
                "",
                "[Boundary] Diagnostic A only permits Diagnostic B. It does not prove that prefix-causal composition beats independent/no-transition capacity controls.",
            ]
        )
    elif decision == "granularity_or_dataset_specific":
        lines.extend(
            [
                "[Decision] The signal is not sufficiently cross-dataset and cross-granularity. Do not implement a future-unit model; first inspect which unit sizes or datasets explain the support.",
            ]
        )
    else:
        lines.extend(
            [
                "[Rollback] Larger future units do not preserve the B9 shared-state pressure. Return B13 to Step 2 and do not implement future-unit composition from the current evidence.",
            ]
        )

    lines.extend(
        [
            "",
            "## Basis-Geometry Confound",
            "",
            f"[Fact] `{len(geometry_confounded)}` main dataset/size settings have gradient-vs-basis pair Spearman `>=0.75`.",
            "",
            "A high value means the gradient relation may be substantially inherited from A6 basis row-subspace geometry. Such settings support future-region heterogeneity but cannot by themselves prove that a compositional generator is necessary.",
            "",
            "## Failure Attribution",
            "",
            "- `hypothesis_false`: not decided unless the large-unit gate fails broadly;",
            "- `basis_geometry_confounded`: explicitly measured by pairwise gradient/basis association;",
            "- `granularity_specific`: applies if support concentrates on isolated unit sizes;",
            "- `capacity_control_explains`: remains untested until Diagnostic B;",
            "- `direction_level_rejection`: Diagnostic A cannot reject all future-unit architectures.",
            "",
            "## Next",
            "",
            "Only if the decision is `partial_pass_large_unit_granularity_robust`, design Diagnostic B with parameter-matched `shared / independent / no-transition / prefix-causal composed` unit states. No model candidate may enter Step 4-6 before that control gate.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def validate_args(args: argparse.Namespace) -> None:
    if args.max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    if args.subspace_rank <= 0:
        raise ValueError("subspace_rank must be positive")
    if len(set(args.unit_sizes)) != len(args.unit_sizes):
        raise ValueError("unit_sizes must be unique")
    for unit_size in args.unit_sizes:
        if unit_size <= 0 or PRED_LEN % unit_size != 0:
            raise ValueError(f"unit_size must divide {PRED_LEN}: {unit_size}")
        if unit_size < args.subspace_rank:
            raise ValueError("unit_size must be at least subspace_rank")
    if not set(args.main_unit_sizes).issubset(set(args.unit_sizes)):
        raise ValueError("main_unit_sizes must be a subset of unit_sizes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B13 future-unit large-granularity stability diagnostic."
    )
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--unit-sizes", nargs="+", type=int, default=list(DEFAULT_UNIT_SIZES))
    parser.add_argument(
        "--main-unit-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_MAIN_UNIT_SIZES),
    )
    parser.add_argument("--subspace-rank", type=int, default=32)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    validate_args(args)
    return args


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    all_batch_rows: list[dict[str, Any]] = []
    all_gradient_pair_rows: list[dict[str, Any]] = []
    all_basis_pair_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    all_bootstrap_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        result = analyze_dataset(args, dataset)
        all_batch_rows.extend(result.batch_rows)
        all_gradient_pair_rows.extend(result.gradient_pair_rows)
        all_basis_pair_rows.extend(result.basis_pair_rows)
        all_summary_rows.extend(result.summary_rows)
        all_bootstrap_rows.extend(result.bootstrap_rows)

    args.analysis_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.analysis_root / "b13_future_unit_gradient_batches.csv", all_batch_rows)
    write_csv(
        args.analysis_root / "b13_future_unit_gradient_pairs.csv",
        all_gradient_pair_rows,
    )
    write_csv(args.analysis_root / "b13_future_unit_basis_pairs.csv", all_basis_pair_rows)
    write_csv(args.analysis_root / "b13_future_unit_summary.csv", all_summary_rows)
    write_csv(args.analysis_root / "b13_future_unit_bootstrap.csv", all_bootstrap_rows)
    write_report(
        args.analysis_root / "b13_future_unit_granularity_report.md",
        all_summary_rows,
        set(args.main_unit_sizes),
    )


if __name__ == "__main__":
    main()

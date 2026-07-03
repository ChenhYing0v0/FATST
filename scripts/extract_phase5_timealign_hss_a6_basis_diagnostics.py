from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch


def arm_from_path(path: Path) -> str:
    prefix = "TimeAlignOfficialUnified720_A6_"
    suffix = "_official-last"
    for part in path.parts:
        if part.startswith(prefix) and part.endswith(suffix):
            return part.removeprefix(prefix).removesuffix(suffix)
    raise ValueError(f"Cannot parse A6 arm from {path}")


def dataset_from_path(path: Path) -> str:
    parts = list(path.parts)
    if "mixed_h96_h192_h336_h720" not in parts:
        raise ValueError(f"Cannot parse dataset from {path}")
    return parts[parts.index("mixed_h96_h192_h336_h720") - 1]


def get_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def cosine_lag_mean(matrix: torch.Tensor, lag: int) -> float:
    if matrix.shape[0] <= lag:
        return float("nan")
    normed = torch.nn.functional.normalize(matrix, dim=1)
    return float((normed[:-lag] * normed[lag:]).sum(dim=1).mean().item())


def rank_for_energy(values: torch.Tensor, threshold: float) -> int:
    if values.numel() == 0:
        return 0
    energy = values.square()
    total = energy.sum()
    if float(total.item()) <= 0.0:
        return 0
    cumulative = torch.cumsum(energy / total, dim=0)
    return int((cumulative < threshold).sum().item() + 1)


def effective_rank(values: torch.Tensor) -> float:
    energy = values.square()
    total = energy.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    prob = energy / total
    entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy).item())


def participation_ratio(values: torch.Tensor) -> float:
    energy = values.square()
    denom = energy.square().sum()
    if float(denom.item()) <= 0.0:
        return 0.0
    return float((energy.sum().square() / denom).item())


def energy_at(values: torch.Tensor, k: int) -> float:
    if values.numel() == 0:
        return 0.0
    energy = values.square()
    total = energy.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    return float((energy[: min(k, values.numel())].sum() / total).item())


def singular_summary(matrix: torch.Tensor, prefix: str) -> dict[str, float | int]:
    values = torch.linalg.svdvals(matrix.float())
    row: dict[str, float | int] = {
        f"{prefix}_singular_count": int(values.numel()),
        f"{prefix}_effective_rank": effective_rank(values),
        f"{prefix}_participation_ratio": participation_ratio(values),
        f"{prefix}_rank90": rank_for_energy(values, 0.90),
        f"{prefix}_rank95": rank_for_energy(values, 0.95),
        f"{prefix}_rank99": rank_for_energy(values, 0.99),
    }
    for k in (8, 16, 32, 64, 128, 256, 512):
        row[f"{prefix}_energy_top{k}"] = energy_at(values, k)
    if values.numel() > 0:
        row[f"{prefix}_top_singular"] = float(values[0].item())
        row[f"{prefix}_last_singular"] = float(values[-1].item())
        row[f"{prefix}_condition_proxy"] = float((values[0] / values[-1].clamp_min(1e-12)).item())
    return row


def analyze_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    state = get_state_dict(checkpoint)
    if "learned_temporal_basis" not in state:
        raise KeyError(f"Missing learned_temporal_basis in {path}")
    if "learned_basis_coeff.weight" not in state:
        raise KeyError(f"Missing learned_basis_coeff.weight in {path}")

    basis = state["learned_temporal_basis"].detach().float()
    coeff_weight = state["learned_basis_coeff.weight"].detach().float()
    bias = state.get("learned_temporal_bias")
    if bias is not None:
        bias = bias.detach().float()

    row_norm = torch.linalg.vector_norm(basis, dim=1)
    row_diff = basis[1:] - basis[:-1]
    diff_norm = torch.linalg.vector_norm(row_diff, dim=1)
    operator = basis @ coeff_weight

    row: dict[str, Any] = {
        "dataset": dataset_from_path(path),
        "arm": arm_from_path(path),
        "checkpoint_path": str(path),
        "basis_horizon": int(basis.shape[0]),
        "basis_rank": int(basis.shape[1]),
        "coeff_out_rank": int(coeff_weight.shape[0]),
        "coeff_in_dim": int(coeff_weight.shape[1]),
        "basis_row_norm_mean": float(row_norm.mean().item()),
        "basis_row_norm_std": float(row_norm.std(unbiased=False).item()),
        "basis_row_norm_min": float(row_norm.min().item()),
        "basis_row_norm_max": float(row_norm.max().item()),
        "basis_adjacent_cosine_mean": cosine_lag_mean(basis, 1),
        "basis_lag24_cosine_mean": cosine_lag_mean(basis, 24),
        "basis_lag48_cosine_mean": cosine_lag_mean(basis, 48),
        "basis_lag96_cosine_mean": cosine_lag_mean(basis, 96),
        "basis_adjacent_l2_mean": float(diff_norm.mean().item()),
        "basis_adjacent_l2_std": float(diff_norm.std(unbiased=False).item()),
        "operator_fro_norm": float(torch.linalg.matrix_norm(operator).item()),
        "coeff_weight_fro_norm": float(torch.linalg.matrix_norm(coeff_weight).item()),
    }
    if bias is not None:
        row["bias_l2_norm"] = float(torch.linalg.vector_norm(bias).item())
        row["bias_mean"] = float(bias.mean().item())
        row["bias_std"] = float(bias.std(unbiased=False).item())
    else:
        row["bias_l2_norm"] = math.nan
        row["bias_mean"] = math.nan
        row["bias_std"] = math.nan

    row.update(singular_summary(basis, "basis"))
    row.update(singular_summary(operator, "operator"))
    row.update(singular_summary(coeff_weight, "coeff_weight"))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract A6 learned-basis checkpoint diagnostics.")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in sorted(args.checkpoint_root.glob("official-last/TimeAlignOfficialUnified720_A6_a6_lbf_*/*/mixed_h96_h192_h336_h720/seed2021/checkpoint.pt")):
        rows.append(analyze_checkpoint(path))
    if not rows:
        raise FileNotFoundError(f"No A6-LBF checkpoints found under {args.checkpoint_root}")
    write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()

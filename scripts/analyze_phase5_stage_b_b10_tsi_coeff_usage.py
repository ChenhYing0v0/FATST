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
SEGMENTS = (
    ("early_0_96", 0, 96),
    ("mid_96_192", 96, 192),
    ("late_192_336", 192, 336),
    ("tail_336_720", 336, 720),
)
RANKS = (32, 64)
DEFAULT_ANALYSIS_ROOT = Path("analysis/phase5_stage_b_b10_tsi_coeff_usage_20260708")
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
    batch_rows: list[dict[str, Any]]
    segment_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]


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


def forward_full_with_coeff(model: Model, batch_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    normalized_output = torch.einsum("hk,bck->bch", basis, coeff) + bias.view(1, 1, -1)
    normalized_output = normalized_output.permute(0, 2, 1)
    output = model.normalization_x(normalized_output, "denorm")
    return output, normalized_output, coeff


def orthonormal_row_space(segment_basis: torch.Tensor, rank: int) -> torch.Tensor:
    _u, singular_values, vh = torch.linalg.svd(segment_basis, full_matrices=False)
    keep = min(rank, int(torch.sum(singular_values > 1e-10).item()), vh.shape[0])
    if keep <= 0:
        return torch.zeros((segment_basis.shape[1], 0), dtype=segment_basis.dtype, device=segment_basis.device)
    return vh[:keep].transpose(0, 1).contiguous()


def build_row_spaces(model: Model, device: torch.device) -> dict[int, dict[str, torch.Tensor]]:
    basis = model.learned_temporal_basis.detach().to(device=device, dtype=torch.float32)
    spaces: dict[int, dict[str, torch.Tensor]] = {}
    for rank in RANKS:
        spaces[rank] = {}
        for name, start, end in SEGMENTS:
            spaces[rank][name] = orthonormal_row_space(basis[start:end], rank)
    return spaces


def safe_mean(values: list[float]) -> float:
    clean = np.asarray([value for value in values if not np.isnan(value)], dtype=np.float64)
    return float(clean.mean()) if clean.size else float("nan")


def safe_entropy(shares: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(shares, min=1e-12)
    return -torch.sum(clipped * torch.log(clipped), dim=-1) / np.log(shares.shape[-1])


def cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = torch.sum(left * right, dim=-1)
    denom = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(right, dim=-1)
    return numerator / torch.clamp(denom, min=1e-12)


def analyze_batch(
    model: Model,
    row_spaces: dict[int, dict[str, torch.Tensor]],
    criterion: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    dataset: str,
    batch_idx: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with torch.no_grad():
        output, normalized_output, coeff = forward_full_with_coeff(model, batch_x)
        target = batch_y[:, -720:, :]
        coeff_flat = coeff.reshape(-1, coeff.shape[-1])
        coeff_norm_sq = torch.sum(coeff_flat * coeff_flat, dim=-1)
        coeff_norm_sq = torch.clamp(coeff_norm_sq, min=1e-12)

        basis = model.learned_temporal_basis.to(device=coeff.device, dtype=coeff.dtype)
        output_energy_by_segment: list[torch.Tensor] = []
        loss_by_segment: dict[str, float] = {}
        normed_loss_by_segment: dict[str, float] = {}
        for name, start, end in SEGMENTS:
            loss_by_segment[name] = float(criterion(output[:, start:end, :], target[:, start:end, :]).cpu())
            normed_loss_by_segment[name] = float(
                torch.mean(normalized_output[:, start:end, :] ** 2).cpu()
            )
            segment_output = torch.einsum("hk,bck->bch", basis[start:end], coeff)
            energy = torch.sum(segment_output * segment_output, dim=-1).reshape(-1)
            output_energy_by_segment.append(energy)

        output_energy = torch.stack(output_energy_by_segment, dim=-1)
        output_share = output_energy / torch.clamp(torch.sum(output_energy, dim=-1, keepdim=True), min=1e-12)
        output_entropy = safe_entropy(output_share)
        output_max_share = torch.max(output_share, dim=-1).values

        batch_rows: list[dict[str, Any]] = []
        segment_rows: list[dict[str, Any]] = []
        for rank, rank_spaces in row_spaces.items():
            projections: dict[str, torch.Tensor] = {}
            projection_shares: dict[str, torch.Tensor] = {}
            for name, _start, _end in SEGMENTS:
                q = rank_spaces[name]
                coord = coeff_flat @ q
                projected = coord @ q.transpose(0, 1)
                projections[name] = projected
                projection_shares[name] = torch.sum(coord * coord, dim=-1) / coeff_norm_sq

            pair_cosines: list[float] = []
            names = [segment[0] for segment in SEGMENTS]
            pair_fields: dict[str, float] = {}
            for left_idx, left_name in enumerate(names):
                for right_name in names[left_idx + 1 :]:
                    value = float(torch.mean(cosine_rows(projections[left_name], projections[right_name])).cpu())
                    pair_fields[f"projection_cos_{left_name}_vs_{right_name}"] = value
                    pair_cosines.append(value)

            row: dict[str, Any] = {
                "dataset": dataset,
                "batch_idx": batch_idx,
                "rank": rank,
                "coeff_vectors": int(coeff_flat.shape[0]),
                "coeff_norm_mean": float(torch.mean(torch.sqrt(coeff_norm_sq)).cpu()),
                "projection_share_mean": safe_mean(
                    [float(torch.mean(values).cpu()) for values in projection_shares.values()]
                ),
                "projection_share_min": min(float(torch.mean(values).cpu()) for values in projection_shares.values()),
                "projection_share_max": max(float(torch.mean(values).cpu()) for values in projection_shares.values()),
                "projection_pair_cosine_mean": safe_mean(pair_cosines),
                "output_energy_entropy_mean": float(torch.mean(output_entropy).cpu()),
                "output_energy_max_stage_share_mean": float(torch.mean(output_max_share).cpu()),
            }
            row.update(pair_fields)
            batch_rows.append(row)

            for name, _start, _end in SEGMENTS:
                segment_rows.append(
                    {
                        "dataset": dataset,
                        "batch_idx": batch_idx,
                        "rank": rank,
                        "segment": name,
                        "projection_share_mean": float(torch.mean(projection_shares[name]).cpu()),
                        "projection_share_std": float(torch.std(projection_shares[name]).cpu()),
                        "output_energy_share_mean": float(
                            torch.mean(output_share[:, [segment[0] for segment in SEGMENTS].index(name)]).cpu()
                        ),
                        "loss_mse": loss_by_segment[name],
                        "normalized_output_energy": normed_loss_by_segment[name],
                    }
                )

    return batch_rows, segment_rows


def summarize_dataset(dataset: str, batch_rows: list[dict[str, Any]], segment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for rank in RANKS:
        rank_batches = [row for row in batch_rows if row["rank"] == rank]
        rank_segments = [row for row in segment_rows if row["rank"] == rank]
        row: dict[str, Any] = {
            "dataset": dataset,
            "rank": rank,
            "batches": len(rank_batches),
            "projection_share_mean": safe_mean([row["projection_share_mean"] for row in rank_batches]),
            "projection_share_min_mean": safe_mean([row["projection_share_min"] for row in rank_batches]),
            "projection_share_max_mean": safe_mean([row["projection_share_max"] for row in rank_batches]),
            "projection_pair_cosine_mean": safe_mean(
                [row["projection_pair_cosine_mean"] for row in rank_batches]
            ),
            "output_energy_entropy_mean": safe_mean(
                [row["output_energy_entropy_mean"] for row in rank_batches]
            ),
            "output_energy_max_stage_share_mean": safe_mean(
                [row["output_energy_max_stage_share_mean"] for row in rank_batches]
            ),
        }
        for name, _start, _end in SEGMENTS:
            segment = [item for item in rank_segments if item["segment"] == name]
            row[f"projection_share_{name}"] = safe_mean(
                [item["projection_share_mean"] for item in segment]
            )
            row[f"output_energy_share_{name}"] = safe_mean(
                [item["output_energy_share_mean"] for item in segment]
            )
            row[f"loss_mse_{name}"] = safe_mean([item["loss_mse"] for item in segment])
        summary_rows.append(row)
    return summary_rows


def analyze_dataset(args: argparse.Namespace, dataset: str) -> DatasetResult:
    official_args = build_args(args, dataset)
    model = load_model(official_args, checkpoint_path(args.checkpoint_root, dataset))
    device = torch.device(args.device)
    model.to(device)
    row_spaces = build_row_spaces(model, device)
    _data, loader = data_provider(official_args, args.split)
    criterion = nn.MSELoss()

    batch_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for batch_idx, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader):
        if batch_idx >= args.max_batches:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_result, segment_result = analyze_batch(
            model,
            row_spaces,
            criterion,
            batch_x,
            batch_y,
            dataset,
            batch_idx,
        )
        batch_rows.extend(batch_result)
        segment_rows.extend(segment_result)

    if not batch_rows:
        raise RuntimeError(f"No batches analyzed for {dataset}")
    return DatasetResult(
        dataset=dataset,
        batch_rows=batch_rows,
        segment_rows=segment_rows,
        summary_rows=summarize_dataset(dataset, batch_rows, segment_rows),
    )


def fmt_value(value: Any) -> str:
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


def decision_lines(rank64_rows: list[dict[str, Any]]) -> list[str]:
    projection = safe_mean([row["projection_share_mean"] for row in rank64_rows])
    pair_cosine = safe_mean([row["projection_pair_cosine_mean"] for row in rank64_rows])
    entropy = safe_mean([row["output_energy_entropy_mean"] for row in rank64_rows])
    max_share = safe_mean([row["output_energy_max_stage_share_mean"] for row in rank64_rows])
    lines = [
        f"[Fact] Rank64 下，平均 `projection_share_mean` 为 `{projection:.4f}`，说明每个 `coeff` 在多个 stage row subspaces 上都有可观投影。",
        f"[Fact] `projection_pair_cosine_mean` 为 `{pair_cosine:.4f}`，说明这些 stage projections 不是同一方向的简单重复。",
        f"[Fact] `output_energy_entropy_mean` 为 `{entropy:.4f}`，`output_energy_max_stage_share_mean` 为 `{max_share:.4f}`。",
        "",
    ]
    if projection >= 0.20 and entropy >= 0.70 and pair_cosine < 0.60:
        lines.extend(
            [
                "[Decision] `B10-TSI-B` 支持继续：同一个 target-set-blind `coeff` 同时激活多个 stage row subspaces，且 output energy 不是单 stage 主导。这与 B10 的收窄问题一致：requested target set 应进入 `history -> coeff/state` 路径。",
                "",
                "[Boundary] 这仍不是方法通过。下一步必须做 target-set oracle/control，证明 target-set-aware readout 不能被 no-target-set capacity control 解释。",
            ]
        )
    else:
        lines.extend(
            [
                "[Decision] `B10-TSI-B` 不足以支持继续：当前 coeff usage 没有显示稳定的 multi-stage shared-state pressure。",
                "",
                "[Rollback] StageB 应回到 Step 2/3，或只保留 B7 objective optimization 作为小贡献候选。",
            ]
        )
    return lines


def write_report(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    rank64_rows = [row for row in summary_rows if row["rank"] == 64]
    lines = [
        "# Phase5 StageB B10-TSI-B Coefficient Usage Diagnostic",
        "",
        "## 阶段记录",
        "",
        "| 字段 | 内容 |",
        "| --- | --- |",
        "| `candidate_id` | `B10-TCO` |",
        "| `diagnostic_id` | `B10-TSI-B` |",
        "| `current_step` | Step 3：target-set interface diagnostic |",
        "| `problem` | A6 basis 已有 stage-differentiated row-space geometry，但 `learned_basis_coeff(hidden)` 仍生成 target-set-blind coefficient/state |",
        "| `scope` | 读取真实 forward batch 的 `coeff`，分析其在 stage row subspaces 中的使用方式；不训练新模型 |",
        "| `decision` | 见文末；本诊断不能单独升级为 method result |",
        "",
        "## 诊断定义",
        "",
        "对 clean A6 checkpoint，在 test split 上取若干 batch，执行 A6 forward 到：",
        "",
        "```text",
        "coeff = learned_basis_coeff(hidden)              # [B, C, 256]",
        "basis_s = learned_temporal_basis[start:end]      # [L_s, 256]",
        "Q_s = row_space(basis_s)                         # [256, rank]",
        "projection_share_s = ||coeff @ Q_s||^2 / ||coeff||^2",
        "```",
        "",
        "同时计算同一个 `coeff` 通过四个 segment basis 生成的 normalized output energy share。",
        "",
        "## Rank64 Summary",
        "",
        markdown_table(
            rank64_rows,
            [
                "dataset",
                "rank",
                "batches",
                "projection_share_mean",
                "projection_share_min_mean",
                "projection_share_max_mean",
                "projection_pair_cosine_mean",
                "output_energy_entropy_mean",
                "output_energy_max_stage_share_mean",
            ],
        ),
        "",
        "## Rank64 Segment Detail",
        "",
        markdown_table(
            rank64_rows,
            [
                "dataset",
                "projection_share_early_0_96",
                "projection_share_mid_96_192",
                "projection_share_late_192_336",
                "projection_share_tail_336_720",
                "output_energy_share_early_0_96",
                "output_energy_share_mid_96_192",
                "output_energy_share_late_192_336",
                "output_energy_share_tail_336_720",
            ],
        ),
        "",
        "## Decision",
        "",
    ]
    lines.extend(decision_lines(rank64_rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B10 target-set interface coefficient usage diagnostic.")
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    all_batch_rows: list[dict[str, Any]] = []
    all_segment_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        result = analyze_dataset(args, dataset)
        all_batch_rows.extend(result.batch_rows)
        all_segment_rows.extend(result.segment_rows)
        all_summary_rows.extend(result.summary_rows)

    write_csv(args.analysis_root / "b10_tsi_coeff_usage_batches.csv", all_batch_rows)
    write_csv(args.analysis_root / "b10_tsi_coeff_usage_segments.csv", all_segment_rows)
    write_csv(args.analysis_root / "b10_tsi_coeff_usage_summary.csv", all_summary_rows)
    write_report(args.analysis_root / "b10_tsi_coeff_usage_report.md", all_summary_rows)


if __name__ == "__main__":
    main()

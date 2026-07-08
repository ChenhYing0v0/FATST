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
DEFAULT_ANALYSIS_ROOT = Path("analysis/phase5_stage_b_b9_stage_gradient_diagnostic_20260707")
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
    summary_row: dict[str, Any]


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
    model = Model(official_args)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.train()
    return model


def forward_full_with_coeff(model: Model, batch_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    output = output.permute(0, 2, 1)
    output = model.normalization_x(output, "denorm")
    return output, coeff


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = torch.linalg.vector_norm(a_flat) * torch.linalg.vector_norm(b_flat)
    if float(denom.detach().cpu()) <= 1e-12:
        return float("nan")
    return float(torch.dot(a_flat, b_flat).detach().cpu() / denom.detach().cpu())


def analyze_batch(
    model: Model,
    criterion: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    dataset: str,
    batch_idx: int,
) -> dict[str, Any]:
    output, coeff = forward_full_with_coeff(model, batch_x)
    target = batch_y[:, -720:, :]

    grads: dict[str, torch.Tensor] = {}
    losses: dict[str, float] = {}
    norms: dict[str, float] = {}
    for name, start, end in SEGMENTS:
        loss = criterion(output[:, start:end, :], target[:, start:end, :])
        grad = torch.autograd.grad(loss, coeff, retain_graph=True)[0]
        grads[name] = grad.detach()
        losses[name] = float(loss.detach().cpu())
        norms[name] = float(torch.linalg.vector_norm(grad.detach()).cpu())

    pair_values: list[float] = []
    row: dict[str, Any] = {"dataset": dataset, "batch_idx": batch_idx}
    for name, _start, _end in SEGMENTS:
        row[f"loss_{name}"] = losses[name]
        row[f"grad_norm_{name}"] = norms[name]

    for left_idx, (left, _ls, _le) in enumerate(SEGMENTS):
        for right, _rs, _re in SEGMENTS[left_idx + 1 :]:
            value = cosine(grads[left], grads[right])
            row[f"cos_{left}_vs_{right}"] = value
            if not np.isnan(value):
                pair_values.append(value)

    if pair_values:
        row["mean_pairwise_cosine"] = float(np.mean(pair_values))
        row["min_pairwise_cosine"] = float(np.min(pair_values))
        row["negative_pair_rate"] = float(np.mean(np.asarray(pair_values) < 0.0))
    else:
        row["mean_pairwise_cosine"] = float("nan")
        row["min_pairwise_cosine"] = float("nan")
        row["negative_pair_rate"] = float("nan")
    row["early_tail_cosine"] = row["cos_early_0_96_vs_tail_336_720"]

    norm_values = np.asarray(list(norms.values()), dtype=np.float64)
    row["max_min_grad_norm_ratio"] = (
        float(norm_values.max() / max(norm_values.min(), 1e-12)) if len(norm_values) else float("nan")
    )
    return row


def summarize_dataset(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_field(name: str) -> float:
        values = np.asarray([row[name] for row in rows if not np.isnan(row[name])], dtype=np.float64)
        return float(values.mean()) if len(values) else float("nan")

    summary = {
        "dataset": dataset,
        "batches": len(rows),
        "mean_pairwise_cosine": mean_field("mean_pairwise_cosine"),
        "min_pairwise_cosine_mean": mean_field("min_pairwise_cosine"),
        "negative_pair_rate_mean": mean_field("negative_pair_rate"),
        "early_tail_cosine_mean": mean_field("early_tail_cosine"),
        "max_min_grad_norm_ratio_mean": mean_field("max_min_grad_norm_ratio"),
    }
    for name, _start, _end in SEGMENTS:
        summary[f"loss_{name}_mean"] = mean_field(f"loss_{name}")
        summary[f"grad_norm_{name}_mean"] = mean_field(f"grad_norm_{name}")
    return summary


def analyze_dataset(args: argparse.Namespace, dataset: str) -> DatasetResult:
    official_args = build_args(args, dataset)
    model = load_model(official_args, checkpoint_path(args.checkpoint_root, dataset))
    device = torch.device(args.device)
    model.to(device)
    _data, loader = data_provider(official_args, args.split)
    criterion = nn.MSELoss()

    rows: list[dict[str, Any]] = []
    for batch_idx, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader):
        if batch_idx >= args.max_batches:
            break
        model.zero_grad(set_to_none=True)
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        rows.append(analyze_batch(model, criterion, batch_x, batch_y, dataset, batch_idx))

    if not rows:
        raise RuntimeError(f"No batches analyzed for {dataset}")
    return DatasetResult(dataset=dataset, batch_rows=rows, summary_row=summarize_dataset(dataset, rows))


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for field in fields:
            value = row[field]
            values.append(fmt_float(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def decision_lines(summary_rows: list[dict[str, Any]]) -> list[str]:
    support = [
        row
        for row in summary_rows
        if row["mean_pairwise_cosine"] < 0.5 or row["early_tail_cosine_mean"] < 0.35
    ]
    strong_conflict = [
        row
        for row in summary_rows
        if row["negative_pair_rate_mean"] >= 0.15 or row["min_pairwise_cosine_mean"] < 0.0
    ]
    lines = [
        "[Fact] 本诊断没有拟合 residual，也没有设计 correction module；它只看训练信号中不同 future stages 对同一个 A6 coefficient 的梯度方向是否一致。",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- {row['dataset']}: mean pairwise cosine = `{row['mean_pairwise_cosine']:.3f}`, "
            f"early-tail cosine = `{row['early_tail_cosine_mean']:.3f}`, "
            f"negative pair rate = `{row['negative_pair_rate_mean']:.3f}`."
        )
    lines.append("")
    if len(support) >= 2:
        lines.extend(
            [
                "[Decision] `B9-SGC` 暂定通过 problem-candidate gate：至少两个 dataset 显示不同 future stages 对共享 coefficient 的梯度方向不够一致，支持 native future-stage-aware representation/operator 的问题存在。",
                "",
                "[Next] 进入 Step 4-6 前仍需设计 narrative gate：方法必须是 primary prediction path，不得是 residual correction；并需要定义 capacity-preserving initialization。",
            ]
        )
        if strong_conflict:
            lines.append(
                f"[Strength] 其中 {len(strong_conflict)} 个 dataset 出现负向 pair 或明显低 min cosine，说明不只是 norm imbalance。"
            )
    else:
        lines.extend(
            [
                "[Decision] `B9-SGC` 未通过 problem-candidate gate：stage gradients 对共享 coefficient 的方向基本一致，native stage-specific architecture 的必要性不足。",
                "",
                "[Rollback] StageB 继续回到 Step 2/3，不实现 B9-FSN。",
            ]
        )
    return lines


def write_report(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase5 StageB B9-SGC Stage Gradient Diagnostic",
        "",
        "## 阶段记录",
        "",
        "| 字段 | 内容 |",
        "| --- | --- |",
        "| `candidate_id` | `B9-FSN` |",
        "| `diagnostic_id` | `B9-SGC` |",
        "| `current_step` | Step 2/3：native future-stage-aware problem diagnostic |",
        "| `problem` | A6-LBF 用一个 `coeff[b,c]` 服务所有 future stages，可能造成不同 stage 的训练信号冲突 |",
        "| `residual_policy` | 不拟合 residual，不设计 residual correction；只分析 primary prediction path 的 stage gradients |",
        "",
        "## 诊断定义",
        "",
        "对 clean A6 checkpoint，在 train split 上取若干 batch，手动执行 A6 forward 到：",
        "",
        "```text",
        "coeff = learned_basis_coeff(hidden)  # [B, C, 256]",
        "prediction = learned_temporal_basis @ coeff",
        "```",
        "",
        "然后分别计算四个 non-overlap future stages 的 MSE loss，并求每个 stage loss 对同一个 `coeff` 的梯度：",
        "",
        "```text",
        "stages = [0,96), [96,192), [192,336), [336,720)",
        "g_s = d loss_s / d coeff",
        "```",
        "",
        "若不同 `g_s` 的 cosine 较低或为负，说明一个共享 coefficient 同时服务所有 future stages 存在 native stage pressure；这支持 B9-FSN 的问题存在。",
        "",
        "## Summary",
        "",
        markdown_table(
            summary_rows,
            [
                "dataset",
                "batches",
                "mean_pairwise_cosine",
                "min_pairwise_cosine_mean",
                "negative_pair_rate_mean",
                "early_tail_cosine_mean",
                "max_min_grad_norm_ratio_mean",
            ],
        ),
        "",
        "## Decision",
        "",
    ]
    lines.extend(decision_lines(summary_rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B9 native future-stage gradient diagnostic.")
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    all_batch_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        result = analyze_dataset(args, dataset)
        all_batch_rows.extend(result.batch_rows)
        summary_rows.append(result.summary_row)

    write_csv(args.analysis_root / "b9_stage_gradient_batches.csv", all_batch_rows)
    write_csv(args.analysis_root / "b9_stage_gradient_summary.csv", summary_rows)
    write_report(args.analysis_root / "b9_stage_gradient_report.md", summary_rows)


if __name__ == "__main__":
    main()

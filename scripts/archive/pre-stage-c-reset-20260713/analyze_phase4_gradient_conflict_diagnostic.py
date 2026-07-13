from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "baselines" / "patch_encoder_target_set_decoder"
sys.path.insert(0, str(BASELINE_ROOT))

from dataset import DATASETS, ForecastDataset  # noqa: E402
from model import PatchEncoderTargetSetDecoder  # noqa: E402


REGION_GROUPS = {
    "early_1_96": (0, 96),
    "middle_97_192": (96, 192),
    "middle_193_336": (192, 336),
    "late_337_720": (336, 720),
}

PARAMETER_GROUP_PREFIXES = {
    "all_shared": (),
    "encoder": ("patch_embedding", "encoder", "pos_embedding"),
    "target_path": ("target_feature_embedding", "target_pos_embedding", "target_decoder", "target_interaction"),
    "readout_head": ("history_projector", "condition_head", "segment_output"),
}


@dataclass(frozen=True)
class BatchGroups:
    losses: dict[str, torch.Tensor]
    block_labels: dict[str, int]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def masked_loss(pred: torch.Tensor, true: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return torch.mean((pred[:, start:end, :] - true[:, start:end, :]) ** 2)


def block_scores(
    true: torch.Tensor,
    history: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    horizon = true.shape[1]
    novelty_scores = []
    variation_scores = []
    last_history = history[:, -1:, :]
    for start in range(0, horizon, block_size):
        end = min(start + block_size, horizon)
        block = true[:, start:end, :]
        reference = last_history.expand(-1, end - start, -1)
        novelty_scores.append(torch.mean((block - reference) ** 2))
        if end - start <= 1:
            variation_scores.append(torch.zeros((), device=true.device, dtype=true.dtype))
        else:
            diff = block[:, 1:, :] - block[:, :-1, :]
            variation_scores.append(torch.mean(diff * diff))
    return torch.stack(novelty_scores), torch.stack(variation_scores)


def batch_group_losses(
    pred: torch.Tensor,
    true: torch.Tensor,
    history: torch.Tensor,
    block_size: int,
    top_ratio: float,
) -> BatchGroups:
    losses: dict[str, torch.Tensor] = {
        "full_1_720": torch.mean((pred - true) ** 2),
    }
    for label, (start, end) in REGION_GROUPS.items():
        if pred.shape[1] > start:
            losses[label] = masked_loss(pred, true, start, min(end, pred.shape[1]))

    with torch.no_grad():
        novelty, variation = block_scores(true, history, block_size)
        top_blocks = max(1, min(len(novelty), int(round(len(novelty) * top_ratio))))
        novelty_top = set(torch.topk(novelty, k=top_blocks, largest=True).indices.tolist())
        variation_top = set(torch.topk(variation, k=top_blocks, largest=True).indices.tolist())
        noisy = sorted(novelty_top & variation_top)
        learnable = sorted(novelty_top - variation_top)
        if not learnable:
            learnable = sorted(novelty_top)
        predictable = sorted(set(range(len(novelty))) - novelty_top)

    for label, selected in {
        "learnable_hard": learnable,
        "noisy_hard": noisy,
        "predictable_easy": predictable,
    }.items():
        if not selected:
            continue
        step_mask = torch.zeros(pred.shape[1], device=pred.device, dtype=pred.dtype)
        for block_index in selected:
            start = int(block_index) * block_size
            end = min(start + block_size, pred.shape[1])
            step_mask[start:end] = 1.0
        denom = torch.sum(step_mask).clamp_min(1.0) * pred.shape[0] * pred.shape[2]
        losses[label] = torch.sum((pred - true) ** 2 * step_mask.view(1, -1, 1)) / denom

    return BatchGroups(
        losses=losses,
        block_labels={
            "top_blocks": top_blocks,
            "learnable_blocks": len(learnable),
            "noisy_blocks": len(noisy),
            "predictable_blocks": len(predictable),
        },
    )


def parameter_groups(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    named_parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {}
    for group_name, prefixes in PARAMETER_GROUP_PREFIXES.items():
        if not prefixes:
            groups[group_name] = named_parameters
            continue
        groups[group_name] = [
            (name, param)
            for name, param in named_parameters
            if any(name.startswith(prefix) for prefix in prefixes)
        ]
    return groups


def flatten_gradients(parameters: Iterable[tuple[str, nn.Parameter]]) -> torch.Tensor:
    parts = []
    for _, param in parameters:
        if param.grad is None:
            parts.append(torch.zeros_like(param, memory_format=torch.preserve_format).reshape(-1))
        else:
            parts.append(param.grad.detach().reshape(-1))
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


def cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    left_cpu = left.detach().float().cpu()
    right_cpu = right.detach().float().cpu()
    denom = torch.linalg.vector_norm(left_cpu) * torch.linalg.vector_norm(right_cpu)
    if float(denom) <= eps:
        return 0.0
    return float(torch.dot(left_cpu, right_cpu) / denom)


def build_model(args: argparse.Namespace, dataset: str, device: torch.device) -> PatchEncoderTargetSetDecoder:
    return PatchEncoderTargetSetDecoder(
        args.seq_len,
        DATASETS[dataset].channels,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        encoder_layers=args.encoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        segment_len=args.segment_len,
        max_pred_len=args.pred_len,
        target_layers=args.target_layers,
        target_heads=args.target_heads,
        target_d_ff=args.target_d_ff,
        readout_dim=args.readout_dim,
    ).to(device)


def load_checkpoint_if_requested(
    model: PatchEncoderTargetSetDecoder,
    checkpoint: str,
    dataset: str,
) -> bool:
    if not checkpoint:
        return False
    path_text = checkpoint.format(dataset=dataset)
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return True


def train_warmup(
    model: PatchEncoderTargetSetDecoder,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    learning_rate: float,
) -> None:
    if steps <= 0:
        return
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iterator = iter(loader)
    for _ in range(steps):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x = x.float().to(device)
        y = y.float().to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, pred_len=y.shape[1])
        if isinstance(pred, dict):
            raise TypeError("Expected tensor prediction.")
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()


def diagnose_dataset(args: argparse.Namespace, dataset: str, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_set = ForecastDataset(args.dataset_root, dataset, "train", args.seq_len, args.pred_len)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = build_model(args, dataset, device)
    checkpoint_loaded = load_checkpoint_if_requested(model, args.checkpoint, dataset)
    if not checkpoint_loaded:
        train_warmup(model, loader, device, args.warmup_steps, args.learning_rate)
    model.train()
    param_groups = parameter_groups(model)

    iterator = iter(loader)
    pair_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for batch_index in range(1, args.diagnostic_batches + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x = x.float().to(device)
        y = y.float().to(device)
        pred = model(x, pred_len=args.pred_len)
        if isinstance(pred, dict):
            raise TypeError("Expected tensor prediction.")
        groups = batch_group_losses(pred, y, x, args.block_size, args.top_ratio)
        block_rows.append(
            {
                "dataset": dataset,
                "batch_index": batch_index,
                "checkpoint_loaded": checkpoint_loaded,
                **groups.block_labels,
            }
        )

        gradients: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        group_items = list(groups.losses.items())
        for loss_index, (loss_name, loss) in enumerate(group_items):
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=loss_index < len(group_items) - 1)
            for param_group_name, params in param_groups.items():
                gradients[loss_name][param_group_name] = flatten_gradients(params)

        names = sorted(gradients)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                for param_group_name in sorted(param_groups):
                    pair_rows.append(
                        {
                            "dataset": dataset,
                            "batch_index": batch_index,
                            "checkpoint_loaded": checkpoint_loaded,
                            "left_group": left_name,
                            "right_group": right_name,
                            "parameter_group": param_group_name,
                            "gradient_cosine": cosine(
                                gradients[left_name][param_group_name],
                                gradients[right_name][param_group_name],
                            ),
                        }
                    )
    return pair_rows, block_rows


def summarize_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["left_group"],
            row["right_group"],
            row["parameter_group"],
        )
        buckets[key].append(float(row["gradient_cosine"]))
    summary = []
    for (dataset, left_group, right_group, parameter_group), values in sorted(buckets.items()):
        summary.append(
            {
                "dataset": dataset,
                "left_group": left_group,
                "right_group": right_group,
                "parameter_group": parameter_group,
                "batches": len(values),
                "mean_gradient_cosine": mean(values),
                "min_gradient_cosine": min(values),
                "negative_share": sum(1 for value in values if value < 0.0) / len(values),
                "low_conflict_share_lt_0p1": sum(1 for value in values if value < 0.1) / len(values),
            }
        )
    return summary


def pair_lookup(summary: list[dict[str, Any]], dataset: str, left: str, right: str, parameter_group: str) -> dict[str, Any] | None:
    aliases = {(left, right), (right, left)}
    for row in summary:
        if row["dataset"] != dataset or row["parameter_group"] != parameter_group:
            continue
        if (row["left_group"], row["right_group"]) in aliases:
            return row
    return None


def write_report(path: Path, args: argparse.Namespace, summary: list[dict[str, Any]], block_rows: list[dict[str, Any]]) -> None:
    focus_pairs = [
        ("noisy_hard", "early_1_96"),
        ("noisy_hard", "predictable_easy"),
        ("learnable_hard", "early_1_96"),
        ("late_337_720", "early_1_96"),
    ]
    lines = [
        "# Phase4 Gradient Conflict Diagnostic",
        "",
        "## 11-Step Record",
        "",
        "| Field | Content |",
        "| --- | --- |",
        "| `current_step` | Step 5/6: theoretical feasibility and concrete diagnostic design |",
        "| `problem` | Loss-only HSS may fail because difficult future units update the same shared representation path |",
        "| `existence_evidence` | SRP-style step-specific representation argument plus Phase4 S1/S2 Weather collapse |",
        "| `idea` | Test whether future-unit supervision groups produce conflicting gradients on shared parameters |",
        "| `theory_check` | If noisy-hard gradients conflict with early/predictable gradients, gradient-routing HSS is justified |",
        "| `design` | Short warmup/full-time checkpoint, then per-group backward passes and gradient cosine by parameter group |",
        "| `gate` | Weather noisy-hard conflicts should be stronger than ETTh2, especially on encoder/target/readout shared paths |",
        f"| `artifacts` | `{path.parent}` |",
        "| `decision` | Pending: use the diagnostics below to decide whether to implement adapter-isolated HSS |",
        "",
        "## Run Config",
        "",
        f"- datasets: `{args.datasets}`",
        f"- pred_len: `{args.pred_len}`",
        f"- warmup_steps: `{args.warmup_steps}`",
        f"- diagnostic_batches: `{args.diagnostic_batches}`",
        f"- block_size: `{args.block_size}`",
        f"- top_ratio: `{args.top_ratio}`",
        f"- checkpoint: `{args.checkpoint or 'none'}`",
        "",
        "## Focus Pair Summary",
        "",
        "| Dataset | Pair | Parameter group | Mean cosine | Negative share | Low share < 0.1 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for dataset in args.datasets.split(","):
        dataset = dataset.strip()
        if not dataset:
            continue
        for left, right in focus_pairs:
            for parameter_group in ["encoder", "target_path", "readout_head", "all_shared"]:
                row = pair_lookup(summary, dataset, left, right, parameter_group)
                if row is None:
                    continue
                lines.append(
                    "| {dataset} | `{left}` vs `{right}` | `{parameter_group}` | {mean_cos:.4f} | {neg:.2f} | {low:.2f} |".format(
                        dataset=dataset,
                        left=left,
                        right=right,
                        parameter_group=parameter_group,
                        mean_cos=float(row["mean_gradient_cosine"]),
                        neg=float(row["negative_share"]),
                        low=float(row["low_conflict_share_lt_0p1"]),
                    )
                )

    lines += [
        "",
        "## Block Bucket Trace",
        "",
        "| Dataset | Mean learnable blocks | Mean noisy blocks | Mean predictable blocks |",
        "| --- | ---: | ---: | ---: |",
    ]
    for dataset in sorted({row["dataset"] for row in block_rows}):
        subset = [row for row in block_rows if row["dataset"] == dataset]
        lines.append(
            "| {dataset} | {learnable:.2f} | {noisy:.2f} | {predictable:.2f} |".format(
                dataset=dataset,
                learnable=mean(float(row["learnable_blocks"]) for row in subset),
                noisy=mean(float(row["noisy_blocks"]) for row in subset),
                predictable=mean(float(row["predictable_blocks"]) for row in subset),
            )
        )

    lines += [
        "",
        "## How To Read",
        "",
        "[Fact] `gradient_cosine < 0` means two supervision groups push the same parameter group in opposing directions on that batch.",
        "",
        "[Inference] If Weather has a high negative/low-cosine share for `noisy_hard` vs `early_1_96` or `predictable_easy`, then S2's scalar downweight was too weak; the next candidate should isolate noisy-hard gradients into adapters or detached auxiliary branches.",
        "",
        "[Counter-check] If cosines are high and positive, the failure is less likely to be representation interference; rollback should prioritize a better predictability proxy rather than architecture-level gradient routing.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Phase4 supervision gradient conflict.")
    parser.add_argument("--dataset-root", default="/Users/river/PaperResearch/Project/datasets")
    parser.add_argument("--datasets", default="ETTh2,Weather")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=720)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--diagnostic-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--block-size", type=int, default=48)
    parser.add_argument("--top-ratio", type=float, default=0.25)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument("--segment-len", type=int, default=48)
    parser.add_argument("--target-layers", type=int, default=1)
    parser.add_argument("--target-heads", type=int, default=8)
    parser.add_argument("--target-d-ff", type=int, default=256)
    parser.add_argument("--readout-dim", type=int, default=256)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-root",
        default=f"analysis/phase4_gradient_conflict_diagnostic_{datetime.now().strftime('%Y%m%d')}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not 0 < args.top_ratio <= 1:
        raise ValueError("top ratio must be in (0, 1].")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    for dataset in datasets:
        rows, blocks = diagnose_dataset(args, dataset, device)
        pair_rows.extend(rows)
        block_rows.extend(blocks)

    summary = summarize_pairs(pair_rows)
    write_csv(output_root / "phase4_gradient_conflict_pairs.csv", pair_rows)
    write_csv(output_root / "phase4_gradient_conflict_summary.csv", summary)
    write_csv(output_root / "phase4_gradient_conflict_block_trace.csv", block_rows)
    (output_root / "phase4_gradient_conflict_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False)
    )
    write_report(output_root / "phase4_gradient_conflict_report.md", args, summary, block_rows)
    print(output_root / "phase4_gradient_conflict_report.md")


if __name__ == "__main__":
    main()

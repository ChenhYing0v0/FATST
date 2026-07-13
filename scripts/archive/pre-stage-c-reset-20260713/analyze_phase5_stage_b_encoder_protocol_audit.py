#!/usr/bin/env python3
"""Audit A6 encoder use, checkpoint drift, and inherited dataset presets."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


DATASETS = ("ETTh2", "ETTm1", "Weather")
HORIZONS = (96, 192, 336, 720)


class ZeroResidual(nn.Module):
    """Return a zero update while preserving the residual-branch interface."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=repo_root
        / "analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/raw/official-last"
        / "TimeAlignOfficialUnified720_a6_clean_official-last",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=repo_root / "analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root.parent / "datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "analysis/phase5_stage_b_encoder_protocol_audit_20260710",
    )
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
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


def local_dataset_path(dataset_root: Path, dataset: str) -> Path:
    if dataset in {"ETTh2", "ETTm1"}:
        return dataset_root / "ETT-small"
    if dataset == "Weather":
        return dataset_root / "weather"
    raise ValueError(f"Unsupported dataset: {dataset}")


def run_dir(checkpoint_root: Path, dataset: str) -> Path:
    candidates = sorted(
        (checkpoint_root / dataset).glob(
            "mixed_h96_h192_h336_h720/seed2021"
        )
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one clean A6 run for {dataset}, found {len(candidates)}"
        )
    return candidates[0]


def parameter_count(model: nn.Module, prefixes: tuple[str, ...]) -> int:
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    )


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    official_args: Namespace,
    max_batches: int,
) -> dict[int, dict[str, float]]:
    totals = {horizon: [0.0, 0.0, 0] for horizon in HORIZONS}
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, _x_mark, _y_mark) in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            batch_x = batch_x.float()
            target = batch_y.float()[:, -official_args.pred_len :, :]
            for horizon in HORIZONS:
                output, _recon, _align = model(
                    batch_x,
                    target,
                    is_training=False,
                    target_prefix=horizon,
                )
                error = output[:, :horizon, :] - target[:, :horizon, :]
                totals[horizon][0] += float(error.square().sum())
                totals[horizon][1] += float(error.abs().sum())
                totals[horizon][2] += error.numel()
    return {
        horizon: {
            "mse": sums[0] / sums[2],
            "mae": sums[1] / sums[2],
        }
        for horizon, sums in totals.items()
    }


def branch_statistics(
    model: nn.Module,
    first_batch: torch.Tensor,
    official_args: Namespace,
) -> list[dict[str, float | int]]:
    model.eval()
    with torch.no_grad():
        normalized = model.normalization_x(first_batch.float(), "norm")
        tokens = model.patch_emb_x(
            normalized.permute(0, 2, 1).reshape(
                -1, official_args.enc_in * official_args.seq_len
            )
        )
        rows: list[dict[str, float | int]] = []
        for layer_idx, layer in enumerate(model.encoder):
            branch = layer(tokens)
            norm_ratio = (
                branch.norm(dim=-1) / (tokens.norm(dim=-1) + 1e-12)
            ).mean()
            cosine = nn.functional.cosine_similarity(
                branch, tokens, dim=-1
            ).mean()
            rows.append(
                {
                    "layer": layer_idx + 1,
                    "branch_to_input_norm": float(norm_ratio),
                    "branch_input_cosine": float(cosine),
                }
            )
            tokens = tokens + branch
            if model.layer_norm:
                tokens = model.norm_x[layer_idx](tokens)
    return rows


def load_model_and_loader(
    repo_root: Path,
    checkpoint_root: Path,
    dataset_root: Path,
    dataset: str,
    batch_size: int,
) -> tuple[nn.Module, torch.utils.data.DataLoader, Namespace, dict[str, Any]]:
    sys.path.insert(0, str(repo_root / "baselines/timealign_official"))
    from data_provider.data_factory import data_provider
    from models import TimeAlign

    artifact_dir = run_dir(checkpoint_root, dataset)
    config = read_json(artifact_dir / "effective_config.json")
    official_args = Namespace(**config["official_args"])
    official_args.root_path = str(local_dataset_path(dataset_root, dataset))
    official_args.device = torch.device("cpu")
    official_args.use_gpu = False
    official_args.gpu_type = "cpu"
    official_args.batch_size = batch_size
    official_args.num_workers = 0
    _test_data, test_loader = data_provider(official_args, "test")

    model = TimeAlign.Model(official_args).float()
    state = torch.load(
        artifact_dir / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    return model, test_loader, official_args, config


def checkpoint_drift_rows(training_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for log_path in sorted(training_root.rglob("training_log.csv")):
        config = read_json(log_path.parent / "effective_config.json")
        with log_path.open(encoding="utf-8") as handle:
            training = list(csv.DictReader(handle))
        values = [float(row["val_mean_mse"]) for row in training]
        best_value = min(values)
        last_value = values[-1]
        rows.append(
            {
                "dataset": config["adapter"]["dataset"],
                "epochs_ran": len(values),
                "best_epoch": values.index(best_value) + 1,
                "best_val_mean_mse": best_value,
                "last_val_mean_mse": last_value,
                "last_minus_best_val_mse_pct":
                    (last_value / best_value - 1.0) * 100.0,
                "checkpoint_policy": config["adapter"]["checkpoint_policy"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        model, loader, official_args, config = load_model_and_loader(
            args.repo_root,
            args.checkpoint_root,
            args.dataset_root,
            dataset,
            args.batch_size,
        )
        preset = config["official_preset"]
        active_prefixes = (
            "patch_emb_x.",
            "encoder.",
            "norm_x.",
            "learned_basis_coeff.",
            "learned_temporal_basis",
            "learned_temporal_bias",
        )
        dropout = float(preset["dropout"])
        config_rows.append(
            {
                "dataset": dataset,
                "seq_len": official_args.seq_len,
                "patch_num": preset["patch_num"],
                "patch_len": official_args.seq_len // preset["patch_num"],
                "d_model": preset["d_model"],
                "d_ff": preset["d_ff"],
                "readout_dim": preset["patch_num"] * preset["d_model"],
                "dropout": dropout,
                "dropout_variance_factor": dropout / (1.0 - dropout),
                "expected_kept_d_ff_units": preset["d_ff"] * (1.0 - dropout),
                "total_model_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "active_forward_parameters": parameter_count(
                    model, active_prefixes
                ),
                "unused_proj_x_parameters": parameter_count(model, ("proj_x.",)),
            }
        )

        first_batch = next(iter(loader))[0]
        for row in branch_statistics(model, first_batch, official_args):
            branch_rows.append({"dataset": dataset, **row})

        variants = {"full": model}
        no_mlp = copy.deepcopy(model)
        no_mlp.encoder = nn.ModuleList(
            [ZeroResidual() for _layer in no_mlp.encoder]
        )
        variants["no_mlp_keep_norm"] = no_mlp
        embed_only = copy.deepcopy(no_mlp)
        embed_only.layer_norm = False
        variants["embed_only"] = embed_only
        metrics = {
            name: evaluate(
                variant,
                loader,
                official_args,
                args.max_batches,
            )
            for name, variant in variants.items()
        }
        for variant_name, variant_metrics in metrics.items():
            for horizon, values in variant_metrics.items():
                full_mse = metrics["full"][horizon]["mse"]
                metric_rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant_name,
                        "target_horizon": horizon,
                        "mse": values["mse"],
                        "mae": values["mae"],
                        "mse_delta_vs_full_pct":
                            (values["mse"] / full_mse - 1.0) * 100.0,
                        "evaluated_batches": args.max_batches,
                    }
                )

    write_csv(args.output_dir / "encoder_preset_audit.csv", config_rows)
    write_csv(args.output_dir / "checkpoint_drift.csv", checkpoint_drift_rows(args.training_root))
    write_csv(args.output_dir / "encoder_branch_statistics.csv", branch_rows)
    write_csv(args.output_dir / "encoder_branch_ablation.csv", metric_rows)


if __name__ == "__main__":
    main()

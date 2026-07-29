from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.dlinear.dataset import DATASETS, ForecastDataset
from baselines.intro_evidence_neutral.model import (
    NeutralSharingExtentForecaster,
)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    histories: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for history, target in loader:
            history = history.float().to(device)
            target = target.float().to(device)
            prediction = model(history)
            histories.append(history.cpu().numpy())
            predictions.append(prediction.cpu().numpy())
            targets.append(target.cpu().numpy())
    history_np = np.concatenate(histories, axis=0)
    prediction_np = np.concatenate(predictions, axis=0)
    target_np = np.concatenate(targets, axis=0)
    error = prediction_np - target_np
    metrics = {
        "mse": float(np.mean(error * error)),
        "mae": float(np.mean(np.abs(error))),
    }
    return metrics, prediction_np, target_np, history_np


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    element_count = 0
    with torch.no_grad():
        for history, target in loader:
            history = history.float().to(device)
            target = target.float().to(device)
            difference = model(history) - target
            squared_error += float(torch.sum(difference * difference))
            absolute_error += float(torch.sum(torch.abs(difference)))
            element_count += difference.numel()
    if element_count == 0:
        raise RuntimeError("evaluation loader produced no elements")
    return {
        "mse": squared_error / element_count,
        "mae": absolute_error / element_count,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the neutral single-sharing-extent diagnostic."
    )
    parser.add_argument(
        "--dataset-root",
        default="/Users/river/PaperResearch/Project/datasets",
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="Weather")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=720)
    parser.add_argument(
        "--sharing-extent",
        type=int,
        choices=[1, 8, 32, 128, 720],
        required=True,
    )
    parser.add_argument("--history-dim", type=int, default=64)
    parser.add_argument("--step-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--run-name", default="NeutralSharingExtent")
    parser.add_argument(
        "--output-root",
        default="artifacts/runs/intro_evidence_visualization",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run one synthetic optimization step without loading data.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_model(args: argparse.Namespace) -> NeutralSharingExtentForecaster:
    return NeutralSharingExtentForecaster(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        sharing_extent=args.sharing_extent,
        history_dim=args.history_dim,
        step_dim=args.step_dim,
        hidden_dim=args.hidden_dim,
        state_dim=args.state_dim,
    )


def run_synthetic_smoke(args: argparse.Namespace, device: torch.device) -> None:
    model = build_model(args).to(device)
    history = torch.randn(2, args.seq_len, 3, device=device)
    target = torch.randn(2, args.pred_len, 3, device=device)
    prediction, candidate, pooled = model.forward_with_states(history)
    loss = torch.mean((prediction - target) ** 2)
    loss.backward()
    gradient_count = sum(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in model.parameters()
    )
    result = {
        "prediction_shape": list(prediction.shape),
        "candidate_shape": list(candidate.shape),
        "pooled_shape": list(pooled.shape),
        "parameter_count": count_parameters(model),
        "gradient_parameter_count": gradient_count,
        "all_parameters_have_finite_nonzero_gradient": gradient_count
        == len(list(model.parameters())),
        "finite": bool(torch.isfinite(prediction).all()),
    }
    print(json.dumps(result, indent=2))
    if not result["all_parameters_have_finite_nonzero_gradient"]:
        raise RuntimeError("synthetic gradient contract failed")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    if args.synthetic_smoke:
        run_synthetic_smoke(args, device)
        return

    train_set = ForecastDataset(
        args.dataset_root,
        args.dataset,
        "train",
        args.seq_len,
        args.pred_len,
    )
    val_set = ForecastDataset(
        args.dataset_root,
        args.dataset,
        "val",
        args.seq_len,
        args.pred_len,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model(args).to(device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    run_dir = (
        Path(args.output_root)
        / args.run_name
        / args.dataset
        / f"s{args.sharing_extent}"
        / f"seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    log_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for history, target in train_loader:
            history = history.float().to(device)
            target = target.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_metrics(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
        }
        log_rows.append(row)
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    val_metrics, prediction, target, history = evaluate(model, val_loader, device)

    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    np.savez_compressed(
        run_dir / "predictions_val.npz",
        pred=prediction,
        true=target,
        history=history,
        origin_index=np.arange(len(prediction), dtype=np.int64),
        train_mean=train_set.scaler.mean,
        train_std=train_set.scaler.std,
    )
    (run_dir / "metrics_val.json").write_text(
        json.dumps(val_metrics, indent=2),
        encoding="utf-8",
    )
    write_csv(run_dir / "training_log.csv", log_rows)
    config = {
        **vars(args),
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "split_role": "validation_explanatory_visualization_only",
        "test_accessed": False,
    }
    (run_dir / "effective_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

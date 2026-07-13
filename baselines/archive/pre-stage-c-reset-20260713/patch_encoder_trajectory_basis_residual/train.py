from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import DATASETS, ForecastDataset
from model import PatchEncoderTrajectoryBasisResidual


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics_by_horizon(pred: np.ndarray, true: np.ndarray) -> list[dict[str, float]]:
    diff = pred - true
    mse = np.mean(diff * diff, axis=(0, 2))
    mae = np.mean(np.abs(diff), axis=(0, 2))
    return [{"horizon": i + 1, "mse": float(mse[i]), "mae": float(mae[i])} for i in range(len(mse))]


def metrics_by_segment(pred: np.ndarray, true: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for start, end in [(1, 96), (97, 192), (193, 336), (337, 720)]:
        if pred.shape[1] < start:
            continue
        segment_pred = pred[:, start - 1 : min(end, pred.shape[1]), :]
        segment_true = true[:, start - 1 : min(end, true.shape[1]), :]
        diff = segment_pred - segment_true
        rows.append(
            {
                "segment": f"{start}-{min(end, pred.shape[1])}",
                "mse": float(np.mean(diff * diff)),
                "mae": float(np.mean(np.abs(diff))),
            }
        )
    return rows


def residual_smoothness_torch(residual: torch.Tensor) -> torch.Tensor:
    if residual.shape[1] < 3:
        return residual.new_tensor(0.0)
    second_diff = residual[:, 2:, :] - 2.0 * residual[:, 1:-1, :] + residual[:, :-2, :]
    return torch.mean(second_diff * second_diff)


def residual_smoothness_np(residual: np.ndarray) -> float:
    if residual.shape[1] < 3:
        return 0.0
    second_diff = residual[:, 2:, :] - 2.0 * residual[:, 1:-1, :] + residual[:, :-2, :]
    return float(np.mean(second_diff * second_diff))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    component_rows: dict[str, list[np.ndarray]] = {
        "base_prediction": [],
        "residual": [],
        "raw_residual": [],
        "residual_norm": [],
        "raw_residual_norm": [],
        "residual_gate": [],
        "coefficients": [],
    }
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader, start=1):
            x = x.float().to(device)
            y = y.float().to(device)
            output = model(x, return_components=True)
            if not isinstance(output, dict):
                raise TypeError("PatchEncoderTrajectoryBasisResidual must return component dict.")
            pred = output["prediction"]
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
            for name in component_rows:
                component_rows[name].append(output[name].cpu().numpy())
            if max_batches is not None and batch_index >= max_batches:
                break
    pred_np = np.concatenate(preds, axis=0)
    true_np = np.concatenate(trues, axis=0)
    components = {name: np.concatenate(values, axis=0) for name, values in component_rows.items()}
    diff = pred_np - true_np
    return {"mse": float(np.mean(diff * diff)), "mae": float(np.mean(np.abs(diff)))}, pred_np, true_np, components


def residual_stats(pred: np.ndarray, components: dict[str, np.ndarray]) -> list[dict[str, float]]:
    base = components["base_prediction"]
    residual = components["residual"]
    raw_residual = components["raw_residual"]
    residual_norm = components["residual_norm"]
    gate = components["residual_gate"]
    coefficients = components["coefficients"]
    base_abs = np.mean(np.abs(base)) + 1e-12
    pred_abs = np.mean(np.abs(pred)) + 1e-12
    rows = [
        {
            "scope": "all",
            "residual_mse": float(np.mean(residual * residual)),
            "residual_mae": float(np.mean(np.abs(residual))),
            "raw_residual_mae": float(np.mean(np.abs(raw_residual))),
            "residual_to_base_mae_ratio": float(np.mean(np.abs(residual)) / base_abs),
            "residual_to_prediction_mae_ratio": float(np.mean(np.abs(residual)) / pred_abs),
            "mean_gate": float(np.mean(gate)),
            "max_gate": float(np.max(gate)),
            "coefficient_l2": float(np.mean(np.linalg.norm(coefficients, axis=-1))),
            "residual_norm_smoothness": residual_smoothness_np(residual_norm),
        }
    ]
    for start, end in [(1, 96), (97, 192), (193, 336), (337, 720)]:
        if residual.shape[1] < start:
            continue
        segment = residual[:, start - 1 : min(end, residual.shape[1]), :]
        rows.append(
            {
                "scope": f"{start}-{min(end, residual.shape[1])}",
                "residual_mse": float(np.mean(segment * segment)),
                "residual_mae": float(np.mean(np.abs(segment))),
                "raw_residual_mae": float(
                    np.mean(np.abs(raw_residual[:, start - 1 : min(end, raw_residual.shape[1]), :]))
                ),
                "residual_to_base_mae_ratio": float(np.mean(np.abs(segment)) / base_abs),
                "residual_to_prediction_mae_ratio": float(np.mean(np.abs(segment)) / pred_abs),
                "mean_gate": float(np.mean(gate[:, start - 1 : min(end, gate.shape[1]), :])),
                "max_gate": float(np.max(gate[:, start - 1 : min(end, gate.shape[1]), :])),
                "coefficient_l2": float(np.mean(np.linalg.norm(coefficients, axis=-1))),
                "residual_norm_smoothness": residual_smoothness_np(
                    residual_norm[:, start - 1 : min(end, residual_norm.shape[1]), :]
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PatchEncoderTrajectoryBasisResidual Phase 1 candidate.")
    parser.add_argument("--dataset-root", default="/Users/river/PaperResearch/Project/datasets")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ETTh2")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, choices=[96, 192, 336, 720], default=96)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument("--basis-count", type=int, default=8)
    parser.add_argument("--residual-gate-init", type=float, default=-4.0)
    parser.add_argument("--residual-penalty", type=float, default=1e-4)
    parser.add_argument("--smoothness-penalty", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--run-name", default="PatchEncoderTrajectoryBasisResidual")
    parser.add_argument("--output-root", default="artifacts/runs/phase1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    train_set = ForecastDataset(args.dataset_root, args.dataset, "train", args.seq_len, args.pred_len)
    val_set = ForecastDataset(args.dataset_root, args.dataset, "val", args.seq_len, args.pred_len)
    test_set = ForecastDataset(args.dataset_root, args.dataset, "test", args.seq_len, args.pred_len)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = PatchEncoderTrajectoryBasisResidual(
        args.seq_len,
        args.pred_len,
        DATASETS[args.dataset].channels,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        encoder_layers=args.encoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        basis_count=args.basis_count,
        residual_gate_init=args.residual_gate_init,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    run_dir = Path(args.output_root) / args.run_name / args.dataset / f"h{args.pred_len}" / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_state = None
    stale_epochs = 0
    log_rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        mse_losses = []
        residual_losses = []
        smoothness_losses = []
        for batch_index, (x, y) in enumerate(train_loader, start=1):
            x = x.float().to(device)
            y = y.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x, return_components=True)
            if not isinstance(output, dict):
                raise TypeError("PatchEncoderTrajectoryBasisResidual must return component dict.")
            pred = output["prediction"]
            residual_norm = output["residual_norm"]
            mse_loss = criterion(pred, y)
            residual_loss = torch.mean(residual_norm * residual_norm)
            smoothness_loss = residual_smoothness_torch(residual_norm)
            loss = (
                mse_loss
                + args.residual_penalty * residual_loss
                + args.smoothness_penalty * smoothness_loss
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            mse_losses.append(float(mse_loss.detach().cpu()))
            residual_losses.append(float(residual_loss.detach().cpu()))
            smoothness_losses.append(float(smoothness_loss.detach().cpu()))
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break

        val_metrics, _, _, _ = evaluate(model, val_loader, device, args.max_eval_batches)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_mse_loss": float(np.mean(mse_losses)),
            "train_residual_loss": float(np.mean(residual_losses)),
            "train_smoothness_loss": float(np.mean(smoothness_losses)),
            **val_metrics,
        }
        log_rows.append(row)
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, pred_np, true_np, components = evaluate(model, test_loader, device, args.max_eval_batches)

    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    np.savez_compressed(run_dir / "predictions_test.npz", pred=pred_np, true=true_np)
    (run_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2))
    write_csv(run_dir / "metrics_by_horizon.csv", metrics_by_horizon(pred_np, true_np))
    write_csv(run_dir / "metrics_by_segment.csv", metrics_by_segment(pred_np, true_np))
    write_csv(run_dir / "trajectory_residual_stats.csv", residual_stats(pred_np, components))
    write_csv(run_dir / "training_log.csv", log_rows)
    (run_dir / "effective_config.json").write_text(json.dumps(vars(args), indent=2))
    env = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    (run_dir / "environment.json").write_text(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()

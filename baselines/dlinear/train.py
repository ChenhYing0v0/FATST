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
from model import DLinear


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(pred: torch.Tensor, true: torch.Tensor) -> dict[str, float]:
    diff = pred - true
    return {"mse": float(torch.mean(diff * diff)), "mae": float(torch.mean(torch.abs(diff)))}


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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    histories: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.float().to(device)
            y = y.float().to(device)
            pred = model(x)
            histories.append(x.cpu().numpy())
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
    history_np = np.concatenate(histories, axis=0)
    pred_np = np.concatenate(preds, axis=0)
    true_np = np.concatenate(trues, axis=0)
    diff = pred_np - true_np
    return (
        {"mse": float(np.mean(diff * diff)), "mae": float(np.mean(np.abs(diff)))},
        pred_np,
        true_np,
        history_np,
    )


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
        for x, y in loader:
            x = x.float().to(device)
            y = y.float().to(device)
            difference = model(x) - y
            squared_error += float(torch.sum(difference * difference))
            absolute_error += float(torch.sum(torch.abs(difference)))
            element_count += difference.numel()
    if element_count == 0:
        raise RuntimeError("evaluation loader produced no elements")
    return {
        "mse": squared_error / element_count,
        "mae": absolute_error / element_count,
    }


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DLinear Phase 0 baseline.")
    parser.add_argument("--dataset-root", default="/Users/river/PaperResearch/Project/datasets")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ETTh2")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, choices=[96, 192, 336, 720], default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--init-mode", choices=["average", "pytorch_default"], default="average")
    parser.add_argument("--run-name", default="DLinear")
    parser.add_argument("--output-root", default="artifacts/runs/phase0")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Save validation predictions without accessing the test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    train_set = ForecastDataset(
        args.dataset_root,
        args.dataset,
        "train",
        args.seq_len,
        args.pred_len,
    )
    val_set = ForecastDataset(args.dataset_root, args.dataset, "val", args.seq_len, args.pred_len)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = None
    if not args.skip_test:
        test_set = ForecastDataset(
            args.dataset_root,
            args.dataset,
            "test",
            args.seq_len,
            args.pred_len,
        )
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = DLinear(
        args.seq_len,
        args.pred_len,
        DATASETS[args.dataset].channels,
        init_mode=args.init_mode,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    run_dir = (
        Path(args.output_root)
        / args.run_name
        / args.dataset
        / f"h{args.pred_len}"
        / f"seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_state = None
    stale_epochs = 0
    log_rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.float().to(device)
            y = y.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_metrics(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **val_metrics}
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
    val_metrics, val_pred, val_true, val_history = evaluate(model, val_loader, device)

    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    np.savez_compressed(
        run_dir / "predictions_val.npz",
        pred=val_pred,
        true=val_true,
        history=val_history,
        origin_index=np.arange(len(val_pred), dtype=np.int64),
        train_mean=train_set.scaler.mean,
        train_std=train_set.scaler.std,
    )
    (run_dir / "metrics_val.json").write_text(json.dumps(val_metrics, indent=2))
    if test_loader is not None:
        test_metrics, pred_np, true_np, history_np = evaluate(
            model,
            test_loader,
            device,
        )
        np.savez_compressed(
            run_dir / "predictions_test.npz",
            pred=pred_np,
            true=true_np,
            history=history_np,
            origin_index=np.arange(len(pred_np), dtype=np.int64),
            train_mean=train_set.scaler.mean,
            train_std=train_set.scaler.std,
        )
        (run_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2))
        write_csv(run_dir / "metrics_by_horizon.csv", metrics_by_horizon(pred_np, true_np))
        write_csv(run_dir / "metrics_by_segment.csv", metrics_by_segment(pred_np, true_np))
    write_csv(run_dir / "training_log.csv", log_rows)
    (run_dir / "effective_config.json").write_text(json.dumps(vars(args), indent=2))
    env = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
    }
    (run_dir / "environment.json").write_text(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()

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

from dataset import DATASETS, ForecastDataset
from model import TimeAlignCarrier


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_horizons(value: str, pred_len: int) -> list[int]:
    horizons = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not horizons:
        raise ValueError("At least one target horizon is required.")
    invalid = [horizon for horizon in horizons if horizon <= 0 or horizon > pred_len]
    if invalid:
        raise ValueError(f"target horizons must be in [1, pred_len={pred_len}], got {invalid}.")
    return horizons


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_pair(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    diff = pred - true
    return {
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
    }


def metrics_by_step(pred: np.ndarray, true: np.ndarray) -> list[dict[str, float]]:
    diff = pred - true
    mse = np.mean(diff * diff, axis=(0, 2))
    mae = np.mean(np.abs(diff), axis=(0, 2))
    return [{"horizon": i + 1, "mse": float(mse[i]), "mae": float(mae[i])} for i in range(pred.shape[1])]


def metrics_by_segment(pred: np.ndarray, true: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for start, end in [(1, 96), (97, 192), (193, 336), (337, 720)]:
        if pred.shape[1] < start:
            continue
        stop = min(end, pred.shape[1])
        segment_pred = pred[:, start - 1 : stop, :]
        segment_true = true[:, start - 1 : stop, :]
        row = {"segment": f"{start}-{stop}", **metric_pair(segment_pred, segment_true)}
        rows.append(row)
    return rows


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_horizons: list[int],
    max_eval_batches: int | None = None,
) -> tuple[list[dict[str, Any]], dict[int, tuple[np.ndarray, np.ndarray]]]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if max_eval_batches is not None and batch_idx >= max_eval_batches:
                break
            x = x.float().to(device)
            y = y.float().to(device)
            output = model(x)
            preds.append(output["prediction"].detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())
    pred_np = np.concatenate(preds, axis=0)
    true_np = np.concatenate(trues, axis=0)
    rows = []
    arrays = {}
    for horizon in target_horizons:
        horizon_pred = pred_np[:, :horizon, :]
        horizon_true = true_np[:, :horizon, :]
        rows.append({"target_horizon": horizon, **metric_pair(horizon_pred, horizon_true)})
        arrays[horizon] = (horizon_pred, horizon_true)
    return rows, arrays


def mean_mse(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([float(row["mse"]) for row in rows]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a repo-local TimeAlign carrier baseline.")
    parser.add_argument("--dataset-root", default="/Users/river/PaperResearch/Project/datasets")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ETTh2")
    parser.add_argument("--seq-len", type=int, default=720)
    parser.add_argument("--pred-len", type=int, choices=[96, 192, 336, 720], default=96)
    parser.add_argument("--target-horizons", default="")
    parser.add_argument("--patch-num", type=int, default=48)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-ff", type=int, default=32)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-layer-norm", action="store_true")
    parser.add_argument("--no-positional", action="store_true")
    parser.add_argument("--disable-local-align", action="store_true")
    parser.add_argument("--disable-global-align", action="store_true")
    parser.add_argument("--local-margin", type=float, default=0.0)
    parser.add_argument("--global-margin", type=float, default=0.0)
    parser.add_argument("--w-align", type=float, default=0.1)
    parser.add_argument("--w-recon", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-name", default="TimeAlignCarrier")
    parser.add_argument("--output-root", default="artifacts/runs/timealign_carrier")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    target_horizons = parse_horizons(args.target_horizons or str(args.pred_len), args.pred_len)
    max_train_batches = args.max_train_batches if args.max_train_batches > 0 else None
    max_eval_batches = args.max_eval_batches if args.max_eval_batches > 0 else None

    train_set = ForecastDataset(args.dataset_root, args.dataset, "train", args.seq_len, args.pred_len)
    val_set = ForecastDataset(args.dataset_root, args.dataset, "val", args.seq_len, args.pred_len)
    test_set = ForecastDataset(args.dataset_root, args.dataset, "test", args.seq_len, args.pred_len)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = TimeAlignCarrier(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        channels=DATASETS[args.dataset].channels,
        patch_num=args.patch_num,
        d_model=args.d_model,
        d_ff=args.d_ff,
        e_layers=args.e_layers,
        dropout=args.dropout,
        use_layer_norm=not args.no_layer_norm,
        use_positional=not args.no_positional,
        use_local_align=not args.disable_local_align,
        use_global_align=not args.disable_global_align,
        local_margin=args.local_margin,
        global_margin=args.global_margin,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.L1Loss()

    horizon_label = "mixed_" + "_".join(f"h{horizon}" for horizon in target_horizons)
    run_dir = Path(args.output_root) / args.run_name / args.dataset / horizon_label / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "effective_config.json").write_text(json.dumps(vars(args), indent=2))
    env = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "dataset": args.dataset,
    }
    (run_dir / "environment.json").write_text(json.dumps(env, indent=2))

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    log_rows: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        pred_losses = []
        recon_losses = []
        align_losses = []
        local_losses = []
        global_losses = []
        for batch_idx, (x, y) in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            x = x.float().to(device)
            y = y.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x, y)
            pred_loss = criterion(output["prediction"], y)
            recon_loss = criterion(output["reconstruction"], y)
            align_loss = output["alignment_loss"]
            loss = pred_loss + args.w_recon * recon_loss + args.w_align * align_loss
            loss.backward()
            optimizer.step()
            pred_losses.append(float(pred_loss.detach().cpu()))
            recon_losses.append(float(recon_loss.detach().cpu()))
            align_losses.append(float(align_loss.detach().cpu()))
            local_losses.append(float(output["local_alignment_loss"].detach().cpu()))
            global_losses.append(float(output["global_alignment_loss"].detach().cpu()))

        val_rows, _ = evaluate(model, val_loader, device, target_horizons, max_eval_batches)
        val_mean_mse = mean_mse(val_rows)
        row = {
            "epoch": epoch,
            "train_prediction_l1": float(np.mean(pred_losses)),
            "train_reconstruction_l1": float(np.mean(recon_losses)),
            "train_alignment_loss": float(np.mean(align_losses)),
            "train_local_alignment_loss": float(np.mean(local_losses)),
            "train_global_alignment_loss": float(np.mean(global_losses)),
            "val_mean_mse": val_mean_mse,
            "val_mean_mae": float(np.mean([float(item["mae"]) for item in val_rows])),
        }
        for item in val_rows:
            row[f"val_h{item['target_horizon']}_mse"] = item["mse"]
            row[f"val_h{item['target_horizon']}_mae"] = item["mae"]
        log_rows.append(row)
        print(
            "epoch_progress "
            f"run_name={args.run_name} dataset={args.dataset} epoch={epoch}/{args.epochs} "
            f"val_mean_mse={val_mean_mse:.6f}",
            flush=True,
        )
        if val_mean_mse < best_val:
            best_val = val_mean_mse
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_rows, test_arrays = evaluate(model, test_loader, device, target_horizons, max_eval_batches=None)
    write_csv(run_dir / "training_log.csv", log_rows)
    write_csv(run_dir / "metrics_by_target_horizon.csv", test_rows)
    (run_dir / "metrics.json").write_text(
        json.dumps({"mean_mse": mean_mse(test_rows), "best_epoch": best_epoch}, indent=2)
    )
    torch.save(model.state_dict(), run_dir / "checkpoint.pt")

    diagnostic_rows = []
    for horizon, (pred_np, true_np) in test_arrays.items():
        horizon_dir = run_dir / f"h{horizon}"
        write_csv(horizon_dir / "metrics_by_horizon.csv", metrics_by_step(pred_np, true_np))
        write_csv(horizon_dir / "metrics_by_segment.csv", metrics_by_segment(pred_np, true_np))
        diagnostic_rows.append(
            {
                "target_horizon": horizon,
                "best_epoch": best_epoch,
                "official_best_epoch": best_epoch,
                "official_gap_to_selector_best_pct": 0.0,
            }
        )
        if args.save_predictions:
            np.savez_compressed(horizon_dir / "predictions_test.npz", pred=pred_np, true=true_np)
    write_csv(run_dir / "checkpoint_selection_diagnostics.csv", diagnostic_rows)


if __name__ == "__main__":
    main()

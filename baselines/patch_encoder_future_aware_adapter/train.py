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
from model import PatchEncoderFutureAwareAdapter


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


def segment_state_similarity(model: PatchEncoderFutureAwareAdapter) -> list[dict[str, float]]:
    queries = model.segment_queries.detach().cpu().squeeze(0)
    if queries.shape[0] < 2:
        return [{"segment_a": 0, "segment_b": 0, "cosine": 1.0}]
    normed = torch.nn.functional.normalize(queries, dim=-1)
    sims = normed @ normed.T
    rows = []
    for i in range(sims.shape[0]):
        for j in range(i + 1, sims.shape[1]):
            rows.append({"segment_a": i, "segment_b": j, "cosine": float(sims[i, j])})
    return rows


def evaluate(
    model: PatchEncoderFutureAwareAdapter,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    component_rows: dict[str, list[np.ndarray]] = {
        "base_prediction": [],
        "gamma": [],
        "beta": [],
    }
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader, start=1):
            x = x.float().to(device)
            y = y.float().to(device)
            output = model(x, return_components=True)
            if not isinstance(output, dict):
                raise TypeError("PatchEncoderFutureAwareAdapter must return component dict.")
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


def future_alignment_stats(
    model: PatchEncoderFutureAwareAdapter,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> list[dict[str, float]]:
    model.eval()
    align_losses = []
    recon_losses = []
    leakage_values = []
    cosines = []
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader, start=1):
            x = x.float().to(device)
            y = y.float().to(device)
            no_future = model(x, return_components=True)
            with_future = model(x, y=y, return_components=True)
            if not isinstance(no_future, dict) or not isinstance(with_future, dict):
                raise TypeError("PatchEncoderFutureAwareAdapter must return component dict.")
            leakage_values.append(
                float((no_future["prediction"] - with_future["prediction"]).abs().max().detach().cpu())
            )
            align_losses.append(float(with_future["alignment_loss"].detach().cpu()))
            recon_losses.append(float(with_future["reconstruction_loss"].detach().cpu()))
            student = torch.nn.functional.normalize(with_future["student_aligned_state"], dim=-1)
            teacher = torch.nn.functional.normalize(with_future["teacher_state"], dim=-1)
            cosines.append(float((student * teacher).sum(dim=-1).mean().detach().cpu()))
            if max_batches is not None and batch_index >= max_batches:
                break
    return [
        {
            "scope": "test",
            "alignment_loss": float(np.mean(align_losses)),
            "reconstruction_loss": float(np.mean(recon_losses)),
            "teacher_student_cosine": float(np.mean(cosines)),
            "prediction_leakage_max_abs": float(np.max(leakage_values)),
        }
    ]


def adapter_delta_stats(pred: np.ndarray, components: dict[str, np.ndarray]) -> list[dict[str, float]]:
    base = components["base_prediction"]
    gamma = components["gamma"]
    beta = components["beta"]
    delta = pred - base
    base_abs = np.mean(np.abs(base)) + 1e-12
    return [
        {
            "scope": "all",
            "delta_mse_to_base": float(np.mean(delta * delta)),
            "delta_mae_to_base": float(np.mean(np.abs(delta))),
            "mean_abs_gamma": float(np.mean(np.abs(gamma))),
            "mean_abs_beta": float(np.mean(np.abs(beta))),
            "delta_to_base_mae_ratio": float(np.mean(np.abs(delta)) / base_abs),
        }
    ]


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PatchEncoderFutureAwareAdapter Phase 1 baseline.")
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
    parser.add_argument("--segment-len", type=int, default=48)
    parser.add_argument("--adapter-layers", type=int, default=1)
    parser.add_argument("--adapter-heads", type=int, default=8)
    parser.add_argument("--adapter-d-ff", type=int, default=256)
    parser.add_argument("--teacher-layers", type=int, default=1)
    parser.add_argument("--teacher-heads", type=int, default=8)
    parser.add_argument("--teacher-d-ff", type=int, default=256)
    parser.add_argument("--align-weight", type=float, default=0.05)
    parser.add_argument("--recon-weight", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--run-name", default="PatchEncoderFutureAwareAdapter")
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

    model = PatchEncoderFutureAwareAdapter(
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
        segment_len=args.segment_len,
        adapter_layers=args.adapter_layers,
        adapter_heads=args.adapter_heads,
        adapter_d_ff=args.adapter_d_ff,
        teacher_layers=args.teacher_layers,
        teacher_heads=args.teacher_heads,
        teacher_d_ff=args.teacher_d_ff,
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
        pred_losses = []
        align_losses = []
        recon_losses = []
        for batch_index, (x, y) in enumerate(train_loader, start=1):
            x = x.float().to(device)
            y = y.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x, y=y, return_components=True)
            if not isinstance(output, dict):
                raise TypeError("PatchEncoderFutureAwareAdapter must return component dict.")
            pred = output["prediction"]
            pred_loss = criterion(pred, y)
            align_loss = output["alignment_loss"]
            recon_loss = output["reconstruction_loss"]
            loss = pred_loss + args.align_weight * align_loss + args.recon_weight * recon_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            pred_losses.append(float(pred_loss.detach().cpu()))
            align_losses.append(float(align_loss.detach().cpu()))
            recon_losses.append(float(recon_loss.detach().cpu()))
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break

        val_metrics, _, _, _ = evaluate(model, val_loader, device, args.max_eval_batches)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_pred_loss": float(np.mean(pred_losses)),
            "train_alignment_loss": float(np.mean(align_losses)),
            "train_reconstruction_loss": float(np.mean(recon_losses)),
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
    write_csv(run_dir / "adapter_query_similarity.csv", segment_state_similarity(model))
    write_csv(run_dir / "adapter_delta_stats.csv", adapter_delta_stats(pred_np, components))
    write_csv(run_dir / "future_alignment_stats.csv", future_alignment_stats(model, test_loader, device, args.max_eval_batches))
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

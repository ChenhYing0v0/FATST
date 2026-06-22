from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DATASETS, ForecastDataset
from model import PatchEncoderTargetSetDecoder


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_horizons(value: str) -> list[int]:
    horizons = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not horizons:
        raise ValueError("At least one target horizon is required.")
    allowed = {96, 192, 336, 720}
    unknown = sorted(set(horizons) - allowed)
    if unknown:
        raise ValueError(f"Unsupported target horizons: {unknown}")
    return sorted(set(horizons))


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


def target_state_similarity(components: dict[str, np.ndarray]) -> list[dict[str, float]]:
    states = components["target_states"]
    if states.shape[2] < 2:
        return [{"segment_a": 0, "segment_b": 0, "cosine": 1.0}]
    flat = states.reshape(-1, states.shape[2], states.shape[3])
    norm = np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-12
    normed = flat / norm
    rows = []
    for i in range(states.shape[2]):
        for j in range(i + 1, states.shape[2]):
            cosine = np.sum(normed[:, i, :] * normed[:, j, :], axis=-1)
            rows.append({"segment_a": i, "segment_b": j, "cosine": float(np.mean(cosine))})
    return rows


def target_conditioning_stats(components: dict[str, np.ndarray]) -> list[dict[str, float]]:
    states = components["target_states"]
    gamma = components["gamma"]
    beta = components["beta"]
    history = components["history_readout"]
    rows = [
        {
            "scope": "all",
            "mean_abs_gamma": float(np.mean(np.abs(gamma))),
            "mean_abs_beta": float(np.mean(np.abs(beta))),
            "mean_target_state_norm": float(np.mean(np.linalg.norm(states, axis=-1))),
            "std_target_state_norm": float(np.std(np.linalg.norm(states, axis=-1))),
            "mean_history_readout_norm": float(np.mean(np.linalg.norm(history, axis=-1))),
        }
    ]
    for segment_index in range(states.shape[2]):
        rows.append(
            {
                "scope": f"segment_{segment_index}",
                "mean_abs_gamma": float(np.mean(np.abs(gamma[:, :, segment_index, :]))),
                "mean_abs_beta": float(np.mean(np.abs(beta[:, :, segment_index, :]))),
                "mean_target_state_norm": float(
                    np.mean(np.linalg.norm(states[:, :, segment_index, :], axis=-1))
                ),
                "std_target_state_norm": float(
                    np.std(np.linalg.norm(states[:, :, segment_index, :], axis=-1))
                ),
                "mean_history_readout_norm": float(np.mean(np.linalg.norm(history, axis=-1))),
            }
        )
    return rows


def prefix_residual_stats(components: dict[str, np.ndarray]) -> list[dict[str, float]]:
    residual = components["prefix_residual_norm"]
    rows = [
        {
            "scope": "all",
            "residual_mse_norm": float(np.mean(residual * residual)),
            "residual_mae_norm": float(np.mean(np.abs(residual))),
            "max_abs_residual_norm": float(np.max(np.abs(residual))),
        }
    ]
    for start, end in [(1, 96), (97, 192), (193, 336), (337, 720)]:
        if residual.shape[1] < start:
            continue
        segment = residual[:, start - 1 : min(end, residual.shape[1]), :]
        rows.append(
            {
                "scope": f"{start}-{min(end, residual.shape[1])}",
                "residual_mse_norm": float(np.mean(segment * segment)),
                "residual_mae_norm": float(np.mean(np.abs(segment))),
                "max_abs_residual_norm": float(np.max(np.abs(segment))),
            }
        )
    return rows


def weighted_mse_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    max_pred_len: int,
    mode: str,
    alpha: float,
) -> torch.Tensor:
    if mode == "uniform":
        return torch.mean((pred - true) ** 2)
    if mode != "prefix_risk":
        raise ValueError(f"Unknown step loss weighting mode: {mode}")
    if alpha < 0:
        raise ValueError("step_loss_alpha must be non-negative.")

    horizon = pred.shape[1]
    step = torch.arange(1, horizon + 1, device=pred.device, dtype=pred.dtype)
    full_step = torch.arange(1, max_pred_len + 1, device=pred.device, dtype=pred.dtype)
    weights = torch.pow(step / float(max_pred_len), -alpha)
    full_weights = torch.pow(full_step / float(max_pred_len), -alpha)
    weights = weights / torch.mean(full_weights)
    return torch.mean((pred - true) ** 2 * weights.view(1, horizon, 1))


def evaluate(
    model: PatchEncoderTargetSetDecoder,
    loader: DataLoader,
    device: torch.device,
    pred_len: int,
    max_batches: int | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    component_rows: dict[str, list[np.ndarray]] = {
        "target_states": [],
        "gamma": [],
        "beta": [],
        "history_readout": [],
        "prefix_residual_norm": [],
    }
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader, start=1):
            x = x.float().to(device)
            y = y.float().to(device)
            output = model(x, pred_len=pred_len, return_components=True)
            if not isinstance(output, dict):
                raise TypeError("PatchEncoderTargetSetDecoder must return component dict.")
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


def prefix_consistency(
    model: PatchEncoderTargetSetDecoder,
    loader: DataLoader,
    device: torch.device,
    short_horizons: list[int],
    long_horizon: int,
    max_batches: int | None = None,
) -> list[dict[str, float]]:
    model.eval()
    mismatch: dict[int, list[np.ndarray]] = {horizon: [] for horizon in short_horizons}
    with torch.no_grad():
        for batch_index, (x, _) in enumerate(loader, start=1):
            x = x.float().to(device)
            long_pred = model(x, pred_len=long_horizon)
            if isinstance(long_pred, dict):
                raise TypeError("Expected tensor prediction.")
            for horizon in short_horizons:
                short_pred = model(x, pred_len=horizon)
                if isinstance(short_pred, dict):
                    raise TypeError("Expected tensor prediction.")
                diff = short_pred - long_pred[:, :horizon, :]
                mismatch[horizon].append(diff.cpu().numpy())
            if max_batches is not None and batch_index >= max_batches:
                break
    rows = []
    for horizon, values in mismatch.items():
        diff_np = np.concatenate(values, axis=0)
        rows.append(
            {
                "short_horizon": horizon,
                "long_horizon": long_horizon,
                "prefix_mismatch_mse": float(np.mean(diff_np * diff_np)),
                "prefix_mismatch_mae": float(np.mean(np.abs(diff_np))),
                "truth_alignment_mse": 0.0,
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


def build_loaders(
    dataset_root: str,
    dataset: str,
    split: str,
    seq_len: int,
    horizons: list[int],
    batch_size: int,
    drop_last: bool,
) -> dict[int, DataLoader]:
    return {
        horizon: DataLoader(
            ForecastDataset(dataset_root, dataset, split, seq_len, horizon),
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=drop_last,
        )
        for horizon in horizons
    }


def next_batch(
    loaders: dict[int, DataLoader],
    iterators: dict[int, Iterator[tuple[torch.Tensor, torch.Tensor]]],
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return next(iterators[horizon])
    except StopIteration:
        iterators[horizon] = iter(loaders[horizon])
        return next(iterators[horizon])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PatchEncoderTargetSetDecoder Phase1-R candidate.")
    parser.add_argument("--dataset-root", default="/Users/river/PaperResearch/Project/datasets")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ETTh2")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--target-horizons", default="96,192,336,720")
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument("--segment-len", type=int, default=48)
    parser.add_argument("--max-pred-len", type=int, default=720)
    parser.add_argument("--target-layers", type=int, default=1)
    parser.add_argument("--target-heads", type=int, default=8)
    parser.add_argument("--target-d-ff", type=int, default=256)
    parser.add_argument("--target-interaction-layers", type=int, default=0)
    parser.add_argument("--target-interaction-heads", type=int, default=0)
    parser.add_argument("--target-interaction-d-ff", type=int, default=0)
    parser.add_argument("--readout-dim", type=int, default=256)
    parser.add_argument("--prefix-residual-segments", type=int, default=0)
    parser.add_argument("--prefix-residual-dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--step-loss-weighting", choices=["uniform", "prefix_risk"], default="uniform")
    parser.add_argument("--step-loss-alpha", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--run-name", default="PatchEncoderTargetSetDecoder")
    parser.add_argument("--output-root", default="artifacts/runs/phase1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_horizons = parse_horizons(args.target_horizons)
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    train_loaders = build_loaders(
        args.dataset_root,
        args.dataset,
        "train",
        args.seq_len,
        target_horizons,
        args.batch_size,
        drop_last=True,
    )
    val_loaders = build_loaders(
        args.dataset_root,
        args.dataset,
        "val",
        args.seq_len,
        target_horizons,
        args.batch_size,
        drop_last=False,
    )
    test_loaders = build_loaders(
        args.dataset_root,
        args.dataset,
        "test",
        args.seq_len,
        target_horizons,
        args.batch_size,
        drop_last=False,
    )

    model = PatchEncoderTargetSetDecoder(
        args.seq_len,
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
        max_pred_len=args.max_pred_len,
        target_layers=args.target_layers,
        target_heads=args.target_heads,
        target_d_ff=args.target_d_ff,
        target_interaction_layers=args.target_interaction_layers,
        target_interaction_heads=args.target_interaction_heads or None,
        target_interaction_d_ff=args.target_interaction_d_ff or None,
        readout_dim=args.readout_dim,
        prefix_residual_segments=args.prefix_residual_segments,
        prefix_residual_dropout=args.prefix_residual_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    horizon_label = "mixed_" + "_".join(f"h{horizon}" for horizon in target_horizons)
    run_dir = Path(args.output_root) / args.run_name / args.dataset / horizon_label / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is None:
        steps_per_epoch = sum(len(loader) for loader in train_loaders.values())
    if args.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_train_batches)

    best_val = float("inf")
    best_state = None
    stale_epochs = 0
    log_rows: list[dict[str, float]] = []
    rng = random.Random(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        iterators = {horizon: iter(loader) for horizon, loader in train_loaders.items()}
        losses = []
        horizon_counts = {horizon: 0 for horizon in target_horizons}
        for _ in range(steps_per_epoch):
            horizon = rng.choice(target_horizons)
            x, y = next_batch(train_loaders, iterators, horizon)
            x = x.float().to(device)
            y = y.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, pred_len=horizon)
            if isinstance(pred, dict):
                raise TypeError("Expected tensor prediction.")
            loss = weighted_mse_loss(
                pred,
                y,
                args.max_pred_len,
                args.step_loss_weighting,
                args.step_loss_alpha,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            horizon_counts[horizon] += 1

        val_rows = []
        for horizon in target_horizons:
            metrics, _, _, _ = evaluate(model, val_loaders[horizon], device, horizon, args.max_eval_batches)
            val_rows.append({"target_horizon": horizon, **metrics})
        mean_val_mse = float(np.mean([row["mse"] for row in val_rows]))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_mean_mse": mean_val_mse}
        for horizon in target_horizons:
            row[f"train_steps_h{horizon}"] = horizon_counts[horizon]
        for val_row in val_rows:
            horizon = int(val_row["target_horizon"])
            row[f"val_mse_h{horizon}"] = float(val_row["mse"])
            row[f"val_mae_h{horizon}"] = float(val_row["mae"])
        log_rows.append(row)

        if mean_val_mse < best_val:
            best_val = mean_val_mse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    target_metric_rows = []
    for horizon in target_horizons:
        metrics, pred_np, true_np, components = evaluate(
            model,
            test_loaders[horizon],
            device,
            horizon,
            args.max_eval_batches,
        )
        target_metric_rows.append({"target_horizon": horizon, **metrics})
        eval_dir = run_dir / f"h{horizon}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        if args.save_predictions:
            np.savez_compressed(eval_dir / "predictions_test.npz", pred=pred_np, true=true_np)
        (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        write_csv(eval_dir / "metrics_by_horizon.csv", metrics_by_horizon(pred_np, true_np))
        write_csv(eval_dir / "metrics_by_segment.csv", metrics_by_segment(pred_np, true_np))
        write_csv(eval_dir / "target_state_similarity.csv", target_state_similarity(components))
        write_csv(eval_dir / "target_conditioning_stats.csv", target_conditioning_stats(components))
        if args.prefix_residual_segments > 0:
            write_csv(eval_dir / "prefix_residual_stats.csv", prefix_residual_stats(components))

    long_horizon = max(target_horizons)
    short_horizons = [horizon for horizon in target_horizons if horizon < long_horizon]
    if short_horizons:
        prefix_rows = prefix_consistency(
            model,
            test_loaders[long_horizon],
            device,
            short_horizons,
            long_horizon,
            args.max_eval_batches,
        )
        write_csv(run_dir / "prefix_consistency.csv", prefix_rows)

    torch.save(model.state_dict(), run_dir / "checkpoint.pt")
    write_csv(run_dir / "metrics_by_target_horizon.csv", target_metric_rows)
    write_csv(run_dir / "training_log.csv", log_rows)
    effective_config = vars(args)
    effective_config["target_horizons"] = target_horizons
    effective_config["steps_per_epoch_effective"] = steps_per_epoch
    (run_dir / "effective_config.json").write_text(json.dumps(effective_config, indent=2))
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

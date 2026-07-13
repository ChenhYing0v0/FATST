from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_BASELINE_DIR = REPO_ROOT / "baselines" / "patch_encoder_fixed_head"
sys.path.insert(0, str(PATCH_BASELINE_DIR))

from dataset import DATASETS, ForecastDataset  # noqa: E402
from model import PatchEncoderFixedHead  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze best checkpoint per forecast segment.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--model", default="PatchEncoderFixedHead")
    parser.add_argument("--datasets", nargs="+", default=["ETTh2", "ETTm1", "Weather"])
    parser.add_argument("--checkpoint-horizons", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--target-horizon", type=int, default=720)
    parser.add_argument("--segment-len", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model(
    checkpoint_root: Path,
    model_name: str,
    dataset: str,
    horizon: int,
    seed: int,
    seq_len: int,
    device: torch.device,
) -> PatchEncoderFixedHead:
    checkpoint = checkpoint_root / model_name / dataset / f"h{horizon}" / f"seed{seed}" / "checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = PatchEncoderFixedHead(seq_len, horizon, DATASETS[dataset].channels)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def rolling_forecast(model: PatchEncoderFixedHead, x: torch.Tensor, target_horizon: int) -> torch.Tensor:
    context = x
    outputs: list[torch.Tensor] = []
    while sum(part.shape[1] for part in outputs) < target_horizon:
        pred = model(context)
        outputs.append(pred)
        context = torch.cat([context, pred], dim=1)[:, -model.seq_len :, :]
    return torch.cat(outputs, dim=1)[:, :target_horizon, :]


def make_segments(target_horizon: int, segment_len: int) -> list[tuple[int, int]]:
    return [(start, min(start + segment_len, target_horizon)) for start in range(0, target_horizon, segment_len)]


def evaluate_checkpoint(
    model: PatchEncoderFixedHead,
    loader: DataLoader,
    segments: list[tuple[int, int]],
    target_horizon: int,
    device: torch.device,
) -> list[dict[str, float]]:
    sqerr = np.zeros(len(segments), dtype=np.float64)
    abserr = np.zeros(len(segments), dtype=np.float64)
    counts = np.zeros(len(segments), dtype=np.int64)

    for x, y in loader:
        x = x.float().to(device)
        y = y.float().to(device)
        pred = rolling_forecast(model, x, target_horizon)
        for i, (start, end) in enumerate(segments):
            diff = pred[:, start:end, :] - y[:, start:end, :]
            sqerr[i] += float(torch.sum(diff * diff).cpu())
            abserr[i] += float(torch.sum(torch.abs(diff)).cpu())
            counts[i] += diff.numel()

    return [{"mse": float(sqerr[i] / counts[i]), "mae": float(abserr[i] / counts[i])} for i in range(len(segments))]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_dataset_heatmap(
    report_dir: Path,
    dataset: str,
    segments: list[tuple[int, int]],
    horizons: list[int],
    metric_matrix: np.ndarray,
    winners: list[int],
) -> str:
    fig_width = max(10.0, len(segments) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    image = ax.imshow(metric_matrix, aspect="auto", cmap="viridis")
    ax.set_title(f"{dataset}: segment MSE by checkpoint horizon")
    ax.set_xlabel("Forecast segment")
    ax.set_ylabel("Checkpoint pred_len")
    ax.set_xticks(range(len(segments)))
    ax.set_xticklabels([f"{start}-{end}" for start, end in segments], rotation=45, ha="right")
    ax.set_yticks(range(len(horizons)))
    ax.set_yticklabels([str(h) for h in horizons])
    for j, winner in enumerate(winners):
        i = horizons.index(winner)
        ax.scatter(j, i, marker="*", s=130, color="white", edgecolor="black", linewidth=0.8)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("MSE")
    fig.tight_layout()
    path = report_dir / f"phase0_segment_oracle_{dataset}_mse_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_dataset_gap(
    report_dir: Path,
    dataset: str,
    segments: list[tuple[int, int]],
    winner_rows: list[dict[str, object]],
) -> str:
    fig_width = max(10.0, len(segments) * 0.7)
    fig, ax1 = plt.subplots(figsize=(fig_width, 4.2))
    x = np.arange(len(segments))
    margins = np.array([float(row["relative_margin_to_second"]) * 100.0 for row in winner_rows])
    winners = [str(row["best_pred_len"]) for row in winner_rows]
    bars = ax1.bar(x, margins, color="#4c78a8")
    ax1.set_title(f"{dataset}: winner margin over second-best checkpoint")
    ax1.set_xlabel("Forecast segment")
    ax1.set_ylabel("Relative MSE margin (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{start}-{end}" for start, end in segments], rotation=45, ha="right")
    for bar, label in zip(bars, winners):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    path = report_dir / f"phase0_segment_oracle_{dataset}_winner_margin.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def summarize_winners(
    rows: list[dict[str, object]],
    datasets: list[str],
    segments: list[tuple[int, int]],
    horizons: list[int],
) -> list[dict[str, object]]:
    winner_rows: list[dict[str, object]] = []
    for dataset in datasets:
        for start, end in segments:
            segment_rows = [
                row for row in rows if row["dataset"] == dataset and row["segment_start"] == start
            ]
            segment_rows = sorted(segment_rows, key=lambda row: float(row["mse"]))
            best = segment_rows[0]
            second = segment_rows[1]
            best_mse = float(best["mse"])
            second_mse = float(second["mse"])
            winner_rows.append(
                {
                    "dataset": dataset,
                    "segment_start": start,
                    "segment_end": end,
                    "best_pred_len": best["checkpoint_pred_len"],
                    "best_mse": best_mse,
                    "best_mae": best["mae"],
                    "second_pred_len": second["checkpoint_pred_len"],
                    "second_mse": second_mse,
                    "mse_margin_to_second": second_mse - best_mse,
                    "relative_margin_to_second": (second_mse - best_mse) / best_mse,
                    "candidate_pred_lens": ";".join(str(horizon) for horizon in horizons),
                }
            )
    return winner_rows


def write_markdown_report(
    report_dir: Path,
    datasets: list[str],
    winner_rows: list[dict[str, object]],
    heatmaps: dict[str, str],
    gap_plots: dict[str, str],
    target_horizon: int,
    segment_len: int,
) -> Path:
    path = report_dir / "phase0_segment_oracle_report.md"
    lines = [
        "# Phase0 Segment-wise Checkpoint Oracle",
        "",
        "## Setup",
        "",
        f"- Target horizon: `{target_horizon}`",
        f"- Segment length: `{segment_len}`",
        "- Short checkpoints are extended to the target horizon with rolling autoregression.",
        "- Winner is selected by segment MSE.",
        "",
        "## Winner Table",
        "",
        "| Dataset | Segment | Best pred_len | Best MSE | Second pred_len | Second MSE | Relative margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in winner_rows:
        lines.append(
            "| {dataset} | {segment_start}-{segment_end} | {best_pred_len} | {best_mse:.6f} | "
            "{second_pred_len} | {second_mse:.6f} | {margin:.2%} |".format(
                dataset=row["dataset"],
                segment_start=row["segment_start"],
                segment_end=row["segment_end"],
                best_pred_len=row["best_pred_len"],
                best_mse=float(row["best_mse"]),
                second_pred_len=row["second_pred_len"],
                second_mse=float(row["second_mse"]),
                margin=float(row["relative_margin_to_second"]),
            )
        )
    lines.extend(["", "## Figures", ""])
    for dataset in datasets:
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"![{dataset} MSE heatmap]({Path(heatmaps[dataset]).name})",
                "",
                f"![{dataset} winner margin]({Path(gap_plots[dataset]).name})",
                "",
            ]
        )
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_root = Path(args.checkpoint_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    segments = make_segments(args.target_horizon, args.segment_len)

    metric_rows: list[dict[str, object]] = []
    heatmaps: dict[str, str] = {}
    gap_plots: dict[str, str] = {}

    for dataset in args.datasets:
        dataset_obj = ForecastDataset(
            args.dataset_root,
            dataset,
            "test",
            args.seq_len,
            args.target_horizon,
        )
        loader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=False)
        matrix_rows = []
        for checkpoint_horizon in args.checkpoint_horizons:
            model = load_model(
                checkpoint_root,
                args.model,
                dataset,
                checkpoint_horizon,
                args.seed,
                args.seq_len,
                device,
            )
            segment_metrics = evaluate_checkpoint(model, loader, segments, args.target_horizon, device)
            matrix_rows.append([metrics["mse"] for metrics in segment_metrics])
            for (start, end), metrics in zip(segments, segment_metrics):
                roll_steps = math.ceil(args.target_horizon / checkpoint_horizon)
                metric_rows.append(
                    {
                        "model": args.model,
                        "dataset": dataset,
                        "segment_start": start,
                        "segment_end": end,
                        "checkpoint_pred_len": checkpoint_horizon,
                        "target_horizon": args.target_horizon,
                        "roll_steps": roll_steps,
                        "mse": metrics["mse"],
                        "mae": metrics["mae"],
                    }
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        dataset_winners = summarize_winners(
            metric_rows,
            [dataset],
            segments,
            args.checkpoint_horizons,
        )
        winners = [int(row["best_pred_len"]) for row in dataset_winners]
        matrix = np.array(matrix_rows, dtype=np.float64)
        heatmaps[dataset] = plot_dataset_heatmap(
            report_dir,
            dataset,
            segments,
            args.checkpoint_horizons,
            matrix,
            winners,
        )
        gap_plots[dataset] = plot_dataset_gap(report_dir, dataset, segments, dataset_winners)

    winner_rows = summarize_winners(metric_rows, args.datasets, segments, args.checkpoint_horizons)
    metric_path = report_dir / "phase0_segment_oracle_metrics.csv"
    winner_path = report_dir / "phase0_segment_oracle_winners.csv"
    summary_path = report_dir / "phase0_segment_oracle_summary.json"
    write_csv(metric_path, metric_rows)
    write_csv(winner_path, winner_rows)
    report_path = write_markdown_report(
        report_dir,
        args.datasets,
        winner_rows,
        heatmaps,
        gap_plots,
        args.target_horizon,
        args.segment_len,
    )
    counts: dict[str, dict[str, int]] = {}
    for row in winner_rows:
        dataset = str(row["dataset"])
        pred_len = str(row["best_pred_len"])
        counts.setdefault(dataset, {})
        counts[dataset][pred_len] = counts[dataset].get(pred_len, 0) + 1
    summary = {
        "metric_csv": str(metric_path),
        "winner_csv": str(winner_path),
        "report_md": str(report_path),
        "winner_counts": counts,
        "heatmaps": heatmaps,
        "gap_plots": gap_plots,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from data_provider.data_factory import data_provider  # noqa: E402
from models import TimeAlign  # noqa: E402
from train_repo import OFFICIAL_PRESETS, build_official_args, parse_horizons, set_seed  # noqa: E402
from utils.metrics import MAE, MSE  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_model(args: argparse.Namespace, official_args: argparse.Namespace, checkpoint_path: Path) -> torch.nn.Module:
    model = TimeAlign.Model(official_args).float().to(official_args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state_dict-like mapping")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def mse_tensor(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float(torch.mean((pred - true) ** 2).detach().cpu())


def mae_tensor(pred: torch.Tensor, true: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - true)).detach().cpu())


def stack_metric(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def model_outputs(
    model: torch.nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    official_args: argparse.Namespace,
    target_prefix: int,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {"is_training": False}
    if getattr(official_args, "readout_mode", "official") != "official":
        kwargs["target_prefix"] = target_prefix
    outputs, _recon, _alignment = model(
        batch_x,
        batch_y[:, -official_args.pred_len :, :],
        **kwargs,
    )
    f_dim = -1 if official_args.features == "MS" else 0
    return outputs[:, -official_args.pred_len :, f_dim:]


def evaluate_validation_diagnostics(
    model: torch.nn.Module,
    teacher: torch.nn.Module | None,
    loader: torch.utils.data.DataLoader,
    official_args: argparse.Namespace,
    horizons: list[int],
    max_batches: int,
) -> list[dict[str, Any]]:
    f_dim = -1 if official_args.features == "MS" else 0
    rows: list[dict[str, Any]] = []
    full_prefix = official_args.pred_len
    can_compare_full_context = getattr(official_args, "readout_mode", "official") != "official"

    for horizon in horizons:
        prefix_mse_values: list[float] = []
        prefix_mae_values: list[float] = []
        full_context_mse_values: list[float] = []
        prefix_vs_full_mse_values: list[float] = []
        teacher_mse_values: list[float] = []
        teacher_mae_values: list[float] = []
        residual_abs_mean_values: list[float] = []
        residual_std_values: list[float] = []
        sample_count = 0

        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader):
                if max_batches and batch_idx >= max_batches:
                    break
                batch_x = batch_x.float().to(official_args.device)
                batch_y = batch_y.float().to(official_args.device)
                target = batch_y[:, -official_args.pred_len :, f_dim:]

                outputs = model_outputs(model, batch_x, batch_y, official_args, horizon)
                pred_h = outputs[:, :horizon, :]
                true_h = target[:, :horizon, :]
                residual = pred_h - true_h

                prefix_mse_values.append(mse_tensor(pred_h, true_h))
                prefix_mae_values.append(mae_tensor(pred_h, true_h))
                residual_abs_mean_values.append(float(torch.mean(torch.abs(residual)).detach().cpu()))
                residual_std_values.append(float(torch.std(residual).detach().cpu()))
                sample_count += int(batch_x.shape[0])

                if can_compare_full_context and horizon != full_prefix:
                    outputs_full = model_outputs(model, batch_x, batch_y, official_args, full_prefix)
                    full_pred_h = outputs_full[:, :horizon, :]
                    full_context_mse_values.append(mse_tensor(full_pred_h, true_h))
                    prefix_vs_full_mse_values.append(mse_tensor(pred_h, full_pred_h))

                if teacher is not None:
                    teacher_outputs = model_outputs(teacher, batch_x, batch_y, official_args, horizon)
                    teacher_h = teacher_outputs[:, :horizon, :]
                    teacher_mse_values.append(mse_tensor(pred_h, teacher_h))
                    teacher_mae_values.append(mae_tensor(pred_h, teacher_h))

        rows.append(
            {
                "target_horizon": horizon,
                "validation_prefix_mse": stack_metric(prefix_mse_values),
                "validation_prefix_mae": stack_metric(prefix_mae_values),
                "full_context_prefix_mse": stack_metric(full_context_mse_values),
                "prefix_vs_full_mse": stack_metric(prefix_vs_full_mse_values),
                "teacher_student_mse": stack_metric(teacher_mse_values),
                "teacher_student_mae": stack_metric(teacher_mae_values),
                "residual_abs_mean": stack_metric(residual_abs_mean_values),
                "residual_std": stack_metric(residual_std_values),
                "num_samples": sample_count,
                "max_eval_batches": max_batches,
            }
        )
    return rows


def build_args_for_diagnostic(args: argparse.Namespace) -> argparse.Namespace:
    preset = OFFICIAL_PRESETS[args.dataset][args.pred_len if args.mode == "fixed" else 720]
    adapter_args = argparse.Namespace(
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        mode=args.mode,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        target_horizons=args.target_horizons,
        e_layers=args.e_layers,
        w_recon=args.w_recon,
        batch_size=args.batch_size,
        epochs=1,
        patience=1,
        seed=args.seed,
        num_workers=args.num_workers,
        max_train_batches=0,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
        output_dir=args.output_dir,
        device=args.device,
        use_amp=False,
        official_test_mode=False,
        checkpoint_policy="official-last",
        readout_mode=args.readout_mode,
        warm_start_checkpoint=None,
        teacher_checkpoint=None,
        teacher_readout_mode="target-set-decoder",
        teacher_loss_weight=0.0,
        pred_loss_mode="multi-prefix",
        prefix_samples=1,
        continuous_min_prefix=32,
        continuous_prefix_step=32,
    )
    official_args = build_official_args(adapter_args, preset)
    return official_args


def run(args: argparse.Namespace) -> None:
    if args.dataset not in OFFICIAL_PRESETS:
        raise ValueError(f"Unsupported dataset {args.dataset}. Choose from {sorted(OFFICIAL_PRESETS)}")
    if args.mode == "fixed" and args.target_horizons != [args.pred_len]:
        raise ValueError("fixed mode expects target_horizons == [pred_len]")
    if args.mode == "unified" and args.pred_len != 720:
        raise ValueError("unified mode expects pred_len=720")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    official_args = build_args_for_diagnostic(args)
    model = load_model(args, official_args, args.checkpoint)
    teacher = None
    if args.teacher_checkpoint is not None:
        teacher_args = argparse.Namespace(**vars(official_args))
        teacher_args.readout_mode = args.teacher_readout_mode
        teacher = load_model(args, teacher_args, args.teacher_checkpoint)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    _val_data, val_loader = data_provider(official_args, "val")
    rows = evaluate_validation_diagnostics(
        model,
        teacher,
        val_loader,
        official_args,
        args.target_horizons,
        max_batches=args.max_eval_batches,
    )
    for row in rows:
        row["dataset"] = args.dataset
        row["mode"] = args.mode
        row["run_name"] = args.run_name
        row["readout_mode"] = args.readout_mode
        row["pred_len"] = args.pred_len
        row["checkpoint"] = str(args.checkpoint)
        row["teacher_checkpoint"] = str(args.teacher_checkpoint) if args.teacher_checkpoint is not None else ""
        row["teacher_readout_mode"] = args.teacher_readout_mode if args.teacher_checkpoint is not None else ""

    write_csv(args.output_dir / "validation_prefix_diagnostics.csv", rows)
    dump_json(
        args.output_dir / "validation_prefix_diagnostics_config.json",
        {
            "adapter": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "official_args": {
                key: (str(value) if isinstance(value, torch.device) else value)
                for key, value in vars(official_args).items()
                if key != "device_ids"
            },
            "official_preset": asdict(OFFICIAL_PRESETS[args.dataset][args.pred_len if args.mode == "fixed" else 720]),
        },
    )
    print(f"diagnostic_done output_dir={args.output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export validation prefix diagnostics for TimeAlign checkpoints.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(OFFICIAL_PRESETS), required=True)
    parser.add_argument("--mode", choices=["fixed", "unified"], required=True)
    parser.add_argument("--seq-len", type=int, default=720)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--target-horizons", type=parse_horizons, required=True)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--w-recon", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--readout-mode", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-readout-mode", choices=["target-set-decoder"], default="target-set-decoder")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

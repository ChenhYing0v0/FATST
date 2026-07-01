from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import optim

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data_provider.data_factory import data_provider  # noqa: E402
from models import TimeAlign  # noqa: E402
from utils.metrics import MAE, MSE  # noqa: E402
from utils.tools import adjust_learning_rate  # noqa: E402


HORIZONS = [96, 192, 336, 720]


@dataclass(frozen=True)
class OfficialPreset:
    data: str
    data_path: str
    relative_root: str
    freq: str
    enc_in: int
    dec_in: int
    c_out: int
    d_model: int
    d_ff: int
    learning_rate: float
    dropout: float
    w_align: float
    patch_num: int
    local_margin: float
    global_margin: float
    layer_norm: int


OFFICIAL_PRESETS: dict[str, dict[int, OfficialPreset]] = {
    "ETTh2": {
        horizon: OfficialPreset(
            data="ETTh2",
            data_path="ETTh2.csv",
            relative_root="ETT-small",
            freq="h",
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=32,
            d_ff=32,
            learning_rate=0.0005,
            dropout=0.1,
            w_align=0.1,
            patch_num=48,
            local_margin=0.0,
            global_margin=0.0,
            layer_norm=1,
        )
        for horizon in HORIZONS
    },
    "ETTm2": {
        96: OfficialPreset(
            data="ETTm2",
            data_path="ETTm2.csv",
            relative_root="ETT-small",
            freq="t",
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=128,
            learning_rate=0.0001,
            dropout=0.3,
            w_align=1.0,
            patch_num=12,
            local_margin=0.0,
            global_margin=0.0,
            layer_norm=1,
        ),
        192: OfficialPreset(
            data="ETTm2",
            data_path="ETTm2.csv",
            relative_root="ETT-small",
            freq="t",
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=128,
            learning_rate=0.0001,
            dropout=0.3,
            w_align=1.0,
            patch_num=12,
            local_margin=0.0,
            global_margin=0.0,
            layer_norm=1,
        ),
        336: OfficialPreset(
            data="ETTm2",
            data_path="ETTm2.csv",
            relative_root="ETT-small",
            freq="t",
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=128,
            learning_rate=0.0001,
            dropout=0.9,
            w_align=1.0,
            patch_num=12,
            local_margin=0.0,
            global_margin=0.0,
            layer_norm=1,
        ),
        720: OfficialPreset(
            data="ETTm2",
            data_path="ETTm2.csv",
            relative_root="ETT-small",
            freq="t",
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=128,
            learning_rate=0.0001,
            dropout=0.9,
            w_align=1.0,
            patch_num=12,
            local_margin=0.0,
            global_margin=0.0,
            layer_norm=1,
        ),
    },
    "ETTm1": {
        96: OfficialPreset(
            data="ETTm1",
            data_path="ETTm1.csv",
            relative_root="ETT-small",
            freq="t",
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=256,
            learning_rate=0.0001,
            dropout=0.2,
            w_align=0.1,
            patch_num=1,
            local_margin=0.5,
            global_margin=0.0,
            layer_norm=1,
        ),
        192: OfficialPreset(
            data="ETTm1",
            data_path="ETTm1.csv",
            relative_root="ETT-small",
            freq="t",
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=256,
            learning_rate=0.0001,
            dropout=0.2,
            w_align=0.1,
            patch_num=1,
            local_margin=0.5,
            global_margin=0.0,
            layer_norm=1,
        ),
        336: OfficialPreset(
            data="ETTm1",
            data_path="ETTm1.csv",
            relative_root="ETT-small",
            freq="t",
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=128,
            d_ff=256,
            learning_rate=0.0001,
            dropout=0.8,
            w_align=0.1,
            patch_num=1,
            local_margin=0.5,
            global_margin=0.0,
            layer_norm=1,
        ),
        720: OfficialPreset(
            data="ETTm1",
            data_path="ETTm1.csv",
            relative_root="ETT-small",
            freq="t",
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=256,
            d_ff=256,
            learning_rate=0.0001,
            dropout=0.9,
            w_align=0.1,
            patch_num=1,
            local_margin=0.5,
            global_margin=0.0,
            layer_norm=1,
        ),
    },
    "Weather": {
        96: OfficialPreset(
            data="custom",
            data_path="weather.csv",
            relative_root="weather",
            freq="h",
            enc_in=21,
            dec_in=21,
            c_out=21,
            d_model=128,
            d_ff=256,
            learning_rate=0.0001,
            dropout=0.1,
            w_align=0.1,
            patch_num=48,
            local_margin=0.5,
            global_margin=0.0,
            layer_norm=0,
        ),
        192: OfficialPreset(
            data="custom",
            data_path="weather.csv",
            relative_root="weather",
            freq="h",
            enc_in=21,
            dec_in=21,
            c_out=21,
            d_model=128,
            d_ff=256,
            learning_rate=0.0001,
            dropout=0.1,
            w_align=0.1,
            patch_num=48,
            local_margin=0.5,
            global_margin=0.0,
            layer_norm=0,
        ),
        336: OfficialPreset(
            data="custom",
            data_path="weather.csv",
            relative_root="weather",
            freq="h",
            enc_in=21,
            dec_in=21,
            c_out=21,
            d_model=128,
            d_ff=256,
            learning_rate=0.0001,
            dropout=0.1,
            w_align=0.1,
            patch_num=48,
            local_margin=0.5,
            global_margin=0.0,
            layer_norm=0,
        ),
        720: OfficialPreset(
            data="custom",
            data_path="weather.csv",
            relative_root="weather",
            freq="h",
            enc_in=21,
            dec_in=21,
            c_out=21,
            d_model=128,
            d_ff=128,
            learning_rate=0.0001,
            dropout=0.5,
            w_align=0.1,
            patch_num=48,
            local_margin=0.5,
            global_margin=0.0,
            layer_norm=0,
        ),
    },
}


def parse_horizons(value: str) -> list[int]:
    horizons = [int(item) for item in value.replace(" ", "").split(",") if item]
    if not horizons:
        raise argparse.ArgumentTypeError("at least one horizon is required")
    return horizons


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_dataset_root(dataset_root: Path, preset: OfficialPreset) -> Path:
    direct = dataset_root / preset.data_path
    nested = dataset_root / preset.relative_root / preset.data_path
    if direct.exists():
        return dataset_root
    if nested.exists():
        return dataset_root / preset.relative_root
    raise FileNotFoundError(
        f"Cannot find {preset.data_path} under {dataset_root} or {dataset_root / preset.relative_root}"
    )


def build_official_args(args: argparse.Namespace, preset: OfficialPreset) -> argparse.Namespace:
    root_path = resolve_dataset_root(args.dataset_root, preset)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    official = argparse.Namespace(
        task_name="long_term_forecast",
        is_training=1,
        model_id=f"{args.dataset}_{args.seq_len}_{args.pred_len}",
        model="TimeAlign",
        data=preset.data,
        root_path=str(root_path),
        data_path=preset.data_path,
        features="M",
        target="OT",
        freq=preset.freq,
        checkpoints=str(args.output_dir / "_official_checkpoints"),
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        seasonal_patterns="Monthly",
        inverse=False,
        enc_in=preset.enc_in,
        dec_in=preset.dec_in,
        c_out=preset.c_out,
        d_model=preset.d_model,
        n_heads=8,
        e_layers=args.e_layers,
        d_layers=1,
        d_ff=preset.d_ff,
        factor=3,
        dropout=preset.dropout,
        embed="timeF",
        distil=True,
        expand=2,
        d_conv=4,
        num_workers=args.num_workers,
        itr=1,
        train_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=preset.learning_rate,
        des="Exp",
        loss="MSE",
        lradj="cosine",
        use_amp=args.use_amp,
        use_gpu=device.type == "cuda",
        gpu=0,
        gpu_type=device.type,
        use_multi_gpu=False,
        device_ids=[],
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        use_dtw=False,
        augmentation_ratio=0,
        seed=args.seed,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag="",
        w_align=preset.w_align,
        w_recon=args.w_recon,
        local_margin=preset.local_margin,
        global_margin=preset.global_margin,
        patch_num=preset.patch_num,
        layer_norm=preset.layer_norm,
        pos=1,
        loc=1,
        glo=1,
        device=device,
        readout_mode=args.readout_mode,
        target_horizons=args.target_horizons,
    )
    return official


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


def warm_start_nested_from_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state_dict-like mapping")

    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    incompatible_keys = sorted(key for key in state if key not in compatible)
    load_result = model.load_state_dict(compatible, strict=False)

    if not hasattr(model, "nested_segment_heads") or not hasattr(model, "nested_boundaries"):
        raise ValueError("Warm-started nested checkpoint requires nested_segment_heads")
    if "proj_x.weight" not in state or "proj_x.bias" not in state:
        raise KeyError("Checkpoint must contain proj_x.weight and proj_x.bias")

    previous = 0
    copied_segments = []
    with torch.no_grad():
        for boundary, head in zip(model.nested_boundaries, model.nested_segment_heads):
            head.weight.copy_(state["proj_x.weight"][previous:boundary].to(device=head.weight.device, dtype=head.weight.dtype))
            head.bias.copy_(state["proj_x.bias"][previous:boundary].to(device=head.bias.device, dtype=head.bias.dtype))
            copied_segments.append({"start": previous, "end": boundary, "width": boundary - previous})
            previous = boundary

    model.to(device)
    return {
        "checkpoint_path": str(checkpoint_path),
        "compatible_keys": len(compatible),
        "incompatible_keys": len(incompatible_keys),
        "missing_keys": sorted(load_result.missing_keys),
        "unexpected_keys": sorted(load_result.unexpected_keys),
        "copied_segments": copied_segments,
    }


def load_teacher_model(
    official_args: argparse.Namespace,
    checkpoint_path: Path,
    readout_mode: str,
) -> nn.Module:
    teacher_args = argparse.Namespace(**vars(official_args))
    teacher_args.readout_mode = readout_mode
    teacher = TimeAlign.Model(teacher_args).float().to(official_args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("Teacher checkpoint does not contain a state_dict-like mapping")
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def metric_rows(preds: np.ndarray, trues: np.ndarray, horizons: list[int]) -> list[dict[str, Any]]:
    rows = []
    for horizon in horizons:
        pred_h = preds[:, :horizon, :]
        true_h = trues[:, :horizon, :]
        rows.append(
            {
                "target_horizon": horizon,
                "mse": float(MSE(pred_h, true_h)),
                "mae": float(MAE(pred_h, true_h)),
                "num_samples": int(preds.shape[0]),
                "num_channels": int(preds.shape[-1]),
                "eval_prefix_steps": horizon,
            }
        )
    return rows


def segment_rows(preds: np.ndarray, trues: np.ndarray, target_horizon: int, segment_len: int = 96) -> list[dict[str, Any]]:
    rows = []
    for start in range(0, target_horizon, segment_len):
        end = min(start + segment_len, target_horizon)
        pred_s = preds[:, start:end, :]
        true_s = trues[:, start:end, :]
        rows.append(
            {
                "target_horizon": target_horizon,
                "segment_start": start,
                "segment_end": end,
                "mse": float(MSE(pred_s, true_s)),
                "mae": float(MAE(pred_s, true_s)),
            }
        )
    return rows


def prediction_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    horizons: list[int],
    mode: str,
    prefix_samples: int,
    continuous_min_prefix: int,
    continuous_prefix_step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    full_loss = criterion(outputs, targets)
    losses: dict[str, torch.Tensor] = {"full": full_loss}
    if mode == "full":
        return full_loss, losses
    if mode == "balanced-step":
        segment_losses = []
        start = 0
        for horizon in sorted(set(horizons)):
            if horizon <= start:
                continue
            segment_loss = criterion(outputs[:, start:horizon, :], targets[:, start:horizon, :])
            losses[f"seg{start + 1}_{horizon}"] = segment_loss
            segment_losses.append(segment_loss)
            start = horizon
        if not segment_losses:
            raise ValueError("balanced-step produced no supervision segments")
        return torch.stack(segment_losses).mean(), losses
    else:
        selected_horizons = select_prediction_horizons(
            horizons,
            mode,
            prefix_samples,
            continuous_min_prefix,
            continuous_prefix_step,
            pred_len=outputs.shape[1],
        )
    prefix_losses = []
    for horizon in selected_horizons:
        horizon_loss = criterion(outputs[:, :horizon, :], targets[:, :horizon, :])
        key = f"h{horizon}"
        if key in losses:
            key = f"{key}_repeat{len([name for name in losses if name.startswith(key)])}"
        losses[key] = horizon_loss
        prefix_losses.append(horizon_loss)
    return torch.stack(prefix_losses).mean(), losses


def select_prediction_horizons(
    horizons: list[int],
    mode: str,
    prefix_samples: int,
    continuous_min_prefix: int,
    continuous_prefix_step: int,
    pred_len: int,
) -> list[int]:
    sorted_horizons = sorted(set(horizons))
    if mode == "full":
        return [pred_len]
    if mode == "multi-prefix":
        return sorted_horizons
    if mode == "stochastic-prefix":
        sample_count = max(prefix_samples, 1)
        if sample_count <= len(sorted_horizons):
            return random.sample(sorted_horizons, k=sample_count)
        return random.choices(sorted_horizons, k=sample_count)
    if mode == "continuous-prefix":
        pool = list(range(max(1, continuous_min_prefix), pred_len + 1, max(continuous_prefix_step, 1)))
        if pred_len not in pool:
            pool.append(pred_len)
        sample_count = max(prefix_samples, 1)
        if sample_count <= len(pool):
            return random.sample(pool, k=sample_count)
        return random.choices(pool, k=sample_count)
    raise ValueError(f"Unsupported prediction loss mode for prefix-conditioned readout: {mode}")


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    official_args: argparse.Namespace,
    horizons: list[int],
    max_batches: int,
    is_training_flag: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    if getattr(official_args, "readout_mode", "official") != "official":
        return evaluate_prefix_conditioned(model, loader, official_args, horizons, max_batches, is_training_flag)
    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            batch_x = batch_x.float().to(official_args.device)
            batch_y = batch_y.float().to(official_args.device)
            outputs, _recon, _alignment = model(
                batch_x,
                batch_y[:, -official_args.pred_len :, :],
                is_training=is_training_flag,
            )
            f_dim = -1 if official_args.features == "MS" else 0
            outputs = outputs[:, -official_args.pred_len :, f_dim:]
            batch_y = batch_y[:, -official_args.pred_len :, f_dim:]
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
    if not preds:
        raise RuntimeError("evaluation produced no batches")
    pred_np = np.concatenate(preds, axis=0)
    true_np = np.concatenate(trues, axis=0)
    main_rows = metric_rows(pred_np, true_np, horizons)
    all_segments: list[dict[str, Any]] = []
    for horizon in horizons:
        all_segments.extend(segment_rows(pred_np, true_np, horizon))
    return main_rows, all_segments, pred_np, true_np


def evaluate_prefix_conditioned(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    official_args: argparse.Namespace,
    horizons: list[int],
    max_batches: int,
    is_training_flag: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    main_rows: list[dict[str, Any]] = []
    all_segments: list[dict[str, Any]] = []
    pred_for_npz: np.ndarray | None = None
    true_for_npz: np.ndarray | None = None
    f_dim = -1 if official_args.features == "MS" else 0
    model.eval()
    with torch.no_grad():
        for horizon in horizons:
            preds = []
            trues = []
            for batch_idx, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader):
                if max_batches and batch_idx >= max_batches:
                    break
                batch_x = batch_x.float().to(official_args.device)
                batch_y = batch_y.float().to(official_args.device)
                outputs, _recon, _alignment = model(
                    batch_x,
                    batch_y[:, -official_args.pred_len :, :],
                    is_training=is_training_flag,
                    target_prefix=horizon,
                )
                outputs = outputs[:, -official_args.pred_len :, f_dim:]
                target = batch_y[:, -official_args.pred_len :, f_dim:]
                preds.append(outputs.detach().cpu().numpy())
                trues.append(target.detach().cpu().numpy())
            if not preds:
                raise RuntimeError("evaluation produced no batches")
            pred_np = np.concatenate(preds, axis=0)
            true_np = np.concatenate(trues, axis=0)
            main_rows.extend(metric_rows(pred_np, true_np, [horizon]))
            all_segments.extend(segment_rows(pred_np, true_np, horizon))
            if horizon == max(horizons):
                pred_for_npz = pred_np
                true_for_npz = true_np
    if pred_for_npz is None or true_for_npz is None:
        raise RuntimeError("prefix-conditioned evaluation did not produce final-horizon predictions")
    return main_rows, all_segments, pred_for_npz, true_for_npz


def validation_mean_mse(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    official_args: argparse.Namespace,
    horizons: list[int],
    max_batches: int,
) -> float:
    rows, _segments, _preds, _trues = evaluate(
        model,
        loader,
        official_args,
        horizons,
        max_batches=max_batches,
        is_training_flag=False,
    )
    return float(np.mean([row["mse"] for row in rows]))


def train(args: argparse.Namespace, official_args: argparse.Namespace) -> tuple[nn.Module, list[dict[str, Any]]]:
    train_data, train_loader = data_provider(official_args, "train")
    vali_data, vali_loader = data_provider(official_args, "val")
    test_data, test_loader = data_provider(official_args, "test")
    del train_data, vali_data, test_data, test_loader

    model = TimeAlign.Model(official_args).float().to(official_args.device)
    warm_start_info = None
    if args.warm_start_checkpoint is not None:
        warm_start_info = warm_start_nested_from_checkpoint(model, args.warm_start_checkpoint, official_args.device)
        print(f"warm_start_info={json.dumps(warm_start_info, sort_keys=True)}", flush=True)
    teacher_model = None
    if args.teacher_checkpoint is not None:
        teacher_model = load_teacher_model(official_args, args.teacher_checkpoint, args.teacher_readout_mode)
        print(
            f"teacher_info={json.dumps({'checkpoint_path': str(args.teacher_checkpoint), 'readout_mode': args.teacher_readout_mode, 'loss_weight': args.teacher_loss_weight}, sort_keys=True)}",
            flush=True,
        )
    optimizer = optim.AdamW(model.parameters(), lr=official_args.learning_rate)
    criterion = nn.L1Loss()
    training_rows: list[dict[str, Any]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        total_loss = []
        pred_loss_values = []
        pred_full_loss_values = []
        pred_component_values: dict[str, list[float]] = {}
        teacher_loss_values = []
        recon_loss_values = []
        alignment_values = []
        train_steps = len(train_loader)
        for batch_idx, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            optimizer.zero_grad()
            batch_x = batch_x.float().to(official_args.device)
            batch_y = batch_y.float().to(official_args.device)

            f_dim = -1 if official_args.features == "MS" else 0
            target_y = batch_y[:, -official_args.pred_len :, f_dim:]

            if args.readout_mode == "official":
                outputs, recon, alignment_loss = model(
                    batch_x,
                    batch_y[:, -official_args.pred_len :, :],
                    is_training=True,
                )
                outputs = outputs[:, -official_args.pred_len :, f_dim:]
                pred_loss, pred_components = prediction_loss(
                    outputs,
                    target_y,
                    criterion,
                    args.target_horizons,
                    args.pred_loss_mode,
                    args.prefix_samples,
                    args.continuous_min_prefix,
                    args.continuous_prefix_step,
                )
                recon_loss = criterion(recon, target_y)
            else:
                if args.pred_loss_mode == "balanced-step":
                    raise ValueError("balanced-step is not supported for prefix-conditioned readout modes")
                selected_horizons = select_prediction_horizons(
                    args.target_horizons,
                    args.pred_loss_mode,
                    args.prefix_samples,
                    args.continuous_min_prefix,
                    args.continuous_prefix_step,
                    pred_len=official_args.pred_len,
                )
                prefix_losses = []
                teacher_losses = []
                recon_losses = []
                alignment_losses = []
                pred_components = {}
                for horizon in selected_horizons:
                    outputs, recon, alignment_loss = model(
                        batch_x,
                        batch_y[:, -official_args.pred_len :, :],
                        is_training=True,
                        target_prefix=horizon,
                    )
                    outputs = outputs[:, -official_args.pred_len :, f_dim:]
                    horizon_loss = criterion(outputs[:, :horizon, :], target_y[:, :horizon, :])
                    key = f"h{horizon}"
                    if key in pred_components:
                        key = f"{key}_repeat{len([name for name in pred_components if name.startswith(key)])}"
                    pred_components[key] = horizon_loss
                    prefix_losses.append(horizon_loss)
                    if teacher_model is not None:
                        with torch.no_grad():
                            teacher_outputs, _teacher_recon, _teacher_alignment = teacher_model(
                                batch_x,
                                batch_y[:, -official_args.pred_len :, :],
                                is_training=True,
                                target_prefix=horizon,
                            )
                            teacher_outputs = teacher_outputs[:, -official_args.pred_len :, f_dim:]
                        teacher_losses.append(criterion(outputs[:, :horizon, :], teacher_outputs[:, :horizon, :]))
                    recon_losses.append(criterion(recon, target_y))
                    alignment_losses.append(alignment_loss)
                    if horizon == official_args.pred_len:
                        pred_components["full"] = criterion(outputs, target_y)
                if "full" not in pred_components:
                    with torch.no_grad():
                        outputs_full, _recon_full, _alignment_full = model(
                            batch_x,
                            batch_y[:, -official_args.pred_len :, :],
                            is_training=True,
                            target_prefix=official_args.pred_len,
                        )
                        outputs_full = outputs_full[:, -official_args.pred_len :, f_dim:]
                        pred_components["full"] = criterion(outputs_full, target_y)
                pred_loss = torch.stack(prefix_losses).mean()
                teacher_loss = (
                    torch.stack(teacher_losses).mean()
                    if teacher_losses
                    else torch.zeros((), device=official_args.device)
                )
                recon_loss = torch.stack(recon_losses).mean()
                alignment_loss = torch.stack(alignment_losses).mean()
            if args.readout_mode == "official":
                teacher_loss = torch.zeros((), device=official_args.device)
            loss = (
                pred_loss
                + args.teacher_loss_weight * teacher_loss
                + official_args.w_recon * recon_loss
                + official_args.w_align * alignment_loss
            )
            loss.backward()
            optimizer.step()

            total_loss.append(float(loss.detach().cpu()))
            pred_loss_values.append(float(pred_loss.detach().cpu()))
            pred_full_loss_values.append(float(pred_components["full"].detach().cpu()))
            teacher_loss_values.append(float(teacher_loss.detach().cpu()))
            for name, component in pred_components.items():
                if name == "full":
                    continue
                pred_component_values.setdefault(name, []).append(float(component.detach().cpu()))
            recon_loss_values.append(float(recon_loss.detach().cpu()))
            alignment_values.append(float(alignment_loss.detach().cpu()))

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"\titers: {batch_idx + 1}, epoch: {epoch + 1} | loss: {float(loss.detach().cpu()):.7f}",
                    flush=True,
                )

        val_mean_mse = validation_mean_mse(
            model,
            vali_loader,
            official_args,
            args.target_horizons,
            max_batches=args.max_eval_batches,
        )
        if val_mean_mse < best_val:
            best_val = val_mean_mse
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        row = {
            "epoch": epoch + 1,
            "train_steps": train_steps if not args.max_train_batches else min(train_steps, args.max_train_batches),
            "train_loss": float(np.mean(total_loss)),
            "train_prediction_l1": float(np.mean(pred_loss_values)),
            "train_prediction_full_l1": float(np.mean(pred_full_loss_values)),
            "train_teacher_l1": float(np.mean(teacher_loss_values)),
            "teacher_loss_weight": args.teacher_loss_weight,
            "train_reconstruction_l1": float(np.mean(recon_loss_values)),
            "train_alignment_loss": float(np.mean(alignment_values)),
            "pred_loss_mode": args.pred_loss_mode,
            "prefix_samples": args.prefix_samples,
            "continuous_min_prefix": args.continuous_min_prefix,
            "continuous_prefix_step": args.continuous_prefix_step,
            "val_mean_mse": val_mean_mse,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.time() - epoch_start,
        }
        if warm_start_info is not None:
            row["warm_start_checkpoint"] = warm_start_info["checkpoint_path"]
            row["warm_start_compatible_keys"] = warm_start_info["compatible_keys"]
            row["warm_start_incompatible_keys"] = warm_start_info["incompatible_keys"]
        if args.teacher_checkpoint is not None:
            row["teacher_checkpoint"] = str(args.teacher_checkpoint)
            row["teacher_readout_mode"] = args.teacher_readout_mode
        for name, values in sorted(pred_component_values.items()):
            if values:
                row[f"train_prediction_{name}_l1"] = float(np.mean(values))
        training_rows.append(row)
        print(
            "Epoch: {epoch}, Steps: {steps} | Train Loss: {train_loss:.7f} Vali Loss: {val:.7f}".format(
                epoch=epoch + 1,
                steps=row["train_steps"],
                train_loss=row["train_loss"],
                val=val_mean_mse,
            ),
            flush=True,
        )
        adjust_learning_rate(optimizer, epoch + 1, official_args)

    if args.checkpoint_policy == "best-val":
        if best_state is None:
            raise RuntimeError("best-val policy requested but no checkpoint was captured")
        model.load_state_dict(best_state)
    return model, training_rows


def run(args: argparse.Namespace) -> None:
    if args.dataset not in OFFICIAL_PRESETS:
        raise ValueError(f"Unsupported dataset {args.dataset}. Choose from {sorted(OFFICIAL_PRESETS)}")
    preset_key = args.pred_len if args.mode == "fixed" else 720
    preset = OFFICIAL_PRESETS[args.dataset][preset_key]
    official_args = build_official_args(args, preset)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    adapter_config = vars(args) | {"dataset_root": str(args.dataset_root), "output_dir": str(args.output_dir)}
    if args.warm_start_checkpoint is not None:
        adapter_config["warm_start_checkpoint"] = str(args.warm_start_checkpoint)
    if args.teacher_checkpoint is not None:
        adapter_config["teacher_checkpoint"] = str(args.teacher_checkpoint)
    dump_json(
        args.output_dir / "effective_config.json",
        {
            "adapter": adapter_config,
            "official_args": {
                key: (str(value) if isinstance(value, torch.device) else value)
                for key, value in vars(official_args).items()
                if key != "device_ids"
            },
            "official_preset": asdict(preset),
            "source_note": "Official TimeAlign source vendored under baselines/timealign_official; train_repo.py is a thin repo adapter.",
        },
    )
    dump_json(
        args.output_dir / "environment.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": str(official_args.device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    )

    print(
        f"run_start dataset={args.dataset} mode={args.mode} pred_len={args.pred_len} "
        f"target_horizons={args.target_horizons} output_dir={args.output_dir}",
        flush=True,
    )
    model, training_rows = train(args, official_args)
    write_csv(args.output_dir / "training_log.csv", training_rows)
    torch.save(model.state_dict(), args.output_dir / "checkpoint.pt")

    _test_data, test_loader = data_provider(official_args, "test")
    main_rows, segment_metric_rows, preds, trues = evaluate(
        model,
        test_loader,
        official_args,
        args.target_horizons,
        max_batches=args.max_eval_batches,
        is_training_flag=args.official_test_mode,
    )
    for row in main_rows:
        row["mode"] = args.mode
        row["run_name"] = args.run_name
        row["dataset"] = args.dataset
        row["pred_len"] = args.pred_len
        row["checkpoint_policy"] = args.checkpoint_policy
        row["official_test_mode"] = int(args.official_test_mode)
    for row in segment_metric_rows:
        row["mode"] = args.mode
        row["run_name"] = args.run_name
        row["dataset"] = args.dataset
        row["pred_len"] = args.pred_len
    write_csv(args.output_dir / "metrics_by_target_horizon.csv", main_rows)
    write_csv(args.output_dir / "metrics_by_segment.csv", segment_metric_rows)
    np.savez_compressed(args.output_dir / "predictions_test.npz", pred=preds, true=trues)
    print(f"run_done output_dir={args.output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin FATST adapter for official TimeAlign.")
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--official-test-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-policy", choices=["official-last", "best-val"], default="official-last")
    parser.add_argument(
        "--readout-mode",
        choices=[
            "official",
            "prefix-conditioned-head",
            "target-set-decoder",
            "target-set-prefix-head",
            "prefix-token-decoder",
            "dense-prefix-residual-adapter",
            "row-gated-dense-head",
            "prefix-adapter-shared-dense",
            "dense-row-initialized-prefix-decoder",
            "nested-segment-decoder",
            "dense-initialized-nested-segment-decoder",
            "target-conditioned-nested-residual-decoder",
            "checkpoint-initialized-nested-segment-decoder",
            "target-conditioned-nested-segment-decoder",
        ],
        default="official",
    )
    parser.add_argument("--warm-start-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-readout-mode", choices=["target-set-decoder"], default="target-set-decoder")
    parser.add_argument("--teacher-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--pred-loss-mode",
        choices=["full", "multi-prefix", "balanced-step", "stochastic-prefix", "continuous-prefix"],
        default="full",
    )
    parser.add_argument("--prefix-samples", type=int, default=1)
    parser.add_argument("--continuous-min-prefix", type=int, default=32)
    parser.add_argument("--continuous-prefix-step", type=int, default=32)
    args = parser.parse_args()
    if max(args.target_horizons) > args.pred_len:
        raise ValueError("target horizons cannot exceed pred_len")
    if args.mode == "fixed" and args.target_horizons != [args.pred_len]:
        raise ValueError("fixed mode expects target_horizons == [pred_len]")
    if args.mode == "unified" and args.pred_len != 720:
        raise ValueError("unified mode currently expects pred_len=720")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

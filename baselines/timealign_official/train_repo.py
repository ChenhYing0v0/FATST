from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
PREFIX_READOUT_MODES = {
    "learned-basis-forecast-operator",
    "stage-native-coefficient-field",
    "stage-native-coefficient-field-no-stage",
    "basis-conditioned-coefficient-field",
    "basis-conditioned-coefficient-field-no-basis",
    "basis-conditioned-coefficient-field-shuffled-basis",
    "basis-conditioned-coefficient-field-constant-slot",
    "subspace-tiled-basis-operator-shared",
    "subspace-tiled-basis-operator-bank",
    "subspace-tiled-basis-operator-dct",
    "subspace-tiled-basis-operator-independent",
    "pmfo-rct",
    "pmfo-rct-no-transition",
    "pmfo-rct-no-conservation",
    "dense-mlp-matched",
    "plgo-paf-geo-c256",
    "plgo-paf-perm-c256",
    "plgo-paf-random-c256",
    "plgo-paf-geo-m694",
    "plgo-paf-perm-m694",
    "plgo-paf-random-m694",
    "japo-joint-geo",
    "japo-uniform",
    "japo-history",
    "japo-atom",
    "japo-joint-perm",
    "japo-joint-random",
}

STAGE_C_ACTIVE_READOUTS = {
    "learned-basis-forecast-operator",
    "pmfo-rct",
    "pmfo-rct-no-transition",
    "pmfo-rct-no-conservation",
    "dense-mlp-matched",
    "plgo-paf-geo-c256",
    "plgo-paf-perm-c256",
    "plgo-paf-random-c256",
    "plgo-paf-geo-m694",
    "plgo-paf-perm-m694",
    "plgo-paf-random-m694",
    "japo-joint-geo",
    "japo-uniform",
    "japo-history",
    "japo-atom",
    "japo-joint-perm",
    "japo-joint-random",
}

ACTIVE_STAGE_C_CONTRACT = {
    "mode": "unified",
    "encoder_mode": "timealign-token-mlp",
    "readout_mode": "learned-basis-forecast-operator",
    "pred_loss_mode": "full",
}


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
    "ETTh1": {
        horizon: OfficialPreset(
            data="ETTh1",
            data_path="ETTh1.csv",
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
            patch_num=24,
            local_margin=0.5,
            global_margin=0.0,
            layer_norm=1,
        )
        for horizon in HORIZONS
    },
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
        96: OfficialPreset("ETTm2", "ETTm2.csv", "ETT-small", "t", 7, 7, 7, 128, 128, 0.0001, 0.3, 1.0, 12, 0.0, 0.0, 1),
        192: OfficialPreset("ETTm2", "ETTm2.csv", "ETT-small", "t", 7, 7, 7, 128, 128, 0.0001, 0.3, 1.0, 12, 0.0, 0.0, 1),
        336: OfficialPreset("ETTm2", "ETTm2.csv", "ETT-small", "t", 7, 7, 7, 128, 128, 0.0001, 0.9, 1.0, 12, 0.0, 0.0, 1),
        720: OfficialPreset("ETTm2", "ETTm2.csv", "ETT-small", "t", 7, 7, 7, 128, 128, 0.0001, 0.9, 1.0, 12, 0.0, 0.0, 1),
    },
    "ETTm1": {
        96: OfficialPreset("ETTm1", "ETTm1.csv", "ETT-small", "t", 7, 7, 7, 128, 256, 0.0001, 0.2, 0.1, 1, 0.5, 0.0, 1),
        192: OfficialPreset("ETTm1", "ETTm1.csv", "ETT-small", "t", 7, 7, 7, 128, 256, 0.0001, 0.2, 0.1, 1, 0.5, 0.0, 1),
        336: OfficialPreset("ETTm1", "ETTm1.csv", "ETT-small", "t", 7, 7, 7, 128, 256, 0.0001, 0.8, 0.1, 1, 0.5, 0.0, 1),
        720: OfficialPreset("ETTm1", "ETTm1.csv", "ETT-small", "t", 7, 7, 7, 256, 256, 0.0001, 0.9, 0.1, 1, 0.5, 0.0, 1),
    },
    "Weather": {
        96: OfficialPreset("custom", "weather.csv", "weather", "h", 21, 21, 21, 128, 256, 0.0001, 0.1, 0.1, 48, 0.5, 0.0, 0),
        192: OfficialPreset("custom", "weather.csv", "weather", "h", 21, 21, 21, 128, 256, 0.0001, 0.1, 0.1, 48, 0.5, 0.0, 0),
        336: OfficialPreset("custom", "weather.csv", "weather", "h", 21, 21, 21, 128, 256, 0.0001, 0.1, 0.1, 48, 0.5, 0.0, 0),
        720: OfficialPreset("custom", "weather.csv", "weather", "h", 21, 21, 21, 128, 128, 0.0001, 0.5, 0.1, 48, 0.5, 0.0, 0),
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
    is_prefix_readout = args.readout_mode in PREFIX_READOUT_MODES
    encoder_mode = getattr(args, "encoder_mode", "timealign-token-mlp")
    contextual_encoder = encoder_mode in {
        "contextual-patch-transformer",
        "global-anchored-patch-transformer",
    }
    global_anchored_encoder = encoder_mode == "global-anchored-patch-transformer"
    legacy_d_model = preset.d_model if args.legacy_d_model is None else args.legacy_d_model
    legacy_d_ff = preset.d_ff if args.legacy_d_ff is None else args.legacy_d_ff
    legacy_dropout = preset.dropout if args.legacy_dropout is None else args.legacy_dropout
    legacy_patch_num = preset.patch_num if args.legacy_patch_num is None else args.legacy_patch_num
    legacy_layer_norm = (
        preset.layer_norm
        if args.legacy_layer_norm is None
        else args.legacy_layer_norm
    )
    if global_anchored_encoder:
        effective_dropout = args.history_ffn_dropout
    elif contextual_encoder:
        effective_dropout = args.history_dropout
    else:
        effective_dropout = legacy_dropout
    return argparse.Namespace(
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
        d_model=(
            getattr(args, "history_d_model", 128)
            if contextual_encoder
            else legacy_d_model
        ),
        n_heads=(
            getattr(args, "history_n_heads", 16) if contextual_encoder else 8
        ),
        e_layers=(
            getattr(args, "history_e_layers", 3)
            if contextual_encoder
            else args.e_layers
        ),
        d_layers=1,
        d_ff=(
            getattr(args, "history_d_ff", 256)
            if contextual_encoder
            else legacy_d_ff
        ),
        factor=3,
        dropout=effective_dropout,
        embed="timeF",
        distil=True,
        expand=2,
        d_conv=4,
        num_workers=args.num_workers,
        itr=1,
        train_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=(
            preset.learning_rate
            if getattr(args, "learning_rate", None) is None
            else args.learning_rate
        ),
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
        w_align=0.0 if is_prefix_readout else (preset.w_align if args.w_align is None else args.w_align),
        w_recon=0.0 if is_prefix_readout else args.w_recon,
        local_margin=preset.local_margin,
        global_margin=preset.global_margin,
        patch_num=legacy_patch_num,
        layer_norm=legacy_layer_norm,
        pos=1,
        loc=1,
        glo=1,
        device=device,
        readout_mode=args.readout_mode,
        encoder_mode=encoder_mode,
        history_patch_len=getattr(args, "history_patch_len", 48),
        history_patch_stride=getattr(args, "history_patch_stride", 24),
        history_token_dropout=getattr(args, "history_token_dropout", 0.0),
        history_attn_dropout=getattr(args, "history_attn_dropout", 0.0),
        history_attn_residual_dropout=getattr(
            args,
            "history_attn_residual_dropout",
            0.1,
        ),
        history_ffn_dropout=getattr(args, "history_ffn_dropout", 0.1),
        history_ffn_residual_dropout=getattr(
            args,
            "history_ffn_residual_dropout",
            0.1,
        ),
        history_res_attention=getattr(args, "history_res_attention", True),
        target_horizons=args.target_horizons,
        basis_rank=args.basis_rank,
        stage_token_dim=args.stage_token_dim,
        stage_field_rank=args.stage_field_rank,
        stage_gate_init=args.stage_gate_init,
        basis_field_window_len=args.basis_field_window_len,
        basis_field_stride=args.basis_field_stride,
        basis_field_rank=args.basis_field_rank,
        basis_field_tau=args.basis_field_tau,
        basis_field_gate_init=args.basis_field_gate_init,
        stbo_tile_len=args.stbo_tile_len,
        stbo_rank=args.stbo_rank,
        stbo_bank_count=args.stbo_bank_count,
        stbo_basis_init_std=args.stbo_basis_init_std,
        pmfo_state_dim=args.pmfo_state_dim,
        pmfo_dense_hidden_dim=args.pmfo_dense_hidden_dim,
        plgo_global_rank=args.plgo_global_rank,
        plgo_latent_width=args.plgo_latent_width,
        plgo_permutation_seed=args.plgo_permutation_seed,
        plgo_random_descriptor_seed=args.plgo_random_descriptor_seed,
        japo_expert_count=args.japo_expert_count,
        japo_expert_rank=args.japo_expert_rank,
        japo_router_width=args.japo_router_width,
        japo_router_output_init_std=args.japo_router_output_init_std,
        evaluation_prefix_mode=getattr(args, "evaluation_prefix_mode", "native"),
        segment_horizons=getattr(
            args,
            "segment_horizons",
            getattr(args, "evaluation_horizons", args.target_horizons),
        ),
    )


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


def _tensor_hash(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialization_contract(model: nn.Module) -> dict[str, Any]:
    named_parameters = list(model.named_parameters())
    encoder = [
        parameter
        for name, parameter in named_parameters
        if name.startswith(("patch_emb_x.", "encoder.", "norm_x."))
    ]
    payload: dict[str, Any] = {
        "encoder_initialization_hash": _tensor_hash(encoder),
        "readout_mode": getattr(model, "readout_mode", ""),
    }
    if hasattr(model, "japo_readout"):
        readout = model.japo_readout
        experts = [
            parameter
            for name, parameter in named_parameters
            if name.startswith(
                (
                    "japo_readout.expert_branches.",
                    "japo_readout.atom_basis",
                    "japo_readout.coefficient_bias",
                )
            )
        ]
        hidden = torch.linspace(
            -1.0,
            1.0,
            steps=next(iter(readout.expert_branches)).in_features,
            device=readout.atom_basis.device,
        ).view(1, 1, -1)
        with torch.no_grad():
            gates = readout.gates(hidden)
            entropy = -(
                gates * gates.clamp_min(1e-12).log()
            ).sum(dim=-1).mean() / math.log(readout.expert_count)
            usage = gates.mean(dim=(0, 1, 2))
        payload.update(
            {
                "expert_bank_initialization_hash": _tensor_hash(experts),
                "basis_hash": _tensor_hash([readout.basis_rows]),
                "descriptor_hash": _tensor_hash([readout.descriptors]),
                "expert_pair_max_abs_difference": float(
                    (
                        readout.expert_branches[0].weight
                        - readout.expert_branches[1].weight
                    )
                    .abs()
                    .max()
                    .detach()
                    .item()
                ),
                "initial_gate_entropy": float(entropy.detach().item()),
                "initial_expert_usage": usage.detach().cpu().tolist(),
            }
        )
    return payload


def model_diagnostics(model: nn.Module) -> dict[str, Any]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    payload: dict[str, Any] = {
        "total_parameters": total_parameters,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "frozen_parameter_tensors": sum(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "readout_mode": getattr(model, "readout_mode", ""),
        "encoder_mode": getattr(model, "encoder_mode", ""),
        "patch_num": int(getattr(model, "patch_num", 0)),
        "d_model": int(getattr(model, "d_model", 0)),
    }
    if getattr(model, "readout_mode", "") == "learned-basis-forecast-operator":
        active_prefixes = (
            "patch_emb_x.",
            "encoder.",
            "norm_x.",
            "history_encoder.",
            "learned_basis_coeff.",
            "learned_temporal_basis",
            "learned_temporal_bias",
        )
        active_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith(active_prefixes)
        )
        unused_proj_x_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("proj_x.")
        )
        payload.update(
            {
                "active_forward_parameters": active_parameters,
                "unused_proj_x_parameters": unused_proj_x_parameters,
                "inactive_or_other_parameters": total_parameters - active_parameters,
            }
        )
    if hasattr(model, "history_encoder"):
        history_payload = {
            "history_patch_len": int(model.history_encoder.patch_len),
            "history_patch_stride": int(model.history_encoder.stride),
            "history_res_attention": bool(model.history_encoder.residual_attention),
        }
        if hasattr(model.history_encoder, "patch_num"):
            history_payload["history_local_patch_num"] = int(
                model.history_encoder.patch_num
            )
        for field in (
            "token_dropout_p",
            "attn_dropout_p",
            "attn_residual_dropout_p",
            "ffn_dropout_p",
            "ffn_residual_dropout_p",
        ):
            if hasattr(model.history_encoder, field):
                history_payload[f"history_{field}"] = float(
                    getattr(model.history_encoder, field)
                )
        payload.update(history_payload)
    if hasattr(model, "retrieval_memory"):
        payload.update(
            {
                "retrieval_patch_num": int(model.retrieval_memory.patch_num),
                "retrieval_patch_len": int(model.retrieval_memory.patch_len),
                "retrieval_patch_stride": int(model.retrieval_memory.stride),
                "retrieval_token_dim": int(model.retrieval_memory.patch_len),
            }
        )
    if hasattr(model, "stage_gate_logits"):
        with torch.no_grad():
            gate = torch.sigmoid(model.stage_gate_logits.detach().cpu()).view(-1).tolist()
            payload.update(
                {
                    "stage_count": int(model.stage_count),
                    "stage_boundaries": [int(value) for value in model.stage_boundaries],
                    "stage_token_dim": int(model.stage_token_dim),
                    "stage_field_rank": int(model.stage_field_rank),
                    "stage_gate_sigmoid": gate,
                    "stage_gate_mean": float(np.mean(gate)),
                    "stage_gate_min": float(np.min(gate)),
                    "stage_gate_max": float(np.max(gate)),
                    "stage_token_l2": float(model.stage_tokens.detach().cpu().norm().item()),
                    "stage_coeff_down_l2": float(model.stage_coeff_down.weight.detach().cpu().norm().item()),
                    "stage_coeff_up_l2": float(model.stage_coeff_up.weight.detach().cpu().norm().item()),
                }
            )
    if hasattr(model, "basis_field_gate_logit"):
        with torch.no_grad():
            gate = float(torch.sigmoid(model.basis_field_gate_logit.detach().cpu()).item())
            payload.update(
                {
                    "basis_field_window_len": int(model.basis_field_window_len),
                    "basis_field_stride": int(model.basis_field_stride),
                    "basis_field_window_count": int(model.basis_field_window_count),
                    "basis_field_rank": int(model.basis_field_rank),
                    "basis_field_tau": float(model.basis_field_tau),
                    "basis_field_gate_sigmoid": gate,
                    "basis_field_desc_proj_l2": float(model.basis_field_desc_proj.weight.detach().cpu().norm().item()),
                    "basis_field_state_proj_l2": float(model.basis_field_state_proj.weight.detach().cpu().norm().item()),
                    "basis_field_delta_l2": float(model.basis_field_delta.weight.detach().cpu().norm().item()),
                }
            )
    if hasattr(model, "stbo_coeff"):
        payload.update(
            {
                "stbo_tile_len": int(model.stbo_tile_len),
                "stbo_tile_count": int(model.stbo_tile_count),
                "stbo_rank": int(model.stbo_rank),
                "stbo_coeff_l2": float(model.stbo_coeff.weight.detach().cpu().norm().item()),
            }
        )
        if hasattr(model, "stbo_shared_basis"):
            payload["stbo_shared_basis_l2"] = float(model.stbo_shared_basis.detach().cpu().norm().item())
        if hasattr(model, "stbo_basis_bank"):
            with torch.no_grad():
                weights = torch.softmax(model.stbo_tile_bank_logits.detach().cpu(), dim=-1)
                payload.update(
                    {
                        "stbo_bank_count": int(model.stbo_bank_count),
                        "stbo_basis_bank_l2": float(model.stbo_basis_bank.detach().cpu().norm().item()),
                        "stbo_tile_bank_entropy_mean": float(
                            (-weights * torch.log(torch.clamp(weights, min=1e-12))).sum(dim=-1).mean().item()
                            / np.log(model.stbo_bank_count)
                        ),
                    }
                )
        if hasattr(model, "stbo_tile_basis"):
            payload["stbo_tile_basis_l2"] = float(model.stbo_tile_basis.detach().cpu().norm().item())
        if hasattr(model, "stbo_dct_basis"):
            payload["stbo_dct_basis_l2"] = float(model.stbo_dct_basis.detach().cpu().norm().item())
    if hasattr(model, "pmfo_readout"):
        payload.update(
            {
                "pmfo_decoder_parameters": sum(
                    parameter.numel()
                    for parameter in model.pmfo_readout.parameters()
                ),
                "pmfo_state_dim": getattr(model.pmfo_readout, "state_dim", None),
                "pmfo_dense_hidden_dim": getattr(
                    model.pmfo_readout,
                    "hidden_dim",
                    None,
                ),
                "pmfo_conservative": getattr(
                    model.pmfo_readout,
                    "conservative",
                    None,
                ),
            }
        )
    if hasattr(model, "plgo_paf_readout"):
        readout = model.plgo_paf_readout
        active_prefixes = (
            "patch_emb_x.",
            "encoder.",
            "norm_x.",
            "plgo_paf_readout.",
        )
        active_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith(active_prefixes)
        )
        payload.update(
            {
                "active_forward_parameters": active_parameters,
                "unused_proj_x_parameters": sum(
                    parameter.numel()
                    for name, parameter in model.named_parameters()
                    if name.startswith("proj_x.")
                ),
                "plgo_paf_decoder_parameters": sum(
                    parameter.numel() for parameter in readout.parameters()
                ),
                "plgo_descriptor_family": readout.descriptor_name,
                "plgo_trunk_width": int(readout.trunk_width),
                "plgo_latent_width": int(readout.latent_width),
                "plgo_global_rank": int(readout.global_rank),
            }
        )
    if hasattr(model, "japo_readout"):
        readout = model.japo_readout
        active_prefixes: tuple[str, ...] = (
            "patch_emb_x.",
            "encoder.",
            "norm_x.",
            "japo_readout.expert_branches.",
            "japo_readout.atom_basis",
            "japo_readout.coefficient_bias",
        )
        if readout.gate_mode in {"joint", "history"}:
            active_prefixes += ("japo_readout.history_projection.",)
        if readout.gate_mode in {"joint", "atom"}:
            active_prefixes += ("japo_readout.descriptor_projection.",)
        if readout.gate_mode != "uniform":
            active_prefixes += ("japo_readout.gate_weight",)
        active_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith(active_prefixes)
        )
        payload.update(
            {
                "active_forward_parameters": active_parameters,
                "unused_proj_x_parameters": sum(
                    parameter.numel()
                    for name, parameter in model.named_parameters()
                    if name.startswith("proj_x.")
                ),
                "japo_decoder_parameters": sum(
                    parameter.numel() for parameter in readout.parameters()
                ),
                "japo_gate_mode": readout.gate_mode,
                "japo_descriptor_family": readout.descriptor_name,
                "japo_expert_count": int(readout.expert_count),
                "japo_expert_rank": int(readout.expert_rank),
                "japo_router_width": int(readout.router_width),
                "plgo_global_rank": int(readout.global_rank),
            }
        )
    return payload


def patch_interface_diagnostics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    official_args: argparse.Namespace,
    max_batches: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray | None]:
    """Measure patch usage without changing checkpoint selection."""
    readout_mode = str(getattr(model, "readout_mode", ""))
    if readout_mode == "learned-basis-forecast-operator":
        projection = model.learned_basis_coeff
        interface = "a6_coefficient_projection"
    elif hasattr(model, "plgo_paf_readout"):
        projection = model.plgo_paf_readout.branch
        interface = "plgo_shared_latent_branch"
    elif hasattr(model, "japo_readout"):
        projection = None
        interface = "japo_independent_expert_branches"
    else:
        raise ValueError("patch diagnostics require A6, PLGO-PAF, or JAPO readout")

    patch_num = int(model.patch_num)
    d_model = int(model.d_model)
    if projection is not None:
        latent_width = int(projection.out_features)
        if projection.in_features != patch_num * d_model:
            raise ValueError("projection width does not match patch tensor contract")
        blocks = projection.weight.reshape(latent_width, patch_num, d_model)
        projection_bias = projection.bias
    else:
        readout = model.japo_readout
        latent_width = int(readout.expert_count * readout.expert_rank)
        blocks = torch.stack(
            [branch.weight for branch in readout.expert_branches],
            dim=0,
        ).reshape(latent_width, patch_num, d_model)
        projection_bias = torch.stack(
            [branch.bias for branch in readout.expert_branches],
            dim=0,
        ).reshape(-1)
    weight_norm = blocks.square().sum(dim=(0, 2)).sqrt()
    weight_share = weight_norm / weight_norm.sum().clamp_min(1e-12)
    contribution_energy = torch.zeros(
        patch_num,
        device=official_args.device,
        dtype=torch.float64,
    )
    contribution_elements = 0
    flatten_block_sum_max_abs = 0.0
    gate_probability_sum = None
    gate_entropy_sum = 0.0
    gate_rows = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_x, _batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            batch_x = batch_x.float().to(official_args.device)
            memory = model.encode_history(batch_x)
            hidden = memory.flatten(start_dim=-2)
            contributions = torch.einsum(
                "bcpd,kpd->bcpk",
                memory,
                blocks,
            )
            contribution_energy += contributions.double().square().sum(
                dim=(0, 1, 3)
            )
            contribution_elements += int(
                contributions.shape[0]
                * contributions.shape[1]
                * contributions.shape[3]
            )
            if projection is not None:
                direct = projection(hidden)
            else:
                direct = model.japo_readout.expert_latents(hidden).flatten(
                    start_dim=-2
                )
            explicit = contributions.sum(dim=2) + projection_bias
            flatten_block_sum_max_abs = max(
                flatten_block_sum_max_abs,
                float((direct - explicit).abs().max()),
            )
            if hasattr(model, "japo_readout"):
                gates = model.japo_readout.gates(hidden)
                reduced = gates.sum(dim=(0, 1, 2)).double()
                gate_probability_sum = (
                    reduced
                    if gate_probability_sum is None
                    else gate_probability_sum + reduced
                )
                entropy = -(
                    gates * gates.clamp_min(1e-12).log()
                ).sum(dim=-1) / math.log(model.japo_readout.expert_count)
                gate_entropy_sum += float(entropy.double().sum())
                gate_rows += int(entropy.numel())
    if contribution_elements == 0:
        raise RuntimeError("patch diagnostics received no validation batches")
    contribution_energy /= contribution_elements
    contribution_share = contribution_energy / contribution_energy.sum().clamp_min(
        1e-12
    )
    entropy = float(
        (
            -(contribution_share * contribution_share.clamp_min(1e-12).log()).sum()
            / math.log(patch_num)
        ).item()
    )
    rows = [
        {
            "patch_index": index,
            "weight_norm_share": float(weight_share[index]),
            "latent_contribution_share": float(contribution_share[index]),
        }
        for index in range(patch_num)
    ]
    payload: dict[str, Any] = {
        "readout_mode": readout_mode,
        "interface": interface,
        "source_split": "validation",
        "max_batches": max_batches,
        "patch_num": patch_num,
        "d_model": d_model,
        "state_width": patch_num * d_model,
        "latent_width": latent_width,
        "flatten_block_sum_max_abs": flatten_block_sum_max_abs,
        "patch_contribution_entropy": entropy,
        "finite": all(
            math.isfinite(value)
            for value in (
                flatten_block_sum_max_abs,
                entropy,
                *weight_share.detach().cpu().tolist(),
                *contribution_share.detach().cpu().tolist(),
            )
        ),
    }
    atom_patch_jacobian: np.ndarray | None = None
    if hasattr(model, "plgo_paf_readout"):
        readout = model.plgo_paf_readout
        atom_features = readout.atom_features()
        jacobian = torch.einsum("nk,kpd->npd", atom_features, blocks)
        jacobian_norm = jacobian.square().sum(dim=-1).sqrt()
        atom_patch_jacobian = jacobian_norm.detach().cpu().numpy()
        group_profiles = []
        for group_id in torch.unique(readout.atom_group_ids).tolist():
            mask = readout.atom_group_ids == int(group_id)
            profile = jacobian_norm[mask].mean(dim=0)
            profile = profile / profile.sum().clamp_min(1e-12)
            group_profiles.append(profile)
        distances = []
        for left in range(len(group_profiles)):
            for right in range(left + 1, len(group_profiles)):
                distances.append(
                    float((group_profiles[left] - group_profiles[right]).abs().mean())
                )
        payload.update(
            {
                "atom_group_count": len(group_profiles),
                "atom_patch_profile_diversity": (
                    float(np.mean(distances)) if distances else 0.0
                ),
                "atom_patch_jacobian_shape": list(jacobian_norm.shape),
            }
        )
    if hasattr(model, "japo_readout"):
        readout = model.japo_readout
        expert_gap = float(
            (
                readout.expert_branches[0].weight
                - readout.expert_branches[1].weight
            )
            .abs()
            .max()
            .detach()
            .item()
        )
        payload.update(
            {
                "japo_gate_mode": readout.gate_mode,
                "japo_descriptor_family": readout.descriptor_name,
                "japo_expert_pair_max_abs_difference": expert_gap,
                "japo_gate_entropy": gate_entropy_sum / gate_rows,
                "japo_expert_usage": (
                    gate_probability_sum / gate_probability_sum.sum()
                ).detach().cpu().tolist(),
            }
        )
    return payload, rows, atom_patch_jacobian


def metric_rows(preds: np.ndarray, trues: np.ndarray, horizons: list[int]) -> list[dict[str, Any]]:
    maximum_horizon = max(horizons)
    if preds.shape[0] != trues.shape[0] or preds.shape[2] != trues.shape[2]:
        raise ValueError("prediction and target batch/channel shapes must match")
    if preds.shape[1] < maximum_horizon or trues.shape[1] < maximum_horizon:
        raise ValueError("prediction or target length is shorter than requested horizon")
    errors = preds[:, :maximum_horizon] - trues[:, :maximum_horizon]
    squared_by_step = np.mean(errors**2, axis=(0, 2))
    absolute_by_step = np.mean(np.abs(errors), axis=(0, 2))
    cumulative_squared = np.cumsum(squared_by_step, dtype=np.float64)
    cumulative_absolute = np.cumsum(absolute_by_step, dtype=np.float64)
    rows = []
    for horizon in horizons:
        if horizon <= 0 or horizon > preds.shape[1]:
            raise ValueError("evaluation horizon exceeds prediction length")
        rows.append(
            {
                "target_horizon": horizon,
                "mse": float(cumulative_squared[horizon - 1] / horizon),
                "mae": float(cumulative_absolute[horizon - 1] / horizon),
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
        rows.append(
            {
                "target_horizon": target_horizon,
                "segment_start": start,
                "segment_end": end,
                "mse": float(MSE(preds[:, start:end, :], trues[:, start:end, :])),
                "mae": float(MAE(preds[:, start:end, :], trues[:, start:end, :])),
            }
        )
    return rows


def select_prediction_horizons(mode: str, horizons: list[int], pred_len: int) -> list[int]:
    if mode == "full":
        return [pred_len]
    if mode == "multi-prefix":
        return sorted(set(horizons))
    raise ValueError(f"Unsupported prediction loss mode: {mode}")


def prediction_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    horizons: list[int],
    mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    selected_horizons = select_prediction_horizons(mode, horizons, outputs.shape[1])
    losses: dict[str, torch.Tensor] = {"full": criterion(outputs, targets)}
    prefix_losses = []
    for horizon in selected_horizons:
        loss = criterion(outputs[:, :horizon, :], targets[:, :horizon, :])
        losses[f"h{horizon}"] = loss
        prefix_losses.append(loss)
    return torch.stack(prefix_losses).mean(), losses


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    official_args: argparse.Namespace,
    horizons: list[int],
    max_batches: int,
    is_training_flag: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    if official_args.readout_mode in PREFIX_READOUT_MODES:
        return evaluate_a6_lbf(model, loader, official_args, horizons, max_batches, is_training_flag)

    preds = []
    trues = []
    f_dim = -1 if official_args.features == "MS" else 0
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
            preds.append(outputs[:, -official_args.pred_len :, f_dim:].detach().cpu().numpy())
            trues.append(batch_y[:, -official_args.pred_len :, f_dim:].detach().cpu().numpy())

    if not preds:
        raise RuntimeError("evaluation produced no batches")
    pred_np = np.concatenate(preds, axis=0)
    true_np = np.concatenate(trues, axis=0)
    main_rows = metric_rows(pred_np, true_np, horizons)
    all_segments: list[dict[str, Any]] = []
    for horizon in official_args.segment_horizons:
        all_segments.extend(segment_rows(pred_np, true_np, horizon))
    return main_rows, all_segments, pred_np, true_np


def evaluate_a6_lbf(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    official_args: argparse.Namespace,
    horizons: list[int],
    max_batches: int,
    is_training_flag: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    if official_args.evaluation_prefix_mode == "full-crop":
        preds = []
        trues = []
        f_dim = -1 if official_args.features == "MS" else 0
        model.eval()
        with torch.no_grad():
            for batch_idx, (
                batch_x,
                batch_y,
                _batch_x_mark,
                _batch_y_mark,
            ) in enumerate(loader):
                if max_batches and batch_idx >= max_batches:
                    break
                batch_x = batch_x.float().to(official_args.device)
                batch_y = batch_y.float().to(official_args.device)
                outputs, _recon, _alignment = model(
                    batch_x,
                    batch_y[:, -official_args.pred_len :, :],
                    is_training=is_training_flag,
                    target_prefix=official_args.pred_len,
                )
                preds.append(outputs[:, :, f_dim:].detach().cpu().numpy())
                trues.append(
                    batch_y[:, -official_args.pred_len :, f_dim:]
                    .detach()
                    .cpu()
                    .numpy()
                )
        if not preds:
            raise RuntimeError("full-crop evaluation produced no batches")
        pred_np = np.concatenate(preds, axis=0)
        true_np = np.concatenate(trues, axis=0)
        main_rows = metric_rows(pred_np, true_np, horizons)
        all_segments = []
        for horizon in official_args.segment_horizons:
            all_segments.extend(segment_rows(pred_np, true_np, horizon))
        return main_rows, all_segments, pred_np, true_np

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
                preds.append(outputs[:, -official_args.pred_len :, f_dim:].detach().cpu().numpy())
                trues.append(batch_y[:, -official_args.pred_len :, f_dim:].detach().cpu().numpy())
            if not preds:
                raise RuntimeError("evaluation produced no batches")
            pred_np = np.concatenate(preds, axis=0)
            true_np = np.concatenate(trues, axis=0)
            main_rows.extend(metric_rows(pred_np, true_np, [horizon]))
            if horizon in official_args.segment_horizons:
                all_segments.extend(segment_rows(pred_np, true_np, horizon))
            if horizon == max(horizons):
                pred_for_npz = pred_np
                true_for_npz = true_np

    if pred_for_npz is None or true_for_npz is None:
        raise RuntimeError("A6-LBF evaluation did not produce final-horizon predictions")
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


def early_stopping_update(
    value: float,
    best_value: float,
    epochs_without_improvement: int,
    min_delta: float,
) -> tuple[bool, int]:
    """Return whether validation improved and the updated patience counter."""
    if value < best_value - min_delta:
        return True, 0
    return False, epochs_without_improvement + 1


def train(
    args: argparse.Namespace,
    official_args: argparse.Namespace,
) -> tuple[
    nn.Module,
    list[dict[str, Any]],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    train_data, train_loader = data_provider(official_args, "train")
    vali_data, vali_loader = data_provider(official_args, "val")
    del train_data, vali_data

    model = TimeAlign.Model(official_args).float().to(official_args.device)
    dump_json(
        args.output_dir / "initialization_contract.json",
        initialization_contract(model),
    )
    optimizer = optim.AdamW(model.parameters(), lr=official_args.learning_rate)
    criterion = nn.L1Loss()
    training_rows: list[dict[str, Any]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        total_loss = []
        pred_loss_values = []
        pred_full_loss_values = []
        pred_component_values: dict[str, list[float]] = {}
        recon_loss_values = []
        alignment_values = []
        train_steps = len(train_loader)
        effective_train_steps = (
            train_steps
            if not args.max_train_batches
            else min(train_steps, args.max_train_batches)
        )
        optimizer.zero_grad()

        for batch_idx, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
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
                )
                if recon is None:
                    raise RuntimeError("official TimeAlign training requires reconstruction output")
                recon_loss = criterion(recon, target_y)
            else:
                selected_horizons = select_prediction_horizons(
                    args.pred_loss_mode,
                    args.target_horizons,
                    official_args.pred_len,
                )
                prefix_losses = []
                pred_components = {}
                for horizon in selected_horizons:
                    outputs, _recon, _alignment_loss = model(
                        batch_x,
                        batch_y[:, -official_args.pred_len :, :],
                        is_training=True,
                        target_prefix=horizon,
                    )
                    outputs = outputs[:, -official_args.pred_len :, f_dim:]
                    horizon_loss = criterion(outputs[:, :horizon, :], target_y[:, :horizon, :])
                    pred_components[f"h{horizon}"] = horizon_loss
                    prefix_losses.append(horizon_loss)
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
                recon_loss = pred_loss.new_zeros(())
                alignment_loss = pred_loss.new_zeros(())

            loss = pred_loss + official_args.w_recon * recon_loss + official_args.w_align * alignment_loss
            (loss / args.gradient_accumulation_steps).backward()
            should_step = (
                (batch_idx + 1) % args.gradient_accumulation_steps == 0
                or batch_idx + 1 == effective_train_steps
            )
            if should_step:
                optimizer.step()
                optimizer.zero_grad()

            total_loss.append(float(loss.detach().cpu()))
            pred_loss_values.append(float(pred_loss.detach().cpu()))
            pred_full_loss_values.append(float(pred_components["full"].detach().cpu()))
            recon_loss_values.append(float(recon_loss.detach().cpu()))
            alignment_values.append(float(alignment_loss.detach().cpu()))
            for name, component in pred_components.items():
                if name != "full":
                    pred_component_values.setdefault(name, []).append(float(component.detach().cpu()))

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"\titers: {batch_idx + 1}, epoch: {epoch + 1} | loss: {float(loss.detach().cpu()):.7f}",
                    flush=True,
                )

        val_mean_mse = validation_mean_mse(
            model,
            vali_loader,
            official_args,
            args.validation_horizons,
            max_batches=args.max_eval_batches,
        )
        improved, epochs_without_improvement = early_stopping_update(
            val_mean_mse,
            best_val,
            epochs_without_improvement,
            args.early_stopping_min_delta,
        )
        if improved:
            best_val = val_mean_mse
            best_epoch = epoch + 1
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        stop_triggered = (
            args.enable_early_stopping
            and epochs_without_improvement >= args.patience
        )

        row = {
            "epoch": epoch + 1,
            "train_steps": train_steps if not args.max_train_batches else min(train_steps, args.max_train_batches),
            "train_loss": float(np.mean(total_loss)),
            "train_prediction_l1": float(np.mean(pred_loss_values)),
            "train_prediction_full_l1": float(np.mean(pred_full_loss_values)),
            "train_reconstruction_l1": float(np.mean(recon_loss_values)),
            "train_alignment_loss": float(np.mean(alignment_values)),
            "train_weighted_reconstruction_l1": official_args.w_recon * float(np.mean(recon_loss_values)),
            "train_weighted_alignment_loss": official_args.w_align * float(np.mean(alignment_values)),
            "pred_loss_mode": args.pred_loss_mode,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": (
                args.batch_size * args.gradient_accumulation_steps
            ),
            "validation_horizons": ",".join(
                str(horizon) for horizon in args.validation_horizons
            ),
            "val_mean_mse": val_mean_mse,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.time() - epoch_start,
            "early_stopping_enabled": int(args.enable_early_stopping),
            "early_stopping_patience": args.patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "best_epoch_so_far": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "stop_triggered": int(stop_triggered),
        }
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
        if stop_triggered:
            print(
                f"Early stopping at epoch {epoch + 1}; restoring best epoch {best_epoch}",
                flush=True,
            )
            break
        adjust_learning_rate(optimizer, epoch + 1, official_args)

    if best_state is None:
        raise RuntimeError("training completed without capturing a validation checkpoint")
    last_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    if args.checkpoint_policy == "best-val":
        model.load_state_dict(best_state)
    return model, training_rows, last_state, best_state


def annotate_evaluation_rows(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    checkpoint_policy: str,
) -> None:
    for row in rows:
        row["mode"] = args.mode
        row["run_name"] = args.run_name
        row["dataset"] = args.dataset
        row["pred_len"] = args.pred_len
        row["checkpoint_policy"] = checkpoint_policy
        row["evaluation_split"] = args.final_evaluation_split
        row["official_test_mode"] = int(args.official_test_mode)
        row["protocol_class"] = args.protocol_class
        row["protocol_profile"] = args.protocol_profile
        row["profile_hash"] = args.profile_hash


def run(args: argparse.Namespace) -> None:
    if args.dataset not in OFFICIAL_PRESETS:
        raise ValueError(f"Unsupported dataset {args.dataset}. Choose from {sorted(OFFICIAL_PRESETS)}")
    preset_key = args.pred_len if args.mode == "fixed" else 720
    preset = OFFICIAL_PRESETS[args.dataset][preset_key]
    official_args = build_official_args(args, preset)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    dump_json(
        args.output_dir / "effective_config.json",
        {
            "adapter": vars(args) | {"dataset_root": str(args.dataset_root), "output_dir": str(args.output_dir)},
            "training_contract": {
                "initialization": "from_scratch",
                "checkpoint_input": None,
                "encoder_trainable": True,
                "decoder_trainable": True,
                "expected_frozen_parameter_tensors": 0,
            },
            "official_args": {
                key: (str(value) if isinstance(value, torch.device) else value)
                for key, value in vars(official_args).items()
                if key != "device_ids"
            },
            "official_preset": asdict(preset),
            "source_note": (
                "StageC end-to-end architecture method screening."
                if args.protocol_class == "method_screening"
                else (
                    "StageC standardized mechanism-control carrier calibration."
                    if args.protocol_class == "mechanism_control"
                    else (
                        "B14 prerequisite: PatchTST-derived contextual history encoder plus A6-LBF-r256 operator."
                        if args.encoder_mode == "contextual-patch-transformer"
                        else (
                            "C1 carrier normalization: full-window global anchor plus valid local patch tokens and explicit dropout sites."
                            if args.encoder_mode == "global-anchored-patch-transformer"
                            else (
                                "B14 prerequisite: accepted A6 carrier plus parameter-free hierarchical patch memory."
                                if args.encoder_mode == "hierarchical-patch-memory"
                                else "Clean TimeAlign adapter: official baseline plus A6-LBF-r256 unified carrier."
                            )
                        )
                    )
                )
            ),
            "a6_lbf_auxiliary_policy": (
                "Prefix-native learned-basis readouts disable TimeAlign future reconstruction/alignment branches "
                "and train with prediction loss only."
                if args.readout_mode in PREFIX_READOUT_MODES
                else "Official TimeAlign keeps the inherited reconstruction/alignment objective."
            ),
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
        f"target_horizons={args.target_horizons} encoder_mode={args.encoder_mode} "
        f"readout_mode={args.readout_mode} output_dir={args.output_dir}",
        flush=True,
    )
    model, training_rows, last_state, best_state = train(args, official_args)
    write_csv(args.output_dir / "training_log.csv", training_rows)
    torch.save(model.state_dict(), args.output_dir / "checkpoint.pt")
    dump_json(args.output_dir / "model_diagnostics.json", model_diagnostics(model))
    if args.readout_mode in {
        "learned-basis-forecast-operator",
        *TimeAlign.PLGO_PAF_READOUTS,
        *TimeAlign.JAPO_READOUTS,
    }:
        patch_data, patch_loader = data_provider(official_args, "val")
        del patch_data
        patch_payload, patch_rows, atom_patch_jacobian = patch_interface_diagnostics(
            model,
            patch_loader,
            official_args,
            max_batches=args.patch_diagnostic_batches,
        )
        dump_json(args.output_dir / "patch_diagnostics.json", patch_payload)
        write_csv(
            args.output_dir / "patch_diagnostics_by_patch.csv",
            patch_rows,
        )
        if atom_patch_jacobian is not None:
            np.savez_compressed(
                args.output_dir / "atom_patch_jacobian_norm.npz",
                norm=atom_patch_jacobian,
            )

    if args.final_evaluation_split == "none":
        print(f"run_done output_dir={args.output_dir} evaluation_split=none", flush=True)
        return

    evaluation_data, evaluation_loader = data_provider(
        official_args,
        args.final_evaluation_split,
    )
    del evaluation_data
    if args.evaluate_dual_checkpoints:
        torch.save(last_state, args.output_dir / "checkpoint_last.pt")
        torch.save(best_state, args.output_dir / "checkpoint_best_val.pt")
        evaluations: dict[
            str,
            tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray],
        ] = {}
        for file_label, policy_label, state in (
            ("last", "official-last", last_state),
            ("best_val", "best-val", best_state),
        ):
            model.load_state_dict(state)
            evaluation = evaluate(
                model,
                evaluation_loader,
                official_args,
                args.evaluation_horizons,
                max_batches=args.max_eval_batches,
                is_training_flag=args.official_test_mode,
            )
            checkpoint_rows, checkpoint_segments, _preds, _trues = evaluation
            annotate_evaluation_rows(checkpoint_rows, args, policy_label)
            annotate_evaluation_rows(checkpoint_segments, args, policy_label)
            write_csv(
                args.output_dir / f"metrics_{file_label}_by_target_horizon.csv",
                checkpoint_rows,
            )
            write_csv(
                args.output_dir / f"metrics_{file_label}_by_segment.csv",
                checkpoint_segments,
            )
            evaluations[file_label] = evaluation
        selected_label = "last" if args.checkpoint_policy == "official-last" else "best_val"
        main_rows, segment_metric_rows, preds, trues = evaluations[selected_label]
        selected_state = last_state if selected_label == "last" else best_state
        model.load_state_dict(selected_state)
    else:
        main_rows, segment_metric_rows, preds, trues = evaluate(
            model,
            evaluation_loader,
            official_args,
            args.evaluation_horizons,
            max_batches=args.max_eval_batches,
            is_training_flag=args.official_test_mode,
        )
        annotate_evaluation_rows(main_rows, args, args.checkpoint_policy)
        annotate_evaluation_rows(segment_metric_rows, args, args.checkpoint_policy)
    write_csv(args.output_dir / "metrics_by_target_horizon.csv", main_rows)
    write_csv(args.output_dir / "metrics_by_segment.csv", segment_metric_rows)
    if getattr(args, "save_predictions", True):
        np.savez_compressed(
            args.output_dir / f"predictions_{args.final_evaluation_split}.npz",
            pred=preds,
            true=trues,
        )
    print(
        f"run_done output_dir={args.output_dir} "
        f"evaluation_split={args.final_evaluation_split}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean FATST adapter for official TimeAlign and A6-LBF.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(OFFICIAL_PRESETS), required=True)
    parser.add_argument("--mode", choices=["fixed", "unified"], required=True)
    parser.add_argument("--seq-len", type=int, default=720)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--target-horizons", type=parse_horizons, required=True)
    parser.add_argument("--validation-horizons", type=parse_horizons, default=None)
    parser.add_argument("--evaluation-horizons", type=parse_horizons, default=None)
    parser.add_argument("--segment-horizons", type=parse_horizons, default=None)
    parser.add_argument(
        "--evaluation-prefix-mode",
        choices=["native", "full-crop"],
        default="native",
    )
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument(
        "--encoder-mode",
        choices=[
            "timealign-token-mlp",
            "contextual-patch-transformer",
            "global-anchored-patch-transformer",
            "hierarchical-patch-memory",
        ],
        default="timealign-token-mlp",
    )
    parser.add_argument("--history-patch-len", type=int, default=16)
    parser.add_argument("--history-patch-stride", type=int, default=8)
    parser.add_argument("--history-d-model", type=int, default=128)
    parser.add_argument("--history-n-heads", type=int, default=16)
    parser.add_argument("--history-d-ff", type=int, default=256)
    parser.add_argument("--history-e-layers", type=int, default=3)
    parser.add_argument("--history-dropout", type=float, default=0.2)
    parser.add_argument("--history-token-dropout", type=float, default=0.0)
    parser.add_argument("--history-attn-dropout", type=float, default=0.0)
    parser.add_argument("--history-attn-residual-dropout", type=float, default=0.1)
    parser.add_argument("--history-ffn-dropout", type=float, default=0.1)
    parser.add_argument("--history-ffn-residual-dropout", type=float, default=0.1)
    parser.add_argument(
        "--history-res-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--legacy-patch-num", type=int, default=None)
    parser.add_argument("--legacy-d-model", type=int, default=None)
    parser.add_argument("--legacy-d-ff", type=int, default=None)
    parser.add_argument("--legacy-dropout", type=float, default=None)
    parser.add_argument("--legacy-layer-norm", type=int, choices=[0, 1], default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--w-recon", type=float, default=1.0)
    parser.add_argument("--w-align", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--enable-early-stopping",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--official-test-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--final-evaluation-split",
        choices=["train", "val", "test", "none"],
        default="test",
    )
    parser.add_argument(
        "--protocol-class",
        choices=[
            "source",
            "mechanism_control",
            "method_screening",
            "native_external",
        ],
        default="source",
    )
    parser.add_argument("--protocol-profile", default="")
    parser.add_argument("--profile-hash", default="")
    parser.add_argument("--checkpoint-policy", choices=["official-last", "best-val"], default="official-last")
    parser.add_argument(
        "--evaluate-dual-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--readout-mode",
        choices=sorted({"official", *PREFIX_READOUT_MODES}),
        default="official",
    )
    parser.add_argument("--basis-rank", type=int, default=256)
    parser.add_argument("--stage-token-dim", type=int, default=32)
    parser.add_argument("--stage-field-rank", type=int, default=32)
    parser.add_argument("--stage-gate-init", type=float, default=-5.0)
    parser.add_argument("--basis-field-window-len", type=int, default=96)
    parser.add_argument("--basis-field-stride", type=int, default=48)
    parser.add_argument("--basis-field-rank", type=int, default=32)
    parser.add_argument("--basis-field-tau", type=float, default=1.0)
    parser.add_argument("--basis-field-gate-init", type=float, default=-5.0)
    parser.add_argument("--stbo-tile-len", type=int, default=48)
    parser.add_argument("--stbo-rank", type=int, default=16)
    parser.add_argument("--stbo-bank-count", type=int, default=4)
    parser.add_argument("--stbo-basis-init-std", type=float, default=0.0)
    parser.add_argument("--pmfo-state-dim", type=int, default=32)
    parser.add_argument("--pmfo-dense-hidden-dim", type=int, default=144)
    parser.add_argument("--plgo-global-rank", type=int, default=16)
    parser.add_argument("--plgo-latent-width", type=int, default=256)
    parser.add_argument("--plgo-permutation-seed", type=int, default=7101)
    parser.add_argument("--plgo-random-descriptor-seed", type=int, default=7102)
    parser.add_argument("--japo-expert-count", type=int, default=2)
    parser.add_argument("--japo-expert-rank", type=int, default=256)
    parser.add_argument("--japo-router-width", type=int, default=32)
    parser.add_argument("--japo-router-output-init-std", type=float, default=0.01)
    parser.add_argument("--patch-diagnostic-batches", type=int, default=8)
    parser.add_argument(
        "--pred-loss-mode",
        choices=["full", "multi-prefix"],
        default="full",
    )
    parser.add_argument(
        "--allow-archived-research-modes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt in to historical modes that are not active StageC research paths.",
    )
    args = parser.parse_args()
    if not args.allow_archived_research_modes:
        active_values = {
            "mode": args.mode,
            "encoder_mode": args.encoder_mode,
            "readout_mode": args.readout_mode,
            "pred_loss_mode": args.pred_loss_mode,
        }
        inactive = {
            name: value
            for name, value in active_values.items()
            if (
                value != ACTIVE_STAGE_C_CONTRACT[name]
                and not (
                    name == "readout_mode"
                    and value in STAGE_C_ACTIVE_READOUTS
                )
            )
        }
        if inactive:
            formatted = ", ".join(
                f"{name}={value!r}" for name, value in inactive.items()
            )
            raise ValueError(
                "Inactive pre-StageC research mode requested: "
                f"{formatted}. The default active contract is "
                f"{ACTIVE_STAGE_C_CONTRACT}. Use "
                "--allow-archived-research-modes only for an explicitly "
                "authorized historical reproduction."
            )
    args.validation_horizons = args.validation_horizons or list(args.target_horizons)
    args.evaluation_horizons = args.evaluation_horizons or list(args.target_horizons)
    args.segment_horizons = args.segment_horizons or list(
        args.evaluation_horizons
    )
    if max(args.target_horizons) > args.pred_len:
        raise ValueError("target horizons cannot exceed pred_len")
    if max(args.validation_horizons) > args.pred_len:
        raise ValueError("validation horizons cannot exceed pred_len")
    if max(args.evaluation_horizons) > args.pred_len:
        raise ValueError("evaluation horizons cannot exceed pred_len")
    if max(args.segment_horizons) > args.pred_len:
        raise ValueError("segment horizons cannot exceed pred_len")
    if args.mode == "fixed" and args.target_horizons != [args.pred_len]:
        raise ValueError("fixed mode expects target_horizons == [pred_len]")
    if args.mode == "unified" and args.pred_len != 720:
        raise ValueError("unified mode currently expects pred_len=720")
    if args.w_recon < 0.0:
        raise ValueError("w_recon must be non-negative")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient accumulation steps must be positive")
    if args.patience <= 0:
        raise ValueError("patience must be positive")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("early stopping min delta must be non-negative")
    if args.enable_early_stopping and args.checkpoint_policy != "best-val":
        raise ValueError("early stopping requires checkpoint_policy=best-val")
    if args.protocol_class in {"mechanism_control", "method_screening"}:
        if not args.protocol_profile or not args.profile_hash:
            raise ValueError(
                "controlled protocols require protocol profile and profile hash"
            )
    if args.protocol_class in {"mechanism_control", "method_screening"}:
        if args.final_evaluation_split == "test":
            raise ValueError(
                f"{args.protocol_class} cannot evaluate the test split"
            )
    if args.history_patch_len <= 0 or args.history_patch_stride <= 0:
        raise ValueError("history patch length and stride must be positive")
    if args.history_patch_len > args.seq_len + args.history_patch_stride:
        raise ValueError("history patch length cannot exceed padded sequence length")
    if args.history_d_model <= 0 or args.history_n_heads <= 0:
        raise ValueError("history d_model and n_heads must be positive")
    if args.history_d_model % args.history_n_heads != 0:
        raise ValueError("history d_model must be divisible by history n_heads")
    if args.history_d_ff <= 0 or args.history_e_layers <= 0:
        raise ValueError("history d_ff and e_layers must be positive")
    if not 0.0 <= args.history_dropout < 1.0:
        raise ValueError("history dropout must be in [0, 1)")
    if not 0.0 <= args.history_attn_dropout < 1.0:
        raise ValueError("history attention dropout must be in [0, 1)")
    named_history_dropouts = {
        "history token dropout": args.history_token_dropout,
        "history attention residual dropout": args.history_attn_residual_dropout,
        "history FFN dropout": args.history_ffn_dropout,
        "history FFN residual dropout": args.history_ffn_residual_dropout,
    }
    for name, value in named_history_dropouts.items():
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")
    legacy_overrides = (
        args.legacy_patch_num,
        args.legacy_d_model,
        args.legacy_d_ff,
        args.legacy_dropout,
        args.legacy_layer_norm,
    )
    if any(value is not None for value in legacy_overrides):
        if args.encoder_mode not in {
            "timealign-token-mlp",
            "hierarchical-patch-memory",
        }:
            raise ValueError("legacy encoder overrides require a token-MLP encoder mode")
        if args.readout_mode not in STAGE_C_ACTIVE_READOUTS:
            raise ValueError(
                "legacy encoder overrides are restricted to active StageC readouts"
            )
    if args.legacy_patch_num is not None:
        if args.legacy_patch_num <= 0:
            raise ValueError("legacy patch_num must be positive")
        if args.seq_len % args.legacy_patch_num != 0:
            raise ValueError("legacy patch_num must divide seq_len")
    if args.legacy_d_model is not None and args.legacy_d_model <= 0:
        raise ValueError("legacy d_model must be positive")
    if args.legacy_d_model is not None and args.legacy_d_model % 2 != 0:
        raise ValueError("legacy d_model must be even for sinusoidal positional encoding")
    if args.legacy_d_ff is not None and args.legacy_d_ff <= 0:
        raise ValueError("legacy d_ff must be positive")
    if args.legacy_dropout is not None and not 0.0 <= args.legacy_dropout < 1.0:
        raise ValueError("legacy dropout must be in [0, 1)")
    if args.learning_rate is not None and args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    if args.w_align is not None and args.w_align < 0.0:
        raise ValueError("w_align must be non-negative")
    if args.basis_rank <= 0:
        raise ValueError("basis_rank must be positive")
    if args.pmfo_state_dim <= 0:
        raise ValueError("pmfo_state_dim must be positive")
    if args.pmfo_dense_hidden_dim <= 0:
        raise ValueError("pmfo_dense_hidden_dim must be positive")
    if args.plgo_global_rank <= 0 or args.plgo_global_rank > args.pred_len:
        raise ValueError("plgo_global_rank must lie in [1, pred_len]")
    if args.plgo_latent_width <= 0:
        raise ValueError("plgo_latent_width must be positive")
    if args.japo_expert_count != 2:
        raise ValueError("SC1-JAPO Step7A requires japo_expert_count=2")
    if args.japo_expert_rank != 256:
        raise ValueError("SC1-JAPO Step7A requires japo_expert_rank=256")
    if args.japo_router_width != 32:
        raise ValueError("SC1-JAPO Step7A requires japo_router_width=32")
    if args.japo_router_output_init_std != 0.01:
        raise ValueError(
            "SC1-JAPO Step7A requires japo_router_output_init_std=0.01"
        )
    if args.patch_diagnostic_batches <= 0:
        raise ValueError("patch_diagnostic_batches must be positive")
    if args.stage_token_dim <= 0:
        raise ValueError("stage_token_dim must be positive")
    if args.stage_field_rank <= 0:
        raise ValueError("stage_field_rank must be positive")
    if args.basis_field_window_len <= 0:
        raise ValueError("basis_field_window_len must be positive")
    if args.basis_field_stride <= 0:
        raise ValueError("basis_field_stride must be positive")
    if args.basis_field_rank <= 0:
        raise ValueError("basis_field_rank must be positive")
    if args.basis_field_tau <= 0.0:
        raise ValueError("basis_field_tau must be positive")
    if args.stbo_tile_len <= 0:
        raise ValueError("stbo_tile_len must be positive")
    if args.pred_len % args.stbo_tile_len != 0:
        raise ValueError("stbo_tile_len must divide pred_len")
    if args.stbo_rank <= 0 or args.stbo_rank > args.stbo_tile_len:
        raise ValueError("stbo_rank must be in [1, stbo_tile_len]")
    if args.stbo_bank_count < 2:
        raise ValueError("stbo_bank_count must be at least 2")
    if args.stbo_basis_init_std < 0.0:
        raise ValueError("stbo_basis_init_std must be non-negative")
    if args.stbo_basis_init_std == 0.0:
        args.stbo_basis_init_std = args.stbo_rank ** -0.5
    if args.readout_mode in PREFIX_READOUT_MODES and args.mode != "unified":
        raise ValueError("Prefix-native learned-basis readouts require --mode unified --pred-len 720")
    if args.encoder_mode in {
        "contextual-patch-transformer",
        "global-anchored-patch-transformer",
    }:
        if args.readout_mode != "learned-basis-forecast-operator":
            raise ValueError(
                "contextual patch encoders currently require "
                "readout-mode=learned-basis-forecast-operator"
            )
        if args.mode != "unified" or args.pred_len != 720:
            raise ValueError("contextual patch encoders require unified A6-LBF")
    if args.encoder_mode == "hierarchical-patch-memory":
        if args.readout_mode != "learned-basis-forecast-operator":
            raise ValueError(
                "hierarchical-patch-memory currently requires "
                "readout-mode=learned-basis-forecast-operator"
            )
        if args.mode != "unified" or args.pred_len != 720:
            raise ValueError("hierarchical-patch-memory is a unified A6 carrier prerequisite")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

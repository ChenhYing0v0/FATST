#!/usr/bin/env python3
"""Audit per-scope shared-field gradients at one PCSD/SIFF checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from layers.PCC import prefix_measure  # noqa: E402
from models import TimeAlign  # noqa: E402
from train_repo import initialization_contract  # noqa: E402

from evaluate_stage_c_pcsd_cf_checkpoint import (  # noqa: E402
    load_model,
    sequential_loader,
)


SCALES = (1, 48, 144, 360, 720)
SHARED_NAMES = (
    "mode_weight",
    "mode_bias",
    "identity_synthesis",
    "nonlinear_synthesis",
    "temporal_bias",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def shared_parameters(model: TimeAlign.Model) -> list[torch.nn.Parameter]:
    readout = model.pcsd_readout
    return [getattr(readout, name) for name in SHARED_NAMES]


def scope_gradient_payload(
    model: TimeAlign.Model,
    batch_x: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    output, _recon, _align, details = model(
        batch_x,
        target,
        is_training=False,
        target_prefix=720,
        return_pcsd_training_details=True,
    )
    arms = details["arm_forecasts"]
    policy = details["policy"]
    measure = prefix_measure(
        720,
        device=target.device,
        dtype=target.dtype,
    )
    parameters = shared_parameters(model)
    gradient_vectors = []
    scope_losses = []
    scope_norms = []
    for scope_index in range(len(SCALES)):
        error = (arms[:, :, scope_index, :] - target.permute(0, 2, 1)).abs()
        loss = (error * measure.view(1, 1, -1)).sum(dim=-1).mean()
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=scope_index + 1 < len(SCALES),
            allow_unused=False,
        )
        vector = torch.cat([gradient.reshape(-1) for gradient in gradients])
        gradient_vectors.append(vector.detach())
        scope_losses.append(float(loss.detach().cpu()))
        scope_norms.append(float(vector.norm().detach().cpu()))

    pair_rows = []
    cosine_values = []
    for left in range(len(SCALES)):
        for right in range(left + 1, len(SCALES)):
            denominator = gradient_vectors[left].norm() * gradient_vectors[right].norm()
            cosine = float(
                (
                    torch.dot(gradient_vectors[left], gradient_vectors[right])
                    / denominator.clamp_min(1e-12)
                )
                .detach()
                .cpu()
            )
            cosine_values.append(cosine)
            pair_rows.append(
                {
                    "left_scope": SCALES[left],
                    "right_scope": SCALES[right],
                    "cosine": cosine,
                }
            )

    all_finite = bool(
        torch.isfinite(output).all()
        and torch.isfinite(arms).all()
        and torch.isfinite(policy).all()
        and all(torch.isfinite(vector).all() for vector in gradient_vectors)
    )
    all_nonzero = all(value > 0.0 for value in scope_norms)
    return {
        "snapshot": "best-val-h720",
        "batch": "first-sequential-train-row",
        "rows": int(batch_x.shape[0]),
        "shared_parameter_names": list(SHARED_NAMES),
        "shared_parameter_count": sum(parameter.numel() for parameter in parameters),
        "scope_losses": dict(zip((str(value) for value in SCALES), scope_losses)),
        "scope_gradient_norms": dict(
            zip((str(value) for value in SCALES), scope_norms)
        ),
        "pairwise_cosines": pair_rows,
        "pairwise_cosine_mean": sum(cosine_values) / len(cosine_values),
        "pairwise_cosine_min": min(cosine_values),
        "pairwise_cosine_max": max(cosine_values),
        "all_finite": all_finite,
        "all_scope_gradients_nonzero": all_nonzero,
        "gradient_surgery_applied": False,
        "pass": all_finite and all_nonzero,
    }


def synthetic_model() -> TimeAlign.Model:
    config = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode="pcsd-coupling-field",
        e_layers=2,
        patch_num=12,
        d_model=64,
        d_ff=128,
        dropout=0.1,
        pos=1,
        layer_norm=1,
        enc_in=7,
        pcsd_coordinate_dim=4,
        pcsd_mode_rank=256,
        pcsd_policy_history_dim=32,
        pcsd_policy_hidden_dim=64,
        pcsd_policy_mode="direct",
        pcsd_fixed_scale=720,
        pcsd_partition="canonical",
        pcsd_partition_seed=15101,
        pcsd_group_chunk_size=64,
        pcsd_target_chunk_size=128,
    )
    torch.manual_seed(2021)
    return TimeAlign.Model(config).float().eval()


def main() -> None:
    args = parse_args()
    if args.rows <= 0:
        raise ValueError("rows must be positive")
    if args.synthetic_smoke:
        model = synthetic_model()
        x = torch.randn(1, 720, 7)
        target = torch.randn(1, 720, 7)
        payload = scope_gradient_payload(model, x, target)
        if not payload["pass"] or len(payload["pairwise_cosines"]) != 10:
            raise RuntimeError(f"PCC gradient synthetic smoke failed: {payload}")
        print("pcc_shared_gradient_synthetic_smoke=pass")
        return
    if args.run_dir is None:
        raise ValueError("run-dir is required outside synthetic smoke")

    device = torch.device(args.device)
    model, config, official_args = load_model(args.run_dir, device)
    adapter = config["adapter"]
    if (
        adapter.get("pcc_objective_mode") == "off"
        or adapter.get("readout_mode") not in TimeAlign.COUPLING_READOUTS
    ):
        raise ValueError(
            "gradient audit requires a trained PCSD/SIFF coupling run"
        )
    loader = sequential_loader(official_args, "train")
    batch_x, batch_y, _batch_x_mark, _batch_y_mark = next(iter(loader))
    batch_x = batch_x[: args.rows].float().to(device)
    target = batch_y[: args.rows, -720:, :].float().to(device)
    payload = scope_gradient_payload(model, batch_x, target)
    payload.update(
        {
            "candidate": "SC1-SIFF-v1/SC2-MCCA-v1",
            "dataset": adapter["dataset"],
            "objective_mode": adapter["pcc_objective_mode"],
            "checkpoint_pcsd_parameter_hash": initialization_contract(model).get(
                "pcsd_initialization_hash",
                "",
            ),
        }
    )
    output = args.run_dir / "pcc_shared_gradient_diagnostics.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not payload["pass"] or not math.isfinite(payload["pairwise_cosine_mean"]):
        raise RuntimeError(f"PCC shared gradient audit failed: {payload}")
    print(
        f"pcc_shared_gradient=pass dataset={payload['dataset']} "
        f"mode={payload['objective_mode']}"
    )


if __name__ == "__main__":
    main()

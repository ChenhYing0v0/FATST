#!/usr/bin/env python3
"""Local smoke checks for Step 7B dense evaluation and readout routing."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

import train_repo  # noqa: E402
import analyze_stage_c_step7b_pmfo_rct as step7b_analyzer  # noqa: E402
from models import TimeAlign  # noqa: E402


READOUTS = (
    "learned-basis-forecast-operator",
    "dense-mlp-matched",
    "pmfo-rct-no-transition",
    "pmfo-rct-no-conservation",
    "pmfo-rct",
)
HORIZONS = [1, 48, 96, 192, 336, 720]


def model_config(readout_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode=readout_mode,
        e_layers=2,
        patch_num=24,
        d_model=32,
        d_ff=64,
        dropout=0.1,
        pos=1,
        layer_norm=1,
        enc_in=7,
        basis_rank=256,
        pmfo_state_dim=32,
        pmfo_dense_hidden_dim=144,
    )


def evaluation_args(prefix_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        features="M",
        device=torch.device("cpu"),
        pred_len=720,
        segment_horizons=[48, 96, 192, 336, 720],
        evaluation_prefix_mode=prefix_mode,
    )


def check_metric_equivalence(generator: np.random.Generator) -> float:
    pred = generator.normal(size=(3, 720, 7)).astype(np.float32)
    true = generator.normal(size=(3, 720, 7)).astype(np.float32)
    rows = train_repo.metric_rows(pred, true, HORIZONS)
    errors = []
    for row in rows:
        horizon = int(row["target_horizon"])
        errors.extend(
            [
                abs(row["mse"] - train_repo.MSE(pred[:, :horizon], true[:, :horizon])),
                abs(row["mae"] - train_repo.MAE(pred[:, :horizon], true[:, :horizon])),
            ]
        )
    maximum = max(errors)
    if maximum > 1e-6:
        raise AssertionError(f"cumulative metric mismatch: {maximum}")
    return maximum


def check_full_crop_equivalence(generator: torch.Generator) -> float:
    batch_x = torch.randn(2, 720, 7, generator=generator)
    batch_y = torch.randn(2, 720, 7, generator=generator)
    marks = torch.zeros(2, 720, 1)
    loader = [(batch_x, batch_y, marks, marks)]
    maximum = 0.0
    for readout in READOUTS:
        model = TimeAlign.Model(model_config(readout)).float().eval()
        native = train_repo.evaluate_a6_lbf(
            model,
            loader,
            evaluation_args("native"),
            HORIZONS,
            max_batches=0,
            is_training_flag=False,
        )[0]
        cropped = train_repo.evaluate_a6_lbf(
            model,
            loader,
            evaluation_args("full-crop"),
            HORIZONS,
            max_batches=0,
            is_training_flag=False,
        )[0]
        for native_row, cropped_row in zip(native, cropped, strict=True):
            maximum = max(
                maximum,
                abs(float(native_row["mse"]) - float(cropped_row["mse"])),
                abs(float(native_row["mae"]) - float(cropped_row["mae"])),
            )
    if maximum > 1e-6:
        raise AssertionError(f"native/full-crop metric mismatch: {maximum}")
    return maximum


def check_cli_contract() -> None:
    original = sys.argv
    sys.argv = [
        "train_repo.py",
        "--dataset-root",
        "/tmp/fatst-dataset-smoke",
        "--dataset",
        "ETTm1",
        "--mode",
        "unified",
        "--pred-len",
        "720",
        "--target-horizons",
        "720",
        "--validation-horizons",
        "720",
        "--evaluation-horizons",
        "1,48,720",
        "--segment-horizons",
        "48,720",
        "--evaluation-prefix-mode",
        "full-crop",
        "--run-name",
        "step7b_cli_smoke",
        "--output-dir",
        "/tmp/fatst-step7b-cli-smoke",
        "--readout-mode",
        "pmfo-rct",
        "--pred-loss-mode",
        "full",
        "--protocol-class",
        "method_screening",
        "--protocol-profile",
        "stage_c_pmfo_rct_step7b_v1",
        "--profile-hash",
        "smoke-hash",
        "--final-evaluation-split",
        "test",
        "--checkpoint-policy",
        "best-val",
        "--legacy-patch-num",
        "24",
        "--legacy-d-model",
        "32",
        "--legacy-d-ff",
        "64",
        "--no-save-predictions",
    ]
    try:
        args = train_repo.parse_args()
    finally:
        sys.argv = original
    if args.protocol_class != "method_screening":
        raise AssertionError("method_screening protocol was not retained")
    if args.evaluation_prefix_mode != "full-crop" or args.save_predictions:
        raise AssertionError("Step 7B evaluation CLI contract was not retained")


def check_gate_logic() -> None:
    arm_auc = {
        "a6": 1.0,
        "dense_mlp_matched": 0.995,
        "pmfo_no_transition": 0.993,
        "pmfo_no_conservation": 0.994,
        "pmfo_rct": 0.98,
    }
    rows = [
        {
            "dataset": dataset,
            "arm": arm,
            "status": "ok",
            "dense_mse_auc": auc,
            "invariant_pass": True,
        }
        for dataset in step7b_analyzer.DATASETS
        for arm, auc in arm_auc.items()
    ]
    gate = step7b_analyzer.decide_gate(rows)
    if gate["decision"] != "partial_pass":
        raise AssertionError(f"unexpected synthetic gate: {gate}")


def main() -> None:
    np_generator = np.random.default_rng(20260713)
    torch_generator = torch.Generator().manual_seed(20260713)
    metric_gap = check_metric_equivalence(np_generator)
    crop_gap = check_full_crop_equivalence(torch_generator)
    check_cli_contract()
    check_gate_logic()
    print(
        "stage_c_step7b_local=pass "
        f"metric_gap={metric_gap:.3e} full_crop_gap={crop_gap:.3e}"
    )


if __name__ == "__main__":
    main()

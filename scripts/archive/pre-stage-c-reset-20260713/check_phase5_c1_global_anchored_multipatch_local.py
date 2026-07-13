#!/usr/bin/env python3
"""Local contract and one-batch smoke for the C1 global-anchored carrier."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from models.TimeAlign import Model  # noqa: E402
from train_repo import model_diagnostics  # noqa: E402


SCALES = {"p16s8": (16, 8, 89), "p48s24": (48, 24, 29)}


def make_config(patch_len: int, stride: int, enc_in: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        patch_num=1,
        d_model=256,
        n_heads=8,
        e_layers=1,
        d_ff=512,
        dropout=0.1,
        layer_norm=1,
        pos=1,
        enc_in=enc_in,
        readout_mode="learned-basis-forecast-operator",
        encoder_mode="global-anchored-patch-transformer",
        history_patch_len=patch_len,
        history_patch_stride=stride,
        history_token_dropout=0.0,
        history_attn_dropout=0.0,
        history_attn_residual_dropout=0.1,
        history_ffn_dropout=0.1,
        history_ffn_residual_dropout=0.1,
        history_res_attention=True,
        basis_rank=256,
        target_horizons=[96, 192, 336, 720],
        local_margin=0.0,
        global_margin=0.0,
        loc=1,
        glo=1,
    )


def finite_gradient(parameter: torch.nn.Parameter, name: str) -> None:
    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
        raise AssertionError(f"missing or non-finite gradient: {name}")


def structural_checks() -> None:
    torch.manual_seed(2021)
    torch.set_num_threads(1)
    x = torch.randn(2, 720, 3)
    y = torch.randn(2, 720, 3)

    for scale, (patch_len, stride, expected_patches) in SCALES.items():
        config = make_config(patch_len, stride)
        model = Model(config).eval()
        global_memory = model.encode_history(x)
        local_memory = model.encode_retrieval_memory(x)
        assert global_memory.shape == (2, 3, 1, 256), scale
        assert local_memory.shape == (2, 3, expected_patches, 256), scale

        encoder = model.history_encoder
        assert encoder.token_dropout.p == 0.0
        assert encoder.layers[0].attention.attn_dropout.p == 0.0
        assert encoder.layers[0].attention_residual_dropout.p == 0.1
        assert encoder.layers[0].feed_forward[2].p == 0.1
        assert encoder.layers[0].feed_forward_residual_dropout.p == 0.1

        with torch.no_grad():
            prefix = model(x, y, is_training=False, target_prefix=96)[0]
            full = model(x, y, is_training=False, target_prefix=720)[0]
        torch.testing.assert_close(prefix, full[:, :96, :], rtol=0.0, atol=0.0)

        reloaded = Model(config).eval()
        reloaded.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            actual = reloaded(x, y, is_training=False, target_prefix=720)[0]
        torch.testing.assert_close(actual, full, rtol=0.0, atol=0.0)

        train_model = Model(config).train()
        output = train_model(x, y, is_training=True, target_prefix=96)[0]
        output.square().mean().backward()
        finite_gradient(
            train_model.history_encoder.global_projection.weight,
            f"{scale}.global_projection",
        )
        finite_gradient(
            train_model.history_encoder.local_projection.weight,
            f"{scale}.local_projection",
        )
        finite_gradient(
            train_model.history_encoder.layers[0].attention.query.weight,
            f"{scale}.attention_query",
        )
        finite_gradient(
            train_model.history_encoder.layers[0].feed_forward[0].weight,
            f"{scale}.feed_forward",
        )

        diagnostics = model_diagnostics(model)
        assert diagnostics["patch_num"] == 1
        assert diagnostics["history_local_patch_num"] == expected_patches
        assert diagnostics["unused_proj_x_parameters"] == 0
        assert diagnostics["inactive_or_other_parameters"] == 0


def end_to_end_smoke() -> None:
    dataset_root = REPO_ROOT.parent / "datasets"
    dataset_file = dataset_root / "ETT-small" / "ETTm1.csv"
    if not dataset_file.exists():
        raise FileNotFoundError(f"local smoke requires {dataset_file}")

    with tempfile.TemporaryDirectory(prefix="fatst-c1-smoke-") as temp_dir:
        output_dir = Path(temp_dir) / "run"
        command = [
            sys.executable,
            str(TIMEALIGN_ROOT / "train_repo.py"),
            "--dataset-root",
            str(dataset_root),
            "--dataset",
            "ETTm1",
            "--mode",
            "unified",
            "--seq-len",
            "720",
            "--pred-len",
            "720",
            "--target-horizons",
            "96,192,336,720",
            "--batch-size",
            "2",
            "--epochs",
            "1",
            "--seed",
            "2021",
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
            "--num-workers",
            "0",
            "--run-name",
            "c1_local_smoke",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--checkpoint-policy",
            "official-last",
            "--evaluate-dual-checkpoints",
            "--encoder-mode",
            "global-anchored-patch-transformer",
            "--history-patch-len",
            "48",
            "--history-patch-stride",
            "24",
            "--history-d-model",
            "256",
            "--history-n-heads",
            "8",
            "--history-d-ff",
            "512",
            "--history-e-layers",
            "1",
            "--history-token-dropout",
            "0.0",
            "--history-attn-dropout",
            "0.0",
            "--history-attn-residual-dropout",
            "0.1",
            "--history-ffn-dropout",
            "0.1",
            "--history-ffn-residual-dropout",
            "0.1",
            "--readout-mode",
            "learned-basis-forecast-operator",
            "--basis-rank",
            "256",
            "--pred-loss-mode",
            "multi-prefix",
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)

        required = (
            "checkpoint_last.pt",
            "checkpoint_best_val.pt",
            "metrics_last_by_target_horizon.csv",
            "metrics_best_val_by_target_horizon.csv",
            "model_diagnostics.json",
            "effective_config.json",
        )
        missing = [name for name in required if not (output_dir / name).exists()]
        if missing:
            raise AssertionError(f"missing C1 smoke artifacts: {missing}")
        diagnostics = json.loads((output_dir / "model_diagnostics.json").read_text())
        assert diagnostics["history_local_patch_num"] == 29
        assert diagnostics["history_token_dropout_p"] == 0.0
        assert diagnostics["history_attn_dropout_p"] == 0.0
        assert diagnostics["history_attn_residual_dropout_p"] == 0.1
        assert diagnostics["history_ffn_dropout_p"] == 0.1
        assert diagnostics["history_ffn_residual_dropout_p"] == 0.1


def main() -> None:
    structural_checks()
    end_to_end_smoke()
    print("phase5_c1_global_anchored_multipatch_local_check=pass")


if __name__ == "__main__":
    main()

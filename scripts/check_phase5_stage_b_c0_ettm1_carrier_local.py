#!/usr/bin/env python3
"""Local structural and dual-checkpoint smoke for the StageB C0 Encoder control."""

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


ARMS = {
    "p1_d256_f256_d09": (1, 256, 256, 0.9, 699_600),
    "p1_d384_f96_d09": (1, 384, 96, 0.9, 710_416),
    "p5_d52_f256_d09": (5, 52, 256, 0.9, 313_468),
    "p5_d52_f2048_d09": (5, 52, 2048, 0.9, 689_788),
    "p1_d256_f256_d02": (1, 256, 256, 0.2, 699_600),
    "p5_d52_f2048_d02": (5, 52, 2048, 0.2, 689_788),
}


def make_config(patch_num: int, d_model: int, d_ff: int, dropout: float) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        patch_num=patch_num,
        d_model=d_model,
        n_heads=8,
        e_layers=2,
        d_ff=d_ff,
        dropout=dropout,
        layer_norm=1,
        pos=1,
        enc_in=7,
        readout_mode="learned-basis-forecast-operator",
        encoder_mode="timealign-token-mlp",
        basis_rank=256,
        target_horizons=[96, 192, 336, 720],
        local_margin=0.5,
        global_margin=0.0,
        loc=1,
        glo=1,
    )


def structural_checks() -> None:
    torch.manual_seed(2021)
    torch.set_num_threads(1)
    x = torch.randn(2, 720, 7)
    y = torch.randn(2, 720, 7)
    for arm, (patch_num, d_model, d_ff, dropout, expected_active) in ARMS.items():
        config = make_config(patch_num, d_model, d_ff, dropout)
        model = Model(config).eval()
        diagnostics = model_diagnostics(model)
        if diagnostics["active_forward_parameters"] != expected_active:
            raise AssertionError(
                f"{arm}: expected {expected_active} active parameters, "
                f"got {diagnostics['active_forward_parameters']}"
            )
        memory = model.encode_history(x)
        assert memory.shape == (2, 7, patch_num, d_model)
        patch_len = 720 // patch_num
        boundaries = [index * patch_len for index in range(patch_num + 1)]
        assert boundaries[0] == 0 and boundaries[-1] == 720

        with torch.no_grad():
            prefix = model(x, y, is_training=False, target_prefix=96)[0]
            full = model(x, y, is_training=False, target_prefix=720)[0]
        torch.testing.assert_close(prefix, full[:, :96, :], rtol=0.0, atol=0.0)

        reloaded = Model(config).eval()
        reloaded.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            actual = reloaded(x, y, is_training=False, target_prefix=720)[0]
        torch.testing.assert_close(actual, full, rtol=0.0, atol=0.0)


def end_to_end_dual_checkpoint_smoke() -> None:
    dataset_root = REPO_ROOT.parent / "datasets"
    dataset_file = dataset_root / "ETT-small" / "ETTm1.csv"
    if not dataset_file.exists():
        raise FileNotFoundError(f"local smoke requires {dataset_file}")

    with tempfile.TemporaryDirectory(prefix="fatst-c0-smoke-") as temp_dir:
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
            "c0_local_smoke",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--checkpoint-policy",
            "official-last",
            "--evaluate-dual-checkpoints",
            "--legacy-patch-num",
            "5",
            "--legacy-d-model",
            "52",
            "--legacy-d-ff",
            "256",
            "--legacy-dropout",
            "0.9",
            "--readout-mode",
            "learned-basis-forecast-operator",
            "--basis-rank",
            "256",
            "--pred-loss-mode",
            "multi-prefix",
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)

        required = (
            "checkpoint.pt",
            "checkpoint_last.pt",
            "checkpoint_best_val.pt",
            "metrics_by_target_horizon.csv",
            "metrics_last_by_target_horizon.csv",
            "metrics_best_val_by_target_horizon.csv",
            "model_diagnostics.json",
            "effective_config.json",
        )
        missing = [name for name in required if not (output_dir / name).exists()]
        if missing:
            raise AssertionError(f"missing dual-checkpoint artifacts: {missing}")

        effective = json.loads((output_dir / "effective_config.json").read_text())
        official = effective["official_args"]
        assert (official["patch_num"], official["d_model"], official["d_ff"]) == (
            5,
            52,
            256,
        )
        assert abs(float(official["dropout"]) - 0.9) < 1e-12
        config = make_config(5, 52, 256, 0.9)
        for checkpoint_name in ("checkpoint_last.pt", "checkpoint_best_val.pt"):
            state = torch.load(
                output_dir / checkpoint_name,
                map_location="cpu",
                weights_only=True,
            )
            Model(config).load_state_dict(state, strict=True)


def main() -> None:
    structural_checks()
    end_to_end_dual_checkpoint_smoke()
    print("phase5_stage_b_c0_ettm1_carrier_local_check=pass")


if __name__ == "__main__":
    main()

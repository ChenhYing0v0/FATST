from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from models import TimeAlign  # noqa: E402


B11_MODES = [
    "basis-conditioned-coefficient-field",
    "basis-conditioned-coefficient-field-no-basis",
    "basis-conditioned-coefficient-field-shuffled-basis",
    "basis-conditioned-coefficient-field-constant-slot",
]


def build_config(readout_mode: str, channels: int = 7) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        patch_num=48,
        d_model=32,
        d_ff=32,
        dropout=0.0,
        e_layers=1,
        layer_norm=1,
        pos=1,
        local_margin=0.0,
        global_margin=0.0,
        loc=1,
        glo=1,
        enc_in=channels,
        readout_mode=readout_mode,
        target_horizons=[96, 192, 336, 720],
        basis_rank=256,
        stage_token_dim=32,
        stage_field_rank=32,
        stage_gate_init=-5.0,
        basis_field_window_len=96,
        basis_field_stride=48,
        basis_field_rank=32,
        basis_field_tau=1.0,
        basis_field_gate_init=-5.0,
    )


def make_model(readout_mode: str, seed: int) -> TimeAlign.Model:
    torch.manual_seed(seed)
    return TimeAlign.Model(build_config(readout_mode)).float()


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, tolerance: float) -> float:
    max_abs = float((actual - expected).abs().max().item())
    if max_abs > tolerance:
        raise AssertionError(f"{name} max_abs={max_abs:.6e} exceeds tolerance={tolerance:.6e}")
    return max_abs


def run_checks(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed + 100)
    x = torch.randn(args.batch_size, 720, 7)
    y = torch.randn(args.batch_size, 720, 7)

    a6 = make_model("learned-basis-forecast-operator", args.seed)
    b11 = make_model("basis-conditioned-coefficient-field", args.seed)
    a6.eval()
    b11.eval()

    with torch.no_grad():
        a6_h96, _recon, _align = a6(x, y, is_training=False, target_prefix=96)
        b11_h96, _recon, _align = b11(x, y, is_training=False, target_prefix=96)
        b11_h720, _recon, _align = b11(x, y, is_training=False, target_prefix=720)

    fallback_diff = assert_close("a6_fallback_h96", b11_h96, a6_h96, args.tolerance)
    prefix_diff = assert_close("b11_prefix_h96_vs_h720", b11_h96, b11_h720[:, :96, :], args.tolerance)
    print(f"a6_fallback_h96_max_abs={fallback_diff:.6e}")
    print(f"b11_prefix_h96_vs_h720_max_abs={prefix_diff:.6e}")

    for mode in B11_MODES:
        model = make_model(mode, args.seed)
        model.train()
        output, recon, align = model(x, y, is_training=True, target_prefix=96)
        if recon is not None:
            raise AssertionError(f"{mode} unexpectedly returned future reconstruction")
        loss = output[:, :96, :].pow(2).mean() + align.abs()
        loss.backward()
        if not torch.isfinite(loss):
            raise AssertionError(f"{mode} produced non-finite loss")
        grad = model.learned_temporal_basis.grad
        if grad is None or not torch.isfinite(grad).all():
            raise AssertionError(f"{mode} produced invalid learned_temporal_basis gradient")
        print(f"backward_ok mode={mode} loss={float(loss.detach().item()):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local B11-BCF fallback/prefix/smoke checks.")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    run_checks(parse_args())

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

from data_provider.data_factory import data_provider  # noqa: E402
from models import TimeAlign  # noqa: E402
from train_repo import OFFICIAL_PRESETS, build_official_args  # noqa: E402


STBO_MODES = [
    "subspace-tiled-basis-operator-shared",
    "subspace-tiled-basis-operator-bank",
    "subspace-tiled-basis-operator-dct",
    "subspace-tiled-basis-operator-independent",
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
        stbo_tile_len=48,
        stbo_rank=16,
        stbo_bank_count=4,
        stbo_basis_init_std=16 ** -0.5,
    )


def make_model(readout_mode: str, seed: int, channels: int = 7) -> TimeAlign.Model:
    torch.manual_seed(seed)
    return TimeAlign.Model(build_config(readout_mode, channels=channels)).float()


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, tolerance: float) -> float:
    max_abs = float((actual - expected).abs().max().item())
    if max_abs > tolerance:
        raise AssertionError(f"{name} max_abs={max_abs:.6e} exceeds tolerance={tolerance:.6e}")
    return max_abs


def run_synthetic_checks(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed + 100)
    x = torch.randn(args.batch_size, 720, 7)
    y = torch.randn(args.batch_size, 720, 7)

    for mode in STBO_MODES:
        model = make_model(mode, args.seed)
        model.eval()
        with torch.no_grad():
            out_h96, recon, align = model(x, y, is_training=False, target_prefix=96)
            out_h720, _recon, _align = model(x, y, is_training=False, target_prefix=720)
        if recon is not None:
            raise AssertionError(f"{mode} unexpectedly returned future reconstruction")
        if not torch.isfinite(out_h96).all() or not torch.isfinite(out_h720).all():
            raise AssertionError(f"{mode} produced non-finite outputs")
        prefix_diff = assert_close(f"{mode}_prefix_h96_vs_h720", out_h96, out_h720[:, :96, :], args.tolerance)
        print(f"prefix_ok mode={mode} max_abs={prefix_diff:.6e} align={float(align.detach().item()):.6e}")

        model.train()
        output, recon, align = model(x, y, is_training=True, target_prefix=96)
        if recon is not None:
            raise AssertionError(f"{mode} unexpectedly returned future reconstruction during training")
        loss = output.pow(2).mean() + align.abs()
        loss.backward()
        if not torch.isfinite(loss):
            raise AssertionError(f"{mode} produced non-finite loss")
        if model.stbo_coeff.weight.grad is None or not torch.isfinite(model.stbo_coeff.weight.grad).all():
            raise AssertionError(f"{mode} produced invalid stbo_coeff gradient")
        if mode != "subspace-tiled-basis-operator-dct":
            basis_grads = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if name.startswith("stbo_") and name != "stbo_coeff.weight" and name != "stbo_coeff.bias"
            ]
            if not basis_grads or any(grad is None or not torch.isfinite(grad).all() for grad in basis_grads):
                raise AssertionError(f"{mode} produced invalid STBO basis gradients")
        print(f"backward_ok mode={mode} loss={float(loss.detach().item()):.6f}")


def build_official_smoke_args(args: argparse.Namespace, readout_mode: str) -> argparse.Namespace:
    preset = OFFICIAL_PRESETS["ETTh2"][720]
    adapter_args = SimpleNamespace(
        dataset_root=args.dataset_root,
        output_dir=Path("artifacts/smoke_phase5_stage_b_b12_stbo_local/_tmp"),
        dataset="ETTh2",
        seq_len=720,
        label_len=48,
        pred_len=720,
        e_layers=2,
        num_workers=0,
        epochs=1,
        batch_size=args.batch_size,
        patience=1,
        use_amp=False,
        seed=args.seed,
        device="cpu",
        readout_mode=readout_mode,
        w_align=None,
        w_recon=0.0,
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
        stbo_tile_len=48,
        stbo_rank=16,
        stbo_bank_count=4,
        stbo_basis_init_std=16 ** -0.5,
    )
    official_args = build_official_args(adapter_args, preset)
    official_args.batch_size = args.batch_size
    official_args.num_workers = 0
    return official_args


def run_etth2_one_batch_smoke(args: argparse.Namespace) -> None:
    for mode in STBO_MODES:
        official_args = build_official_smoke_args(args, mode)
        _data, loader = data_provider(official_args, "train")
        batch_x, batch_y, _batch_x_mark, _batch_y_mark = next(iter(loader))
        model = TimeAlign.Model(official_args).float()
        model.train()
        output, recon, align = model(
            batch_x.float(),
            batch_y[:, -official_args.pred_len :, :].float(),
            is_training=True,
            target_prefix=96,
        )
        if recon is not None:
            raise AssertionError(f"{mode} unexpectedly returned future reconstruction in ETTh2 smoke")
        loss = output[:, :96, :].pow(2).mean() + align.abs()
        loss.backward()
        if model.stbo_coeff.weight.grad is None:
            raise AssertionError(f"{mode} did not backpropagate to stbo_coeff")
        print(f"etth2_one_batch_ok mode={mode} loss={float(loss.detach().item()):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local B12-STBO prefix/backward/smoke checks.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/Users/river/PaperResearch/Project/datasets"))
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--skip-etth2-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_synthetic_checks(args)
    if not args.skip_etth2_smoke:
        run_etth2_one_batch_smoke(args)


if __name__ == "__main__":
    main()

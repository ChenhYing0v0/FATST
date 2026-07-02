from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from models import TimeAlign  # noqa: E402


def build_config(readout_mode: str, **overrides: int | float | str) -> Namespace:
    config = Namespace(
        task_name="long_term_forecast",
        seq_len=96,
        label_len=48,
        pred_len=720,
        patch_num=12,
        d_model=32,
        d_ff=64,
        e_layers=1,
        dropout=0.0,
        local_margin=0.0,
        global_margin=0.0,
        loc=1,
        glo=1,
        layer_norm=1,
        pos=1,
        enc_in=3,
        readout_mode=readout_mode,
        target_horizons=[96, 192, 336, 720],
        basis_rank=64,
        target_query_segment_len=48,
        target_query_heads=4,
        target_query_ff=64,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def check_arm(name: str, config: Namespace) -> dict[str, float | str | dict[str, list[int]]]:
    torch.manual_seed(20260702)
    model = TimeAlign.Model(config).eval()
    x = torch.randn(2, config.seq_len, config.enc_in)
    y = torch.randn(2, config.pred_len, config.enc_in)
    shapes: dict[str, list[int]] = {}
    with torch.no_grad():
        full, _recon, _align = model(x, y, is_training=False, target_prefix=config.pred_len)
        for horizon in config.target_horizons:
            out, _recon, _align = model(x, y, is_training=False, target_prefix=horizon)
            expected = (2, horizon, config.enc_in)
            actual = tuple(out.shape)
            if actual != expected:
                raise AssertionError(f"{name} h{horizon} shape {actual} != {expected}")
            shapes[f"h{horizon}"] = list(actual)
        prefix, _recon, _align = model(x, y, is_training=False, target_prefix=96)
    mismatch = float(torch.max(torch.abs(prefix - full[:, :96, :])).item())
    if mismatch > 1e-5:
        raise AssertionError(f"{name} prefix mismatch {mismatch:.6g} > 1e-5")
    return {"arm": name, "prefix_mismatch_h96_vs_h720": mismatch, "shapes": shapes}


def main() -> None:
    arms = [
        (
            "A5-B-r64",
            build_config("continuous-forecast-basis-operator", basis_rank=64),
        ),
        (
            "A5-B-r128",
            build_config("continuous-forecast-basis-operator", basis_rank=128),
        ),
        (
            "A5-Q-seg48-small",
            build_config("elastic-causal-target-query-decoder", target_query_segment_len=48, target_query_ff=64),
        ),
        (
            "A5-Q-seg24-wide",
            build_config("elastic-causal-target-query-decoder", target_query_segment_len=24, target_query_ff=128),
        ),
    ]
    results = [check_arm(name, config) for name, config in arms]
    print(json.dumps({"status": "ok", "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

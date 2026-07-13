#!/usr/bin/env python3
"""Audit prefix and PMFO algebra invariants for a trained Step 7B checkpoint."""

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

from layers.PMFO import PMFO_BLOCK_SIZES, PMFO_RADICES  # noqa: E402
from models import TimeAlign  # noqa: E402


HORIZONS = (1, 48, 96, 192, 336, 720)
TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--synthetic-readout",
        choices=sorted(TimeAlign.PMFO_READOUTS | {"learned-basis-forecast-operator"}),
    )
    parser.add_argument("--seed", type=int, default=2021)
    return parser.parse_args()


def synthetic_config(readout_mode: str) -> SimpleNamespace:
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


def load_model(run_dir: Path) -> tuple[TimeAlign.Model, int]:
    effective_config = json.loads(
        (run_dir / "effective_config.json").read_text(encoding="utf-8")
    )
    official_args = effective_config["official_args"]
    model = TimeAlign.Model(SimpleNamespace(**official_args)).float()
    state = torch.load(
        run_dir / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    return model, int(official_args["enc_in"])


def pmfo_algebra_metrics(
    model: TimeAlign.Model,
    generator: torch.Generator,
) -> dict[str, float]:
    if model.readout_mode != "pmfo-rct":
        return {}
    synthesis = model.pmfo_readout.synthesis
    recovery_errors = []
    conservation_errors = []
    for level, radix in enumerate(PMFO_RADICES):
        parent = torch.randn(5, 7, generator=generator)
        detail = torch.randn(5, 7, radix - 1, generator=generator)
        children = synthesis.refine(parent, detail, level)
        scaling = getattr(synthesis, f"scaling_{level}")
        contrast = getattr(synthesis, f"contrast_{level}")
        recovered_parent = torch.einsum("bnr,r->bn", children, scaling)
        recovered_detail = torch.einsum("bnr,rd->bnd", children, contrast)
        recovery_errors.extend(
            [
                float((recovered_parent - parent).abs().max()),
                float((recovered_detail - detail).abs().max()),
            ]
        )
        perturbed = detail.clone()
        perturbed[..., 0] += 2.0
        changed = synthesis.refine(parent, perturbed, level)
        changed_parent = torch.einsum("bnr,r->bn", changed, scaling)
        conservation_errors.append(
            float((changed_parent - recovered_parent).abs().max())
        )

    coarse = torch.randn(1, 1, 8, generator=generator)
    parent_counts = [720 // size for size in PMFO_BLOCK_SIZES[:-1]]
    details = tuple(
        torch.randn(1, 1, count, radix - 1, generator=generator)
        for count, radix in zip(parent_counts, PMFO_RADICES, strict=True)
    )
    baseline = synthesis(coarse, details, 720)
    locality_errors = []
    for level, support in enumerate(PMFO_BLOCK_SIZES[:-1]):
        parent_index = parent_counts[level] // 2
        changed_details = [detail.clone() for detail in details]
        changed_details[level][0, 0, parent_index, 0] += 2.0
        changed = synthesis(coarse, tuple(changed_details), 720)
        delta = (changed - baseline).abs()
        start = parent_index * support
        end = start + support
        outside = torch.cat([delta[..., :start], delta[..., end:]], dim=-1)
        locality_errors.append(float(outside.max()) if outside.numel() else 0.0)
    return {
        "refinement_recovery_max_abs": max(recovery_errors),
        "conservation_max_abs": max(conservation_errors),
        "locality_outside_support_max_abs": max(locality_errors),
    }


def audit_model(
    model: TimeAlign.Model,
    channels: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    x = torch.randn(1, 720, channels, generator=generator)
    y = torch.zeros(1, 720, channels)
    prefix_rows = []
    with torch.no_grad():
        full = model(x, y, is_training=False, target_prefix=720)[0]
        for horizon in HORIZONS:
            prefix = model(
                x,
                y,
                is_training=False,
                target_prefix=horizon,
            )[0]
            gap = float((prefix - full[:, :horizon]).abs().max())
            prefix_rows.append(
                {
                    "horizon": horizon,
                    "shape": list(prefix.shape),
                    "full_prefix_max_abs": gap,
                }
            )
    algebra = pmfo_algebra_metrics(model, generator)
    maximum_prefix_gap = max(row["full_prefix_max_abs"] for row in prefix_rows)
    checked_errors = [maximum_prefix_gap, *algebra.values()]
    finite = all(math.isfinite(value) for value in checked_errors)
    passed = finite and max(checked_errors) <= TOLERANCE
    return {
        "readout_mode": model.readout_mode,
        "seed": seed,
        "tolerance": TOLERANCE,
        "prefix_rows": prefix_rows,
        "full_prefix_max_abs": maximum_prefix_gap,
        **algebra,
        "finite": finite,
        "pass": passed,
    }


def main() -> None:
    args = parse_args()
    if (args.run_dir is None) == (args.synthetic_readout is None):
        raise ValueError("provide exactly one of --run-dir or --synthetic-readout")
    if args.run_dir is not None:
        model, channels = load_model(args.run_dir)
        result = audit_model(model, channels, args.seed)
        output_path = args.run_dir / "trained_invariants.json"
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    else:
        model = TimeAlign.Model(synthetic_config(args.synthetic_readout)).float()
        result = audit_model(model, 7, args.seed)
    if not result["pass"]:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

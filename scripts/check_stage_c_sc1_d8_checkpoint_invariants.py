#!/usr/bin/env python3
"""Audit trained SC1-D8 A6/PAF checkpoints and patch-interface artifacts."""

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

from models import TimeAlign  # noqa: E402


HORIZONS = (1, 48, 96, 192, 336, 720)
TOLERANCE = 1e-5
# Trained projections can accumulate larger float32 summation-order error than
# the initialization-time gate while remaining the exact same linear map.
PATCH_EQUIVALENCE_TOLERANCE = 2e-5
READOUTS = TimeAlign.PLGO_PAF_READOUTS | {
    "learned-basis-forecast-operator",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--synthetic-readout", choices=sorted(READOUTS))
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
        patch_num=12,
        d_model=64,
        d_ff=128,
        dropout=0.1,
        pos=1,
        layer_norm=1,
        enc_in=7,
        basis_rank=256,
        plgo_global_rank=16,
        plgo_latent_width=256,
        plgo_permutation_seed=7101,
        plgo_random_descriptor_seed=7102,
    )


def load_model(run_dir: Path) -> tuple[TimeAlign.Model, dict[str, Any]]:
    config = json.loads(
        (run_dir / "effective_config.json").read_text(encoding="utf-8")
    )
    model = TimeAlign.Model(SimpleNamespace(**config["official_args"])).float()
    state = torch.load(
        run_dir / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    return model, config


def audit_model(
    model: TimeAlign.Model,
    channels: int,
    seed: int,
    patch_diagnostics: dict[str, Any] | None,
    training_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    x = torch.randn(1, 720, channels, generator=generator)
    y = torch.zeros(1, 720, channels)
    rows = []
    with torch.no_grad():
        full = model(x, y, is_training=False, target_prefix=720)[0]
        for horizon in HORIZONS:
            prefix = model(x, y, is_training=False, target_prefix=horizon)[0]
            rows.append(
                {
                    "horizon": horizon,
                    "shape": list(prefix.shape),
                    "full_prefix_max_abs": float(
                        (prefix - full[:, :horizon]).abs().max()
                    ),
                }
            )
    maximum_gap = max(row["full_prefix_max_abs"] for row in rows)
    finite = math.isfinite(maximum_gap)
    frozen_tensors = sum(
        not parameter.requires_grad for parameter in model.parameters()
    )
    contract_ok = training_contract is None or (
        training_contract.get("initialization") == "from_scratch"
        and training_contract.get("checkpoint_input") is None
        and training_contract.get("encoder_trainable") is True
        and training_contract.get("decoder_trainable") is True
        and training_contract.get("expected_frozen_parameter_tensors") == 0
    )
    patch_ok = patch_diagnostics is None or (
        patch_diagnostics.get("finite") is True
        and math.isfinite(float(patch_diagnostics["patch_contribution_entropy"]))
        and float(patch_diagnostics["flatten_block_sum_max_abs"])
        <= PATCH_EQUIVALENCE_TOLERANCE
    )
    result = {
        "candidate": "SC1-D8-E2E",
        "readout_mode": model.readout_mode,
        "seed": seed,
        "tolerance": TOLERANCE,
        "patch_equivalence_tolerance": PATCH_EQUIVALENCE_TOLERANCE,
        "prefix_rows": rows,
        "full_prefix_max_abs": maximum_gap,
        "frozen_parameter_tensors": frozen_tensors,
        "from_scratch_contract_pass": contract_ok,
        "patch_diagnostics_present": patch_diagnostics is not None,
        "patch_diagnostics_pass": patch_ok,
        "finite": finite,
    }
    result["pass"] = bool(
        finite
        and maximum_gap <= TOLERANCE
        and frozen_tensors == 0
        and contract_ok
        and patch_ok
    )
    return result


def main() -> None:
    args = parse_args()
    if (args.run_dir is None) == (args.synthetic_readout is None):
        raise ValueError("provide exactly one of --run-dir or --synthetic-readout")
    if args.run_dir is not None:
        model, config = load_model(args.run_dir)
        patch_path = args.run_dir / "patch_diagnostics.json"
        patch_diagnostics = json.loads(patch_path.read_text(encoding="utf-8"))
        result = audit_model(
            model,
            int(config["official_args"]["enc_in"]),
            args.seed,
            patch_diagnostics,
            config.get("training_contract"),
        )
        (args.run_dir / "trained_invariants.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    else:
        model = TimeAlign.Model(synthetic_config(args.synthetic_readout)).float()
        result = audit_model(model, 7, args.seed, None, None)
    if not result["pass"]:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

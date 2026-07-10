from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from data_provider.data_factory import data_provider  # noqa: E402
from models.TimeAlign import Model  # noqa: E402
from train_repo import (  # noqa: E402
    OFFICIAL_PRESETS,
    build_official_args,
    evaluate,
    write_csv,
)


HORIZONS = (96, 192, 336, 720)
PRED_LEN = 720


def read_metrics(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        int(row["target_horizon"]): {
            "mse": float(row["mse"]),
            "mae": float(row["mae"]),
        }
        for row in rows
    }


def adapter_args(args: argparse.Namespace, encoder_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        seq_len=PRED_LEN,
        label_len=48,
        pred_len=PRED_LEN,
        e_layers=2,
        num_workers=0,
        epochs=10,
        batch_size=args.batch_size,
        patience=3,
        use_amp=False,
        seed=2021,
        device=args.device,
        readout_mode="learned-basis-forecast-operator",
        encoder_mode=encoder_mode,
        history_patch_len=48,
        history_patch_stride=24,
        history_d_model=128,
        history_n_heads=16,
        history_d_ff=256,
        history_e_layers=3,
        history_dropout=0.2,
        history_attn_dropout=0.0,
        history_res_attention=True,
        learning_rate=None,
        w_align=None,
        w_recon=0.0,
        target_horizons=list(HORIZONS),
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
        stbo_basis_init_std=16**-0.5,
    )


def build_args(
    args: argparse.Namespace,
    encoder_mode: str,
) -> argparse.Namespace:
    preset = OFFICIAL_PRESETS[args.dataset][PRED_LEN]
    result = build_official_args(adapter_args(args, encoder_mode), preset)
    result.batch_size = args.batch_size
    result.num_workers = 0
    return result


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def compare_first_batch(
    legacy: Model,
    hierarchical: Model,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, list[int]]:
    batch_x, batch_y, _batch_x_mark, _batch_y_mark = next(iter(loader))
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    max_abs_diff = 0.0
    with torch.no_grad():
        for horizon in HORIZONS:
            legacy_output = legacy(
                batch_x,
                batch_y[:, -PRED_LEN:, :],
                is_training=False,
                target_prefix=horizon,
            )[0]
            hierarchical_output = hierarchical(
                batch_x,
                batch_y[:, -PRED_LEN:, :],
                is_training=False,
                target_prefix=horizon,
            )[0]
            max_abs_diff = max(
                max_abs_diff,
                float((legacy_output - hierarchical_output).abs().max().cpu()),
            )
        memory = hierarchical.encode_retrieval_memory(batch_x)
    return max_abs_diff, list(memory.shape)


def metric_comparisons(
    rows: list[dict[str, Any]],
    reference: dict[int, dict[str, float]],
) -> tuple[list[dict[str, Any]], float]:
    comparisons: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    for row in rows:
        horizon = int(row["target_horizon"])
        mse_diff = float(row["mse"]) - reference[horizon]["mse"]
        mae_diff = float(row["mae"]) - reference[horizon]["mae"]
        max_abs_diff = max(max_abs_diff, abs(mse_diff), abs(mae_diff))
        comparisons.append(
            {
                "dataset": row["dataset"],
                "target_horizon": horizon,
                "hierarchical_mse": row["mse"],
                "reference_mse": reference[horizon]["mse"],
                "mse_abs_diff": mse_diff,
                "hierarchical_mae": row["mae"],
                "reference_mae": reference[horizon]["mae"],
                "mae_abs_diff": mae_diff,
            }
        )
    return comparisons, max_abs_diff


def run(args: argparse.Namespace) -> None:
    checkpoint = args.reference_dir / "checkpoint.pt"
    reference_metrics = args.reference_dir / "metrics_by_target_horizon.csv"
    if not checkpoint.exists() or not reference_metrics.exists():
        raise FileNotFoundError(args.reference_dir)

    legacy_args = build_args(args, "timealign-token-mlp")
    hierarchical_args = build_args(args, "hierarchical-patch-memory")
    state = load_state(checkpoint)
    legacy = Model(legacy_args)
    hierarchical = Model(hierarchical_args)
    legacy.load_state_dict(state, strict=True)
    hierarchical.load_state_dict(state, strict=True)
    state_keys_equal = legacy.state_dict().keys() == hierarchical.state_dict().keys()
    legacy_parameters = sum(parameter.numel() for parameter in legacy.parameters())
    hierarchical_parameters = sum(
        parameter.numel() for parameter in hierarchical.parameters()
    )

    device = hierarchical_args.device
    legacy = legacy.float().to(device).eval()
    hierarchical = hierarchical.float().to(device).eval()
    _test_data, test_loader = data_provider(hierarchical_args, "test")
    output_diff, memory_shape = compare_first_batch(
        legacy,
        hierarchical,
        test_loader,
        device,
    )
    main_rows, _segment_rows, _preds, _trues = evaluate(
        hierarchical,
        test_loader,
        hierarchical_args,
        list(HORIZONS),
        max_batches=0,
        is_training_flag=True,
    )
    for row in main_rows:
        row.update(
            {
                "dataset": args.dataset,
                "encoder_mode": "hierarchical-patch-memory",
                "checkpoint": str(checkpoint),
            }
        )
    comparisons, metric_diff = metric_comparisons(
        main_rows,
        read_metrics(reference_metrics),
    )
    passed = bool(
        state_keys_equal
        and legacy_parameters == hierarchical_parameters
        and output_diff == 0.0
        and metric_diff <= 1e-8
        and memory_shape[2:] == [29, 48]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metrics_by_target_horizon.csv", main_rows)
    write_csv(args.output_dir / "equivalence_comparison.csv", comparisons)
    (args.output_dir / "equivalence_summary.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "decision": (
                    "exact_equivalence_pass" if passed else "equivalence_fail"
                ),
                "state_keys_equal": state_keys_equal,
                "legacy_parameters": legacy_parameters,
                "hierarchical_parameters": hierarchical_parameters,
                "first_batch_max_abs_output_diff": output_diff,
                "max_abs_metric_diff": metric_diff,
                "retrieval_memory_shape": memory_shape,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if not passed:
        raise RuntimeError(f"hierarchical patch-memory equivalence failed: {args.dataset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(OFFICIAL_PRESETS), required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

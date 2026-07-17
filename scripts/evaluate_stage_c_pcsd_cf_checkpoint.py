#!/usr/bin/env python3
"""Audit one frozen PCSD-CF checkpoint on sequential validation or test rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from data_provider.data_factory import data_provider  # noqa: E402
from models import TimeAlign  # noqa: E402
from train_repo import model_diagnostics  # noqa: E402


HORIZONS = (1, 48, 96, 192, 336, 720)
PREFIX_TOLERANCE = 2e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--design",
        type=Path,
        default=Path("configs/stage_c_pcsd_cf_native_direct.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-rows", type=int, default=256)
    parser.add_argument(
        "--evaluation-split",
        choices=("val", "test"),
        default="val",
    )
    parser.add_argument(
        "--test-audit-config",
        type=Path,
        default=Path("configs/stage_c_pcsd_cf_test_audit.json"),
    )
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args()


def load_model(
    run_dir: Path,
    device: torch.device,
) -> tuple[TimeAlign.Model, dict[str, Any], SimpleNamespace]:
    config = json.loads(
        (run_dir / "effective_config.json").read_text(encoding="utf-8")
    )
    official = dict(config["official_args"])
    official["device"] = device
    official["use_gpu"] = device.type == "cuda"
    official_args = SimpleNamespace(**official)
    model = TimeAlign.Model(official_args).to(device).float()
    state = torch.load(
        run_dir / "checkpoint.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model, config, official_args


def sequential_loader(
    official_args: SimpleNamespace,
    split: str,
) -> DataLoader:
    evaluation_data, _loader = data_provider(official_args, split)
    return DataLoader(
        evaluation_data,
        batch_size=official_args.batch_size,
        shuffle=False,
        num_workers=official_args.num_workers,
        drop_last=False,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def expected_matrix_size(audit: dict[str, Any]) -> int:
    matrix = audit["matrix"]
    datasets = audit.get("datasets", matrix.get("datasets", []))
    arms = audit.get("arms", matrix.get("arms", []))
    seeds = audit.get("seeds")
    if seeds is None:
        seeds = [matrix["seed"]] if "seed" in matrix else []
    return len(datasets) * len(arms) * len(seeds)


def test_audit_authorized(audit: dict[str, Any]) -> bool:
    authorization = audit["authorization"]
    if audit.get("candidate_version") == "SC1-PCSD-CF-v1":
        return bool(
            audit["status"] in {
                "authorized_prelaunch",
                "completed_test_fail_with_arm_headroom",
            }
            and audit["matrix"]["expected_runs"] == 60
            and authorization["user_authorized"] is True
            and authorization["checkpoint_retraining_allowed"] is False
            and authorization["checkpoint_selection"]
            == "historical-best-validation-h720-mse"
            and authorization["test_role"]
            == "primary-milestone-effectiveness-gate"
            and authorization["formal_test_access_count_for_version"] == 1
        )
    return bool(
        audit.get("status") == "authorized_prelaunch"
        and audit["matrix"]["expected_runs"] == expected_matrix_size(audit)
        and authorization.get("user_authorized") is True
        and authorization.get("authorization_date")
        and authorization.get("test_role")
        == "primary-mechanism-effectiveness-and-paper-benchmark"
        and authorization.get("checkpoint_selection")
        == "best-validation-mean-mse-h96-h192-h336-h720"
        and authorization.get("checkpoint_retraining_allowed") is True
        and authorization.get("checkpoint_mutation_during_test_allowed") is False
        and authorization.get("per_dataset_horizon_or_cell_tuning_allowed")
        is False
        and authorization.get("formal_test_access_count_for_version") == 1
    )


def bin_reduce(
    values: torch.Tensor,
    bins: list[dict[str, Any]],
    reduction: str,
) -> torch.Tensor:
    outputs = []
    for entry in bins:
        start, end = int(entry["start"]), int(entry["end"])
        chunk = values[..., start:end]
        if reduction == "mse":
            outputs.append(chunk.square().mean(dim=-1))
        elif reduction == "mae":
            outputs.append(chunk.abs().mean(dim=-1))
        elif reduction == "mean":
            outputs.append(chunk.mean(dim=-2))
        else:
            raise ValueError(f"unsupported reduction: {reduction}")
    return torch.stack(outputs, dim=-1 if reduction != "mean" else -2)


def prefix_audit(
    model: TimeAlign.Model,
    batch_x: torch.Tensor,
    target: torch.Tensor,
) -> tuple[list[dict[str, Any]], float]:
    rows = []
    with torch.no_grad():
        full = model(
            batch_x,
            target,
            is_training=False,
            target_prefix=720,
        )[0]
        for horizon in HORIZONS:
            prefix = model(
                batch_x,
                target,
                is_training=False,
                target_prefix=horizon,
            )[0]
            gap = float((prefix - full[:, :horizon]).abs().max())
            rows.append(
                {
                    "horizon": horizon,
                    "shape": list(prefix.shape),
                    "full_prefix_max_abs": gap,
                }
            )
    return rows, max(row["full_prefix_max_abs"] for row in rows)


def denormalized_arms(
    model: TimeAlign.Model,
    batch_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    memory = model.encode_history(batch_x)
    hidden = memory.flatten(start_dim=-2)
    _fused, arms, weights = model.pcsd_readout.forward_with_diagnostics(
        hidden,
        720,
    )
    outputs = []
    for scale_index in range(arms.shape[2]):
        normalized = arms[:, :, scale_index].permute(0, 2, 1)
        outputs.append(model.normalization_x(normalized, "denorm"))
    return torch.stack(outputs, dim=2), weights


def evaluate(args: argparse.Namespace) -> None:
    if args.run_dir is None:
        raise ValueError("run-dir is required outside synthetic smoke")
    if args.probe_rows <= 0:
        raise ValueError("probe_rows must be positive")
    device = torch.device(args.device)
    design = json.loads(args.design.read_text(encoding="utf-8"))
    test_audit = None
    if args.evaluation_split == "test":
        test_audit = json.loads(
            args.test_audit_config.read_text(encoding="utf-8")
        )
    bins = design["step7b_protocol"]["future_bins"]
    model, config, official_args = load_model(args.run_dir, device)
    loader = sequential_loader(official_args, args.evaluation_split)
    adapter = config["adapter"]
    training_contract = config["training_contract"]
    initialization = json.loads(
        (args.run_dir / "initialization_contract.json").read_text(
            encoding="utf-8"
        )
    )
    diagnostics = model_diagnostics(model)

    fused_bin_mse: list[np.ndarray] = []
    fused_bin_mae: list[np.ndarray] = []
    persistence_bin_mse: list[np.ndarray] = []
    arm_bin_mse: list[np.ndarray] = []
    arm_bin_mae: list[np.ndarray] = []
    policy_bin_usage: list[np.ndarray] = []
    probe_arms: list[np.ndarray] = []
    probe_fused: list[np.ndarray] = []
    probe_targets: list[np.ndarray] = []
    probe_count = 0
    all_finite = True
    prefix_rows: list[dict[str, Any]] | None = None
    prefix_gap = 0.0
    step_squared_error = torch.zeros(720, dtype=torch.float64)
    step_absolute_error = torch.zeros(720, dtype=torch.float64)
    element_rows = 0

    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            target = batch_y[:, -720:, :].float().to(device)
            if prefix_rows is None:
                prefix_rows, prefix_gap = prefix_audit(model, batch_x[:1], target[:1])
            fused = model(batch_x, target, is_training=False, target_prefix=720)[0]
            errors = (fused - target).detach().to(torch.float64).cpu()
            step_squared_error += errors.square().sum(dim=(0, 2))
            step_absolute_error += errors.abs().sum(dim=(0, 2))
            element_rows += int(errors.shape[0] * errors.shape[2])
            fused_rows = (fused - target).permute(0, 2, 1).reshape(-1, 720)
            persistence = batch_x[:, -1:, :].expand_as(target)
            persistence_rows = (
                persistence - target
            ).permute(0, 2, 1).reshape(-1, 720)
            fused_bin_mse.append(
                bin_reduce(fused_rows, bins, "mse").cpu().numpy()
            )
            fused_bin_mae.append(
                bin_reduce(fused_rows, bins, "mae").cpu().numpy()
            )
            persistence_bin_mse.append(
                bin_reduce(persistence_rows, bins, "mse").cpu().numpy()
            )

            if hasattr(model, "pcsd_readout"):
                arms, weights = denormalized_arms(model, batch_x)
                arm_errors = arms - target.unsqueeze(2)
                arm_rows = arm_errors.permute(0, 3, 2, 1).reshape(
                    -1,
                    arms.shape[2],
                    720,
                )
                arm_bin_mse.append(
                    torch.stack(
                        [
                            arm_rows[..., int(entry["start"]): int(entry["end"])]
                            .square()
                            .mean(dim=-1)
                            for entry in bins
                        ],
                        dim=1,
                    ).cpu().numpy()
                )
                arm_bin_mae.append(
                    torch.stack(
                        [
                            arm_rows[..., int(entry["start"]): int(entry["end"])]
                            .abs()
                            .mean(dim=-1)
                            for entry in bins
                        ],
                        dim=1,
                    ).cpu().numpy()
                )
                weight_rows = weights.reshape(-1, 720, weights.shape[-1])
                policy_bin_usage.append(
                    torch.stack(
                        [
                            weight_rows[
                                :, int(entry["start"]): int(entry["end"])
                            ].mean(dim=1)
                            for entry in bins
                        ],
                        dim=1,
                    ).cpu().numpy()
                )
                if probe_count < args.probe_rows:
                    rows_available = arm_rows.shape[0]
                    count = min(args.probe_rows - probe_count, rows_available)
                    probe_arms.append(
                        arms.permute(0, 3, 2, 1)
                        .reshape(-1, arms.shape[2], 720)[:count]
                        .cpu()
                        .numpy()
                    )
                    probe_fused.append(
                        fused.permute(0, 2, 1).reshape(-1, 720)[:count]
                        .cpu()
                        .numpy()
                    )
                    probe_targets.append(
                        target.permute(0, 2, 1).reshape(-1, 720)[:count]
                        .cpu()
                        .numpy()
                    )
                    probe_count += count
                all_finite = all_finite and bool(
                    torch.isfinite(arms).all() and torch.isfinite(weights).all()
                )
            all_finite = all_finite and bool(
                torch.isfinite(fused).all() and torch.isfinite(target).all()
            )

    if not fused_bin_mse or prefix_rows is None or element_rows <= 0:
        raise RuntimeError(
            f"{args.evaluation_split} evaluation produced no rows"
        )
    payload: dict[str, np.ndarray] = {
        "fused_row_bin_mse": np.concatenate(fused_bin_mse).astype(np.float32),
        "fused_row_bin_mae": np.concatenate(fused_bin_mae).astype(np.float32),
        "persistence_row_bin_mse": np.concatenate(persistence_bin_mse).astype(
            np.float32
        ),
        "bin_names": np.asarray([entry["name"] for entry in bins]),
        "scales": np.asarray(design["coupling_scales"], dtype=np.int64),
    }
    if arm_bin_mse:
        payload.update(
            {
                "arm_row_bin_mse": np.concatenate(arm_bin_mse).astype(
                    np.float32
                ),
                "arm_row_bin_mae": np.concatenate(arm_bin_mae).astype(
                    np.float32
                ),
                "policy_row_bin_usage": np.concatenate(policy_bin_usage).astype(
                    np.float32
                ),
                "probe_arms": np.concatenate(probe_arms).astype(np.float32),
                "probe_fused": np.concatenate(probe_fused).astype(np.float32),
                "probe_targets": np.concatenate(probe_targets).astype(np.float32),
            }
        )
    artifact_prefix = (
        "validation" if args.evaluation_split == "val" else "test_audit"
    )
    np.savez_compressed(
        args.run_dir / f"pcsd_{artifact_prefix}_diagnostics.npz",
        **payload,
    )

    if args.evaluation_split == "test":
        cumulative_squared = torch.cumsum(step_squared_error, dim=0)
        cumulative_absolute = torch.cumsum(step_absolute_error, dim=0)
        metric_rows = []
        for horizon in range(1, 721):
            denominator = float(element_rows * horizon)
            metric_rows.append(
                {
                    "target_horizon": horizon,
                    "mse": float(cumulative_squared[horizon - 1] / denominator),
                    "mae": float(cumulative_absolute[horizon - 1] / denominator),
                    "num_rows_channels": element_rows,
                    "evaluation_split": "test",
                    "checkpoint_policy": adapter["checkpoint_policy"],
                    "candidate_version": test_audit["candidate_version"],
                }
            )
        write_csv(
            args.run_dir / "test_audit_metrics_by_target_horizon.csv",
            metric_rows,
        )

    test_authorized = (
        args.evaluation_split == "val"
        or test_audit is not None
        and test_audit_authorized(test_audit)
    )
    expected_validation_horizons = [720]
    expected_final_split = "val"
    if test_audit is not None and "training" in test_audit:
        expected_validation_horizons = test_audit["training"][
            "validation_horizons"
        ]
        expected_final_split = test_audit["training"][
            "training_final_evaluation_split"
        ]

    protocol_pass = bool(
        adapter["mode"] == "unified"
        and int(adapter["pred_len"]) == 720
        and adapter["target_horizons"] == [720]
        and adapter["validation_horizons"] == expected_validation_horizons
        and adapter["checkpoint_policy"] == "best-val"
        and adapter["pred_loss_mode"] == "full"
        and adapter["protocol_class"] == "method_screening"
        and adapter["evaluation_prefix_mode"] == "full-crop"
        and adapter["final_evaluation_split"] == expected_final_split
        and training_contract["initialization"] == "from_scratch"
        and training_contract["checkpoint_input"] is None
        and diagnostics["frozen_parameter_tensors"] == 0
        and test_authorized
    )
    readout_contract_pass = True
    if hasattr(model, "pcsd_readout"):
        readout_contract_pass = bool(
            initialization.get("pcsd_initialization_hash")
            and initialization.get("pcsd_coordinate_hash")
            and initialization.get("pcsd_partition_hash")
            and diagnostics.get("pcsd_scales") == design["coupling_scales"]
            and diagnostics.get("pcsd_policy_mode")
            == adapter["pcsd_policy_mode"]
            and diagnostics.get("pcsd_partition") == adapter["pcsd_partition"]
        )
    elif hasattr(model, "pcsd_m0_readout"):
        readout_contract_pass = bool(
            initialization.get("pcsd_m0_initialization_hash")
            and initialization.get("operator_initialization_hash")
            and diagnostics.get("pcsd_m0_mode_rank") == 256
        )
    elif hasattr(model, "pcsd_dense_readout"):
        dense_gap_limit = design.get("gates", {}).get(
            "dense_parameter_gap_max",
            design.get("step7b_protocol", {}).get(
                "dense_parameter_gap_max",
                0.005,
            ),
        )
        readout_contract_pass = bool(
            initialization.get("pcsd_dense_initialization_hash")
            and diagnostics.get("pcsd_dense_parameter_relative_gap", 1.0)
            <= dense_gap_limit
        )

    invariant = {
        "candidate_id": design.get(
            "candidate_id",
            design.get("candidate_version", "PCSD-CF"),
        ),
        "diagnostic_id": design.get(
            "diagnostic_id",
            design.get("audit_id", "PCSD-CF-checkpoint-audit"),
        ),
        "dataset": adapter["dataset"],
        "readout_mode": adapter["readout_mode"],
        "policy_mode": adapter.get("pcsd_policy_mode", "control"),
        "fixed_scale": adapter.get("pcsd_fixed_scale"),
        "partition": adapter.get("pcsd_partition", "control"),
        "prefix_rows": prefix_rows,
        "full_prefix_max_abs": prefix_gap,
        "evaluation_split": args.evaluation_split,
        "evaluation_rows": int(payload["fused_row_bin_mse"].shape[0]),
        "arm_diagnostics_present": "arm_row_bin_mse" in payload,
        "probe_rows": int(payload.get("probe_arms", np.empty((0,))).shape[0]),
        "all_finite": all_finite,
        "protocol_pass": protocol_pass,
        "readout_contract_pass": readout_contract_pass,
        "uses_test_split": args.evaluation_split == "test",
        "test_access_authorized": test_authorized,
        "checkpoint_sha256": file_sha256(args.run_dir / "checkpoint.pt"),
        "checkpoint_retrained": bool(
            test_audit is not None
            and test_audit["authorization"].get(
                "checkpoint_retraining_allowed",
                False,
            )
        ),
        "pass": bool(
            all_finite
            and math.isfinite(prefix_gap)
            and prefix_gap <= PREFIX_TOLERANCE
            and protocol_pass
            and readout_contract_pass
        ),
    }
    invariant_name = (
        "trained_invariants.json"
        if args.evaluation_split == "val"
        else "test_audit_invariants.json"
    )
    (args.run_dir / invariant_name).write_text(
        json.dumps(invariant, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not invariant["pass"]:
        raise RuntimeError(f"trained invariant failed: {invariant}")
    print(
        f"pcsd_checkpoint=pass dataset={invariant['dataset']} "
        f"readout={invariant['readout_mode']} split={args.evaluation_split} "
        f"rows={invariant['evaluation_rows']}"
    )


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
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
        model = TimeAlign.Model(config).float().eval()
        x = torch.randn(2, 720, 7)
        target = torch.randn(2, 720, 7)
        rows, gap = prefix_audit(model, x[:1], target[:1])
        with torch.no_grad():
            arms, weights = denormalized_arms(model, x)
        if (
            len(rows) != len(HORIZONS)
            or gap != 0.0
            or tuple(arms.shape) != (2, 720, 5, 7)
            or tuple(weights.shape) != (2, 7, 720, 5)
            or not bool(torch.isfinite(arms).all())
        ):
            raise RuntimeError("PCSD checkpoint evaluator synthetic smoke failed")
        print("pcsd_checkpoint_evaluator_synthetic_smoke=pass")
        return
    evaluate(args)


if __name__ == "__main__":
    main()

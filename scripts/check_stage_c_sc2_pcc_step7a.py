#!/usr/bin/env python3
"""Run the SC2-PCC-v1-TI Step 7A local implementation gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from layers.PCC import (  # noqa: E402
    PCC_FINAL_ROUTE_WEIGHT,
    PCC_FINAL_SKILL_FLOOR,
    PCC_OBJECTIVE_MODES,
    PCC_RAMP_FRACTION,
    pcc_schedule,
    prefix_measure,
    projective_coupling_credit_loss,
    standardized_capability,
    transport_prefix_credit,
)
from models import TimeAlign  # noqa: E402
import train_repo as training_adapter  # noqa: E402


EXPECTED_DIAGNOSTICS = {
    "pcc_total_loss",
    "pcc_fused_measure_l1",
    "pcc_skill_loss",
    "pcc_route_kl",
    "pcc_weighted_skill_loss",
    "pcc_weighted_route_loss",
    "pcc_skill_floor",
    "pcc_route_weight",
    "pcc_credit_normalized_entropy",
    "pcc_policy_normalized_entropy",
    "pcc_policy_usage_max",
    "pcc_credit_policy_kl",
    "pcc_credit_argmax_accuracy",
    "pcc_credit_min",
    "pcc_credit_max",
    "pcc_arm_s0_measure_l1",
    "pcc_arm_s1_measure_l1",
    "pcc_arm_s2_measure_l1",
    "pcc_arm_s3_measure_l1",
    "pcc_arm_s4_measure_l1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_sc2_pcc_step6.json"),
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path("configs/stage_c_five_dataset_natural_profiles.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage_c_sc2_pcc_step7a_local_20260716"),
    )
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def record(
    rows: list[dict[str, Any]],
    case: str,
    value: Any,
    threshold: str,
    passed: bool,
) -> None:
    rows.append(
        {
            "case": case,
            "value": value,
            "threshold": threshold,
            "pass": bool(passed),
        }
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "value", "threshold", "pass"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def direct_prefix_risk(errors: Tensor) -> Tensor:
    risks = [
        errors[:, :, : horizon + 1].mean(dim=-2)
        for horizon in range(errors.shape[-2])
    ]
    return torch.stack(risks, dim=-2)


def direct_transport(capability: Tensor) -> Tensor:
    transported = []
    for target_index in range(capability.shape[-2]):
        horizons = torch.arange(
            target_index + 1,
            capability.shape[-2] + 1,
            device=capability.device,
            dtype=capability.dtype,
        )
        weights = 1.0 / horizons
        credit = capability[:, :, target_index:, :]
        transported.append(
            (credit * weights.view(1, 1, -1, 1)).sum(dim=-2)
            / weights.sum()
        )
    return torch.stack(transported, dim=-2)


def synthetic_math_gate(rows: list[dict[str, Any]], seed: int) -> None:
    torch.manual_seed(seed)
    dtype = torch.float64
    errors = torch.rand(2, 3, 23, 5, dtype=dtype, requires_grad=True)
    prefix_risk, capability = standardized_capability(
        errors,
        prefix_risk=True,
        standardization_epsilon=1e-12,
    )
    direct_risk = direct_prefix_risk(errors)
    risk_gap = float((prefix_risk - direct_risk).abs().max())
    record(
        rows,
        "vectorized_prefix_risk_vs_nested_loop",
        risk_gap,
        "<=1e-12",
        risk_gap <= 1e-12,
    )

    transported = transport_prefix_credit(capability)
    direct_credit = direct_transport(capability)
    transport_gap = float((transported - direct_credit).abs().max())
    record(
        rows,
        "vectorized_transport_vs_nested_loop",
        transport_gap,
        "<=1e-12",
        transport_gap <= 1e-12,
    )

    measure = prefix_measure(23, device=errors.device, dtype=dtype)
    left = (capability * prefix_risk).sum(dim=-1).mean(dim=-1).mean()
    right = (
        measure.view(1, 1, -1)
        * (transported * errors).sum(dim=-1)
    ).sum(dim=-1).mean()
    identity_gap = float((left - right).abs())
    record(rows, "nested_risk_transport_identity", identity_gap, "<=1e-12", identity_gap <= 1e-12)

    identical = torch.ones(2, 2, 23, 5, dtype=dtype)
    _risk, identical_capability = standardized_capability(
        identical,
        prefix_risk=True,
        standardization_epsilon=1e-12,
    )
    uniform_gap = float((identical_capability - 0.2).abs().max())
    record(
        rows,
        "identical_arms_uniform_capability",
        uniform_gap,
        "<=1e-12",
        uniform_gap <= 1e-12,
    )

    floored = transport_prefix_credit(
        (1.0 - PCC_FINAL_SKILL_FLOOR) * capability
        + PCC_FINAL_SKILL_FLOOR / 5
    )
    floor_min = float(floored.min())
    record(rows, "transported_skill_floor", floor_min, ">=0.04", floor_min >= 0.04 - 1e-12)

    _risk_stop, credit_stop = standardized_capability(
        errors,
        prefix_risk=True,
        standardization_epsilon=1e-12,
        stop_gradient=True,
    )
    _risk_live, credit_live = standardized_capability(
        errors,
        prefix_risk=True,
        standardization_epsilon=1e-12,
        stop_gradient=False,
    )
    stop_value_gap = float((credit_stop - credit_live).abs().max())
    record(rows, "stopgrad_value_invariance", stop_value_gap, "<=1e-12", stop_value_gap <= 1e-12)
    record(
        rows,
        "authorized_credit_has_no_gradient",
        credit_stop.requires_grad,
        "False",
        not credit_stop.requires_grad,
    )
    record(
        rows,
        "conditional_no_stopgrad_has_gradient",
        credit_live.requires_grad,
        "True",
        credit_live.requires_grad,
    )


def objective_control_gate(rows: list[dict[str, Any]], seed: int) -> None:
    torch.manual_seed(seed + 1)
    dtype = torch.float64
    batch, channels, length, scopes = 2, 3, 19, 5
    arms = torch.randn(batch, channels, length, scopes, dtype=dtype, requires_grad=True)
    logits = torch.randn(batch, channels, length, scopes, dtype=dtype, requires_grad=True)
    policy = torch.softmax(logits, dim=-1)
    target = torch.randn(batch, length, channels, dtype=dtype)
    fused = (arms * policy).sum(dim=-1).permute(0, 2, 1)
    skill_modes = {
        "equal_skill",
        "pointwise_capability_skill_only",
        "pointwise_prior_composed",
        "pointwise_pcc_v0",
        "transport_skill_only",
        "pcc_transport_full",
    }
    route_modes = {
        "pointwise_route_only",
        "pointwise_prior_composed",
        "pointwise_pcc_v0",
        "transport_route_only",
        "pcc_transport_full",
    }
    expected_modes = sorted(PCC_OBJECTIVE_MODES)
    for mode in expected_modes:
        result = projective_coupling_credit_loss(
            fused,
            arms,
            policy,
            target,
            mode=mode,
            progress=1.0,
        )
        expected_total = result.fused_loss
        if mode in skill_modes:
            expected_total = expected_total + result.skill_loss
        if mode in route_modes:
            expected_total = (
                expected_total + PCC_FINAL_ROUTE_WEIGHT * result.route_loss
            )
        decomposition_gap = float((result.total_loss - expected_total).abs())
        active_exact = (
            (mode in skill_modes) == bool(float(result.skill_loss) > 0.0)
            and (mode in route_modes) == bool(float(result.route_loss) > 0.0)
        )
        diagnostics_exact = set(result.diagnostics) == EXPECTED_DIAGNOSTICS
        record(
            rows,
            f"control_decomposition_{mode}",
            decomposition_gap,
            "<=1e-12 and exact active terms/diagnostics",
            decomposition_gap <= 1e-12 and active_exact and diagnostics_exact,
        )

    pointwise = projective_coupling_credit_loss(
        fused,
        arms,
        policy,
        target,
        mode="pointwise_pcc_v0",
        progress=1.0,
    )
    transported = projective_coupling_credit_loss(
        fused,
        arms,
        policy,
        target,
        mode="pcc_transport_full",
        progress=1.0,
    )
    specificity_gap = float(
        (pointwise.route_credit - transported.route_credit).abs().max()
    )
    record(rows, "transport_not_pointwise_control", specificity_gap, ">=1e-3", specificity_gap >= 1e-3)


def schedule_gate(rows: list[dict[str, Any]]) -> None:
    progress_values = [index / 100 for index in range(101)]
    schedules = [pcc_schedule(progress) for progress in progress_values]
    floors = [value.skill_floor for value in schedules]
    routes = [value.route_weight for value in schedules]
    endpoints = (
        abs(floors[0] - 1.0) <= 1e-12
        and abs(routes[0]) <= 1e-12
        and abs(floors[-1] - PCC_FINAL_SKILL_FLOOR) <= 1e-12
        and abs(routes[-1] - PCC_FINAL_ROUTE_WEIGHT) <= 1e-12
    )
    monotonic = all(
        floors[index + 1] <= floors[index] + 1e-12
        and routes[index + 1] >= routes[index] - 1e-12
        for index in range(len(schedules) - 1)
    )
    ramp = pcc_schedule(PCC_RAMP_FRACTION)
    reaches_at_ramp = (
        abs(ramp.skill_floor - PCC_FINAL_SKILL_FLOOR) <= 1e-12
        and abs(ramp.route_weight - PCC_FINAL_ROUTE_WEIGHT) <= 1e-12
    )
    record(rows, "continuous_schedule_endpoints", endpoints, "True", endpoints)
    record(rows, "continuous_schedule_monotonicity", monotonic, "True", monotonic)
    record(rows, "continuous_schedule_reaches_final_at_ramp", reaches_at_ramp, "True", reaches_at_ramp)


def model_config(profile: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=720,
        pred_len=720,
        encoder_mode="timealign-token-mlp",
        readout_mode="pcsd-coupling-field",
        e_layers=2,
        patch_num=int(profile["patch_num"]),
        d_model=int(profile["d_model"]),
        d_ff=int(profile["d_ff"]),
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


def real_model_gate(
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    seed: int,
) -> None:
    torch.manual_seed(seed + 2)
    model = TimeAlign.Model(model_config(profile)).double().eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    x = torch.randn(1, 720, 7, dtype=torch.float64)
    target = torch.randn(1, 720, 7, dtype=torch.float64)

    with torch.no_grad():
        plain_result = model(x, target, is_training=False, target_prefix=720)
        detailed_result = model(
            x,
            target,
            is_training=False,
            target_prefix=720,
            return_pcsd_training_details=True,
        )
        prefix_result = model(x, target, is_training=False, target_prefix=337)
    plain = plain_result[0]
    detailed = detailed_result[0]
    details = detailed_result[3]
    arms_bcst = details["arm_forecasts"]
    policy = details["policy"]
    reconstructed = (
        arms_bcst * policy.permute(0, 1, 3, 2)
    ).sum(dim=2).permute(0, 2, 1)
    output_gap = float((plain - detailed).abs().max())
    reconstruction_gap = float((detailed - reconstructed).abs().max())
    prefix_gap = float((prefix_result[0] - plain[:, :337]).abs().max())
    parameter_count_after = sum(parameter.numel() for parameter in model.parameters())
    record(rows, "default_forward_signature_unchanged", len(plain_result), "3", len(plain_result) == 3)
    record(rows, "details_forward_output_invariance", output_gap, "<=1e-12", output_gap <= 1e-12)
    record(rows, "raw_scale_arm_fusion_identity", reconstruction_gap, "<=1e-10", reconstruction_gap <= 1e-10)
    record(rows, "arbitrary_prefix_is_full_crop", prefix_gap, "<=1e-12", prefix_gap <= 1e-12)
    record(rows, "parameter_count_unchanged", parameter_count_after, str(parameter_count), parameter_count_after == parameter_count)

    model.zero_grad(set_to_none=True)
    detailed_train = model(
        x,
        target,
        is_training=True,
        target_prefix=720,
        return_pcsd_training_details=True,
    )
    train_arms_bcst = detailed_train[3]["arm_forecasts"]
    train_arms = train_arms_bcst.permute(0, 1, 3, 2)
    train_policy = detailed_train[3]["policy"]
    objective = projective_coupling_credit_loss(
        detailed_train[0],
        train_arms,
        train_policy,
        target,
        mode="pcc_transport_full",
        progress=1.0,
    )
    arm_gradient = torch.autograd.grad(
        objective.weighted_skill_loss,
        train_arms_bcst,
        retain_graph=True,
    )[0]
    scope_gradient = arm_gradient.abs().sum(dim=(0, 1, 3))
    all_scopes_nonzero = bool((scope_gradient > 0.0).all())
    objective.total_loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    gradients_finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    record(rows, "real_pcsd_all_scope_aux_gradients_nonzero", float(scope_gradient.min()), ">0", all_scopes_nonzero)
    record(rows, "real_pcsd_one_batch_parameter_gradients_finite", gradients_finite, "True", gradients_finite)


def protocol_gate(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    expected_modes = set(config["phase_a"]["training_arms"])
    record(rows, "frozen_nine_arm_matrix_exact", len(expected_modes), "9 exact modes", expected_modes == set(PCC_OBJECTIVE_MODES))
    contract = config["contract"]
    local_boundary = (
        contract["one_forward"]
        and contract["one_stage_training"]
        and not contract["requested_horizon_feature"]
        and not contract["inference_graph_changed"]
        and not config["remote_authorized"]
        and not config["test_access_authorized"]
    )
    record(rows, "local_only_protocol_boundary", local_boundary, "True", local_boundary)

    base_argv = [
        "train_repo.py",
        "--dataset-root",
        "/tmp",
        "--dataset",
        "ETTh2",
        "--mode",
        "unified",
        "--pred-len",
        "720",
        "--target-horizons",
        "96,192,336,720",
        "--run-name",
        "pcc_step7a_cli_smoke",
        "--output-dir",
        "/tmp/pcc_step7a_cli_smoke",
        "--final-evaluation-split",
        "none",
        "--readout-mode",
        "pcsd-coupling-field",
        "--pcsd-policy-mode",
        "direct",
        "--pcc-objective-mode",
        "pcc_transport_full",
    ]
    original_argv = sys.argv
    try:
        sys.argv = base_argv
        parsed = training_adapter.parse_args()
        valid_cli = parsed.pcc_objective_mode == "pcc_transport_full"
        sys.argv = [
            value
            if value != "pcsd-coupling-field"
            else "learned-basis-forecast-operator"
            for value in base_argv
        ]
        mismatch_rejected = False
        try:
            training_adapter.parse_args()
        except ValueError:
            mismatch_rejected = True
    finally:
        sys.argv = original_argv
    record(rows, "pcc_cli_contract_parses", valid_cli, "True", valid_cli)
    record(
        rows,
        "pcc_cli_rejects_non_pcsd_readout",
        mismatch_rejected,
        "True",
        mismatch_rejected,
    )


def training_adapter_gate(
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    output_dir: Path,
    seed: int,
) -> None:
    """Exercise one actual adapter update without opening a dataset split."""
    torch.manual_seed(seed + 3)
    x = torch.randn(1, 720, 7)
    target = torch.randn(1, 720, 7)
    marks = torch.zeros(1, 720, 1)
    loader = [(x, target, marks, marks)]
    requested_splits: list[str] = []

    def fake_data_provider(_args: argparse.Namespace, split: str) -> tuple[None, list[Any]]:
        requested_splits.append(split)
        if split not in {"train", "val"}:
            raise AssertionError(f"unexpected split access: {split}")
        return None, loader

    adapter_args = SimpleNamespace(
        output_dir=output_dir / "adapter_smoke",
        epochs=1,
        max_train_batches=1,
        max_eval_batches=1,
        gradient_accumulation_steps=1,
        readout_mode="pcsd-coupling-field",
        pcc_objective_mode="pcc_transport_full",
        target_horizons=[96, 192, 336, 720],
        validation_horizons=[96, 192, 336, 720],
        pred_loss_mode="full",
        enable_early_stopping=False,
        early_stopping_min_delta=0.0,
        patience=5,
        checkpoint_policy="best-val",
        batch_size=1,
    )
    adapter_args.output_dir.mkdir(parents=True, exist_ok=True)
    official_args = model_config(profile)
    official_args.device = torch.device("cpu")
    official_args.learning_rate = 1e-4
    official_args.features = "M"
    official_args.w_recon = 0.0
    official_args.w_align = 0.0
    official_args.lradj = "cosine"
    official_args.train_epochs = 1

    original_provider = training_adapter.data_provider
    original_validation = training_adapter.validation_mean_mse
    training_adapter.data_provider = fake_data_provider
    training_adapter.validation_mean_mse = lambda *_args, **_kwargs: 1.0
    try:
        _model, training_rows, _last_state, _best_state = training_adapter.train(
            adapter_args,
            official_args,
        )
    finally:
        training_adapter.data_provider = original_provider
        training_adapter.validation_mean_mse = original_validation

    logged = training_rows[0]
    diagnostics_logged = all(
        f"train_{name}" in logged for name in EXPECTED_DIAGNOSTICS
    )
    finite_loss = torch.isfinite(torch.tensor(logged["train_loss"])).item()
    record(
        rows,
        "training_adapter_one_update_finite",
        logged["train_loss"],
        "finite",
        bool(finite_loss),
    )
    record(
        rows,
        "training_adapter_diagnostics_complete",
        diagnostics_logged,
        "True",
        diagnostics_logged,
    )
    record(
        rows,
        "training_adapter_never_accesses_test",
        ",".join(requested_splits),
        "train,val",
        requested_splits == ["train", "val"],
    )


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    profiles_path = resolve(args.profiles)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    synthetic_math_gate(rows, args.seed)
    objective_control_gate(rows, args.seed)
    schedule_gate(rows)
    real_model_gate(rows, profiles["dataset_profiles"]["ETTh2"], args.seed)
    protocol_gate(rows, config)
    training_adapter_gate(
        rows,
        profiles["dataset_profiles"]["ETTh2"],
        output_dir,
        args.seed,
    )

    passed = sum(bool(row["pass"]) for row in rows)
    payload = {
        "candidate": config["candidate"],
        "current_step": "Step7A local implementation gate",
        "cases_passed": passed,
        "cases_total": len(rows),
        "all_pass": passed == len(rows),
        "remote_authorized": False,
        "test_accessed": False,
        "checkpoint_mutated": False,
        "decision": (
            "step7a_pass_prelaunch_audit_next"
            if passed == len(rows)
            else "step7a_fail_return_step5_or_6"
        ),
    }
    write_csv(output_dir / "step7a_cases.csv", rows)
    (output_dir / "local_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

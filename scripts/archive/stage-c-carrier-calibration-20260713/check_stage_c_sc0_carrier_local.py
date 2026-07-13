#!/usr/bin/env python3
"""Verify the StageC SC0 standardized carrier contract before remote launch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "stage_c_mechanism_control.json"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from models.TimeAlign import Model  # noqa: E402
from train_repo import model_diagnostics, select_prediction_horizons  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT.parent / "datasets",
    )
    parser.add_argument("--skip-end-to-end", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def profile_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model_config(
    common: dict[str, Any],
    arm: dict[str, Any],
    enc_in: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=int(common["seq_len"]),
        pred_len=int(common["pred_len"]),
        patch_num=int(arm["patch_num"]),
        d_model=int(arm["d_model"]),
        n_heads=8,
        e_layers=int(common["e_layers"]),
        d_ff=int(arm["d_ff"]),
        dropout=float(common["dropout"]),
        layer_norm=int(common["layer_norm"]),
        pos=1,
        enc_in=enc_in,
        readout_mode=str(common["readout_mode"]),
        encoder_mode=str(common["encoder_mode"]),
        basis_rank=int(common["basis_rank"]),
        target_horizons=list(common["target_horizons"]),
    )


def assert_patch_boundaries(seq_len: int, patch_num: int) -> None:
    patch_len = seq_len // patch_num
    raw = torch.arange(2 * seq_len, dtype=torch.float32).reshape(1, 2, seq_len)
    flattened = raw.reshape(-1, 2 * seq_len)
    patches = flattened.unfold(-1, patch_len, patch_len)
    patches = patches.reshape(1, 2, patch_num, patch_len)
    for channel in range(2):
        for patch_idx in range(patch_num):
            start = patch_idx * patch_len
            end = start + patch_len
            torch.testing.assert_close(
                patches[0, channel, patch_idx],
                raw[0, channel, start:end],
                rtol=0.0,
                atol=0.0,
            )


def structural_checks(config: dict[str, Any]) -> None:
    common = config["common"]
    torch.manual_seed(int(common["initial_seed"]))
    torch.set_num_threads(1)
    x = torch.randn(2, int(common["seq_len"]), 7)
    y = torch.randn(2, int(common["pred_len"]), 7)
    observed_active: list[int] = []

    if select_prediction_horizons("full", [96, 192, 336, 720], 720) != [720]:
        raise AssertionError("full objective must select exactly one 720-step loss")

    for arm_name, arm in config["arms"].items():
        patch_num = int(arm["patch_num"])
        if patch_num <= 1:
            raise AssertionError(f"{arm_name}: patch_num must be greater than one")
        if int(common["seq_len"]) % patch_num != 0:
            raise AssertionError(f"{arm_name}: patch_num must divide seq_len")
        assert_patch_boundaries(int(common["seq_len"]), patch_num)

        model_config = make_model_config(common, arm, enc_in=7)
        model = Model(model_config).eval()
        diagnostics = model_diagnostics(model)
        active = int(diagnostics["active_forward_parameters"])
        observed_active.append(active)
        if active != int(arm["active_forward_parameters"]):
            raise AssertionError(
                f"{arm_name}: active params {active} != "
                f"{arm['active_forward_parameters']}"
            )
        if int(diagnostics["unused_proj_x_parameters"]) != int(
            arm["unused_proj_x_parameters"]
        ):
            raise AssertionError(f"{arm_name}: unexpected unused proj_x count")

        memory = model.encode_history(x)
        expected_memory = (
            2,
            7,
            patch_num,
            int(arm["d_model"]),
        )
        if tuple(memory.shape) != expected_memory:
            raise AssertionError(
                f"{arm_name}: memory {tuple(memory.shape)} != {expected_memory}"
            )
        if memory.shape[-2] * memory.shape[-1] != int(arm["state_width"]):
            raise AssertionError(f"{arm_name}: state width mismatch")

        with torch.no_grad():
            prefix = model(x, y, is_training=False, target_prefix=96)[0]
            full = model(x, y, is_training=False, target_prefix=720)[0]
        if tuple(prefix.shape) != (2, 96, 7):
            raise AssertionError(f"{arm_name}: unexpected prefix output shape")
        torch.testing.assert_close(prefix, full[:, :96, :], rtol=0.0, atol=0.0)

        reloaded = Model(model_config).eval()
        reloaded.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            actual = reloaded(x, y, is_training=False, target_prefix=720)[0]
        torch.testing.assert_close(actual, full, rtol=0.0, atol=0.0)

    relative_spread = (max(observed_active) - min(observed_active)) / min(
        observed_active
    )
    if relative_spread > float(
        config["gates"]["active_parameter_relative_spread_max"]
    ):
        raise AssertionError(f"active parameter spread too large: {relative_spread}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_smoke(
    config_path: Path,
    config: dict[str, Any],
    dataset_root: Path,
    dataset: str,
    arm_name: str,
    arm: dict[str, Any],
    output_dir: Path,
) -> None:
    common = config["common"]
    config_hash = profile_hash(config_path)
    command = [
        sys.executable,
        str(TIMEALIGN_ROOT / "train_repo.py"),
        "--dataset-root",
        str(dataset_root),
        "--dataset",
        dataset,
        "--mode",
        "unified",
        "--seq-len",
        str(common["seq_len"]),
        "--pred-len",
        str(common["pred_len"]),
        "--target-horizons",
        ",".join(str(value) for value in common["target_horizons"]),
        "--validation-horizons",
        ",".join(str(value) for value in common["validation_horizons"]),
        "--evaluation-horizons",
        ",".join(str(value) for value in common["evaluation_horizons"]),
        "--batch-size",
        "2",
        "--gradient-accumulation-steps",
        "1",
        "--epochs",
        "1",
        "--seed",
        str(common["initial_seed"]),
        "--max-train-batches",
        "1",
        "--max-eval-batches",
        "1",
        "--num-workers",
        "0",
        "--run-name",
        f"SC0_local_{dataset}_{arm_name}",
        "--output-dir",
        str(output_dir),
        "--device",
        "cpu",
        "--checkpoint-policy",
        str(common["checkpoint_policy"]),
        "--evaluate-dual-checkpoints",
        "--final-evaluation-split",
        str(common["final_evaluation_split"]),
        "--protocol-class",
        str(config["protocol_class"]),
        "--protocol-profile",
        str(config["protocol_profile"]),
        "--profile-hash",
        config_hash,
        "--legacy-patch-num",
        str(arm["patch_num"]),
        "--legacy-d-model",
        str(arm["d_model"]),
        "--legacy-d-ff",
        str(arm["d_ff"]),
        "--legacy-dropout",
        str(common["dropout"]),
        "--legacy-layer-norm",
        str(common["layer_norm"]),
        "--learning-rate",
        str(common["learning_rate"]),
        "--readout-mode",
        str(common["readout_mode"]),
        "--basis-rank",
        str(common["basis_rank"]),
        "--pred-loss-mode",
        str(common["pred_loss_mode"]),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{dataset}/{arm_name} smoke failed\nSTDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    required = (
        "checkpoint.pt",
        "checkpoint_last.pt",
        "checkpoint_best_val.pt",
        "metrics_by_target_horizon.csv",
        "metrics_last_by_target_horizon.csv",
        "metrics_best_val_by_target_horizon.csv",
        "model_diagnostics.json",
        "effective_config.json",
        "predictions_val.npz",
    )
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise AssertionError(f"{dataset}/{arm_name}: missing artifacts {missing}")
    if (output_dir / "predictions_test.npz").exists():
        raise AssertionError(f"{dataset}/{arm_name}: test predictions are forbidden")

    effective = json.loads((output_dir / "effective_config.json").read_text())
    official = effective["official_args"]
    adapter = effective["adapter"]
    expected = (
        int(arm["patch_num"]),
        int(arm["d_model"]),
        int(arm["d_ff"]),
        float(common["dropout"]),
        int(common["layer_norm"]),
    )
    observed = (
        int(official["patch_num"]),
        int(official["d_model"]),
        int(official["d_ff"]),
        float(official["dropout"]),
        int(official["layer_norm"]),
    )
    if observed != expected:
        raise AssertionError(f"{dataset}/{arm_name}: config {observed} != {expected}")
    if adapter["protocol_class"] != "mechanism_control":
        raise AssertionError(f"{dataset}/{arm_name}: wrong protocol class")
    if adapter["profile_hash"] != config_hash:
        raise AssertionError(f"{dataset}/{arm_name}: wrong profile hash")

    for selector in ("last", "best_val"):
        rows = read_csv(output_dir / f"metrics_{selector}_by_target_horizon.csv")
        horizons = {int(row["target_horizon"]) for row in rows}
        if horizons != set(common["evaluation_horizons"]):
            raise AssertionError(
                f"{dataset}/{arm_name}: evaluation horizons {horizons}"
            )
        if any(row["evaluation_split"] != "val" for row in rows):
            raise AssertionError(f"{dataset}/{arm_name}: non-validation metric")

    enc_in = 21 if dataset == "Weather" else 7
    model_config = make_model_config(common, arm, enc_in=enc_in)
    for checkpoint_name in ("checkpoint_last.pt", "checkpoint_best_val.pt"):
        state = torch.load(
            output_dir / checkpoint_name,
            map_location="cpu",
            weights_only=True,
        )
        Model(model_config).load_state_dict(state, strict=True)


def end_to_end_checks(
    config_path: Path,
    config: dict[str, Any],
    dataset_root: Path,
) -> None:
    with tempfile.TemporaryDirectory(prefix="fatst-stagec-sc0-") as temp_dir:
        root = Path(temp_dir)
        for dataset in config["datasets"]:
            for arm_name, arm in config["arms"].items():
                run_smoke(
                    config_path,
                    config,
                    dataset_root,
                    dataset,
                    arm_name,
                    arm,
                    root / dataset / arm_name,
                )

        analyzer_root = root / "runner_layout"
        for dataset in config["datasets"]:
            for arm_name in config["arms"]:
                source = root / dataset / arm_name
                destination = (
                    analyzer_root
                    / f"SC0_{arm_name}_validation_only"
                    / dataset
                    / "h720_full"
                    / f"seed{config['common']['initial_seed']}"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.rename(destination)

        analysis_dir = root / "analysis"
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "analyze_stage_c_sc0_carrier_calibration.py"),
                "--raw-root",
                str(analyzer_root),
                "--output-dir",
                str(analysis_dir),
                "--config",
                str(config_path),
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SC0 analyzer smoke failed\nSTDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
        summary = json.loads((analysis_dir / "sc0_summary.json").read_text())
        if not summary["complete"]:
            raise AssertionError(f"SC0 analyzer reported incomplete: {summary}")
        if set(summary["winners"]) != {"best_val", "last"}:
            raise AssertionError(f"SC0 analyzer selector output mismatch: {summary}")
        if summary["test_metrics_used_for_selection"]:
            raise AssertionError("SC0 analyzer must not use test metrics")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    structural_checks(config)
    if not args.skip_end_to_end:
        end_to_end_checks(args.config, config, args.dataset_root)
    print(
        "stage_c_sc0_carrier_local_check=pass "
        f"profile_hash={profile_hash(args.config)}"
    )


if __name__ == "__main__":
    main()

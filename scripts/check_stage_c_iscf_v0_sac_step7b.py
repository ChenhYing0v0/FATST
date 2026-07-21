#!/usr/bin/env python3
"""Run the ISCF-v0 SAC Step 7B prelaunch gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from models import TimeAlign  # noqa: E402
import train_repo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_v0_scope_attribution_confirmation.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_post_d21_unconstrained_reset_20260720/"
            "iscf_v0_sac_step7b_prelaunch_20260721"
        ),
    )
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def add(
    rows: list[dict[str, Any]],
    category: str,
    case: str,
    passed: bool,
    value: Any,
    threshold: Any,
) -> None:
    rows.append(
        {
            "category": category,
            "case": case,
            "value": value,
            "threshold": threshold,
            "pass": bool(passed),
        }
    )


def run_command(
    command: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def parsed_model(
    profile: dict[str, Any],
    profile_hash: str,
    rank: int,
    readout: str,
    partition: str,
    seed: int,
    dataset_root: Path,
) -> tuple[TimeAlign.Model, torch.Tensor]:
    argv = [
        "train_repo.py",
        "--dataset-root",
        str(dataset_root),
        "--dataset",
        "ETTh1",
        "--mode",
        "unified",
        "--seq-len",
        "720",
        "--pred-len",
        "720",
        "--target-horizons",
        "720",
        "--validation-horizons",
        "96,192,336,720",
        "--evaluation-horizons",
        "96,192,336,720",
        "--segment-horizons",
        "96,192,336,720",
        "--evaluation-prefix-mode",
        "full-crop",
        "--e-layers",
        "2",
        "--batch-size",
        "2",
        "--epochs",
        "1",
        "--patience",
        "1",
        "--seed",
        str(seed),
        "--num-workers",
        "0",
        "--run-name",
        f"SAC_PRELAUNCH_{readout}_{partition}",
        "--output-dir",
        str(dataset_root / "output"),
        "--device",
        "cpu",
        "--checkpoint-policy",
        "best-val",
        "--protocol-class",
        "method_screening",
        "--protocol-profile",
        "stage_c_iscf_v0_sac_v1",
        "--profile-hash",
        profile_hash,
        "--legacy-patch-num",
        str(profile["patch_num"]),
        "--legacy-d-model",
        str(profile["d_model"]),
        "--legacy-d-ff",
        str(profile["d_ff"]),
        "--readout-mode",
        readout,
        "--pcsd-coordinate-dim",
        "4",
        "--pcsd-mode-rank",
        str(rank),
        "--pcsd-policy-mode",
        "direct",
        "--pcsd-partition",
        partition,
        "--pcsd-partition-seed",
        "15101",
        "--pcc-objective-mode",
        "equal_skill",
        "--pred-loss-mode",
        "full",
        "--final-evaluation-split",
        "val",
        "--no-save-predictions",
    ]
    original = sys.argv
    try:
        sys.argv = argv
        parsed = train_repo.parse_args()
    finally:
        sys.argv = original
    preset = train_repo.OFFICIAL_PRESETS["ETTh1"][720]
    (dataset_root / preset.data_path).parent.mkdir(parents=True, exist_ok=True)
    (dataset_root / preset.data_path).touch()
    official = train_repo.build_official_args(parsed, preset)
    torch.manual_seed(seed)
    model = TimeAlign.Model(official).float().eval()
    post_rng = torch.random.get_rng_state().clone()
    return model, post_rng


def historical_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in config["artifact_sources"]:
        if source["kind"] != "historical":
            continue
        audit_path = ROOT / source["run_audit"]
        if file_hash(audit_path) != source["run_audit_sha256"]:
            raise RuntimeError(f"run audit hash drift: {audit_path}")
        config_path = ROOT / source["config"]
        if file_hash(config_path) != source["config_sha256"]:
            raise RuntimeError(f"source config hash drift: {config_path}")
        aliases = source["arm_aliases"]
        for row in read_csv(audit_path):
            seed = int(row.get("seed", source["seeds"][0]))
            for arm in source["arms"]:
                if row["arm"] == aliases[arm] and seed in source["seeds"]:
                    rows.append(
                        {
                            "source": source["id"],
                            "dataset": row["dataset"],
                            "arm": arm,
                            "source_arm": row["arm"],
                            "seed": seed,
                            "status": row["status"],
                            "protocol_pass": row["protocol_pass"],
                            "checkpoint_sha256": row["checkpoint_sha256"],
                        }
                    )
    return rows


def main() -> None:
    args = parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = ROOT / args.output_dir
    rows: list[dict[str, Any]] = []
    profile_path = ROOT / config["profiles"]["path"]
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))[
        "dataset_profiles"
    ]

    add(
        rows,
        "contract",
        "profile_hash",
        file_hash(profile_path) == config["profiles"]["sha256"],
        file_hash(profile_path),
        config["profiles"]["sha256"],
    )
    expected_launch = {
        (seed, dataset, arm)
        for dataset in config["datasets"]
        for seed in config["seeds"]
        for arm in ["iscf_random_partition"]
    } | {
        (seed, dataset, "iscf_q1_wide")
        for dataset in config["datasets"]
        for seed in [2022, 2023]
    }
    launch = {tuple(row) for row in config["launch_order"]}
    add(rows, "contract", "launch_25_exact", launch == expected_launch, len(launch), 25)
    add(
        rows,
        "contract",
        "matrix_60_effective",
        config["matrix"]["expected_runs"] == 60
        and config["matrix"]["new_training_runs"] == 25
        and config["matrix"]["historical_reference_runs"] == 35,
        config["matrix"]["effective_runs"],
        60,
    )
    add(
        rows,
        "governance",
        "candidate_unchanged",
        config["arms"][0]["id"] == "iscf_v0"
        and all(
            config["arms"][0][key] == value
            for key, value in {
                "readout_mode": "siff-independent-scope-control",
                "policy_mode": "direct",
                "objective_mode": "equal_skill",
                "partition": "canonical",
            }.items()
        ),
        config["arms"][0]["id"],
        "frozen iscf_v0",
    )
    add(
        rows,
        "governance",
        "remote_not_authorized",
        config["authorization"]["remote_training_authorized"] is False,
        config["authorization"]["remote_training_authorized"],
        False,
    )
    add(
        rows,
        "governance",
        "formal_test_not_authorized",
        config["authorization"]["formal_test_access_authorized"] is False,
        config["authorization"]["formal_test_access_authorized"],
        False,
    )
    add(
        rows,
        "gate",
        "primary_margins_frozen",
        [row["mse_macro_gain_percent_min"] for row in config["primary_comparisons"]]
        == [0.5, 0.3],
        [row["mse_macro_gain_percent_min"] for row in config["primary_comparisons"]],
        [0.5, 0.3],
    )

    references = historical_rows(config)
    expected_references = {
        (seed, dataset, arm)
        for dataset in config["datasets"]
        for seed in config["seeds"]
        for arm in ["iscf_v0", "a6_full"]
    } | {
        (2021, dataset, "iscf_q1_wide") for dataset in config["datasets"]
    }
    actual_references = {
        (row["seed"], row["dataset"], row["arm"]) for row in references
    }
    add(
        rows,
        "reference",
        "historical_35_exact",
        actual_references == expected_references and len(references) == 35,
        len(references),
        35,
    )
    add(
        rows,
        "reference",
        "historical_protocol_pass",
        all(
            row["status"] == "ok" and row["protocol_pass"] == "True"
            for row in references
        ),
        sum(row["protocol_pass"] == "True" for row in references),
        35,
    )

    jobs = []
    arms = {arm["id"]: arm for arm in config["arms"]}
    for seed, dataset, arm_id in config["launch_order"]:
        arm = arms[arm_id]
        jobs.append(
            {
                "seed": seed,
                "dataset": dataset,
                "arm": arm_id,
                "role": arm["role"],
                "profile": profiles[dataset]["profile"],
                "rank": config["matched_ranks"][dataset][arm["rank_rule"]],
                "readout_mode": arm["readout_mode"],
                "partition": arm["partition"],
                "objective_mode": arm["objective_mode"],
            }
        )

    parameter_gaps: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="fatst_iscf_sac_step7b_") as temp:
        temp_root = Path(temp)
        contract_pass = True
        endpoint_pass = True
        middle_pass = True
        for dataset in config["datasets"]:
            profile = profiles[dataset]
            independent_rank = config["matched_ranks"][dataset][
                "independent_dataset_matched"
            ]
            canonical, canonical_rng = parsed_model(
                profile,
                config["profiles"]["sha256"],
                independent_rank,
                "siff-independent-scope-control",
                "canonical",
                2021,
                temp_root / dataset / "canonical",
            )
            random_model, random_rng = parsed_model(
                profile,
                config["profiles"]["sha256"],
                independent_rank,
                "siff-independent-scope-control",
                "random",
                2021,
                temp_root / dataset / "random",
            )
            q1, _ = parsed_model(
                profile,
                config["profiles"]["sha256"],
                config["matched_ranks"][dataset]["q1_dataset_matched"],
                "siff-q1-wide-control",
                "canonical",
                2021,
                temp_root / dataset / "q1",
            )
            canonical_init = train_repo.initialization_contract(canonical)
            random_init = train_repo.initialization_contract(random_model)
            canonical_diag = train_repo.model_diagnostics(canonical)
            random_diag = train_repo.model_diagnostics(random_model)
            q1_diag = train_repo.model_diagnostics(q1)
            contract_pass = contract_pass and bool(
                canonical_diag["active_forward_parameters"]
                == random_diag["active_forward_parameters"]
                and canonical_init["pcsd_initialization_hash"]
                == random_init["pcsd_initialization_hash"]
                and canonical_init["encoder_initialization_hash"]
                == random_init["encoder_initialization_hash"]
                and torch.equal(canonical_rng, random_rng)
                and canonical_init["pcsd_partition_hash"]
                != random_init["pcsd_partition_hash"]
            )
            for index, scale in enumerate(config["coupling_scales"]):
                same = torch.equal(
                    canonical.pcsd_readout.group_indices(index),
                    random_model.pcsd_readout.group_indices(index),
                )
                if scale in {1, 720}:
                    endpoint_pass = endpoint_pass and same
                else:
                    middle_pass = middle_pass and not same
            generator = torch.Generator().manual_seed(7101)
            x = torch.randn(2, 720, 7, generator=generator)
            with torch.no_grad():
                canonical_output = canonical(
                    x, torch.zeros_like(x), is_training=False, target_prefix=720
                )[0]
                random_output = random_model(
                    x, torch.zeros_like(x), is_training=False, target_prefix=720
                )[0]
            contract_pass = contract_pass and bool(
                canonical_output.shape == random_output.shape == (2, 720, 7)
                and torch.isfinite(canonical_output).all()
                and torch.isfinite(random_output).all()
                and not torch.equal(canonical_output, random_output)
            )
            iscf_parameters = canonical_diag["active_forward_parameters"]
            q1_parameters = q1_diag["active_forward_parameters"]
            parameter_gaps[dataset] = (
                100.0 * (iscf_parameters - q1_parameters) / iscf_parameters
            )
        add(
            rows,
            "model",
            "canonical_random_exact_contract",
            contract_pass,
            contract_pass,
            True,
        )
        add(rows, "model", "endpoint_partitions_match", endpoint_pass, endpoint_pass, True)
        add(rows, "model", "middle_partitions_differ", middle_pass, middle_pass, True)
        gap_pass = all(
            abs(parameter_gaps[dataset] - config["q1_active_parameter_audit"][dataset])
            <= 1e-10
            for dataset in config["datasets"]
        )
        add(
            rows,
            "model",
            "q1_parameter_gaps",
            gap_pass,
            max(abs(value) for value in parameter_gaps.values()),
            0.4646379821436204,
        )

    syntax = run_command(["bash", "-n", "scripts/remote/run_stage_c_iscf_v0_sac.sh"])
    add(rows, "execution", "runner_syntax", syntax.returncode == 0, syntax.stderr[-200:], 0)
    environment = dict(os.environ)
    environment.update(
        {
            "CONFIG": str(args.config),
            "DRY_RUN": "1",
            "PYTHON_BIN": sys.executable,
        }
    )
    dry_run = run_command(
        ["bash", "scripts/remote/run_stage_c_iscf_v0_sac.sh"], environment
    )
    dry_lines = [line for line in dry_run.stdout.splitlines() if line.count("\t") == 12]
    add(
        rows,
        "execution",
        "runner_dry_run_25",
        dry_run.returncode == 0
        and "iscf_sac_dry_run=pass jobs=25" in dry_run.stdout
        and len(dry_lines) == 25,
        len(dry_lines),
        25,
    )
    blocked_env = dict(os.environ)
    blocked_env.update({"CONFIG": str(args.config), "GPU_IDS": "0"})
    blocked = run_command(
        ["bash", "scripts/remote/run_stage_c_iscf_v0_sac.sh"], blocked_env
    )
    add(
        rows,
        "execution",
        "unauthorized_launch_blocked",
        blocked.returncode == 3 and "not authorized" in blocked.stderr,
        blocked.returncode,
        3,
    )
    analyzer = run_command(
        [
            sys.executable,
            "scripts/analyze_stage_c_iscf_v0_sac.py",
            "--config",
            str(args.config),
            "--synthetic-smoke",
        ]
    )
    add(
        rows,
        "execution",
        "analyzer_synthetic_smoke",
        analyzer.returncode == 0
        and "iscf_v0_sac_analyzer_synthetic_smoke=pass" in analyzer.stdout,
        analyzer.stdout.strip()[-200:],
        "returncode=0",
    )
    runner_text = (ROOT / "scripts/remote/run_stage_c_iscf_v0_sac.sh").read_text(
        encoding="utf-8"
    )
    add(
        rows,
        "execution",
        "remote_log_scanner_fallback",
        "command -v rg" in runner_text and "grep -Ein" in runner_text,
        "rg+grep" if "grep -Ein" in runner_text else "rg-only",
        "rg+grep",
    )

    passed = sum(bool(row["pass"]) for row in rows)
    summary = {
        "candidate": config["candidate_version"],
        "checks": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "pass": passed == len(rows),
        "new_runs": len(jobs),
        "historical_runs": len(references),
        "effective_runs": config["matrix"]["effective_runs"],
        "remote_training_authorized": False,
        "formal_test_authorized": False,
        "decision": (
            "step7b_prelaunch_pass_waiting_remote_authorization"
            if passed == len(rows)
            else "step7b_prelaunch_blocked"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "checks.csv", rows)
    write_csv(output_dir / "jobs.csv", jobs)
    write_csv(output_dir / "historical_reference_audit.csv", references)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not summary["pass"]:
        failed = [row["case"] for row in rows if not row["pass"]]
        raise RuntimeError(f"SAC Step7B gate failed: {failed}")
    print(f"iscf_v0_sac_step7b_prelaunch=pass checks={len(rows)} jobs=25")


if __name__ == "__main__":
    main()

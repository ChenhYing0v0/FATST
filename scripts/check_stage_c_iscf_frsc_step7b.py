#!/usr/bin/env python3
"""Run the local FRSC Step 7B prelaunch gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "baselines" / "timealign_official"
if str(OFFICIAL) not in sys.path:
    sys.path.insert(0, str(OFFICIAL))

from layers.SIFF import SIFFCouplingFieldReadout  # noqa: E402
from layers.SPS import FullRankScopeConditioningReadout  # noqa: E402
from analyze_stage_c_iscf_sps import (  # noqa: E402
    expected_projection_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_c_iscf_frsc_step7b.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis/stage_c_post_d21_unconstrained_reset_20260720/"
            "iscf_frsc_step7b_prelaunch_20260722"
        ),
    )
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for parameter in module.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def make_readout(
    seed: int,
    projection_mode: str,
    partition: str,
    mode_rank: int,
    strength: float,
) -> FullRankScopeConditioningReadout:
    torch.manual_seed(seed)
    return FullRankScopeConditioningReadout(
        readout_dim=32,
        series_length=720,
        coordinate_dim=4,
        mode_rank=mode_rank,
        projection_mode=projection_mode,
        conditioning_strength=strength,
        policy_mode="direct",
        partition=partition,
        partition_seed=15101,
    )


def make_parent(seed: int, mode_rank: int) -> SIFFCouplingFieldReadout:
    torch.manual_seed(seed)
    return SIFFCouplingFieldReadout(
        readout_dim=32,
        series_length=720,
        coordinate_dim=4,
        mode_rank=mode_rank,
        scale_components=5,
        scale_basis_mode="independent",
        policy_mode="direct",
        partition="canonical",
        partition_seed=15101,
    )


def run_command(
    command: list[str],
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, **(extra_env or {})},
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    profile_path = ROOT / config["profiles"]["path"]
    profile_hash = file_hash(profile_path)
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))[
        "dataset_profiles"
    ]
    train_arms = [arm for arm in config["arms"] if arm.get("train", True)]
    reference_arms = [arm for arm in config["arms"] if not arm.get("train", True)]

    add(
        rows,
        "contract",
        "profile_hash",
        profile_hash == config["profiles"]["sha256"],
        profile_hash,
        config["profiles"]["sha256"],
    )
    add(
        rows,
        "contract",
        "new_matrix_20_effective_25",
        config["matrix"]["expected_runs"] == 20
        and config["matrix"]["effective_runs"] == 25,
        f"{config['matrix']['expected_runs']}/{config['matrix']['effective_runs']}",
        "20/25",
    )
    add(
        rows,
        "contract",
        "four_new_one_frozen_reference",
        len(train_arms) == 4
        and len(reference_arms) == 1
        and reference_arms[0]["id"] == "sps_identity_canonical",
        f"new={len(train_arms)},reference={len(reference_arms)}",
        "4/1",
    )
    launch = list(map(tuple, config["launch_order"]))
    add(
        rows,
        "contract",
        "unique_launch_20",
        len(launch) == 20
        and len(set(launch)) == 20
        and {arm for _dataset, arm in launch}
        == {arm["id"] for arm in train_arms},
        len(launch),
        20,
    )
    authorization = config["authorization"]
    add(
        rows,
        "governance",
        "remote_authorized_validation_only",
        authorization["remote_training_authorized"] is True
        and authorization["formal_test_access_authorized"] is False,
        f"remote={authorization['remote_training_authorized']},test={authorization['formal_test_access_authorized']}",
        "remote=true,test=false",
    )
    training = config["training"]
    add(
        rows,
        "governance",
        "no_loss_router_requested_h",
        training["new_auxiliary_loss"] is False
        and training["new_router"] is False
        and training["requested_h_input"] is False,
        "loss=false,router=false,H=false",
        "all false",
    )

    hidden = torch.randn(2, 3, 32)
    hashes: list[str] = []
    for dataset in config["datasets"]:
        mode_rank = config["matched_ranks"][dataset][
            "independent_dataset_matched"
        ]
        for arm in train_arms:
            module = make_readout(
                2021,
                arm["projection_mode"],
                arm["partition"],
                mode_rank,
                arm["conditioning_strength"],
            )
            hashes.append(tensor_hash(module))
            with torch.no_grad():
                output = module(hidden)
                tensors = module.projection_diagnostics(hidden)
            expected_ranks, expected_degrees = expected_projection_contract(
                config,
                arm["projection_mode"],
                mode_rank,
            )
            actual_degrees = [
                int(rank * (720 // scale))
                if arm["projection_mode"] != "global"
                else int(rank)
                for rank, scale in zip(
                    module.projection_ranks,
                    config["scales"],
                    strict=True,
                )
            ]
            passed = bool(
                output.shape == (2, 720, 3)
                and all(torch.isfinite(value).all() for value in tensors.values())
                and list(module.projection_ranks) == expected_ranks
                and actual_degrees == expected_degrees
                and abs(
                    module.minimum_operator_eigenvalue
                    - (1.0 - arm["conditioning_strength"])
                )
                <= 1e-12
                and module.minimum_operator_eigenvalue > 0.0
            )
            add(
                rows,
                "model",
                f"{dataset}:{arm['id']}",
                passed,
                f"shape={list(output.shape)},min_eig={module.minimum_operator_eigenvalue}",
                "finite,matched projection,full rank",
            )

        parent = make_parent(2021, mode_rank)
        identity = make_readout(2021, "scope", "canonical", mode_rank, 0.0)
        with torch.no_grad():
            identity_gap = float((parent(hidden) - identity(hidden)).abs().max())
        add(
            rows,
            "model",
            f"{dataset}:alpha0_parent_identity",
            identity_gap <= 1e-6,
            identity_gap,
            1e-6,
        )
    add(
        rows,
        "model",
        "paired_new_arm_initialization",
        len(set(hashes))
        == len(
            {
                config["matched_ranks"][dataset][
                    "independent_dataset_matched"
                ]
                for dataset in config["datasets"]
            }
        ),
        len(set(hashes)),
        "one hash per dataset rank",
    )

    runner = "scripts/remote/run_stage_c_iscf_frsc_step7b.sh"
    syntax = run_command(["bash", "-n", runner])
    dry = run_command(["bash", runner], {"DRY_RUN": "1"})
    test_blocked = run_command(
        ["bash", runner],
        {"EVALUATION_SPLIT": "test", "DRY_RUN": "1"},
    )
    analyzer = run_command(
        [
            sys.executable,
            "scripts/analyze_stage_c_iscf_sps.py",
            "--config",
            str(args.config),
            "--synthetic-smoke",
        ]
    )
    add(rows, "execution", "runner_syntax", syntax.returncode == 0, syntax.returncode, 0)
    add(
        rows,
        "execution",
        "runner_dry_run_20",
        dry.returncode == 0
        and "jobs=20" in dry.stdout
        and "remote_authorized=true" in dry.stdout,
        dry.stdout.strip().splitlines()[-1] if dry.stdout.strip() else "",
        "jobs=20,remote_authorized=true",
    )
    add(
        rows,
        "execution",
        "test_split_blocked",
        test_blocked.returncode == 3
        and "validation-only" in test_blocked.stderr,
        f"{test_blocked.returncode}:{test_blocked.stderr.strip()}",
        "returncode=3",
    )
    add(
        rows,
        "execution",
        "analyzer_synthetic_smoke",
        analyzer.returncode == 0,
        analyzer.stdout.strip()[-200:],
        "returncode=0",
    )
    runner_text = (ROOT / "scripts/remote/run_stage_c_iscf_sps_step7b.sh").read_text(
        encoding="utf-8"
    )
    add(
        rows,
        "execution",
        "gpu_audit_and_failure_scan",
        "nvidia-smi --query-gpu" in runner_text
        and "grep -Ein" in runner_text,
        "nvidia-smi+scanner",
        "present",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "checks.csv", rows)
    jobs = []
    arms = {arm["id"]: arm for arm in train_arms}
    for index, (dataset, arm_id) in enumerate(config["launch_order"], start=1):
        arm = arms[arm_id]
        profile = profiles[dataset]
        jobs.append(
            {
                "job": index,
                "dataset": dataset,
                "arm": arm_id,
                "projection_mode": arm["projection_mode"],
                "partition": arm["partition"],
                "conditioning_strength": arm["conditioning_strength"],
                "mode_rank": config["matched_ranks"][dataset][arm["rank_rule"]],
                "profile": profile["profile"],
                "seed": 2021,
                "split": "validation",
            }
        )
    write_csv(args.output_dir / "jobs.csv", jobs)
    passed = sum(bool(row["pass"]) for row in rows)
    summary = {
        "candidate": config["candidate_version"],
        "checks": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "pass": passed == len(rows),
        "new_jobs": len(jobs),
        "effective_runs": config["matrix"]["effective_runs"],
        "remote_training_authorized": True,
        "formal_test_authorized": False,
        "decision": (
            "step8_remote_validation_authorized"
            if passed == len(rows)
            else "step7b_blocked"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = f"""# FRSC Step7B Prelaunch Audit

## 1. What is tested and why

FRSC-v0 tests whether invertible scope conditioning can preserve the ISCF carrier while making canonical scope geometry useful. The candidate must improve on the frozen identity reference and cannot attribute a gain to scope structure unless it also beats same-alpha global, best-tuned global, and random-binding controls.

## 2. Matrix and artifact construction

- new training: four arms x five datasets x seed2021 = `{len(jobs)}` from-scratch matched runs;
- effective audit: add five historical `sps_identity_canonical` checkpoints = `25` runs;
- checkpoint selector: mean validation MSE over H96/H192/H336/H720;
- required outputs: checkpoint, training log, four-horizon metrics, effective config, initialization contract, model diagnostics, validation diagnostic tensors, and trained invariants;
- split boundary: train optimizes the frozen full-H720 objective; validation selects checkpoints and screens the mechanism; official test is disabled.

## 3. Statistics and controls

`effectiveness_vs_identity` is the primary carrier-preservation comparison. `scope_specificity_vs_global` compares against alpha .45, while `same_alpha_scope_vs_global` isolates geometry at alpha .55. `canonical_binding_vs_random` is attribution-only. Internal health reports conditioned-arm pairwise RMS normalized by target RMS, oracle headroom, normalized policy entropy, future-bin winner count, and conditioning-delta/raw RMS. These diagnostics cannot override a negative effectiveness gate.

## 4. Prelaunch evidence

- decision: `{summary['decision']}`;
- checks: `{passed}/{len(rows)}`;
- all model arms satisfy output shape, finite values, matched projection ranks/degrees, and strictly positive minimum operator eigenvalue;
- alpha0 reproduces the ISCF parent, and trainable initialization is paired within each dataset rank;
- runner dry-run enumerates 20 unique jobs, formal-test split exits with code 3, analyzer synthetic smoke passes, and GPU/log scanning is present.

## 5. Failure attribution and decision

Candidate below identity without pathology means exact FRSC-v0 is not supported and returns to Step4; global tie means scope specificity is unresolved; random tie/loss blocks canonical binding attribution but cannot reject ISCF; diversity/oracle collapse points to intervention/readout design; NaN/OOM/divergence is an optimization or numeric pathology and only blocks the exact execution. Current decision is `{summary['decision']}`: remote validation may start after commit-pinned pull, GPU/process preflight, and resource smoke. Formal test, confirmation seeds, modern baselines, new loss/router, and requested-H input remain unauthorized.
"""
    (args.output_dir / "prelaunch_report.md").write_text(report, encoding="utf-8")
    if not summary["pass"]:
        failed = [row["case"] for row in rows if not row["pass"]]
        raise RuntimeError(f"FRSC Step7B gate failed: {failed}")
    print(
        f"iscf_frsc_step7b_prelaunch=pass checks={len(rows)} jobs={len(jobs)}"
    )


if __name__ == "__main__":
    main()

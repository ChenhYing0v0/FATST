#!/usr/bin/env python3
"""Run the SC-D20-CST Step 7B prelaunch gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_stage_c_pcsd_cf_checkpoint import (  # noqa: E402
    test_audit_authorized,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def check(condition: bool, name: str, details: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(condition), "details": details}


def run_command(command: list[str], env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout + completed.stderr


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    results = []

    contract_rows = []
    contract_pass = True
    for prefix in (
        "step6",
        "step7a",
        "step7a_gate",
        "model",
        "trainer",
        "evaluator",
        "analyzer",
        "runner",
    ):
        path = Path(config["contracts"][f"{prefix}_path"])
        computed = sha256(path)
        frozen = config["contracts"][f"{prefix}_sha256"]
        contract_rows.append(
            {"contract": prefix, "computed": computed, "frozen": frozen}
        )
        contract_pass = contract_pass and computed == frozen
    results.append(check(contract_pass, "frozen_contract_hashes", contract_rows))

    step7a_gate = json.loads(
        Path(config["contracts"]["step7a_gate_path"]).read_text(
            encoding="utf-8"
        )
    )
    results.append(
        check(
            step7a_gate["overall_pass"] is True
            and step7a_gate["checks_passed"] == step7a_gate["checks_total"]
            and step7a_gate["counts"]
            == {
                "cli_cases": 15,
                "model_constructors": 15,
                "shape_prefix_cases": 60,
                "summary_gradient_cases": 10,
            },
            "step7a_gate_dependency",
            {
                "checks": [
                    step7a_gate["checks_passed"],
                    step7a_gate["checks_total"],
                ],
                "counts": step7a_gate["counts"],
            },
        )
    )

    expected_pairs = {
        (dataset, arm["id"])
        for dataset in config["datasets"]
        for arm in config["arms"]
    }
    launch_pairs = {tuple(row) for row in config["launch_order"]}
    results.append(
        check(
            len(config["launch_order"]) == 15
            and launch_pairs == expected_pairs
            and config["matrix"]["expected_runs"] == 15
            and config["matrix"]["official_test_cells"] == 60,
            "complete_unique_matrix",
            config["launch_order"],
        )
    )
    results.append(
        check(
            test_audit_authorized(config),
            "formal_test_authorization_contract",
            config["authorization"],
        )
    )
    results.append(
        check(
            config["authorization"]["confirmation"] is False
            and config["authorization"]["paper_method"] is False
            and config["paper_method_authorized"] is False,
            "promotion_boundary",
            {
                "confirmation": config["authorization"]["confirmation"],
                "paper_method": config["authorization"]["paper_method"],
            },
        )
    )

    runner_path = config["contracts"]["runner_path"]
    run_command(["bash", "-n", runner_path])
    results.append(check(True, "runner_shell_syntax", runner_path))

    dry_env = dict(os.environ)
    dry_env["DRY_RUN"] = "1"
    dry_env["CONFIG"] = str(args.config)
    dry_output = run_command(["bash", runner_path], env=dry_env)
    dry_lines = [
        line for line in dry_output.splitlines() if line and "\t" in line
    ]
    results.append(
        check(
            len(dry_lines) == 15
            and "d20_dry_run=pass jobs=15" in dry_output
            and "remote_authorized=true" in dry_output
            and "test_authorized=true" in dry_output,
            "runner_dry_run",
            {"job_lines": len(dry_lines), "footer": dry_output.splitlines()[-1]},
        )
    )

    evaluator_output = run_command(
        [
            sys.executable,
            config["contracts"]["evaluator_path"],
            "--synthetic-smoke",
        ]
    )
    results.append(
        check(
            "d20_checkpoint_evaluator_synthetic_smoke=pass"
            in evaluator_output,
            "checkpoint_evaluator_smoke",
            evaluator_output.strip().splitlines(),
        )
    )
    analyzer_output = run_command(
        [
            sys.executable,
            config["contracts"]["analyzer_path"],
            "--config",
            str(args.config),
            "--synthetic-smoke",
        ]
    )
    results.append(
        check(
            "d20_analyzer_synthetic_smoke=pass" in analyzer_output,
            "analyzer_smoke",
            analyzer_output.strip(),
        )
    )
    results.append(
        check(
            set(config["effectiveness_gates"])
            == {"transfer_spec_vs_a6", "specificity_spec_vs_random"}
            and set(config["decision_map"])
            == {
                "spec_fails_a6_internal_valid",
                "spec_beats_a6_not_random",
                "internal_invalid",
                "all_pass",
            },
            "gates_and_rollback_map",
            {
                "effectiveness": sorted(config["effectiveness_gates"]),
                "decision_map": sorted(config["decision_map"]),
            },
        )
    )

    summary = {
        "candidate_version": config["candidate_version"],
        "config_path": str(args.config),
        "config_sha256": sha256(args.config),
        "checks_passed": sum(item["pass"] for item in results),
        "checks_total": len(results),
        "overall_pass": all(item["pass"] for item in results),
        "matrix": {
            "runs": 15,
            "official_test_cells": 60,
            "datasets": 5,
            "arms": 3,
            "seed": 2021,
        },
        "authorization": config["authorization"],
        "checks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

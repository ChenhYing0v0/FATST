#!/usr/bin/env python3
"""Check the frozen ETTh1-only H5D HPO contract."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_iscf_bsca_main_v1_hpo import materialize_jobs


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "iscf_bsca_main_v1_hpo_etth1_h5d.json"
HISTORICAL_CONFIGS = (
    "configs/iscf_bsca_main_v1_hpo.json",
    "configs/iscf_bsca_main_v1_hpo_h2.json",
    "configs/iscf_bsca_main_v1_hpo_joint_h4j.json",
    "configs/iscf_bsca_main_v1_hpo_targeted_h4k.json",
    "configs/iscf_bsca_main_v1_hpo_main_ii_h5a.json",
    "configs/iscf_bsca_main_v1_hpo_etth1_h5b.json",
    "configs/iscf_bsca_main_v1_hpo_etth1_h5c.json",
)
PROFILE_FIELDS = (
    "dataset",
    "seq_len",
    "patch_num",
    "d_model",
    "d_ff",
    "dropout",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "gradient_accumulation_steps",
    "mode_rank",
    "layer_norm",
    "max_epochs",
    "early_stopping_patience",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_fingerprint(
    job: dict[str, Any], config: dict[str, Any]
) -> tuple[Any, ...]:
    resolved = dict(job)
    resolved.setdefault("layer_norm", 1)
    resolved.setdefault("max_epochs", config["training"]["max_epochs"])
    resolved.setdefault(
        "early_stopping_patience",
        config["training"]["early_stopping_patience"],
    )
    return tuple(resolved[field] for field in PROFILE_FIELDS)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["matrix"]["phase"] == "H5D"
    assert config["status"] == "frozen_authorized_prelaunch"
    assert config["architecture_search"] is False
    assert config["architecture_invariants"]["inference_graph_changed"] is False
    assert config["training"]["official_test_during_training"] is False
    assert config["matrix"]["expected_training_runs"] == 48
    assert config["matrix"]["expected_test_runs_during_training"] == 0
    assert config["hpo_budget"]["H5D_search_space_frozen"] is True
    authorization = config["authorization"]
    assert authorization["remote_H5D_training_authorized"] is True
    assert (
        authorization[
            "official_test_H5D_execution_authorized_after_complete_training_manifest"
        ]
        is False
    )
    assert authorization["automatic_table_mutation_after_H5D"] is False

    for key in ("main_i_table_data", "main_ii_table_data"):
        path = ROOT / config["frozen_target_artifacts"][key]
        expected = config["frozen_target_artifacts"][f"{key}_sha256"]
        assert sha256(path) == expected
    evidence = config["existing_evidence"]
    for key in (
        "history_scorecard",
        "history_analysis",
        "h5c_config",
        "h5c_selection_result",
    ):
        path = ROOT / evidence[key]
        assert sha256(path) == evidence[f"{key}_sha256"]

    jobs = materialize_jobs(config, CONFIG)
    assert len(jobs) == 48
    assert {job["dataset"] for job in jobs} == {"ETTh1"}
    assert len({job["trial_id"] for job in jobs}) == 48
    assert {job["trial_id"] for job in jobs} == set(
        config["provisional_lpt_order"]
    )
    fingerprints = {effective_fingerprint(job, config) for job in jobs}
    assert len(fingerprints) == 48

    historical_fingerprints: set[tuple[Any, ...]] = set()
    historical_etth1_jobs = 0
    for relative_path in HISTORICAL_CONFIGS:
        path = ROOT / relative_path
        historical_config = json.loads(path.read_text(encoding="utf-8"))
        for job in materialize_jobs(historical_config, path):
            if job["dataset"] != "ETTh1":
                continue
            historical_etth1_jobs += 1
            historical_fingerprints.add(
                effective_fingerprint(job, historical_config)
            )
    assert historical_etth1_jobs == 115
    assert fingerprints.isdisjoint(historical_fingerprints)

    expected_groups = {
        "dropout0_batch_lr_interaction": 12,
        "h192_p19_dropout0_interaction": 8,
        "h336_p21_dropout0_interaction": 8,
        "dropout0_rank_refinement": 8,
        "h192_geometry_rank_dropout0": 8,
        "h336_geometry_rank_dropout0": 4,
    }
    assert Counter(job["source_prior"] for job in jobs) == expected_groups
    search = config["search_space"]
    for job in jobs:
        assert job["seq_len"] in search["seq_len"]
        assert job["patch_num"] in search["patch_num"]
        assert job["seq_len"] % job["patch_num"] == 0
        assert job["d_model"] == job["d_ff"] == 32
        assert job["dropout"] == 0.0
        assert job["learning_rate"] in search["learning_rate"]
        assert job["weight_decay"] == 0.01
        assert job["batch_size"] in search["batch_size"]
        assert job["gradient_accumulation_steps"] == 1
        assert job["mode_rank"] in search["mode_rank"]
        assert job["layer_norm"] == 1
        assert effective_fingerprint(job, config)[-2:] == (120, 24)

    result = subprocess.run(
        ["bash", "scripts/remote/run_iscf_bsca_main_v1_hpo_etth1_h5d.sh"],
        cwd=ROOT,
        env={**os.environ, "MODE": "dry-run"},
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "iscf_bsca_main_h5d_dry_run=pass" in result
    assert "jobs=48" in result and "test_jobs=0" in result
    assert "remote_authorized=true" in result

    print(
        json.dumps(
            {
                "candidate_version": config["candidate_version"],
                "jobs": len(jobs),
                "groups": expected_groups,
                "historical_ETTh1_jobs_audited": historical_etth1_jobs,
                "historical_effective_profile_duplicates": 0,
                "training_test_jobs": 0,
                "remote_gpu_workers": 3,
                "formal_test_after_complete_manifest_authorized": False,
                "automatic_table_mutation_authorized": False,
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

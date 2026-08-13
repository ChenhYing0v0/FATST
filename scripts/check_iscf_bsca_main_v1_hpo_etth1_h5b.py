#!/usr/bin/env python3
"""Check the frozen ETTh1-only H5B HPO contract."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "iscf_bsca_main_v1_hpo_etth1_h5b.json"
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for specification in config["jobs"]:
        job = dict(config["base_profiles"][specification["base_profile_id"]])
        job.update(specification.get("overrides", {}))
        job.update(
            {
                key: value
                for key, value in specification.items()
                if key not in {"base_profile_id", "overrides"}
            }
        )
        job.setdefault("max_epochs", config["training"]["max_epochs"])
        job.setdefault(
            "early_stopping_patience",
            config["training"]["early_stopping_patience"],
        )
        jobs.append(job)
    return jobs


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["matrix"]["phase"] == "H5B"
    assert config["status"] == "frozen_authorized_prelaunch"
    assert config["architecture_search"] is False
    assert config["architecture_invariants"]["inference_graph_changed"] is False
    assert config["training"]["official_test_during_training"] is False
    assert config["matrix"]["expected_training_runs"] == 36
    assert config["matrix"]["expected_test_runs_during_training"] == 0
    assert config["hpo_budget"]["H5B_search_space_frozen"] is True
    assert config["authorization"]["remote_H5B_training_authorized"] is True
    assert config["authorization"]["automatic_table_mutation_after_H5B"] is False

    targets = config["frozen_target_artifacts"]
    main_i = ROOT / targets["main_i_table_data"]
    main_ii = ROOT / targets["main_ii_table_data"]
    assert sha256(main_i) == targets["main_i_table_data_sha256"]
    assert sha256(main_ii) == targets["main_ii_table_data_sha256"]
    ranking = ROOT / config["existing_evidence"]["h5a_profile_ranking"]
    assert sha256(ranking) == config["existing_evidence"]["h5a_profile_ranking_sha256"]
    assert len([row for row in read_csv(ranking) if row["dataset"] == "ETTh1"]) == 25

    jobs = resolve_jobs(config)
    assert len(jobs) == 36
    assert {job["dataset"] for job in jobs} == {"ETTh1"}
    assert len({job["trial_id"] for job in jobs}) == 36
    assert {job["trial_id"] for job in jobs} == set(config["provisional_lpt_order"])
    fingerprints = {tuple(job[field] for field in PROFILE_FIELDS) for job in jobs}
    assert len(fingerprints) == 36
    for job in jobs:
        assert job["seq_len"] in config["search_space"]["seq_len"]
        assert job["patch_num"] in config["search_space"]["patch_num"]
        assert job["seq_len"] % job["patch_num"] == 0
        assert job["d_model"] % 2 == 0
        assert 0.0 <= job["dropout"] < 1.0
        assert job["learning_rate"] > 0.0
        assert job["weight_decay"] >= 0.0
        assert job["batch_size"] * job["gradient_accumulation_steps"] == 32
        assert job["layer_norm"] == 1
        assert job["max_epochs"] == 120
        assert job["early_stopping_patience"] == 24

    assert min(job["seq_len"] for job in jobs) == 576
    assert max(job["seq_len"] for job in jobs) == 960
    assert {job["mode_rank"] for job in jobs} >= {80, 96, 109, 128, 160}
    assert {job["learning_rate"] for job in jobs} >= {
        0.00032,
        0.00033,
        0.00034,
        0.00035,
        0.00036,
        0.00037,
        0.00038,
        0.0004,
        0.00042,
    }

    result = subprocess.run(
        ["bash", "scripts/remote/run_iscf_bsca_main_v1_hpo_etth1_h5b.sh"],
        cwd=ROOT,
        env={**os.environ, "MODE": "dry-run"},
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "iscf_bsca_main_h5b_dry_run=pass" in result
    assert "jobs=36" in result and "test_jobs=0" in result
    assert "remote_authorized=true" in result

    print(
        json.dumps(
            {
                "candidate_version": config["candidate_version"],
                "jobs": len(jobs),
                "datasets": ["ETTh1"],
                "seq_len_range": [
                    min(job["seq_len"] for job in jobs),
                    max(job["seq_len"] for job in jobs),
                ],
                "training_test_jobs": 0,
                "remote_gpu_workers": 3,
                "formal_test_after_complete_manifest_authorized": True,
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

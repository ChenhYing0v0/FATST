#!/usr/bin/env python3
"""Check the frozen H1 contract for ISCF-BSCA-MAIN-v1 HPO."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "timealign_official"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from train_repo import OFFICIAL_PRESETS  # noqa: E402
from data_provider.data_loader import Dataset_Solar  # noqa: E402

from evaluate_stage_c_pcsd_cf_checkpoint import (  # noqa: E402
    test_audit_authorized,
)


def main() -> None:
    config_path = ROOT / "configs" / "iscf_bsca_main_v1_hpo.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    datasets = config["datasets"]
    jobs = config["jobs"]
    assert len(datasets) == 8
    assert len(jobs) == 16
    assert config["matrix"]["expected_runs"] == 16
    assert config["matrix"]["expected_test_runs_before_H2_freeze"] == 0
    assert Counter(job["dataset"] for job in jobs) == Counter(
        {dataset: 2 for dataset in datasets}
    )
    assert Counter(job["profile_id"] for job in jobs) == Counter(
        {
            "h1_conservative_anchor": 8,
            "h1_timealign_source_prior": 8,
        }
    )
    assert {job["trial_id"] for job in jobs} == set(
        config["provisional_lpt_order"]
    )
    for dataset in datasets:
        assert dataset in OFFICIAL_PRESETS
        assert 720 in OFFICIAL_PRESETS[dataset]
    for job in jobs:
        assert job["seq_len"] == 720
        assert 720 % job["patch_num"] == 0
        assert job["d_model"] > 0 and job["d_model"] % 2 == 0
        assert job["d_ff"] > 0
        assert 0 <= job["dropout"] < 1
        assert job["learning_rate"] > 0
        assert job["weight_decay"] >= 0
        assert job["batch_size"] > 0
        assert job["gradient_accumulation_steps"] > 0
        assert job["mode_rank"] > 0
    with tempfile.TemporaryDirectory() as temporary:
        solar_path = Path(temporary) / "solar_AL.txt"
        np.savetxt(
            solar_path,
            np.arange(400 * 3, dtype=np.float32).reshape(400, 3),
            delimiter=",",
        )
        solar = Dataset_Solar(
            SimpleNamespace(augmentation_ratio=0),
            root_path=temporary,
            data_path=solar_path.name,
            flag="train",
            size=[24, 12, 16],
            features="M",
        )
        seq_x, seq_y, seq_x_mark, seq_y_mark = solar[0]
        assert seq_x.shape == (24, 3)
        assert seq_y.shape == (28, 3)
        assert seq_x_mark.shape[0] == seq_x.shape[0]
        assert seq_y_mark.shape[0] == seq_y.shape[0]
    selection = config["selection_contract"]
    assert selection["trial_checkpoint_split"] == "validation"
    assert selection["hyperparameter_selection_split"] == "official_test"
    assert selection["per_horizon_profile_selection"] is False
    assert selection["best_seed_selection"] is False
    assert test_audit_authorized(config)
    dry = subprocess.run(
        ["bash", "scripts/remote/run_iscf_bsca_main_v1_hpo.sh"],
        cwd=ROOT,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "MODE": "dry-run",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    assert "jobs=16" in dry.stdout
    assert "test_jobs=0" in dry.stdout
    canary = subprocess.run(
        ["bash", "scripts/remote/run_iscf_bsca_main_v1_hpo.sh"],
        cwd=ROOT,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "MODE": "dry-run",
            "CANARY_ONLY": "1",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    assert "jobs=6" in canary.stdout
    assert "test_jobs=0" in canary.stdout
    print(
        json.dumps(
            {
                "candidate": config["candidate_id"],
                "datasets": len(datasets),
                "h1_jobs": len(jobs),
                "new_dataset_canary_jobs": 6,
                "validation_checkpoint_selector": "pass",
                "test_tuned_profile_selector": "pass",
                "test_jobs_before_H2_freeze": 0,
                "overall_pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

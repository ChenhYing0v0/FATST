#!/usr/bin/env python3
"""Run local invariants for the D21-EVS diagnostic implementation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from analyze_stage_c_d21_evs import evaluate_policies
from evaluate_stage_c_d21_evs_checkpoint import history_descriptor


REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = REPO_ROOT / "configs" / "stage_c_d21_evidence_validity_surface.json"


def main() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    generator = np.random.default_rng(20260720)
    torch.manual_seed(20260720)

    history = torch.randn(17, 720)
    features, names = history_descriptor(
        history,
        design["history_descriptor"],
    )
    expected_feature_count = 4 + 60 + 48 + 64 + 8 + 8
    assert features.shape == (17, expected_feature_count)
    assert len(names) == expected_feature_count
    assert torch.isfinite(features).all()

    validation_rows = 2048
    test_rows = 1024
    feature_count = 12
    bin_count = 3
    arm_count = 5
    x_validation = generator.normal(size=(validation_rows, feature_count))
    x_test = generator.normal(size=(test_rows, feature_count))

    def make_losses(x: np.ndarray) -> np.ndarray:
        rows = x.shape[0]
        losses = np.full((rows, bin_count, arm_count), 1.0, dtype=np.float64)
        for bin_index in range(bin_count):
            preferred = (
                (x[:, 0] > 0).astype(np.int64) + 2 * bin_index
            ) % arm_count
            losses[np.arange(rows), bin_index, preferred] = 0.55
            losses[:, bin_index, :] += 0.02 * generator.random(
                (rows, arm_count)
            )
        return losses

    validation_losses = make_losses(x_validation)
    test_losses = make_losses(x_test)
    results = evaluate_policies(
        x_validation,
        x_test,
        validation_losses,
        test_losses,
        np.asarray([144.0, 216.0, 360.0]),
        "ridge",
        design["readouts"]["primary"],
        20260720,
    )
    by_name = {result.name: result.mse for result in results}
    assert by_name["evs_interaction"] < by_name["region_fixed"]
    assert by_name["evs_interaction"] < by_name["history_global"]
    assert by_name["evs_interaction"] < by_name["permuted_history"]
    assert by_name["oracle"] <= by_name["evs_interaction"]

    assert design["authorization"]["new_forecasting_model_training"] is False
    assert design["authorization"]["paper_method_implementation"] is False
    assert design["authorization"]["uses_validation_split_for_fit"] is True
    assert design["authorization"]["uses_test_split_for_evaluation"] is True
    print(
        "d21_evs_step7a=pass "
        f"features={expected_feature_count} synthetic_policies=7"
    )


if __name__ == "__main__":
    main()

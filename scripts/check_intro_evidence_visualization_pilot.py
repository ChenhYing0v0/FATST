#!/usr/bin/env python3
"""Local contract check for the Introduction visualization pilot."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.dlinear.dataset import DATASETS
from baselines.intro_evidence_neutral.model import (
    NeutralSharingExtentForecaster,
)


HORIZONS = (96, 192, 336, 720)
SCALES = (1, 8, 32, 128, 720)
FULL_SEARCH_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def check_dataset_registry() -> dict[str, list[str] | bool]:
    missing = sorted(set(FULL_SEARCH_DATASETS) - set(DATASETS))
    if missing:
        raise RuntimeError(f"full-search datasets missing from registry: {missing}")
    invalid_channels = {
        dataset: DATASETS[dataset].channels
        for dataset in FULL_SEARCH_DATASETS
        if DATASETS[dataset].channels <= 0
    }
    if invalid_channels:
        raise RuntimeError(f"invalid dataset channel counts: {invalid_channels}")
    return {
        "registered_datasets": list(FULL_SEARCH_DATASETS),
        "pass": True,
    }


def reference_pooled_states(
    model: NeutralSharingExtentForecaster,
    candidate_states: torch.Tensor,
) -> torch.Tensor:
    pooled_parts = []
    for start in range(0, model.pred_len, model.sharing_extent):
        end = min(start + model.sharing_extent, model.pred_len)
        state = candidate_states[:, :, start:end, :].mean(dim=2)
        state = model.pooled_state_norm(state)
        pooled_parts.append(
            state.unsqueeze(2).expand(-1, -1, end - start, -1)
        )
    return torch.cat(pooled_parts, dim=2)


def check_vectorized_pooling_equivalence() -> dict[str, float | bool]:
    torch.manual_seed(2021)
    maximum_output_gap = 0.0
    maximum_candidate_gradient_gap = 0.0
    maximum_weight_gradient_gap = 0.0
    maximum_bias_gradient_gap = 0.0
    for scale in SCALES:
        model = NeutralSharingExtentForecaster(
            seq_len=96,
            pred_len=720,
            sharing_extent=scale,
        ).double()
        candidate_vector = torch.randn(
            2,
            3,
            720,
            model.state_dim,
            dtype=torch.float64,
            requires_grad=True,
        )
        candidate_reference = (
            candidate_vector.detach().clone().requires_grad_(True)
        )
        upstream = torch.randn_like(candidate_vector)
        vectorized = model.pooled_states(candidate_vector)
        reference = reference_pooled_states(model, candidate_reference)
        maximum_output_gap = max(
            maximum_output_gap,
            float(
                torch.max(
                    torch.abs(vectorized - reference)
                ).detach()
            ),
        )
        vector_gradients = torch.autograd.grad(
            (vectorized * upstream).sum(),
            (
                candidate_vector,
                model.pooled_state_norm.weight,
                model.pooled_state_norm.bias,
            ),
        )
        reference_gradients = torch.autograd.grad(
            (reference * upstream).sum(),
            (
                candidate_reference,
                model.pooled_state_norm.weight,
                model.pooled_state_norm.bias,
            ),
        )
        maximum_candidate_gradient_gap = max(
            maximum_candidate_gradient_gap,
            float(
                torch.max(
                    torch.abs(
                        vector_gradients[0] - reference_gradients[0]
                    )
                ).detach()
            ),
        )
        maximum_weight_gradient_gap = max(
            maximum_weight_gradient_gap,
            float(
                torch.max(
                    torch.abs(
                        vector_gradients[1] - reference_gradients[1]
                    )
                ).detach()
            ),
        )
        maximum_bias_gradient_gap = max(
            maximum_bias_gradient_gap,
            float(
                torch.max(
                    torch.abs(
                        vector_gradients[2] - reference_gradients[2]
                    )
                ).detach()
            ),
        )
    tolerance = 1e-10
    if max(
        maximum_output_gap,
        maximum_candidate_gradient_gap,
        maximum_weight_gradient_gap,
        maximum_bias_gradient_gap,
    ) > tolerance:
        raise RuntimeError(
            "vectorized pooling is not equivalent to reference: "
            f"output={maximum_output_gap}, "
            f"candidate_grad={maximum_candidate_gradient_gap}, "
            f"weight_grad={maximum_weight_gradient_gap}, "
            f"bias_grad={maximum_bias_gradient_gap}"
        )
    return {
        "maximum_output_gap": maximum_output_gap,
        "maximum_candidate_gradient_gap": maximum_candidate_gradient_gap,
        "maximum_weight_gradient_gap": maximum_weight_gradient_gap,
        "maximum_bias_gradient_gap": maximum_bias_gradient_gap,
        "pass": True,
    }


def check_model_contract() -> dict[str, float | int | bool]:
    torch.manual_seed(2021)
    models = {
        scale: NeutralSharingExtentForecaster(
            seq_len=96,
            pred_len=720,
            sharing_extent=scale,
        )
        for scale in SCALES
    }
    state = models[1].state_dict()
    for scale in SCALES[1:]:
        models[scale].load_state_dict(state)
    counts = {parameter_count(model) for model in models.values()}
    if len(counts) != 1:
        raise RuntimeError(f"parameter mismatch: {counts}")

    history = torch.randn(2, 96, 3)
    outputs = {}
    maximum_within_block_gap = 0.0
    for scale, model in models.items():
        prediction, _, pooled = model.forward_with_states(history)
        outputs[scale] = prediction
        if prediction.shape != (2, 720, 3):
            raise RuntimeError(f"unexpected prediction shape for s={scale}")
        for start in range(0, 720, scale):
            end = min(start + scale, 720)
            block = pooled[:, :, start:end]
            maximum_within_block_gap = max(
                maximum_within_block_gap,
                float(
                    torch.max(
                        torch.abs(block - block[:, :, :1])
                    ).detach()
                ),
            )
    endpoint_gap = float(
        torch.max(torch.abs(outputs[1] - outputs[720])).detach()
    )
    if maximum_within_block_gap > 1e-6:
        raise RuntimeError(
            f"pooled-state sharing contract failed: {maximum_within_block_gap}"
        )
    if endpoint_gap <= 1e-6:
        raise RuntimeError("sharing endpoints are functionally indistinguishable")
    return {
        "parameter_count": counts.pop(),
        "maximum_within_block_gap": maximum_within_block_gap,
        "endpoint_prediction_gap": endpoint_gap,
        "pass": True,
    }


def write_prefix_artifacts(root: Path) -> None:
    rng = np.random.default_rng(2021)
    origins = 24
    channels = 5
    history = rng.normal(size=(origins, 96, channels)).astype(np.float32)
    full_target = rng.normal(size=(origins, 720, channels)).astype(np.float32)
    horizon_bias = {96: -0.06, 192: -0.02, 336: 0.02, 720: 0.06}
    for horizon in HORIZONS:
        prediction = (
            full_target[:, :horizon]
            + horizon_bias[horizon]
            + rng.normal(scale=0.03, size=(origins, horizon, channels))
        ).astype(np.float32)
        run_dir = (
            root
            / "IntroDLinearPrefixViz"
            / "Weather"
            / f"h{horizon}"
            / "seed2021"
        )
        run_dir.mkdir(parents=True)
        np.savez_compressed(
            run_dir / "predictions_val.npz",
            pred=prediction,
            true=full_target[:, :horizon],
            history=history,
            origin_index=np.arange(origins, dtype=np.int64),
            train_mean=np.zeros((1, channels), dtype=np.float32),
            train_std=np.ones((1, channels), dtype=np.float32),
        )


def write_sharing_artifacts(root: Path) -> None:
    rng = np.random.default_rng(2021)
    origins = 24
    channels = 5
    history = rng.normal(size=(origins, 96, channels)).astype(np.float32)
    target = rng.normal(size=(origins, 720, channels)).astype(np.float32)
    best_by_region = (1, 1, 8, 8, 32, 32, 128, 128, 720, 720, 32, 8)
    for scale in SCALES:
        error = np.empty_like(target)
        for region, best_scale in enumerate(best_by_region):
            start = region * 60
            end = start + 60
            distance = abs(np.log2(scale) - np.log2(best_scale))
            standard_deviation = 0.18 + 0.025 * distance
            error[:, start:end] = rng.normal(
                scale=standard_deviation,
                size=(origins, 60, channels),
            )
        prediction = target + error
        run_dir = (
            root
            / "NeutralSharingExtent"
            / "Weather"
            / f"s{scale}"
            / "seed2021"
        )
        run_dir.mkdir(parents=True)
        np.savez_compressed(
            run_dir / "predictions_val.npz",
            pred=prediction,
            true=target,
            history=history,
            origin_index=np.arange(origins, dtype=np.int64),
            train_mean=np.zeros((1, channels), dtype=np.float32),
            train_std=np.ones((1, channels), dtype=np.float32),
        )
        (run_dir / "effective_config.json").write_text(
            json.dumps(
                {
                    "parameter_count": 111312,
                    "best_epoch": 3,
                    "test_accessed": False,
                }
            ),
            encoding="utf-8",
        )


def run_analyzers(root: Path) -> dict[str, bool]:
    prefix_root = root / "prefix_artifacts"
    sharing_root = root / "sharing_artifacts"
    prefix_output = root / "prefix_output"
    sharing_output = root / "sharing_output"
    sharing_sample_output = root / "sharing_sample_output"
    write_prefix_artifacts(prefix_root)
    write_sharing_artifacts(sharing_root)

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/analyze_intro_prefix_disagreement.py"),
            "--input-root",
            str(prefix_root),
            "--output-dir",
            str(prefix_output),
            "--dataset",
            "Weather",
            "--seed",
            "2021",
            "--sample-quantile",
            "0.85",
            "--selection-mode",
            "maximum",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/analyze_intro_sharing_demand.py"),
            "--input-root",
            str(sharing_root),
            "--output-dir",
            str(sharing_output),
            "--dataset",
            "Weather",
            "--seed",
            "2021",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts/analyze_intro_sharing_sample_candidates.py"
            ),
            "--input-root",
            str(sharing_root),
            "--output-dir",
            str(sharing_sample_output),
            "--dataset",
            "Weather",
            "--seed",
            "2021",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    required = [
        prefix_output / "summary.json",
        prefix_output / "pair_metrics.csv",
        prefix_output / "origin_channel_candidates.csv",
        prefix_output / "selected_forecast_data.csv",
        prefix_output / "prefix_disagreement_overlay.svg",
        prefix_output / "prefix_disagreement_heatmap.svg",
        sharing_output / "summary.json",
        sharing_output / "step_risk.csv",
        sharing_output / "region_risk.csv",
        sharing_output / "sharing_demand_visualization.svg",
        sharing_sample_output / "summary.json",
        sharing_sample_output / "sample_candidates.csv",
        sharing_sample_output / "selected_region_risk.csv",
        sharing_sample_output / "selected_step_risk.csv",
        sharing_sample_output / "sharing_sample_candidate.svg",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing analyzer outputs: {missing}")
    prefix_summary = json.loads(
        (prefix_output / "summary.json").read_text(encoding="utf-8")
    )
    sharing_summary = json.loads(
        (sharing_output / "summary.json").read_text(encoding="utf-8")
    )
    sharing_sample_summary = json.loads(
        (sharing_sample_output / "summary.json").read_text(encoding="utf-8")
    )
    if (
        prefix_summary["test_accessed"]
        or sharing_summary["test_accessed"]
        or sharing_sample_summary["test_accessed"]
    ):
        raise RuntimeError("synthetic analyzers unexpectedly accessed test")
    if prefix_summary["selection_mode"] != "maximum":
        raise RuntimeError("maximum prefix candidate selection was not used")
    if not sharing_summary["visualization_signal"]:
        raise RuntimeError("synthetic sharing signal was not detected")
    if (
        sharing_sample_summary["selected"]["distinct_winner_count"]
        < 2
    ):
        raise RuntimeError("sample-level sharing diversity was not detected")

    ranking_analysis = root / "ranking_analysis"
    ranking_output = root / "ranking_output"
    for index, dataset in enumerate(
        ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
    ):
        prefix_dir = ranking_analysis / dataset / "prefix"
        sharing_dir = ranking_analysis / dataset / "sharing_sample"
        prefix_dir.mkdir(parents=True)
        sharing_dir.mkdir(parents=True)
        ranked_prefix = dict(prefix_summary)
        ranked_prefix["dataset"] = dataset
        ranked_prefix["selected_joint_score"] = (
            prefix_summary["selected_joint_score"] + index * 0.01
        )
        (prefix_dir / "summary.json").write_text(
            json.dumps(ranked_prefix),
            encoding="utf-8",
        )
        ranked_sharing = json.loads(json.dumps(sharing_sample_summary))
        ranked_sharing["dataset"] = dataset
        ranked_sharing["selected"]["winner_entropy"] = min(
            1.0,
            sharing_sample_summary["selected"]["winner_entropy"]
            + index * 0.01,
        )
        (sharing_dir / "summary.json").write_text(
            json.dumps(ranked_sharing),
            encoding="utf-8",
        )
    subprocess.run(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts/rank_intro_visualization_candidates.py"
            ),
            "--analysis-root",
            str(ranking_analysis),
            "--output-dir",
            str(ranking_output),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    ranking_required = [
        ranking_output / "prefix_candidate_ranking.csv",
        ranking_output / "sharing_candidate_ranking.csv",
        ranking_output / "selection_summary.json",
    ]
    if not all(path.is_file() for path in ranking_required):
        raise RuntimeError("cross-dataset ranking outputs are incomplete")
    return {
        "prefix_outputs_complete": True,
        "sharing_outputs_complete": True,
        "sharing_sample_outputs_complete": True,
        "cross_dataset_ranking_complete": True,
        "test_accessed": False,
    }


def main() -> None:
    dataset_registry = check_dataset_registry()
    pooling_equivalence = check_vectorized_pooling_equivalence()
    model_contract = check_model_contract()
    with tempfile.TemporaryDirectory(prefix="fatst_intro_viz_") as directory:
        analyzer_contract = run_analyzers(Path(directory))
    print(
        json.dumps(
            {
                "dataset_registry": dataset_registry,
                "pooling_equivalence": pooling_equivalence,
                "model_contract": model_contract,
                "analyzer_contract": analyzer_contract,
                "pass": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

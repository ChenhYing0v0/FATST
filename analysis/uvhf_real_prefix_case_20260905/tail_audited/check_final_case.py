"""Independently replay the chosen UVHF case and audit plotted source values."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_iscf_bsca_appendix_c_predictions as exporter
from evaluate_weather import fit_metrics


def main(case: Path = OUT / "review_case_0") -> None:
    torch.set_num_threads(4)
    selection = json.loads((case / "selection_audit.json").read_text())
    dataset = selection["dataset"]
    assert dataset in ("ETTh1", "ETTm1")
    key = dataset.lower()
    validation_start = 34560 if dataset == "ETTm1" else 8640
    o = int(selection["selected"]["origin"])
    c = int(selection["selected"]["channel"])
    args = exporter.load_effective_args(
        BASE / f"checkpoints/{key}/effective_config.json",
        dataset,
        Path("/Users/river/PaperResearch/Project/datasets"),
        OUT,
        "cpu",
    )
    official = exporter.train_repo.build_official_args(
        args, exporter.train_repo.OFFICIAL_PRESETS[dataset][720]
    )
    checkpoint = BASE / f"checkpoints/{key}/checkpoint.pt"
    assert (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        == selection["uvhf_checkpoint_sha256"]
    )
    model = exporter.train_repo.TimeAlign.Model(official).float().eval()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    ds, _ = exporter.train_repo.data_provider(official, "val")
    x = torch.as_tensor(ds[o][0]).float()[None]
    dummy = torch.zeros((1, 720, 7))
    with torch.no_grad():
        full = model(x, dummy, is_training=False)[0]
        gaps = {
            str(h): float(
                (
                    model(x, dummy, is_training=False, target_prefix=h)[0]
                    - full[:, :h]
                )
                .abs()
                .max()
            )
            for h in [96, 192, 336, 720]
        }
    assert max(gaps.values()) == 0
    source = pd.read_csv(case / "source_data.csv").set_index("step")
    y = source.loc[1:720, "ground_truth"].to_numpy()
    u = source.loc[1:720, "uvhf"].to_numpy()
    raw_replay = (
        full[0, :, c].numpy() * ds.scaler.scale_[c] + ds.scaler.mean_[c]
    )
    gap = float(np.max(abs(raw_replay - u)))
    assert gap < 1e-4
    raw = pd.read_csv(
        f"/Users/river/PaperResearch/Project/datasets/ETT-small/{dataset}.csv"
    )
    np.testing.assert_allclose(
        source.loc[-719:0, "history"],
        raw.iloc[validation_start + o - 720 : validation_start + o, c + 1],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        y, raw.iloc[validation_start + o : validation_start + o + 720, c + 1], atol=1e-10
    )
    measured = {
        k: float(v[0]) for k, v in fit_metrics(u[None], y[None]).items()
    }
    assert (
        measured["full_r2"] >= 0.35
        and measured["tail_r2"] >= 0.25
        and measured["tail_corr"] >= 0.7
        and measured["last192_r2"] >= 0
        and 0.5 <= measured["tail_amplitude_ratio"] <= 1.5
        and measured["tail_bias_sigma"] <= 0.35
    )
    baseline_checks = {}
    for h in [96, 192, 336, 720]:
        baseline_key = "ettm1_timemixer" if dataset == "ETTm1" else "timemixer"
        folder = BASE / f"matched_checkpoints/{baseline_key}/h{h}"
        a = json.loads((folder / "audit.json").read_text())
        cache = np.load(folder / "predictions_validation.npz")
        pred = (
            cache["pred"][o, :, c] * cache["train_std"][c]
            + cache["train_mean"][c]
        )
        np.testing.assert_allclose(
            pred, source.loc[1:h, f"timemixer_h{h}"], atol=2e-6, rtol=0
        )
        assert source.loc[h + 1 :, f"timemixer_h{h}"].isna().all()
        baseline_checks[str(h)] = a
    report = {
        "raw_gt_and_history_aligned": True,
        "selected_origin": o,
        "channel": c,
        "independent_request_max_gap": gaps,
        "cached_vs_replay_raw_max_gap": gap,
        "fit_metrics_raw_recalculated": measured,
        "native96_baselines": baseline_checks,
        "new_test_access": False,
        "future_label_input": "zeros",
        "status": "numeric audit passed; see reviewer report for visual audit",
    }
    (case / "numeric_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(measured, indent=2))
    print("All independent checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=OUT / "review_case_0")
    main(parser.parse_args().case_dir)

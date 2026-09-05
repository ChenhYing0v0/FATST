"""Check independent horizon requests against the frozen UVHF trajectory."""

from __future__ import annotations

import hashlib
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_iscf_bsca_appendix_c_predictions as exporter


def main(selection_dir: Path = OUT) -> None:
    selection = json.loads(
        (selection_dir / "selection_audit.json").read_text()
    )
    indices = sorted({0, int(selection["selected"]["origin"]), 2160})
    channel = int(selection["selected"]["channel"])
    args = exporter.load_effective_args(
        OUT / "checkpoints/effective_config.json",
        "ETTh2",
        Path("/Users/river/PaperResearch/Project/datasets"),
        OUT,
        "cpu",
    )
    official = exporter.train_repo.build_official_args(
        args, exporter.train_repo.OFFICIAL_PRESETS["ETTh2"][720]
    )
    torch.set_num_threads(4)
    model = exporter.train_repo.TimeAlign.Model(official).float().eval()
    checkpoint = OUT / "checkpoints/checkpoint.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == selection["uvhf_checkpoint_sha256"]
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True),
        strict=True,
    )
    ds, _ = exporter.train_repo.data_provider(official, "val")
    x = torch.stack([torch.as_tensor(ds[i][0]).float() for i in indices])
    dummy_y = torch.zeros((len(indices), 720, 7))
    raw_dir = (
        ROOT
        / "UVHF_CN_patent_PDT_style_20260901/work/figure5_prefix_consistency_20260904/raw"
    )
    if channel in (0, 6):
        saved = dict(np.load(raw_dir / f"uvhf_ch{channel}/candidate_pool.npz"))
        lookup = {
            int(o): j for j, o in enumerate(saved["validation_window_index"])
        }
        cached = np.stack(
            [saved["prediction_scaled"][lookup[i]] for i in indices]
        )
        cache_role = "historical GPU export"
    else:
        cached = np.load(OUT / "raw/uvhf_all_channels.npz")[
            "prediction_scaled"
        ][indices, :, channel]
        cache_role = "all-channel CPU replay"
    baseline = dict(np.load(OUT / "raw/dlinear/h720.npz"))
    history_gap = float(
        np.max(np.abs(x.numpy() - baseline["history"][indices]))
    )
    assert history_gap < 5e-6
    request_gaps = {}
    with torch.no_grad():
        full = model(x, dummy_y, is_training=False)[0]
        for h in (96, 192, 336, 720):
            request = model(x, dummy_y, is_training=False, target_prefix=h)[0]
            request_gaps[str(h)] = float((request - full[:, :h]).abs().max())
    replay_gap = float(np.max(np.abs(full[:, :, channel].numpy() - cached)))
    assert replay_gap < 1e-5 and max(request_gaps.values()) < 1e-5
    report = {
        "lookback": official.seq_len,
        "checkpoint_sha256": digest,
        "validation_origins": indices,
        "channel": channel,
        "request_max_abs_difference_scaled": request_gaps,
        "matched_input_max_scaled_gap": history_gap,
        "cached_vs_cpu_replay_max_abs_gap": replay_gap,
        "cache_role": cache_role,
        "future_labels_input": "zeros",
        "device": "cpu",
        "torch": torch.__version__,
    }
    (selection_dir / "prefix_request_check.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    selection["chpc_status"] = (
        "prefix identity verified by independent requests on selected and "
        "boundary validation origins; see prefix_request_check.json"
    )
    (selection_dir / "selection_audit.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, default=OUT)
    main(selection_dir=parser.parse_args().selection_dir)

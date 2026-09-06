"""Replay the frozen ETTm1 UVHF checkpoint on all validation origins."""

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_iscf_bsca_appendix_c_predictions as exporter


def main() -> None:
    torch.set_num_threads(4)
    checkpoint = BASE / "checkpoints/ettm1/checkpoint.pt"
    expected = "2e8c7e7c7d7545a4edfbe81be857299b6fea0dc7e944464803f273257ecb6948"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected
    args = exporter.load_effective_args(
        BASE / "checkpoints/ettm1/effective_config.json",
        "ETTm1",
        Path("/Users/river/PaperResearch/Project/datasets"),
        OUT,
        "cpu",
    )
    official = exporter.train_repo.build_official_args(
        args, exporter.train_repo.OFFICIAL_PRESETS["ETTm1"][720]
    )
    model = exporter.train_repo.TimeAlign.Model(official).float().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    dataset, _ = exporter.train_repo.data_provider(official, "val")
    assert len(dataset) == 10801
    destination = BASE / "raw/ettm1_all_uvhf.npy"
    output = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float32, shape=(10801, 720, 7)
    )
    start_time = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(dataset), 32):
            stop = min(start + 32, len(dataset))
            x = torch.stack(
                [torch.as_tensor(dataset[i][0]).float() for i in range(start, stop)]
            )
            output[start:stop] = model(
                x, torch.zeros((stop - start, 720, 7)), is_training=False
            )[0].numpy()
            if start % 1024 == 0:
                print(
                    f"Replayed {stop}/10801 in {time.monotonic()-start_time:.1f}s",
                    flush=True,
                )
    output.flush()
    assert np.isfinite(output).all()
    pool = np.load(
        ROOT
        / "analysis/iscf_bsca_appendix_c_prediction_export_20260825/ETTm1/candidate_pool.npz"
    )
    gap = float(
        np.max(
            abs(
                output[pool["validation_window_index"], :, int(pool["channel"])]
                - pool["prediction_scaled"]
            )
        )
    )
    assert gap < 1e-5
    (OUT / "all_replay_audit.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": expected,
                "origins": 10801,
                "channels": 7,
                "shape": list(output.shape),
                "seconds": time.monotonic() - start_time,
                "cached_pool_scaled_max_gap": gap,
                "future_labels": "zeros",
                "new_training": False,
                "new_test_access": False,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

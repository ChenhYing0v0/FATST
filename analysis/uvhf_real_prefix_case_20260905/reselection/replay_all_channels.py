"""Replay frozen UVHF on all validation channels without future labels."""

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
    args = exporter.load_effective_args(
        BASE / "checkpoints/effective_config.json",
        "ETTh2",
        Path("/Users/river/PaperResearch/Project/datasets"),
        OUT,
        "cpu",
    )
    official = exporter.train_repo.build_official_args(
        args, exporter.train_repo.OFFICIAL_PRESETS["ETTh2"][720]
    )
    model = exporter.train_repo.TimeAlign.Model(official).float().eval()
    checkpoint = BASE / "checkpoints/checkpoint.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert (
        digest
        == json.loads((BASE / "selection_audit.json").read_text())[
            "uvhf_checkpoint_sha256"
        ]
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True),
        strict=True,
    )
    ds, _ = exporter.train_repo.data_provider(official, "val")
    baseline = dict(np.load(BASE / "raw/dlinear/h720.npz"))
    predictions = []
    start_time = time.monotonic()
    max_input_gap = 0.0
    with torch.no_grad():
        for start in range(0, 2161, 32):
            stop = min(start + 32, 2161)
            x = torch.stack(
                [torch.as_tensor(ds[i][0]).float() for i in range(start, stop)]
            )
            max_input_gap = max(
                max_input_gap,
                float(
                    np.max(abs(x.numpy() - baseline["history"][start:stop]))
                ),
            )
            predictions.append(
                model(
                    x, torch.zeros((stop - start, 720, 7)), is_training=False
                )[0].numpy()
            )
            if start % 320 == 0:
                print(
                    f"Replayed {stop}/2161 origins; {time.monotonic()-start_time:.1f}s",
                    flush=True,
                )
    prediction = np.concatenate(predictions)
    assert np.isfinite(prediction).all() and max_input_gap < 5e-6
    np.savez_compressed(
        BASE / "raw/uvhf_all_channels.npz", prediction_scaled=prediction
    )
    (OUT / "replay_audit.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": digest,
                "origins": 2161,
                "channels": 7,
                "shape": list(prediction.shape),
                "max_input_gap": max_input_gap,
                "seconds": time.monotonic() - start_time,
                "future_labels": "zeros",
                "test_access": False,
                "training": False,
                "torch": torch.__version__,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

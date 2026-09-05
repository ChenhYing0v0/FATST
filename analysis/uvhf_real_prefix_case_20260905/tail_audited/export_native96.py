"""Export existing native L96 TimeMixer validation forecasts, without training."""

import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

NATIVE = Path("/home/yingch/TimeMixer")
OUT = Path(
    "/home/yingch/exp_outputs/r-2026-fatst/uvhf_prefix_native96_20260905"
)


def main() -> None:
    sys.path.insert(0, str(NATIVE))
    from models.TimeMixer import Model
    from data_provider.data_factory import data_provider

    torch.set_num_threads(4)
    configs = []
    for line in (NATIVE / "etth1.log").read_text().splitlines():
        if line.startswith("Namespace("):
            node = ast.parse(line, mode="eval").body
            configs.append(
                {k.arg: ast.literal_eval(k.value) for k in node.keywords}
            )
    for h in (96, 192, 336, 720):
        matches = [
            c for c in configs if c["seq_len"] == 96 and c["pred_len"] == h
        ]
        assert len(matches) == 1
        config = matches[0]
        config["root_path"] = "/home/yingch/dataset/ETT-small/"
        config["num_workers"] = 0
        args = SimpleNamespace(**config)
        model = Model(args).float().cuda().eval()
        checkpoints = list(
            (NATIVE / "checkpoints").glob(f"*_sl96_pl{h}_*/checkpoint.pth")
        )
        assert len(checkpoints) == 1
        model.load_state_dict(
            torch.load(checkpoints[0], map_location="cpu", weights_only=True),
            strict=True,
        )
        ds, _ = data_provider(args, "val")
        loader = DataLoader(
            torch.utils.data.Subset(ds, range(2161)),
            batch_size=64,
            shuffle=False,
            drop_last=False,
        )
        predictions = []
        with torch.no_grad():
            for x, y, xmark, ymark in loader:
                predictions.append(
                    model(x.float().cuda(), xmark.float().cuda(), None, None)
                    .cpu()
                    .numpy()
                )
        folder = OUT / f"h{h}"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / "predictions_validation.npz",
            pred=np.concatenate(predictions),
            train_mean=ds.scaler.mean_,
            train_std=ds.scaler.scale_,
        )
        (folder / "effective_config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )
        (folder / "audit.json").write_text(
            json.dumps(
                {
                    "source_checkpoint": str(checkpoints[0]),
                    "checkpoint_sha256": hashlib.sha256(
                        checkpoints[0].read_bytes()
                    ).hexdigest(),
                    "native_model_sha256": hashlib.sha256(
                        (NATIVE / "models/TimeMixer.py").read_bytes()
                    ).hexdigest(),
                    "source_training_log_sha256": hashlib.sha256(
                        (NATIVE / "etth1.log").read_bytes()
                    ).hexdigest(),
                    "origins": 2161,
                    "seq_len": 96,
                    "horizon": h,
                    "new_training": False,
                    "new_test_access": False,
                    "historical_training_logged_test_losses": True,
                    "checkpoint_rule": "historical native minimum validation loss",
                    "torch": torch.__version__,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Exported H{h}", flush=True)


if __name__ == "__main__":
    main()

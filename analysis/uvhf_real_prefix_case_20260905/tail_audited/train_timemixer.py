"""Run native TimeMixer training with test access disabled for visualization."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys
import textwrap

import numpy as np
import torch
from torch.utils.data import DataLoader


def main() -> None:
    # Preserve the native NumPy 1.x infinity alias under NumPy 2.x.
    np.Inf = np.inf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--dataset", choices=["ETTh1", "ETTm1"], default="ETTh1")
    parser.add_argument("--seq-len", type=int, default=720)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--source", type=Path, default=Path("/home/yingch/TimeMixer")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/yingch/exp_outputs/r-2026-fatst/uvhf_prefix_tail_timemixer_20260905"
        ),
    )
    options = parser.parse_args()
    sys.path.insert(0, str(options.source))
    import run as native
    import exp.exp_long_term_forecasting as native_exp

    args = native.parser.parse_args(
        [
            "--task_name",
            "long_term_forecast",
            "--is_training",
            "1",
            "--model_id",
            "TailAudit",
            "--model",
            "TimeMixer",
            "--data",
            options.dataset,
            "--root_path",
            "/home/yingch/dataset/ETT-small/",
            "--data_path",
            f"{options.dataset}.csv",
            "--freq",
            "t" if options.dataset == "ETTm1" else "h",
            "--seq_len",
            str(options.seq_len),
            "--label_len",
            "0",
            "--pred_len",
            str(options.horizon),
            "--batch_size",
            str(options.batch_size),
            "--train_epochs",
            "10",
            "--patience",
            "10",
            "--learning_rate",
            str(options.learning_rate),
            "--down_sampling_layers",
            "3",
            "--down_sampling_window",
            "2",
            "--down_sampling_method",
            "avg",
            "--num_workers",
            "0",
            "--checkpoints",
            str(options.output),
        ]
    )
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    torch.set_num_threads(4)
    args.use_gpu = True
    # Honor the launcher mask; native device acquisition overwrites it with GPU 0.
    native_exp.Exp_Long_Term_Forecast._acquire_device = (
        lambda self: torch.device("cuda:0")
    )
    experiment = native_exp.Exp_Long_Term_Forecast(args)
    source = textwrap.dedent(
        inspect.getsource(native_exp.Exp_Long_Term_Forecast.train)
    )
    for line in [
        "    test_data, test_loader = self._get_data(flag='test')\n",
        "        test_loss = self.vali(test_data, test_loader, criterion)\n",
    ]:
        assert line in source
        source = source.replace(line, "")
    source = source.replace(" Test Loss: {4:.7f}", "").replace(
        "train_loss, vali_loss, test_loss", "train_loss, vali_loss"
    )
    assert "test_loader" not in source and "test_loss" not in source
    namespace = dict(vars(native_exp))
    exec(compile(source, "<native-train-without-test>", "exec"), namespace)
    experiment.train = namespace["train"].__get__(experiment)
    original_get = experiment._get_data

    def get_data(flag: str):
        assert flag in ("train", "val"), "Test access is disabled"
        ds, loader = original_get(flag)
        if flag == "val":
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
        return ds, loader

    experiment._get_data = get_data
    setting = f"h{options.horizon}"
    folder = options.output / setting
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "effective_config.json").write_text(
        json.dumps(vars(args), indent=2) + "\n"
    )
    (folder / "native_train_without_test.py").write_text(source)
    experiment.train(setting)
    ds, loader = get_data("val")
    predictions = []
    experiment.model.eval()
    with torch.no_grad():
        for x, y, xmark, ymark in loader:
            predictions.append(
                experiment.model(
                    x.float().to(experiment.device),
                    xmark.float().to(experiment.device),
                    None,
                    None,
                )
                .cpu()
                .numpy()
            )
    np.savez_compressed(
        folder / "predictions_validation.npz",
        pred=np.concatenate(predictions)[: len(ds) - (720 - options.horizon)],
        train_mean=ds.scaler.mean_,
        train_std=ds.scaler.scale_,
    )
    report = {
        "checkpoint_sha256": hashlib.sha256(
            (folder / "checkpoint.pth").read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "python": sys.version,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "native_source_commit": "e24610583b36fdd8c76cc17a8df4e65759a5f460",
        "native_model_sha256": hashlib.sha256(
            (options.source / "models/TimeMixer.py").read_bytes()
        ).hexdigest(),
        "test_access": False,
        "validation_export_origins": len(ds) - (720 - options.horizon),
        "validation_loader": "ordered, drop_last=False",
        "checkpoint_rule": "native minimum validation mean batch MSE",
    }
    (folder / "audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

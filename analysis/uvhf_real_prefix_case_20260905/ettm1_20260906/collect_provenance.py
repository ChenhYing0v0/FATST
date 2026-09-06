"""Collect fixed configurations and complete validation-only learning curves."""

import hashlib
import json
from pathlib import Path
import re

import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def main() -> None:
    runs = {}
    for h in (96, 192, 336, 720):
        folder = BASE / f"matched_checkpoints/ettm1_timemixer/h{h}"
        config = json.loads((folder / "effective_config.json").read_text())
        audit = json.loads((folder / "audit.json").read_text())
        assert (
            config["data"] == "ETTm1"
            and config["seq_len"] == 96
            and config["pred_len"] == h
        )
        assert (
            config["batch_size"] == 16
            and config["d_model"] == 16
            and config["d_ff"] == 32
        )
        assert config["learning_rate"] == 0.01 and config["train_epochs"] == 10
        assert not audit["test_access"] and audit["validation_export_origins"] == 10801
        log = OUT / f"training_logs/h{h}.log"
        rows = re.findall(
            r"Epoch: (\d+), Steps: (\d+) \| Train Loss: ([0-9.e+-]+) Vali Loss: ([0-9.e+-]+)",
            log.read_text(),
        )
        table = pd.DataFrame(
            rows, columns=["epoch", "steps", "train_loss", "validation_loss"]
        ).astype(
            {"epoch": int, "steps": int, "train_loss": float, "validation_loss": float}
        )
        assert len(table) == 10
        table.to_csv(OUT / f"training_logs/h{h}_epochs.csv", index=False)
        best = table.loc[table.validation_loss.idxmin()]
        runs[str(h)] = {
            "config": config,
            "audit": audit,
            "best_validation_epoch": int(best.epoch),
            "best_validation_batch_mean_mse": float(best.validation_loss),
            "epochs": len(table),
            "training_log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            "physical_gpu": 1 if h == 96 else 0,
            "source_commit": "9dc997a7" if h == 96 else "00fac229",
        }
    (OUT / "baseline_provenance.json").write_text(json.dumps(runs, indent=2) + "\n")


if __name__ == "__main__":
    main()

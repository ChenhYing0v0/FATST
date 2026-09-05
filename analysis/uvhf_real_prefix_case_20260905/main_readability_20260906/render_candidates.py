"""Render alternative cases without altering the provisionally accepted figure."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    builder = load_module("case_builder", BASE / "tail_audited/build_final_case.py")
    plot = load_module("case_plot", BASE / "plot_single_panel.py")
    builder.OUT = OUT
    builder.BASE = BASE
    candidates = pd.read_csv(OUT / "review_candidates.csv")
    candidates.to_csv(OUT / "timemixer_review_candidates.csv", index=False)
    original = BASE / "tail_audited/review_case_0"
    old_settings = json.loads((original / "figure_settings.json").read_text())
    old_source = pd.read_csv(original / "source_data.csv")
    columns = ["ground_truth", "uvhf", *[f"timemixer_h{h}" for h in builder.HORIZONS]]
    old_values = old_source.loc[old_source.step > 0, columns].to_numpy()
    old_min, old_range = np.nanmin(old_values), np.nanmax(old_values) - np.nanmin(
        old_values
    )
    for rank in range(1, len(candidates)):
        builder.main(rank=rank)
        case = OUT / f"review_case_{rank}"
        source = pd.read_csv(case / "source_data.csv")
        values = source.loc[source.step > 0, columns].to_numpy()
        low, extent = np.nanmin(values), np.nanmax(values) - np.nanmin(values)
        settings = json.loads((case / "figure_settings.json").read_text())
        for key in ["main_ylim", "main_yticks"]:
            settings[key] = [
                float(low + (v - old_min) * extent / old_range)
                for v in old_settings[key]
            ]
        # Use round tick labels while preserving the reviewed axis range.
        settings["main_yticks"] = (
            np.round(np.asarray(settings["main_yticks"]) / 5) * 5
        ).tolist()
        settings["prefix_yticks"] = (
            np.round(np.linspace(*settings["prefix_ylim"], 4) / 5) * 5
        ).tolist()
        settings["horizon_y"] = {
            h: float(low + (v - old_min) * extent / old_range)
            for h, v in old_settings["horizon_y"].items()
        }
        settings["annotation_y"] = float(
            low + (old_settings["annotation_y"] - old_min) * extent / old_range
        )
        settings["prefix_label_y"] = settings["prefix_ylim"][1] + 0.02 * extent
        endpoints = source.loc[source.step == 720, ["uvhf", "timemixer_h720"]].iloc[0]
        if abs(endpoints.uvhf - endpoints.timemixer_h720) < 0.11 * extent:
            center = endpoints.mean()
            direction = np.sign(endpoints.uvhf - endpoints.timemixer_h720)
            settings["endpoint_label_y"] = {
                "UVHF": float(center + direction * 0.055 * extent),
                "TimeMixer": float(center - direction * 0.055 * extent),
            }
        (case / "figure_settings.json").write_text(
            json.dumps(settings, indent=2) + "\n"
        )
        plot.main(zoom=True, output=case)


if __name__ == "__main__":
    main()

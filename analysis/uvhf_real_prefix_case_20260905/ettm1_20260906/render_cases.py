"""Render ETTm1 candidates at the same physical size with correct time units."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "case_plot", BASE / "plot_single_panel.py"
    )
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    original = BASE / "tail_audited/review_case_0"
    old_settings = json.loads((original / "figure_settings.json").read_text())
    columns = ["ground_truth", "uvhf", *[f"timemixer_h{h}" for h in plot.HORIZONS]]
    old_source = pd.read_csv(original / "source_data.csv").set_index("step")
    old_values = old_source.loc[1:720, columns].to_numpy()
    old_min, old_range = np.nanmin(old_values), np.nanmax(old_values) - np.nanmin(
        old_values
    )
    for case in sorted(OUT.glob("review_case_*")):
        source = pd.read_csv(case / "source_data.csv").set_index("step")
        values = source.loc[1:720, columns].to_numpy()
        low, extent = np.nanmin(values), np.nanmax(values) - np.nanmin(values)
        prefix = source.loc[1:96, columns].to_numpy()
        lo, hi = np.nanmin(prefix), np.nanmax(prefix)
        step = int(
            source.loc[24:80, [f"timemixer_h{h}" for h in plot.HORIZONS]]
            .apply(np.ptp, axis=1)
            .idxmax()
        )
        settings = {
            "baseline": "TimeMixer",
            "baseline_prefix": "timemixer",
            "subtitle": "ETTm1 · HUFL · selected validation example",
            "unit": "units",
            "ylabel": "HUFL (original scale)",
            "xlabel": "Forecast step (15 min per step)",
            "main_ylim": [
                float(low + (v - old_min) * extent / old_range)
                for v in old_settings["main_ylim"]
            ],
            "main_yticks": (
                np.round(np.linspace(low, low + extent, 4) / 5) * 5
            ).tolist(),
            "prefix_ylim": [float(lo - 0.06 * (hi - lo)), float(hi + 0.06 * (hi - lo))],
            "prefix_yticks": (np.round(np.linspace(lo, hi, 4) / 5) * 5).tolist(),
            "prefix_label_y": float(hi + 0.08 * (hi - lo)),
            "horizon_y": {
                h: float(low + (v - old_min) * extent / old_range)
                for h, v in old_settings["horizon_y"].items()
            },
            "annotation_y": float(
                low + (old_settings["annotation_y"] - old_min) * extent / old_range
            ),
            "annotation_step": step,
            "connector_corner": [0, 0],
        }
        endpoints = source.loc[720, ["uvhf", "timemixer_h720"]]
        if abs(endpoints.uvhf - endpoints.timemixer_h720) < 0.11 * extent:
            center = endpoints.mean()
            direction = np.sign(endpoints.uvhf - endpoints.timemixer_h720)
            settings["endpoint_label_y"] = {
                "UVHF": float(center + direction * 0.055 * extent),
                "TimeMixer": float(center - direction * 0.055 * extent),
            }
        settings_path = case / "figure_settings.json"
        if not settings_path.exists():
            settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        plot.main(zoom=True, output=case)


if __name__ == "__main__":
    main()

"""Verify the selected figure's plotted arrays, layout, metrics, and exports."""

import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
CASE = BASE / "tail_audited/review_case_0"


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "case_plot", BASE / "plot_single_panel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with patch.object(Figure, "savefig"), patch.object(
        Path, "write_text"
    ), patch.object(plt, "close"):
        module.main(zoom=True, output=CASE)
        fig = plt.gcf()
    fig.canvas.draw()
    assert len(fig.axes) == 1 and len(fig.axes[0].child_axes) == 1
    ax = fig.axes[0]
    inset = ax.child_axes[0]
    source = pd.read_csv(CASE / "source_data.csv").set_index("step")
    columns = ["ground_truth"] + [f"timemixer_h{h}" for h in module.HORIZONS] + ["uvhf"]
    for line, col in zip(inset.lines[:6], columns):
        np.testing.assert_array_equal(line.get_xdata(), np.arange(1, 97))
        np.testing.assert_allclose(
            line.get_ydata(), source.loc[1:96, col], rtol=0, atol=0
        )
    for line in ax.lines:
        x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        if len(x) >= 48:
            assert not any(
                inset.bbox.contains(*p) for p in ax.transData.transform(np.c_[x, y])
            )
    renderer = fig.canvas.get_renderer()
    inset_text = [
        *inset.get_xticklabels(),
        *inset.get_yticklabels(),
        inset.xaxis.label,
        inset.yaxis.label,
    ]
    for text in ax.texts:
        if text.get_text().startswith("H="):
            assert not any(
                text.get_window_extent(renderer).overlaps(
                    item.get_window_extent(renderer)
                )
                for item in inset_text
            )
    for text in fig.texts:
        box = text.get_window_extent(renderer)
        assert (
            box.x0 >= 0
            and box.y0 >= 0
            and box.x1 <= fig.bbox.x1
            and box.y1 <= fig.bbox.y1
        )
    np.testing.assert_allclose(fig.get_size_inches() * 25.4, [183, 135])
    for row in pd.read_csv(CASE / "selected_metrics.csv").itertuples():
        col = "uvhf" if row.model == "UVHF" else f"timemixer_h{row.horizon}"
        delta = (
            source.loc[1 : row.horizon, col]
            - source.loc[1 : row.horizon, "ground_truth"]
        )
        np.testing.assert_allclose(
            [np.mean(delta**2), np.mean(abs(delta))],
            [row.mse_raw, row.mae_raw],
            rtol=1e-10,
        )
    pairs = pd.read_csv(CASE / "selected_pair_disagreement.csv")
    for row in pairs.itertuples():
        a, b = row.short_horizon, row.long_horizon
        gap = np.mean(
            abs(source.loc[1:a, f"timemixer_h{a}"] - source.loc[1:a, f"timemixer_h{b}"])
        )
        np.testing.assert_allclose(gap, row.chpd_raw, rtol=1e-10)
    stem = CASE / "uvhf_real_prefix_zoom"
    svg = ET.parse(stem.with_suffix(".svg")).getroot()
    texts = svg.findall(".//{http://www.w3.org/2000/svg}text")
    assert len(texts) > 30
    assert b"/FontFile2" in stem.with_suffix(".pdf").read_bytes()
    exports = {}
    for ext in ["png", "tiff"]:
        with Image.open(stem.with_suffix("." + ext)) as image:
            dpi = np.asarray(image.info["dpi"], dtype=float)
            exports[ext] = {"pixels": list(image.size), "dpi": dpi.tolist()}
            np.testing.assert_allclose(
                np.array(image.size) / dpi * 25.4, [183, 135], atol=0.1
            )
            assert min(dpi) >= (299 if ext == "png" else 999)
    qa = {
        "status": "PASS for selected-case illustration",
        "date": "2026-09-05",
        "backend": "Python matplotlib",
        "dimensions_mm": [183, 135],
        "main_axes": 1,
        "embedded_prefix_axes": 1,
        "six_inset_curves_equal_source": True,
        "main_trajectory_not_occluded_by_inset": True,
        "horizon_labels_clear_of_inset_labels": True,
        "figure_text_inside_canvas": True,
        "metrics_recalculated_from_csv": True,
        "mean_pairwise_disagreement_raw": float(pairs.chpd_raw.mean()),
        "editable_svg_text_elements": len(texts),
        "pdf_embedded_truetype": True,
        "exports": exports,
        "source_preflight": {
            "PASS": 13,
            "WARN": 1,
            "FAIL": 0,
            "warning_resolution": "Static width detector cannot resolve variables; actual figure and exports verified as 183 x 135 mm. Final submission must follow target journal specifications.",
        },
        "visual_review": "Complete 720-step plot and 96-step inset inspected. Deep trough errors remain visible. No smoothing or tail cropping. See reviewer_audit.md for claim limits.",
        "sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(CASE.iterdir())
            if p.is_file() and p.name != "figure_qa.json"
        },
    }
    (CASE / "figure_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()

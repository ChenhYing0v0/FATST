"""Verify final ETTm1 export and retain the explicit visual-review decisions."""

import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    decisions = {
        0: "not selected: late trough mismatch",
        1: "passed selected-case numeric and visual audit",
        2: "secondary candidate: weak H192 gain",
        3: "not selected: late trough mismatch",
    }
    for rank, decision in decisions.items():
        path = OUT / f"review_case_{rank}/selection_audit.json"
        selection = json.loads(path.read_text())
        selection.update(
            status=decision,
            review_date="2026-09-06",
            review_report="../reviewer_audit.md",
        )
        path.write_text(json.dumps(selection, indent=2) + "\n")
    case = OUT / "review_case_1"
    numeric = json.loads((case / "numeric_audit.json").read_text())
    assert max(numeric["independent_request_max_gap"].values()) == 0
    checker = load_module(
        "export_checker", BASE / "tail_audited/check_figure_exports.py"
    )
    checker.CASE = case
    checker.main()
    plot = load_module("case_plot", BASE / "plot_single_panel.py")
    with patch.object(Figure, "savefig"), patch.object(
        Path, "write_text"
    ), patch.object(plt, "close"):
        plot.main(zoom=True, output=case)
        fig = plt.gcf()
    fig.canvas.draw()
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Forecast step (15 min per step)"
    assert ax.child_axes[0].get_xlabel() == ax.get_xlabel()
    labels = [text for text in ax.texts if text.get_text() in ["UVHF", "TimeMixer"]]
    assert len(labels) == 2
    assert (
        not labels[0]
        .get_window_extent(fig.canvas.get_renderer())
        .overlaps(labels[1].get_window_extent(fig.canvas.get_renderer()))
    )
    plt.close(fig)
    qa_path = case / "figure_qa.json"
    qa = json.loads(qa_path.read_text())
    qa.update(
        date="2026-09-06",
        time_unit="15 minutes per step",
        forecast_duration_hours=180,
        endpoint_labels_do_not_overlap=True,
        visual_review="PASS for selected ETTm1 HUFL origin2295. Fewer cycles across720 steps; visible late TimeMixer bias; prefix disagreement retained. Partial peak underestimation remains. See ../reviewer_audit.md.",
    )
    qa_path.write_text(json.dumps(qa, indent=2) + "\n")
    for name, value in qa["sha256"].items():
        assert hashlib.sha256((case / name).read_bytes()).hexdigest() == value
    (OUT / "delivery.json").write_text(
        json.dumps(
            {
                "selected_case": "review_case_1",
                "dataset": "ETTm1",
                "channel": "HUFL",
                "origin": 2295,
                "new_test_access": False,
                "uvhf_retrained": False,
                "timemixer_trained_horizons": [96, 192, 336, 720],
                "status": "numeric, visual and export audits passed",
                "sha256": {
                    str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [
                        OUT / "reviewer_audit.md",
                        OUT / "baseline_provenance.json",
                        case / "figure_qa.json",
                    ]
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

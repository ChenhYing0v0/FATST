"""Reuse export QA and verify that the provisionally accepted bundle is intact."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from render_candidates import load_module

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def main() -> None:
    preserved = json.loads((OUT / "preserved_result.json").read_text())
    original = BASE / preserved["directory"]
    assert all(
        hashlib.sha256((original / name).read_bytes()).hexdigest() == value
        for name, value in preserved["sha256"].items()
    )
    checker = load_module("export_check", BASE / "tail_audited/check_figure_exports.py")
    plot = load_module("case_plot", BASE / "plot_single_panel.py")
    decisions = {
        1: "not recommended: fewer prominent peaks do not imply a longer cycle",
        2: "recommended alternative for full-horizon readability; not an approved replacement",
        3: "secondary alternative balancing prefix and full-horizon readability",
    }
    for rank, decision in decisions.items():
        case = OUT / f"review_case_{rank}"
        numeric = json.loads((case / "numeric_audit.json").read_text())
        assert max(numeric["independent_request_max_gap"].values()) == 0
        assert numeric["raw_gt_and_history_aligned"]
        selection_path = case / "selection_audit.json"
        selection = json.loads(selection_path.read_text())
        selection.update(
            status=decision,
            review_date="2026-09-06",
            review_report="../reviewer_assessment.md",
        )
        selection_path.write_text(json.dumps(selection, indent=2) + "\n")
        checker.CASE = case
        checker.main()
        with patch.object(Figure, "savefig"), patch.object(
            Path, "write_text"
        ), patch.object(plt, "close"):
            plot.main(zoom=True, output=case)
            fig = plt.gcf()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        labels = [
            text
            for text in fig.axes[0].texts
            if text.get_text() in ("UVHF", "TimeMixer")
        ]
        assert len(labels) == 2
        assert (
            not labels[0]
            .get_window_extent(renderer)
            .overlaps(labels[1].get_window_extent(renderer))
        )
        plt.close(fig)
        qa_path = case / "figure_qa.json"
        qa = json.loads(qa_path.read_text())
        qa.update(
            date="2026-09-06",
            status="PASS technical export checks; see separate recommendation",
            endpoint_labels_do_not_overlap=True,
            visual_review=f"{decision}. Full review: ../reviewer_assessment.md. Deep trough errors and 24-step dominant cycle remain visible.",
        )
        qa_path.write_text(json.dumps(qa, indent=2) + "\n")
    (OUT / "preservation_check.json").write_text(
        json.dumps(
            {
                "date": "2026-09-06",
                "preserved_files": len(preserved["sha256"]),
                "all_hashes_unchanged": True,
                "candidate_numeric_and_export_checks": "PASS",
                "approved_replacement": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

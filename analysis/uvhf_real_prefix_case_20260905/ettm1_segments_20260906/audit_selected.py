"""Verify segment-wise improvements and reuse full export QA."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def main() -> None:
    case = OUT / "review_case_1"
    s = pd.read_csv(case / "source_data.csv").set_index("step")
    selection = json.loads((case / "selection_audit.json").read_text())
    rows = []
    for a, b in [(1, 96), (97, 192), (193, 336), (337, 720)]:
        for h in [96, 192, 336, 720]:
            if h < b:
                continue
            eu = s.loc[a:b, "uvhf"] - s.loc[a:b, "ground_truth"]
            eb = s.loc[a:b, f"timemixer_h{h}"] - s.loc[a:b, "ground_truth"]
            m = 1 - np.mean(eu**2) / np.mean(eb**2)
            ma = 1 - np.mean(abs(eu)) / np.mean(abs(eb))
            assert m > 0 and ma > 0
            np.testing.assert_allclose(
                m, selection["selected"][f"mse_gain_{a}_{b}_h{h}"], atol=1e-7
            )
            np.testing.assert_allclose(
                ma, selection["selected"][f"mae_gain_{a}_{b}_h{h}"], atol=1e-7
            )
            rows.append(
                {
                    "start": a,
                    "end": b,
                    "baseline_horizon": h,
                    "mse_gain": m,
                    "mae_gain": ma,
                }
            )
    pd.DataFrame(rows).to_csv(case / "segment_metrics.csv", index=False)
    selection.update(
        status="passed numeric, segment and visual review",
        review_report="../reviewer_audit.md",
    )
    (case / "selection_audit.json").write_text(json.dumps(selection, indent=2) + "\n")
    for n, reason in [
        (0, "prefix fidelity less convincing"),
        (2, "secondary candidate"),
        (3, "late fidelity less convincing"),
        (4, "deep trough mismatch"),
    ]:
        p = OUT / f"review_case_{n}/selection_audit.json"
        x = json.loads(p.read_text())
        x["status"] = reason
        p.write_text(json.dumps(x, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location(
        "qa", BASE / "tail_audited/check_figure_exports.py"
    )
    qa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qa)
    qa.CASE = case
    qa.main()
    p = case / "figure_qa.json"
    q = json.loads(p.read_text())
    q.update(
        date="2026-09-06",
        visual_review="Passed with limits: sharper spikes and final decline remain imperfect. See ../reviewer_audit.md.",
        segment_combinations_verified=10,
    )
    p.write_text(json.dumps(q, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Audit balanced candidate skill, segment gains and rendered exports."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def main() -> None:
    case = OUT / "review_case_3"
    s = pd.read_csv(case / "source_data.csv").set_index("step")
    selection = json.loads((case / "selection_audit.json").read_text())
    r = selection["selected"]
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
            np.testing.assert_allclose(m, r[f"mse_gain_{a}_{b}_h{h}"], atol=1e-7)
            np.testing.assert_allclose(ma, r[f"mae_gain_{a}_{b}_h{h}"], atol=1e-7)
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
    for h in [96, 192, 336, 720]:
        for label, b in [("full", h), ("prefix", 96)]:
            y = s.loc[1:b, "ground_truth"]
            p = s.loc[1:b, f"timemixer_h{h}"]
            score = 1 - np.mean((p - y) ** 2) / np.var(y)
            np.testing.assert_allclose(
                score, r[f"timemixer_{label}_r2_h{h}"], atol=1e-7
            )
            assert score >= (0.5 if label == "full" else 0.3)
    numeric = json.loads((case / "numeric_audit.json").read_text())
    assert max(numeric["independent_request_max_gap"].values()) == 0
    for n, status in [
        (0, "not selected: last192 reversal"),
        (1, "not selected: weaker whole-plot visibility"),
        (2, "not selected: less balanced gain"),
        (3, "recommended balanced illustration after numerical and visual audit"),
    ]:
        p = OUT / f"review_case_{n}/selection_audit.json"
        a = json.loads(p.read_text())
        a.update(status=status, review_report="../reviewer_audit.md")
        p.write_text(json.dumps(a, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location(
        "qa", BASE / "tail_audited/check_figure_exports.py"
    )
    qa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qa)
    qa.CASE = case
    qa.main()
    p = case / "figure_qa.json"
    a = json.loads(p.read_text())
    a.update(
        date="2026-09-06",
        visual_review="Balanced MUFL3738 selected. Both models have skill; local sharp trough errors remain. See ../reviewer_audit.md.",
        segment_combinations_verified=10,
        baseline_skill_recalculated=True,
    )
    p.write_text(json.dumps(a, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Recalculate all segment comparisons and inspect actual exported plot arrays."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def main() -> None:
    case = OUT / "review_case_0"
    source = pd.read_csv(case / "source_data.csv").set_index("step")
    record = json.loads((case / "selection_audit.json").read_text())["selected"]
    rows = []
    for a, b in [(1, 96), (97, 192), (193, 336), (337, 720), (529, 720)]:
        for h in (96, 192, 336, 720):
            if h < b:
                continue
            y = source.loc[a:b, "ground_truth"].to_numpy()
            u = source.loc[a:b, "uvhf"].to_numpy()
            p = source.loc[a:b, f"timemixer_h{h}"].to_numpy()
            mse_gain = 1 - np.mean((u - y) ** 2) / np.mean((p - y) ** 2)
            mae_gain = 1 - np.mean(abs(u - y)) / np.mean(abs(p - y))
            assert mse_gain > 0 and mae_gain > 0
            if a != 529:
                assert mse_gain >= 0.3 and mae_gain >= 0.15
                np.testing.assert_allclose(
                    mse_gain, record[f"mse_gain_{a}_{b}_h{h}"], atol=1e-7
                )
                np.testing.assert_allclose(
                    mae_gain, record[f"mae_gain_{a}_{b}_h{h}"], atol=1e-7
                )
            rows.append(
                dict(
                    start=a,
                    end=b,
                    baseline_horizon=h,
                    mse_gain=mse_gain,
                    mae_gain=mae_gain,
                )
            )
    pd.DataFrame(rows).to_csv(case / "segment_metrics.csv", index=False)
    for h in (96, 192, 336, 720):
        y = source.loc[1:h, "ground_truth"].to_numpy()
        p = source.loc[1:h, f"timemixer_h{h}"].to_numpy()
        r2 = 1 - np.mean((p - y) ** 2) / np.var(y)
        np.testing.assert_allclose(r2, record[f"timemixer_full_r2_h{h}"], atol=1e-7)
        assert r2 >= (0.4 if h == 720 else 0.2)
    numeric = json.loads((case / "numeric_audit.json").read_text())
    assert max(numeric["independent_request_max_gap"].values()) == 0
    for i, status in enumerate(
        [
            "recommended: clear repeated trough advantage and visible prefix disagreement",
            "not selected: lower UVHF peak and trough fidelity",
            "not selected: less uniform middle and final trough fidelity",
        ]
    ):
        path = OUT / f"review_case_{i}/selection_audit.json"
        data = json.loads(path.read_text())
        data.update(status=status, review_report="../reviewer_audit.md")
        path.write_text(json.dumps(data, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location(
        "qa", BASE / "tail_audited/check_figure_exports.py"
    )
    qa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qa)
    qa.CASE = case
    qa.main()
    path = case / "figure_qa.json"
    data = json.loads(path.read_text())
    data.update(
        date="2026-09-06",
        segment_combinations_verified=11,
        visual_review="LUFL8752: repeated trough advantage, explicit prefix divergence, sharp spike misses preserved. See ../reviewer_audit.md.",
    )
    path.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()

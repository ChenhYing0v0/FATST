"""Render the user-specified forecast origin, shifted forward exactly 144 steps."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def main() -> None:
    table = pd.read_csv(BASE / "ettm1_segments_20260906/all_candidate_audit.csv").merge(
        pd.read_csv(BASE / "ettm1_balanced_20260906/baseline_audit.csv"),
        on=["origin", "channel"],
        validate="one_to_one",
    )
    selected = table[(table.origin == 8752 + 144) & (table.channel == 4)]
    assert len(selected) == 1
    selected.to_csv(OUT / "review_candidates.csv", index=False)
    spec = importlib.util.spec_from_file_location(
        "builder", BASE / "ettm1_segments_20260906/build_cases.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.OUT = OUT
    builder.main()
    case = OUT / "review_case_0"
    old = pd.read_csv(
        BASE / "ettm1_advantage_20260906/review_case_0/source_data.csv"
    ).set_index("step")
    new = pd.read_csv(case / "source_data.csv").set_index("step")
    np.testing.assert_allclose(
        new.loc[-143:0, "history"], old.loc[1:144, "ground_truth"]
    )
    np.testing.assert_allclose(
        new.loc[1:576, "ground_truth"], old.loc[145:720, "ground_truth"]
    )
    rows = []
    for a, b in [(1, 96), (97, 192), (193, 336), (337, 720), (529, 720)]:
        for h in (96, 192, 336, 720):
            if h < b:
                continue
            y = new.loc[a:b, "ground_truth"].to_numpy()
            u = new.loc[a:b, "uvhf"].to_numpy()
            p = new.loc[a:b, f"timemixer_h{h}"].to_numpy()
            rows.append(
                dict(
                    start=a,
                    end=b,
                    baseline_horizon=h,
                    mse_gain=1 - np.mean((u - y) ** 2) / np.mean((p - y) ** 2),
                    mae_gain=1 - np.mean(abs(u - y)) / np.mean(abs(p - y)),
                )
            )
    pd.DataFrame(rows).to_csv(case / "segment_metrics.csv", index=False)
    path = case / "selection_audit.json"
    data = json.loads(path.read_text())
    raw = pd.read_csv("/Users/river/PaperResearch/Project/datasets/ETT-small/ETTm1.csv")
    data.update(
        status="user-specified shifted origin; report all results without reselection",
        previous_origin=8752,
        shift_steps=144,
        new_origin=8896,
        new_zero_equals_previous_future_step=144,
        zero_timestamp=str(raw.iloc[34560 + 8896 - 1, 0]),
        first_forecast_timestamp=str(raw.iloc[34560 + 8896, 0]),
        last_forecast_timestamp=str(raw.iloc[34560 + 8896 + 719, 0]),
        history_and_future_shift_verified=True,
    )
    path.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()

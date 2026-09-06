"""Select strong segment advantages with a nonfailed baseline; reuse raw rendering."""
import importlib.util
import json
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def main() -> None:
    table = pd.read_csv(BASE / "ettm1_segments_20260906/all_candidate_audit.csv").merge(
        pd.read_csv(BASE / "ettm1_balanced_20260906/baseline_audit.csv"),
        on=["origin", "channel"],
        validate="one_to_one",
    )
    mask = (
        table.fit_eligible
        & (table.min_segment_mse_gain >= 0.3)
        & (table.min_segment_mae_gain >= 0.15)
        & (table.visibility96 >= 0.12)
        & (table.persistent_fraction96 >= 0.5)
        & (
            table[[f"timemixer_full_r2_h{h}" for h in (96, 192, 336, 720)]].min(axis=1)
            >= 0.2
        )
        & (table.timemixer_full_r2_h720 >= 0.4)
        & (table.last192_r2 > table.timemixer_last192_r2_h720)
    )
    eligible = table[mask].copy()
    eligible["score"] = eligible.min_segment_mse_gain * eligible.visibility96
    eligible = eligible.sort_values(
        ["score", "origin", "channel"], ascending=[False, True, True]
    )
    eligible.to_csv(OUT / "eligible.csv", index=False)
    selected = []
    for _, row in eligible.iterrows():
        if all(
            row.channel != s.channel or abs(row.origin - s.origin) >= 96
            for s in selected
        ):
            selected.append(row)
        if len(selected) == 3:
            break
    pd.DataFrame(selected).to_csv(OUT / "review_candidates.csv", index=False)
    (OUT / "counts.json").write_text(
        json.dumps(
            {
                "input_windows_channels": len(table),
                "eligible": len(eligible),
                "reviewed": len(selected),
            },
            indent=2,
        )
        + "\n"
    )
    spec = importlib.util.spec_from_file_location(
        "builder", BASE / "ettm1_segments_20260906/build_cases.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    module.main()


if __name__ == "__main__":
    main()

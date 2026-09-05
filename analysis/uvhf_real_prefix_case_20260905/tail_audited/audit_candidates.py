"""Audit absolute full and late forecast fidelity before case confirmation."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def main() -> None:
    old = pd.read_csv(BASE / "reselection/all_candidate_scores.csv")
    u = np.load(BASE / "raw/uvhf_all_channels.npz")[
        "prediction_scaled"
    ].astype(float)
    y = np.load(BASE / "raw/dlinear/h720.npz")["true"].astype(float)
    rows = []
    for c in range(7):
        parts = {}
        for label, start in [("full", 0), ("tail", 336), ("last192", 528)]:
            target, pred = y[:, start:, c], u[:, start:, c]
            yc, uc = target - target.mean(
                axis=1, keepdims=True
            ), pred - pred.mean(axis=1, keepdims=True)
            variance = np.mean(yc**2, axis=1)
            assert np.all(variance > 0)
            parts[f"{label}_r2"] = (
                1 - np.mean((pred - target) ** 2, axis=1) / variance
            )
            parts[f"{label}_corr"] = np.sum(yc * uc, axis=1) / np.maximum(
                np.linalg.norm(yc, axis=1) * np.linalg.norm(uc, axis=1), 1e-12
            )
            parts[f"{label}_amplitude_ratio"] = pred.std(axis=1) / np.sqrt(
                variance
            )
            parts[f"{label}_bias_sigma"] = abs(
                (pred - target).mean(axis=1)
            ) / np.sqrt(variance)
        for o in range(len(y)):
            rows.append(
                {
                    "origin": o,
                    "channel": c,
                    **{k: v[o] for k, v in parts.items()},
                }
            )
    table = old.merge(
        pd.DataFrame(rows), on=["origin", "channel"], validate="one_to_one"
    )
    gates = {
        "full_fit": table.full_r2 >= 0.35,
        "tail_fit": table.tail_r2 >= 0.25,
        "tail_shape": table.tail_corr >= 0.70,
        "last192_fit": table.last192_r2 >= 0,
        "tail_amplitude": table.tail_amplitude_ratio.between(0.5, 1.5),
        "tail_bias": table.tail_bias_sigma <= 0.35,
        "prefix_visible": table.visibility96 >= 0.075,
    }
    for name, mask in gates.items():
        table[f"pass_{name}"] = mask
    table["audited_eligible"] = table.accuracy_eligible & table[
        [f"pass_{k}" for k in gates]
    ].all(axis=1)
    table.to_csv(OUT / "all_candidate_audit.csv", index=False)
    ranked = table[table.audited_eligible].sort_values(
        ["visibility96", "tail_r2"], ascending=False
    )
    ranked.to_csv(OUT / "eligible_candidates.csv", index=False)
    chosen = []
    for _, row in ranked.iterrows():
        if all(
            row.channel != r.channel or abs(row.origin - r.origin) >= 96
            for r in chosen
        ):
            chosen.append(row)
        if len(chosen) == 5:
            break
    pd.DataFrame(chosen).to_csv(OUT / "review_candidates.csv", index=False)
    counts = {
        "total": len(table),
        "accuracy_eligible": int(table.accuracy_eligible.sum()),
        "audited_eligible": len(ranked),
        "each_gate_count": {k: int(v.sum()) for k, v in gates.items()},
    }
    (OUT / "gate_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    columns = [
        "origin",
        "channel",
        "visibility96",
        "full_r2",
        "tail_r2",
        "tail_corr",
        "last192_r2",
        "tail_bias_sigma",
    ]
    print(counts)
    print(ranked.head(8)[columns].to_string(index=False))
    print(
        "Rejected previous:",
        table.query("origin==144 and channel==4")[columns].to_string(
            index=False
        ),
    )


if __name__ == "__main__":
    main()

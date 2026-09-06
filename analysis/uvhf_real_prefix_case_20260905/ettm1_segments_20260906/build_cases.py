"""Export unchanged raw forecasts for segment-audited ETTm1 candidates."""
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
HS = (96, 192, 336, 720)


def main() -> None:
    raw = pd.read_csv("/Users/river/PaperResearch/Project/datasets/ETT-small/ETTm1.csv")
    uall = np.load(BASE / "raw/ettm1_all_uvhf.npy", mmap_mode="r")
    ex = {
        h: dict(
            np.load(
                BASE
                / f"matched_checkpoints/ettm1_timemixer/h{h}/predictions_validation.npz"
            )
        )
        for h in HS
    }
    for rank, r in pd.read_csv(OUT / "review_candidates.csv").iterrows():
        o, c = int(r.origin), int(r.channel)
        scale = ex[720]["train_std"][c]
        mean = ex[720]["train_mean"][c]
        y = raw.iloc[34560 + o : 34560 + o + 720, c + 1].to_numpy()
        u = uall[o, :, c].astype(float) * scale + mean
        pred = {h: ex[h]["pred"][o, :, c].astype(float) * scale + mean for h in HS}
        case = OUT / f"review_case_{rank}"
        case.mkdir(exist_ok=True)
        source = pd.DataFrame(
            {
                "step": np.arange(-719, 721),
                "history": np.r_[
                    raw.iloc[34560 + o - 720 : 34560 + o, c + 1], np.full(720, np.nan)
                ],
                "ground_truth": np.r_[np.full(720, np.nan), y],
                "uvhf": np.r_[np.full(720, np.nan), u],
            }
        )
        for h in HS:
            source[f"timemixer_h{h}"] = np.r_[
                np.full(720, np.nan), pred[h], np.full(720 - h, np.nan)
            ]
        source.to_csv(case / "source_data.csv", index=False)
        metrics = []
        for h in HS:
            for name, p in [("UVHF", u[:h]), ("TimeMixer", pred[h])]:
                e = p - y[:h]
                metrics.append(
                    {
                        "model": name,
                        "horizon": h,
                        "mse_raw": np.mean(e**2),
                        "mae_raw": np.mean(abs(e)),
                        "mse_scaled": np.mean(e**2) / scale**2,
                        "mae_scaled": np.mean(abs(e)) / scale,
                    }
                )
        pd.DataFrame(metrics).to_csv(case / "selected_metrics.csv", index=False)
        pd.DataFrame(
            [
                {
                    "short_horizon": a,
                    "long_horizon": b,
                    "chpd_raw": np.mean(abs(pred[a] - pred[b][:a])),
                    "nchpd": np.mean(abs(pred[a] - pred[b][:a])) / scale,
                }
                for a, b in combinations(HS, 2)
            ]
        ).to_csv(case / "selected_pair_disagreement.csv", index=False)
        selection = {
            "dataset": "ETTm1",
            "split": "validation",
            "selected": r.to_dict(),
            "channel_name": raw.columns[c + 1],
            "uvhf_checkpoint_sha256": "2e8c7e7c7d7545a4edfbe81be857299b6fea0dc7e944464803f273257ecb6948",
            "baseline": "TimeMixer",
            "post_hoc_selection": True,
            "new_test_access": False,
            "status": "pending review",
        }
        (case / "selection_audit.json").write_text(
            json.dumps(selection, indent=2) + "\n"
        )
    spec = importlib.util.spec_from_file_location(
        "render", BASE / "ettm1_20260906/render_cases.py"
    )
    render = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render)
    render.OUT = OUT
    render.main()
    spec = importlib.util.spec_from_file_location("plot", BASE / "plot_single_panel.py")
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    for case in OUT.glob("review_case_*"):
        a = json.loads((case / "selection_audit.json").read_text())
        p = case / "figure_settings.json"
        s = json.loads(p.read_text())
        s["subtitle"] = f"ETTm1 · {a['channel_name']} · selected validation example"
        s["ylabel"] = f"{a['channel_name']} (original scale)"
        # Avoid rounding low-amplitude channels to duplicate five-unit ticks.
        s["main_yticks"] = (
            np.linspace(
                s["main_ylim"][0] + 0.08 * (s["main_ylim"][1] - s["main_ylim"][0]),
                s["main_ylim"][0] + 0.35 * (s["main_ylim"][1] - s["main_ylim"][0]),
                4,
            )
            .round(1)
            .tolist()
        )
        s["prefix_yticks"] = np.linspace(*s["prefix_ylim"], 4).round(1).tolist()
        p.write_text(json.dumps(s, indent=2) + "\n")
        plot.main(zoom=True, output=case)


if __name__ == "__main__":
    main()

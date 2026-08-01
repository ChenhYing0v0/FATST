#!/usr/bin/env python3
"""Extract the selected Main-I published rows from TimeAlign Table 6."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import pdfplumber


MODELS = (
    "TimeAlign",
    "CMoS",
    "TimeBase",
    "TVNet",
    "iTransformer",
    "TimeMixer",
    "Leddam",
    "ModernTCN",
    "PatchTST",
    "Crossformer",
    "TimesNet",
    "DLinear",
)
SELECTED_MODELS = (
    "TimeAlign",
    "TimeMixer",
    "DLinear",
    "iTransformer",
    "PatchTST",
)
DATASET_ORDER = (
    "ETTm1",
    "ETTm2",
    "ETTh1",
    "ETTh2",
    "Weather",
    "ECL",
    "Traffic",
    "Solar",
)
SELECTED_DATASETS = tuple(
    dataset for dataset in DATASET_ORDER if dataset != "Traffic"
)
HORIZONS = (96, 192, 336, 720)
EXPECTED_AVERAGES = {
    "TimeAlign": {
        "ETTm1": (0.340, 0.367),
        "ETTm2": (0.243, 0.302),
        "ETTh1": (0.406, 0.420),
        "ETTh2": (0.336, 0.382),
        "Weather": (0.214, 0.244),
        "ECL": (0.154, 0.244),
        "Solar": (0.192, 0.214),
    },
    "TimeMixer": {
        "ETTm1": (0.355, 0.380),
        "ETTm2": (0.257, 0.318),
        "ETTh1": (0.427, 0.441),
        "ETTh2": (0.349, 0.397),
        "Weather": (0.226, 0.264),
        "ECL": (0.185, 0.284),
        "Solar": (0.193, 0.264),
    },
    "DLinear": {
        "ETTm1": (0.356, 0.378),
        "ETTm2": (0.259, 0.324),
        "ETTh1": (0.424, 0.439),
        "ETTh2": (0.431, 0.447),
        "Weather": (0.242, 0.293),
        "ECL": (0.166, 0.264),
        "Solar": (0.224, 0.226),
    },
    "iTransformer": {
        "ETTm1": (0.362, 0.391),
        "ETTm2": (0.269, 0.329),
        "ETTh1": (0.439, 0.448),
        "ETTh2": (0.374, 0.406),
        "Weather": (0.233, 0.271),
        "ECL": (0.164, 0.261),
        "Solar": (0.202, 0.248),
    },
    "PatchTST": {
        "ETTm1": (0.353, 0.382),
        "ETTm2": (0.256, 0.317),
        "ETTh1": (0.418, 0.436),
        "ETTh2": (0.351, 0.404),
        "Weather": (0.226, 0.264),
        "ECL": (0.159, 0.253),
        "Solar": (0.194, 0.245),
    },
}
KNOWN_SOURCE_AVERAGE_ANOMALIES = {
    ("PatchTST", "ETTm1"),
    ("PatchTST", "ETTh2"),
    ("TimeAlign", "Weather"),
    ("TimeMixer", "ECL"),
    ("TimeMixer", "Solar"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def group_words_by_line(words: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        if not groups or abs(groups[-1][0]["top"] - word["top"]) > 0.6:
            groups.append([word])
        else:
            groups[-1].append(word)
    return groups


def extract_rows(pdf_path: Path) -> list[dict[str, object]]:
    source_sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) < 22:
            raise ValueError("TimeAlign source PDF does not contain page 22")
        words = pdf.pages[21].extract_words(x_tolerance=1, y_tolerance=2)

    horizon_lines: list[list[dict]] = []
    for line in group_words_by_line(words):
        horizon_tokens = [
            word
            for word in line
            if 115 <= word["x0"] <= 128 and word["text"].isdigit()
        ]
        if len(horizon_tokens) != 1:
            continue
        if int(horizon_tokens[0]["text"]) not in HORIZONS:
            continue
        numeric = [word for word in line if word["x0"] > 128]
        if len(numeric) == 24:
            horizon_lines.append(line)

    expected_line_count = len(DATASET_ORDER) * len(HORIZONS)
    if len(horizon_lines) != expected_line_count:
        raise ValueError(
            f"expected {expected_line_count} horizon lines, got {len(horizon_lines)}"
        )

    output: list[dict[str, object]] = []
    for line_index, line in enumerate(horizon_lines):
        dataset = DATASET_ORDER[line_index // len(HORIZONS)]
        horizon = int(
            next(
                word["text"]
                for word in line
                if 115 <= word["x0"] <= 128 and word["text"].isdigit()
            )
        )
        values = [
            float(word["text"])
            for word in sorted(line, key=lambda item: item["x0"])
            if word["x0"] > 128
        ]
        if dataset not in SELECTED_DATASETS:
            continue
        for model in SELECTED_MODELS:
            model_index = MODELS.index(model)
            output.append(
                {
                    "source": "TimeAlign ICLR 2026",
                    "source_url": "https://arxiv.org/abs/2509.14181",
                    "source_pdf_sha256": source_sha256,
                    "source_page": 22,
                    "source_table": 6,
                    "evidence_role": "published_context_not_matched_attribution",
                    "model": model,
                    "dataset": dataset,
                    "horizon": horizon,
                    "mse": values[2 * model_index],
                    "mae": values[2 * model_index + 1],
                    "reported_run_aggregation": "three_run_mean",
                }
            )
    return output


def validate(rows: list[dict[str, object]]) -> None:
    expected_rows = len(SELECTED_MODELS) * len(SELECTED_DATASETS) * len(HORIZONS)
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} output rows, got {len(rows)}")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["dataset"]))].append(row)
    unexpected_average_mismatches = []
    for (model, dataset), group in grouped.items():
        if sorted(int(row["horizon"]) for row in group) != list(HORIZONS):
            raise ValueError(f"horizon mismatch for {model}/{dataset}")
        actual = (
            sum(float(row["mse"]) for row in group) / len(group),
            sum(float(row["mae"]) for row in group) / len(group),
        )
        expected = EXPECTED_AVERAGES[model][dataset]
        mismatch = any(
            abs(value - target) > 0.00101
            for value, target in zip(actual, expected)
        )
        if mismatch and (model, dataset) not in KNOWN_SOURCE_AVERAGE_ANOMALIES:
            unexpected_average_mismatches.append(
                f"{model}/{dataset}: actual={actual}, expected={expected}"
            )
    if unexpected_average_mismatches:
        raise ValueError(
            "unexpected Table 1 average cross-check failures: "
            + "; ".join(unexpected_average_mismatches)
        )


def main() -> None:
    args = parse_args()
    rows = extract_rows(args.pdf)
    validate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"wrote {len(rows)} rows: {len(SELECTED_MODELS)} models x "
        f"{len(SELECTED_DATASETS)} datasets x {len(HORIZONS)} horizons"
    )


if __name__ == "__main__":
    main()

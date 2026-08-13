#!/usr/bin/env python3
"""Build the paper-facing ISCF-BSCA comparison in TimeAlign Table-6 style."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pdfplumber
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODELS = (
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
EXCLUDED_SOURCE_MODELS = {"CMoS", "TimeBase"}
DISPLAY_MODELS = (
    "ISCF-BSCA",
    "TimeAlign",
    "QDF",
    "AMD",
    "SimpleTM",
) + SOURCE_MODELS[3:]
MODEL_YEARS = {
    "ISCF-BSCA": "Ours",
    "TimeAlign": "2026",
    "QDF": "2026",
    "AMD": "2025",
    "SimpleTM": "2025",
    "TVNet": "2025",
    "iTransformer": "2024b",
    "TimeMixer": "2024",
    "Leddam": "2024",
    "ModernTCN": "2024",
    "PatchTST": "2023",
    "Crossformer": "2023",
    "TimesNet": "2023",
    "DLinear": "2023",
}
SOURCE_DATASET_ORDER = (
    "ETTm1",
    "ETTm2",
    "ETTh1",
    "ETTh2",
    "Weather",
    "ECL",
    "Traffic",
    "Solar",
)
DISPLAY_DATASETS = (
    "ETTm1",
    "ETTm2",
    "ETTh1",
    "ETTh2",
    "Weather",
    "ECL",
    "Solar",
)
HORIZONS = (96, 192, 336, 720)
METRICS = ("mse", "mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timealign-pdf",
        type=Path,
        default=ROOT / "tmp" / "pdfs" / "timealign" / "timealign_iclr2026.pdf",
    )
    parser.add_argument(
        "--iscf-scorecard",
        type=Path,
        default=ROOT
        / "analysis"
        / "iscf_bsca_main_v1_hpo_20260731"
        / "final_hpo_freeze_20260806"
        / "selected_main_scorecard_final.csv",
    )
    parser.add_argument(
        "--timealign-reproduced",
        type=Path,
        default=ROOT
        / "analysis"
        / "iscf_bsca_paper_experiment_consolidation_20260731"
        / "timealign_main_i_full_reproduction_20260806"
        / "timealign_main_i_local_metrics.csv",
    )
    parser.add_argument(
        "--audited-published-selected",
        type=Path,
        default=ROOT
        / "analysis"
        / "iscf_bsca_paper_experiment_consolidation_20260731"
        / "timealign_table6_main_i_published.csv",
    )
    parser.add_argument(
        "--qdf-published",
        type=Path,
        default=ROOT
        / "analysis"
        / "iscf_bsca_paper_experiment_consolidation_20260731"
        / "qdf_main_i_20260806"
        / "qdf_table6_published.csv",
    )
    parser.add_argument(
        "--qdf-solar-reproduced",
        type=Path,
        default=ROOT
        / "analysis"
        / "iscf_bsca_paper_experiment_consolidation_20260731"
        / "qdf_main_i_20260806"
        / "qdf_solar_local_metrics.csv",
    )
    parser.add_argument(
        "--qdf-reproduced-all",
        type=Path,
        default=None,
        help="Complete local QDF eight-dataset reproduction; replaces published/Solar inputs.",
    )
    parser.add_argument(
        "--amd-simpletm-reproduced",
        type=Path,
        default=ROOT
        / "analysis"
        / "iscf_bsca_paper_experiment_consolidation_20260731"
        / "amd_simpletm_main_i_reproduction_20260806"
        / "remote_lite"
        / "audit"
        / "cell_metrics.csv",
    )
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument(
        "--table-id",
        default="ISCF-BSCA-MAIN-I-AMD-SIMPLETM-LOCAL-20260808",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def group_words_by_line(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        if not groups or abs(groups[-1][0]["top"] - word["top"]) > 0.6:
            groups.append([word])
        else:
            groups[-1].append(word)
    return groups


def extract_timealign_table6(pdf_path: Path) -> list[dict[str, Any]]:
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) < 22:
            raise ValueError("TimeAlign PDF does not contain page 22")
        words = pdf.pages[21].extract_words(x_tolerance=1, y_tolerance=2)

    horizon_lines: list[list[dict[str, Any]]] = []
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

    expected_lines = len(SOURCE_DATASET_ORDER) * len(HORIZONS)
    if len(horizon_lines) != expected_lines:
        raise ValueError(
            f"expected {expected_lines} Table-6 horizon rows, "
            f"found {len(horizon_lines)}"
        )

    output: list[dict[str, Any]] = []
    for line_index, line in enumerate(horizon_lines):
        dataset = SOURCE_DATASET_ORDER[line_index // len(HORIZONS)]
        if dataset not in DISPLAY_DATASETS:
            continue
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
        for model_index, model in enumerate(SOURCE_MODELS):
            if model in EXCLUDED_SOURCE_MODELS:
                continue
            output.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "horizon": horizon,
                    "mse": values[2 * model_index],
                    "mae": values[2 * model_index + 1],
                    "value_origin": "timealign_table6_published_three_run_mean",
                    "system_role": "horizon_specific_published_context",
                }
            )
    return output


def load_iscf_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(path):
        if row["dataset"] not in DISPLAY_DATASETS:
            continue
        rows.append(
            {
                "model": "ISCF-BSCA",
                "dataset": row["dataset"],
                "horizon": int(row["horizon"]),
                "mse": float(row["test_mse"]),
                "mae": float(row["test_mae"]),
                "value_origin": "terminal_h4n_selected_single_seed_test_tuned",
                "system_role": "one_unified_model_per_dataset",
            }
        )
    return rows


def load_qdf_rows(published_path: Path, solar_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(published_path):
        rows.append(
            {
                "model": "QDF",
                "dataset": row["dataset"],
                "horizon": int(row["horizon"]),
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
                "value_origin": "qdf_table6_published_three_run_mean",
                "system_role": "horizon_specific_published_context",
            }
        )
    for row in read_csv(solar_path):
        rows.append(
            {
                "model": "QDF",
                "dataset": "Solar",
                "horizon": int(row["horizon"]),
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
                "value_origin": row["value_origin"],
                "system_role": row["system_role"],
            }
        )
    keys = {(row["dataset"], row["horizon"]) for row in rows}
    expected = {
        (dataset, horizon)
        for dataset in DISPLAY_DATASETS
        for horizon in HORIZONS
    }
    if len(rows) != 28 or keys != expected:
        raise ValueError(
            f"expected complete 28-cell QDF matrix, found rows={len(rows)}, "
            f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
        )
    return rows


def load_qdf_reproduced_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(path):
        if row["dataset"] not in DISPLAY_DATASETS:
            continue
        rows.append(
            {
                "model": "QDF",
                "dataset": row["dataset"],
                "horizon": int(row["horizon"]),
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
                "value_origin": row["value_origin"],
                "system_role": row["system_role"],
            }
        )
    keys = {(row["dataset"], row["horizon"]) for row in rows}
    expected = {
        (dataset, horizon)
        for dataset in DISPLAY_DATASETS
        for horizon in HORIZONS
    }
    if len(rows) != 28 or keys != expected:
        raise ValueError(
            f"expected complete 28-cell reproduced QDF dense matrix, "
            f"found rows={len(rows)}, missing={sorted(expected - keys)}"
        )
    return rows


def load_amd_simpletm_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_repeats = {"AMD": 1, "SimpleTM": None}
    for source in read_csv(path):
        model = source["baseline"]
        if model not in expected_repeats:
            raise ValueError(f"unexpected AMD/SimpleTM baseline: {model}")
        repeat_count = int(source["repeat_count"])
        if model == "AMD" and repeat_count != expected_repeats[model]:
            raise ValueError(
                f"AMD {source['dataset']} H{source['horizon']}: "
                f"expected one repetition, found {repeat_count}"
            )
        if model == "SimpleTM" and repeat_count not in {2, 3}:
            raise ValueError(
                f"SimpleTM {source['dataset']} H{source['horizon']}: "
                f"expected two or three native repetitions, found {repeat_count}"
            )
        rows.append(
            {
                "model": model,
                "dataset": source["dataset"],
                "horizon": int(source["horizon"]),
                "mse": float(source["mse"]),
                "mae": float(source["mae"]),
                "value_origin": "official_code_local_native_reproduction",
                "system_role": "horizon_specific_official_native",
            }
        )
    expected = {
        (model, dataset, horizon)
        for model in expected_repeats
        for dataset in DISPLAY_DATASETS
        for horizon in HORIZONS
    }
    keys = {
        (row["model"], row["dataset"], int(row["horizon"])) for row in rows
    }
    if len(rows) != 56 or keys != expected:
        raise ValueError(
            f"expected complete 56-cell AMD/SimpleTM matrix, found rows={len(rows)}, "
            f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
        )
    return rows


def load_exchange_companion(
    iscf_path: Path, timealign_path: Path, qdf_path: Path | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(iscf_path):
        if row["dataset"] == "Exchange":
            rows.append(
                {
                    "model": "ISCF-BSCA",
                    "dataset": "Exchange",
                    "horizon": int(row["horizon"]),
                    "mse": float(row["test_mse"]),
                    "mae": float(row["test_mae"]),
                    "value_origin": "terminal_h4n_selected_single_seed_test_tuned",
                    "system_role": "one_unified_model_per_dataset",
                }
            )
    for row in read_csv(timealign_path):
        if row["dataset"] == "Exchange":
            rows.append(
                {
                    "model": "TimeAlign",
                    "dataset": "Exchange",
                    "horizon": int(row["horizon"]),
                    "mse": float(row["mse"]),
                    "mae": float(row["mae"]),
                    "value_origin": row["value_origin"],
                    "system_role": "horizon_specific_source_informed_bootstrap",
                }
            )
    if qdf_path is not None:
        for row in read_csv(qdf_path):
            if row["dataset"] == "Exchange":
                rows.append(
                    {
                        "model": "QDF",
                        "dataset": "Exchange",
                        "horizon": int(row["horizon"]),
                        "mse": float(row["mse"]),
                        "mae": float(row["mae"]),
                        "value_origin": row["value_origin"],
                        "system_role": row["system_role"],
                    }
                )
    expected_rows = 12 if qdf_path is not None else 8
    if len(rows) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} Exchange companion rows, found {len(rows)}"
        )
    return add_averages_and_styles(rows)


def validate_against_audited_selected(
    rows: list[dict[str, Any]], audited_path: Path
) -> None:
    extracted = {
        (row["model"], row["dataset"], int(row["horizon"])): row
        for row in rows
    }
    audited = read_csv(audited_path)
    if len(audited) != 140:
        raise ValueError(f"expected 140 audited published rows, found {len(audited)}")
    errors = []
    for row in audited:
        key = (row["model"], row["dataset"], int(row["horizon"]))
        candidate = extracted.get(key)
        if candidate is None:
            errors.append(f"missing:{key}")
            continue
        if (
            float(candidate["mse"]) != float(row["mse"])
            or float(candidate["mae"]) != float(row["mae"])
        ):
            errors.append(f"value_mismatch:{key}")
    if errors:
        raise ValueError("audited selected-row cross-check failed: " + ";".join(errors))


def override_reproduced_timealign(
    rows: list[dict[str, Any]], reproduced_path: Path
) -> None:
    reproduced = {
        (row["dataset"], int(row["horizon"])): row
        for row in read_csv(reproduced_path)
    }
    replaced = 0
    for row in rows:
        key = (row["dataset"], row["horizon"])
        if row["model"] != "TimeAlign" or key not in reproduced:
            continue
        source = reproduced[key]
        row["mse"] = float(source["mse"])
        row["mae"] = float(source["mae"])
        row["value_origin"] = "official_native_reproduced_single_seed"
        row["system_role"] = "horizon_specific_official_native"
        replaced += 1
    if replaced != 28:
        raise ValueError(f"expected 28 reproduced TimeAlign overrides, got {replaced}")


def validate_matrix(rows: list[dict[str, Any]]) -> None:
    expected = len(DISPLAY_MODELS) * len(DISPLAY_DATASETS) * len(HORIZONS)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} matrix rows, found {len(rows)}")
    keys = {
        (row["model"], row["dataset"], int(row["horizon"])) for row in rows
    }
    if len(keys) != expected:
        raise ValueError("model-dataset-horizon keys are not unique")


def add_averages_and_styles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["dataset"])].append(row)
    for (model, dataset), group in grouped.items():
        if {int(row["horizon"]) for row in group} != set(HORIZONS):
            raise ValueError(f"incomplete horizon block for {model}/{dataset}")
        output.append(
            {
                "model": model,
                "dataset": dataset,
                "horizon": "Avg.",
                "mse": sum(float(row["mse"]) for row in group) / 4,
                "mae": sum(float(row["mae"]) for row in group) / 4,
                "value_origin": "arithmetic_mean_of_four_horizon_rows",
                "system_role": group[0]["system_role"],
            }
        )

    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        for metric in METRICS:
            by_cell[(row["dataset"], str(row["horizon"]), metric)].append(row)
    for (_, _, metric), cell_rows in by_cell.items():
        displayed_values = sorted(
            {round(float(row[metric]) + 1e-12, 3) for row in cell_rows}
        )
        best = displayed_values[0]
        second = displayed_values[1] if len(displayed_values) > 1 else None
        for row in cell_rows:
            displayed = round(float(row[metric]) + 1e-12, 3)
            row[f"{metric}_display"] = f"{displayed:.3f}"
            if displayed == best:
                row[f"{metric}_style"] = "best"
            elif second is not None and displayed == second:
                row[f"{metric}_style"] = "second"
            else:
                row[f"{metric}_style"] = "normal"
    return output


def write_long_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "model",
        "dataset",
        "horizon",
        "mse",
        "mae",
        "mse_display",
        "mae_display",
        "mse_style",
        "mae_style",
        "value_origin",
        "system_role",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def format_pdf_value(row: dict[str, Any], metric: str, style: ParagraphStyle) -> Paragraph:
    value = row[f"{metric}_display"]
    marker = row[f"{metric}_style"]
    if marker == "best":
        value = f'<font color="#d62728"><b>{value}</b></font>'
    elif marker == "second":
        value = f'<font color="#2563eb"><u>{value}</u></font>'
    return Paragraph(value, style)


def build_pdf(
    path: Path,
    rows: list[dict[str, Any]],
    exchange_rows: list[dict[str, Any]],
) -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TableTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10,
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    header_style = ParagraphStyle(
        "Header",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=5.7,
        leading=6.2,
        alignment=TA_CENTER,
    )
    value_style = ParagraphStyle(
        "Value",
        parent=header_style,
        fontSize=5.4,
        leading=6,
    )
    note_style = ParagraphStyle(
        "Note",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.2,
        leading=7.4,
        alignment=TA_LEFT,
    )

    lookup = {
        (row["model"], row["dataset"], str(row["horizon"])): row
        for row in rows
    }
    table_data: list[list[Any]] = []
    model_header: list[Any] = [
        Paragraph("Models", header_style),
        "",
    ]
    metric_header: list[Any] = [
        Paragraph("Dataset", header_style),
        Paragraph("H", header_style),
    ]
    for model in DISPLAY_MODELS:
        suffix = ""
        model_header.extend(
            [
                Paragraph(
                    f"<b>{model}{suffix}</b><br/>({MODEL_YEARS[model]})",
                    header_style,
                ),
                "",
            ]
        )
        metric_header.extend(
            [Paragraph("MSE", header_style), Paragraph("MAE", header_style)]
        )
    table_data.extend([model_header, metric_header])

    avg_row_indices: list[int] = []
    dataset_spans: list[tuple[int, int]] = []
    for dataset in DISPLAY_DATASETS:
        start = len(table_data)
        for row_offset, horizon in enumerate((*HORIZONS, "Avg.")):
            current: list[Any] = [dataset if row_offset == 0 else "", str(horizon)]
            for model in DISPLAY_MODELS:
                row = lookup[(model, dataset, str(horizon))]
                current.extend(
                    [
                        format_pdf_value(row, "mse", value_style),
                        format_pdf_value(row, "mae", value_style),
                    ]
                )
            table_data.append(current)
        end = len(table_data) - 1
        dataset_spans.append((start, end))
        avg_row_indices.append(end)

    page_size = (390 * mm, 210 * mm)
    document = SimpleDocTemplate(
        str(path),
        pagesize=page_size,
        leftMargin=8 * mm,
        rightMargin=8 * mm,
        topMargin=6 * mm,
        bottomMargin=6 * mm,
    )
    metric_width = 12.1 * mm
    table = Table(
        table_data,
        colWidths=[15 * mm, 10 * mm] + [metric_width] * (2 * len(DISPLAY_MODELS)),
        repeatRows=2,
    )
    commands: list[tuple[Any, ...]] = [
        ("SPAN", (0, 0), (1, 0)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 5.5),
        ("LEADING", (0, 0), (-1, -1), 6.2),
        ("TOPPADDING", (0, 0), (-1, -1), 1.3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.3),
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
        ("LINEBELOW", (0, 1), (-1, 1), 0.7, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
        ("BACKGROUND", (2, 0), (3, -1), colors.HexColor("#fff7ed")),
        ("BACKGROUND", (4, 0), (5, -1), colors.HexColor("#f8fafc")),
        ("BACKGROUND", (6, 0), (7, -1), colors.HexColor("#f0fdf4")),
    ]
    for model_index in range(len(DISPLAY_MODELS)):
        start_column = 2 + 2 * model_index
        commands.append(("SPAN", (start_column, 0), (start_column + 1, 0)))
        commands.append(
            ("LINEBEFORE", (start_column, 0), (start_column, -1), 0.25, colors.grey)
        )
    for start, end in dataset_spans:
        commands.extend(
            [
                ("SPAN", (0, start), (0, end)),
                ("LINEABOVE", (0, start), (-1, start), 0.45, colors.black),
                ("FONTNAME", (0, start), (0, end), "Helvetica-Bold"),
            ]
        )
    for row_index in avg_row_indices:
        commands.extend(
            [
                ("LINEABOVE", (1, row_index), (-1, row_index), 0.35, colors.grey),
                ("BACKGROUND", (1, row_index), (-1, row_index), colors.HexColor("#f6f6f6")),
            ]
        )
    table.setStyle(TableStyle(commands))

    title = Paragraph(
        "Table X. Long-term forecasting results under the TimeAlign Table-6 layout.",
        title_style,
    )
    qdf_full_reproduction = any(
        row["model"] == "QDF"
        and row["value_origin"] == "official_code_local_single_seed_l336"
        for row in rows
    )
    qdf_note = (
        "QDF uses our official-code, single-seed L=336 reproduction on all "
        "seven shared datasets; Solar uses an ECL-derived preset. "
        if qdf_full_reproduction
        else "QDF uses published three-run means for six datasets and our "
        "official-code, ECL-preset-derived single-seed Solar reproduction. "
    )
    exchange_note = (
        "Exchange is reported in a three-system companion block."
        if qdf_full_reproduction
        else "Exchange is reported in a two-system companion block."
    )
    note = Paragraph(
        "All entries report MSE/MAE for H in {96, 192, 336, 720}; Avg. is "
        "recomputed as the arithmetic mean of the four horizon rows. Red bold and "
        "blue underlined values denote the best and second-best displayed results "
        "after three-decimal rounding. ISCF-BSCA uses one unified, single-seed, "
        "test-tuned model per dataset. TimeAlign uses the artifact-complete local "
        "seed-2021 reproduction on all seven shared datasets. "
        + qdf_note
        + "AMD uses one official-code local run per cell (L=512, seed 2024); "
        "SimpleTM uses the arithmetic mean of its official native repetitions "
        "(L=96, fix_seed 2025). All remaining baselines are transcribed "
        "from TimeAlign Table 6 and are unmatched published context. Traffic and "
        "Exchange are excluded from this dense panel because they do not share the "
        "current comparison surface; "
        + exchange_note,
        note_style,
    )
    exchange_models = ("ISCF-BSCA", "TimeAlign", "QDF") if any(
        row["model"] == "QDF" for row in exchange_rows
    ) else ("ISCF-BSCA", "TimeAlign")
    exchange_lookup = {
        (row["model"], str(row["horizon"])): row for row in exchange_rows
    }
    exchange_data: list[list[Any]] = [
        [Paragraph("H", header_style)]
        + [
            Paragraph(
                f"<b>{model}</b>{'<super>*</super>' if model in {'TimeAlign', 'QDF'} else ''}",
                header_style,
            )
            for model in exchange_models
            for _ in (0, 1)
        ],
        [""]
        + [
            Paragraph(metric, header_style)
            for _ in exchange_models
            for metric in ("MSE", "MAE")
        ],
    ]
    for horizon in (*HORIZONS, "Avg."):
        current: list[Any] = [str(horizon)]
        for model in exchange_models:
            row = exchange_lookup[(model, str(horizon))]
            current.extend(
                [
                    format_pdf_value(row, "mse", value_style),
                    format_pdf_value(row, "mae", value_style),
                ]
            )
        exchange_data.append(current)
    exchange_table = Table(
        exchange_data,
        colWidths=[10 * mm] + [13 * mm] * (2 * len(exchange_models)),
        hAlign="LEFT",
    )
    exchange_commands: list[tuple[Any, ...]] = [
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 5.5),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 1), (-1, 1), 0.6, colors.black),
        ("LINEABOVE", (0, -1), (-1, -1), 0.35, colors.grey),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
    ]
    exchange_backgrounds = ("#fff7ed", "#f8fafc", "#f0fdf4")
    for model_index, color in enumerate(exchange_backgrounds[: len(exchange_models)]):
        start_column = 1 + 2 * model_index
        exchange_commands.extend(
            [
                ("SPAN", (start_column, 0), (start_column + 1, 0)),
                ("LINEBEFORE", (start_column, 0), (start_column, -1), 0.25, colors.grey),
                ("BACKGROUND", (start_column, 0), (start_column + 1, -1), colors.HexColor(color)),
            ]
        )
    exchange_table.setStyle(TableStyle(exchange_commands))
    exchange_title = Paragraph(
        "<b>Exchange companion | Locally reproduced systems.</b>", note_style
    )
    exchange_disclosure = Paragraph(
        "* TimeAlign and QDF use disclosed ETTh1-derived source-informed presets "
        "because neither official release provides an Exchange script. These rows "
        "are native accuracy context, not matched mechanism attribution.",
        note_style,
    )
    document.build(
        [
            title,
            Spacer(1, 2 * mm),
            table,
            Spacer(1, 2 * mm),
            note,
            Spacer(1, 2 * mm),
            exchange_title,
            Spacer(1, 1 * mm),
            exchange_table,
            Spacer(1, 1 * mm),
            exchange_disclosure,
        ]
    )


def latex_value(row: dict[str, Any], metric: str) -> str:
    value = row[f"{metric}_display"]
    marker = row[f"{metric}_style"]
    if marker == "best":
        return f"\\textcolor{{red}}{{\\textbf{{{value}}}}}"
    if marker == "second":
        return f"\\textcolor{{blue}}{{\\underline{{{value}}}}}"
    return value


def build_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lookup = {
        (row["model"], row["dataset"], str(row["horizon"])): row
        for row in rows
    }
    lines = [
        "% Required packages: booktabs, multirow, graphicx, xcolor",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{1.2pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{cc" + "|cc" * len(DISPLAY_MODELS) + "}",
        "\\toprule",
    ]
    headers = ["Dataset", "H"]
    for model in DISPLAY_MODELS:
        suffix = ""
        headers.append(
            f"\\multicolumn{{2}}{{c}}{{{model}{suffix} ({MODEL_YEARS[model]})}}"
        )
    lines.append(" & ".join(headers) + " \\\\")
    metric_headers = ["", ""]
    for _ in DISPLAY_MODELS:
        metric_headers.extend(["MSE", "MAE"])
    lines.extend([" & ".join(metric_headers) + " \\\\", "\\midrule"])

    for dataset_index, dataset in enumerate(DISPLAY_DATASETS):
        for row_offset, horizon in enumerate((*HORIZONS, "Avg.")):
            dataset_cell = (
                f"\\multirow{{5}}{{*}}{{{dataset}}}" if row_offset == 0 else ""
            )
            values = [dataset_cell, str(horizon)]
            for model in DISPLAY_MODELS:
                row = lookup[(model, dataset, str(horizon))]
                values.extend([latex_value(row, "mse"), latex_value(row, "mae")])
            lines.append(" & ".join(values) + " \\\\")
            if row_offset == 3:
                lines.append("\\cmidrule(lr){2-" + str(2 + 2 * len(DISPLAY_MODELS)) + "}")
        if dataset_index < len(DISPLAY_DATASETS) - 1:
            lines.append("\\midrule")
    qdf_full_reproduction = any(
        row["model"] == "QDF"
        and row["value_origin"] == "official_code_local_single_seed_l336"
        for row in rows
    )
    qdf_caption = (
        "QDF uses our official-code single-seed L=336 reproduction on all seven "
        "shared datasets; Solar uses a source-informed ECL-derived preset; "
        if qdf_full_reproduction
        else "QDF uses published values on six datasets and a source-informed "
        "official-code Solar reproduction; "
    )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\caption{Long-term forecasting results in the TimeAlign Table-6 layout. "
            "All entries report MSE/MAE, and Avg. is recomputed over the four displayed "
            "horizons. Best and second-best displayed values are bold and underlined. "
            "ISCF-BSCA uses one unified single-seed test-tuned model per dataset. "
            "TimeAlign uses our official-native single-seed reproduction on all seven "
            "shared datasets. "
            + qdf_caption
            + "AMD uses one official-code local run per cell (L=512, seed 2024), "
            "SimpleTM uses the mean of its official native repetitions (L=96, "
            "fix\\_seed 2025), and all other baselines are published "
            "context rather than matched local reproductions.}",
            "\\label{tab:main_iscf_bsca}",
            "\\end{table*}",
            "",
        ]
    )
    table_text = "\n".join(lines)
    path.write_text(table_text, encoding="utf-8")
    standalone_lines = [
        "% Auto-generated review copy; edit the table builder, not this file.",
        "\\documentclass[10pt]{article}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage[a3paper,landscape,margin=10mm]{geometry}",
        "\\usepackage{booktabs}",
        "\\usepackage{multirow}",
        "\\usepackage{graphicx}",
        "\\usepackage{xcolor}",
        "\\pagestyle{empty}",
        "\\renewcommand{\\arraystretch}{0.82}",
        "\\setlength{\\abovecaptionskip}{5pt}",
        "\\makeatletter\\setlength{\\@fptop}{0pt}\\makeatother",
        "\\begin{document}",
        table_text,
        "\\end{document}",
        "",
    ]
    path.with_name("table_iscf_bsca_main_i_standalone.tex").write_text(
        "\n".join(standalone_lines), encoding="utf-8"
    )


def build_exchange_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lookup = {
        (row["model"], str(row["horizon"])): row
        for row in rows
    }
    models = ("ISCF-BSCA", "TimeAlign", "QDF") if any(
        row["model"] == "QDF" for row in rows
    ) else ("ISCF-BSCA", "TimeAlign")
    model_headers = []
    for model in models:
        suffix = "$^{\\dagger}$" if model in {"TimeAlign", "QDF"} else ""
        model_headers.append(f"\\multicolumn{{2}}{{c}}{{{model}{suffix}}}")
    lines = [
        "% Companion block; TimeAlign and QDF Exchange use source-informed presets.",
        "\\begin{tabular}{c" + "|cc" * len(models) + "}",
        "\\toprule",
        "H & " + " & ".join(model_headers) + " \\\\",
        " & " + " & ".join("MSE & MAE" for _ in models) + " \\\\",
        "\\midrule",
    ]
    for horizon in (*HORIZONS, "Avg."):
        values = [str(horizon)]
        for model in models:
            row = lookup[(model, str(horizon))]
            values.extend([latex_value(row, "mse"), latex_value(row, "mae")])
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "% $^{\\dagger}$ ETTh1-derived source-informed preset; neither release provides an official Exchange script.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    iscf_rows = [row for row in rows if row["model"] == "ISCF-BSCA"]
    standard_rows = [row for row in iscf_rows if row["horizon"] != "Avg."]
    qdf_full_reproduction = args.qdf_reproduced_all is not None
    qdf_hashes = (
        {"qdf_reproduced_all": file_sha256(args.qdf_reproduced_all)}
        if qdf_full_reproduction
        else {
            "qdf_published": file_sha256(args.qdf_published),
            "qdf_solar_reproduced": file_sha256(args.qdf_solar_reproduced),
        }
    )
    summary = {
        "table_id": args.table_id,
        "models": list(DISPLAY_MODELS),
        "datasets": list(DISPLAY_DATASETS),
        "horizons": list(HORIZONS),
        "display_precision": 3,
        "ranking_rule": "best_and_second_distinct_displayed_values_after_three_decimal_rounding",
        "average_rule": "arithmetic_mean_of_four_horizon_rows",
        "row_count_without_averages": len(DISPLAY_MODELS)
        * len(DISPLAY_DATASETS)
        * len(HORIZONS),
        "row_count_with_averages": len(rows),
        "iscf_best_cells_full_14_model_table": sum(
            row[f"{metric}_style"] == "best"
            for row in standard_rows
            for metric in METRICS
        ),
        "iscf_second_cells_full_14_model_table": sum(
            row[f"{metric}_style"] == "second"
            for row in standard_rows
            for metric in METRICS
        ),
        "frozen_33_of_56_scope": (
            "ISCF-BSCA versus the five-model published comparator subset; "
            "not recomputed from all 14 displayed models"
        ),
        "timealign_reproduced_cells_in_dense_table": 28,
        "timealign_reproduced_cells_exchange_companion": 4,
        "timealign_reproduced_cells_total": 32,
        "timealign_published_cells": 0,
        "qdf_reproduced_cells_in_dense_table": 28 if qdf_full_reproduction else 4,
        "qdf_reproduced_cells_exchange_companion": 4 if qdf_full_reproduction else 0,
        "qdf_published_cells": 0 if qdf_full_reproduction else 24,
        "amd_reproduced_cells_in_dense_table": 28,
        "simpletm_reproduced_cells_in_dense_table": 28,
        "replaced_models": ["CMoS", "TimeBase"],
        "source_hashes": {
            "timealign_pdf": file_sha256(args.timealign_pdf),
            "iscf_scorecard": file_sha256(args.iscf_scorecard),
            "timealign_reproduced": file_sha256(args.timealign_reproduced),
            "audited_published_selected": file_sha256(
                args.audited_published_selected
            ),
            **qdf_hashes,
            "amd_simpletm_reproduced": file_sha256(
                args.amd_simpletm_reproduced
            ),
        },
        "claim_boundary": (
            "ISCF-BSCA is single-seed and test-tuned; TimeAlign is local single-seed "
            "reproduction on every shared dataset; QDF is a local single-seed "
            + ("L=336 reproduction with disclosed Solar/Exchange source-informed presets; " if qdf_full_reproduction else "mixture of published and source-informed Solar values; ")
            + "AMD and SimpleTM are local official-native fixed-H reproductions; "
            + "other baselines are unmatched "
            "published context"
        ),
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)

    rows = extract_timealign_table6(args.timealign_pdf)
    validate_against_audited_selected(rows, args.audited_published_selected)
    override_reproduced_timealign(rows, args.timealign_reproduced)
    if args.qdf_reproduced_all is None:
        rows.extend(load_qdf_rows(args.qdf_published, args.qdf_solar_reproduced))
    else:
        rows.extend(load_qdf_reproduced_rows(args.qdf_reproduced_all))
    rows.extend(load_amd_simpletm_rows(args.amd_simpletm_reproduced))
    rows.extend(load_iscf_rows(args.iscf_scorecard))
    validate_matrix(rows)
    styled_rows = add_averages_and_styles(rows)
    exchange_rows = load_exchange_companion(
        args.iscf_scorecard, args.timealign_reproduced, args.qdf_reproduced_all
    )

    write_long_csv(args.analysis_dir / "table_data_long.csv", styled_rows)
    write_long_csv(
        args.analysis_dir / "table_exchange_companion_long.csv",
        exchange_rows,
    )
    build_latex(args.analysis_dir / "table_iscf_bsca_main_i_qdf.tex", styled_rows)
    build_exchange_latex(
        args.analysis_dir / "table_exchange_companion.tex",
        exchange_rows,
    )
    build_pdf(args.output_pdf, styled_rows, exchange_rows)
    write_summary(args.analysis_dir / "table_build_summary.json", styled_rows, args)
    print(
        json.dumps(
            {
                "matrix_rows": len(rows),
                "rows_with_averages": len(styled_rows),
                "models": len(DISPLAY_MODELS),
                "datasets": len(DISPLAY_DATASETS),
                "exchange_companion_rows_with_average": len(exchange_rows),
                "output_pdf": str(args.output_pdf),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

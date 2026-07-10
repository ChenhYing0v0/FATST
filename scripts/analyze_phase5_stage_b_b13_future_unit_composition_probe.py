from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEALIGN_ROOT = REPO_ROOT / "baselines" / "timealign_official"
if str(TIMEALIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMEALIGN_ROOT))

from data_provider.data_factory import data_provider  # noqa: E402
from models.TimeAlign import Model  # noqa: E402
from train_repo import OFFICIAL_PRESETS, build_official_args  # noqa: E402


DATASETS = ("ETTh2", "ETTm1", "Weather")
ARMS = ("parallel_no_transition", "prefix_causal_composed")
HORIZONS = (96, 192, 336, 720)
PRED_LEN = 720
DEFAULT_UNIT_SIZES = (180, 240)
DEFAULT_SEEDS = (2021, 2022, 2023)
DEFAULT_ANALYSIS_ROOT = Path(
    "analysis/phase5_stage_b_b13_future_unit_composition_20260710"
)
DEFAULT_CHECKPOINT_ROOT = (
    Path("analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707")
    / "raw"
    / "TimeAlignOfficialUnified720_A6LBF_r256_main_official-last"
)
DEFAULT_DATASET_ROOT = Path("/Users/river/PaperResearch/Project/datasets")


@dataclass(frozen=True)
class ProbeData:
    memory: torch.Tensor
    target: torch.Tensor


class FutureUnitProbe(nn.Module):
    def __init__(
        self,
        memory_dim: int,
        state_dim: int,
        unit_size: int,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in ARMS:
            raise ValueError(f"Unsupported probe mode: {mode}")
        if PRED_LEN % unit_size != 0:
            raise ValueError("unit_size must divide pred_len")
        self.mode = mode
        self.unit_size = unit_size
        self.unit_count = PRED_LEN // unit_size
        self.input_project = nn.Linear(memory_dim, state_dim)
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(1, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )
        self.transition = nn.GRUCell(state_dim, state_dim)
        self.shared_decoder = nn.Linear(state_dim, unit_size)
        centers = (
            (torch.arange(self.unit_count, dtype=torch.float32) + 0.5)
            * float(unit_size)
            / float(PRED_LEN)
        )
        self.register_buffer("unit_coordinates", centers.view(-1, 1), persistent=False)

    def forward(self, memory: torch.Tensor, max_units: int | None = None) -> torch.Tensor:
        unit_limit = self.unit_count if max_units is None else int(max_units)
        if unit_limit <= 0 or unit_limit > self.unit_count:
            raise ValueError("max_units must be in [1, unit_count]")
        base = torch.tanh(self.input_project(memory))
        previous = base
        outputs: list[torch.Tensor] = []
        for unit_idx in range(unit_limit):
            coordinate = self.coordinate_mlp(self.unit_coordinates[unit_idx : unit_idx + 1])
            transition_input = base + coordinate.expand(base.shape[0], -1)
            transition_state = previous if self.mode == "prefix_causal_composed" else base
            state = self.transition(transition_input, transition_state)
            outputs.append(self.shared_decoder(state))
            if self.mode == "prefix_causal_composed":
                previous = state
        return torch.cat(outputs, dim=-1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw_row.items():
                if value == "True":
                    row[key] = True
                elif value == "False":
                    row[key] = False
                else:
                    try:
                        row[key] = int(value)
                    except ValueError:
                        try:
                            row[key] = float(value)
                        except ValueError:
                            row[key] = value
            rows.append(row)
    return rows


def checkpoint_path(root: Path, dataset: str) -> Path:
    return root / dataset / "mixed_h96_h192_h336_h720" / "seed2021" / "checkpoint.pt"


def build_args(args: argparse.Namespace, dataset: str) -> argparse.Namespace:
    preset = OFFICIAL_PRESETS[dataset][PRED_LEN]
    adapter_args = SimpleNamespace(
        dataset_root=args.dataset_root,
        output_dir=args.analysis_root / "_tmp_official_args",
        dataset=dataset,
        seq_len=PRED_LEN,
        label_len=48,
        pred_len=PRED_LEN,
        e_layers=2,
        num_workers=0,
        epochs=10,
        batch_size=args.extract_batch_size,
        patience=3,
        use_amp=False,
        seed=args.data_seed,
        device=args.device,
        readout_mode="learned-basis-forecast-operator",
        w_align=None,
        w_recon=0.0,
        target_horizons=list(HORIZONS),
        basis_rank=256,
        stage_token_dim=32,
        stage_field_rank=32,
        stage_gate_init=-5.0,
        basis_field_window_len=96,
        basis_field_stride=48,
        basis_field_rank=32,
        basis_field_tau=1.0,
        basis_field_gate_init=-5.0,
        stbo_tile_len=48,
        stbo_rank=16,
        stbo_bank_count=4,
        stbo_basis_init_std=16**-0.5,
    )
    official_args = build_official_args(adapter_args, preset)
    official_args.batch_size = args.extract_batch_size
    official_args.num_workers = 0
    return official_args


def load_a6_model(official_args: argparse.Namespace, checkpoint: Path) -> Model:
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = Model(official_args)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def extract_a6_memory(
    model: Model,
    batch_x: torch.Tensor,
    memory_source: str,
) -> torch.Tensor:
    batch, seq_len, channels = batch_x.shape
    x = model.normalization_x(batch_x, "norm")
    x = model.patch_emb_x(x.permute(0, 2, 1).reshape(-1, channels * seq_len))
    for layer_idx in range(model.e_layers):
        x = x + model.encoder[layer_idx](x)
        if model.layer_norm:
            x = model.norm_x[layer_idx](x)
    hidden = x.reshape(batch, channels, model.patch_num, model.d_model).flatten(start_dim=-2)
    if memory_source == "hidden":
        return hidden
    if memory_source == "coeff":
        return model.learned_basis_coeff(hidden)
    raise ValueError(f"Unsupported memory source: {memory_source}")


def collect_probe_data(
    args: argparse.Namespace,
    model: Model,
    official_args: argparse.Namespace,
    split: str,
    max_rows: int,
) -> ProbeData:
    _data, loader = data_provider(official_args, split)
    device = torch.device(args.device)
    memory_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    collected = 0
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in loader:
            if collected >= max_rows:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            memory = extract_a6_memory(model, batch_x, args.memory_source)
            target = batch_y[:, -PRED_LEN:, :]
            target_norm = (target - model.normalization_x.mean) / model.normalization_x.stdev
            memory_rows = memory.reshape(-1, memory.shape[-1])
            target_rows = target_norm.permute(0, 2, 1).reshape(-1, PRED_LEN)
            take = min(max_rows - collected, memory_rows.shape[0])
            memory_parts.append(memory_rows[:take].detach().cpu())
            target_parts.append(target_rows[:take].detach().cpu())
            collected += take
    if not memory_parts:
        raise RuntimeError(f"No probe rows collected for split={split}")
    return ProbeData(
        memory=torch.cat(memory_parts, dim=0).float(),
        target=torch.cat(target_parts, dim=0).float(),
    )


def standardize_memory(
    train: ProbeData,
    val: ProbeData,
    test: ProbeData,
) -> tuple[ProbeData, ProbeData, ProbeData]:
    mean = train.memory.mean(dim=0, keepdim=True)
    std = train.memory.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-5)

    def transform(data: ProbeData) -> ProbeData:
        return ProbeData(memory=(data.memory - mean) / std, target=data.target)

    return transform(train), transform(val), transform(test)


def batch_indices(
    row_count: int,
    batch_size: int,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    permutation = torch.randperm(row_count, generator=generator)
    return [permutation[start : start + batch_size] for start in range(0, row_count, batch_size)]


def evaluate_probe(
    model: FutureUnitProbe,
    data: ProbeData,
    batch_size: int,
    device: torch.device,
) -> tuple[float, list[float]]:
    model.eval()
    total_squared = 0.0
    total_values = 0
    unit_squared = np.zeros(model.unit_count, dtype=np.float64)
    unit_values = np.zeros(model.unit_count, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, data.memory.shape[0], batch_size):
            end = min(start + batch_size, data.memory.shape[0])
            memory = data.memory[start:end].to(device)
            target = data.target[start:end].to(device)
            prediction = model(memory)
            error = prediction - target
            total_squared += float(torch.sum(error * error).cpu())
            total_values += error.numel()
            for unit_idx in range(model.unit_count):
                unit_start = unit_idx * model.unit_size
                unit_end = unit_start + model.unit_size
                unit_error = error[:, unit_start:unit_end]
                unit_squared[unit_idx] += float(torch.sum(unit_error * unit_error).cpu())
                unit_values[unit_idx] += unit_error.numel()
    return (
        total_squared / max(total_values, 1),
        [
            float(unit_squared[idx] / max(unit_values[idx], 1))
            for idx in range(model.unit_count)
        ],
    )


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def train_probe(
    args: argparse.Namespace,
    dataset: str,
    unit_size: int,
    arm: str,
    seed: int,
    train: ProbeData,
    val: ProbeData,
    test: ProbeData,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    model = FutureUnitProbe(
        memory_dim=train.memory.shape[1],
        state_dim=args.state_dim,
        unit_size=unit_size,
        mode=arm,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    final_epoch_train = float("nan")

    for epoch in range(args.epochs):
        model.train()
        epoch_squared = 0.0
        epoch_values = 0
        for indices in batch_indices(train.memory.shape[0], args.batch_size, generator):
            memory = train.memory[indices].to(device)
            target = train.target[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(memory)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            epoch_squared += float(torch.sum((prediction.detach() - target) ** 2).cpu())
            epoch_values += target.numel()
        final_epoch_train = epoch_squared / max(epoch_values, 1)
        val_mse, _unit_mse = evaluate_probe(model, val, args.batch_size, device)
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch + 1
            best_state = clone_state_dict(model)

    if best_state is None:
        raise RuntimeError("Probe training did not capture a validation checkpoint")
    model.load_state_dict(best_state)
    train_mse, _train_unit_mse = evaluate_probe(model, train, args.batch_size, device)
    val_mse, _val_unit_mse = evaluate_probe(model, val, args.batch_size, device)
    test_mse, test_unit_mse = evaluate_probe(model, test, args.batch_size, device)
    prefix_units = max(1, model.unit_count - 1)
    prefix_rows = min(16, test.memory.shape[0])
    with torch.no_grad():
        prefix_memory = test.memory[:prefix_rows].to(device)
        direct_prefix = model(prefix_memory, max_units=prefix_units)
        full_prefix = model(prefix_memory)[:, : prefix_units * unit_size]
        prefix_max_abs = float(torch.max(torch.abs(direct_prefix - full_prefix)).cpu())

    row: dict[str, Any] = {
        "dataset": dataset,
        "unit_size": unit_size,
        "unit_count": model.unit_count,
        "arm": arm,
        "seed": seed,
        "train_rows": train.memory.shape[0],
        "val_rows": val.memory.shape[0],
        "test_rows": test.memory.shape[0],
        "state_dim": args.state_dim,
        "memory_source": args.memory_source,
        "memory_dim": train.memory.shape[1],
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "final_epoch_train_mse": final_epoch_train,
        "best_train_mse": train_mse,
        "best_val_mse": val_mse,
        "test_mse": test_mse,
        "test_val_ratio": test_mse / max(val_mse, 1e-12),
        "prefix_max_abs": prefix_max_abs,
    }
    for unit_idx, unit_mse in enumerate(test_unit_mse):
        row[f"test_unit_{unit_idx}_mse"] = unit_mse
    return row


def paired_comparisons(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["dataset"], row["unit_size"], row["seed"], row["arm"]): row
        for row in run_rows
    }
    comparisons: list[dict[str, Any]] = []
    datasets = sorted({str(row["dataset"]) for row in run_rows})
    for dataset in datasets:
        unit_sizes = sorted(
            {
                int(row["unit_size"])
                for row in run_rows
                if row["dataset"] == dataset
            }
        )
        for unit_size in unit_sizes:
            seeds = sorted(
                {
                    int(row["seed"])
                    for row in run_rows
                    if row["dataset"] == dataset and row["unit_size"] == unit_size
                }
            )
            for seed in seeds:
                parallel = lookup[(dataset, unit_size, seed, "parallel_no_transition")]
                composed = lookup[(dataset, unit_size, seed, "prefix_causal_composed")]
                relative = (composed["test_mse"] / parallel["test_mse"] - 1.0) * 100.0
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "unit_size": unit_size,
                    "seed": seed,
                    "parallel_test_mse": parallel["test_mse"],
                    "composed_test_mse": composed["test_mse"],
                    "composed_vs_parallel_mse_pct": relative,
                    "composed_win": composed["test_mse"] < parallel["test_mse"],
                    "parameter_delta": (
                        composed["trainable_parameters"] - parallel["trainable_parameters"]
                    ),
                    "parallel_prefix_max_abs": parallel["prefix_max_abs"],
                    "composed_prefix_max_abs": composed["prefix_max_abs"],
                }
                unit_count = int(parallel["unit_count"])
                for unit_idx in range(unit_count):
                    parallel_unit = parallel[f"test_unit_{unit_idx}_mse"]
                    composed_unit = composed[f"test_unit_{unit_idx}_mse"]
                    row[f"unit_{unit_idx}_relative_mse_pct"] = (
                        composed_unit / parallel_unit - 1.0
                    ) * 100.0
                comparisons.append(row)
    return comparisons


def summarize_comparisons(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    settings = sorted(
        {(str(row["dataset"]), int(row["unit_size"])) for row in comparisons}
    )
    for dataset, unit_size in settings:
        selected = [
            row
            for row in comparisons
            if row["dataset"] == dataset and row["unit_size"] == unit_size
        ]
        relative = np.asarray(
            [float(row["composed_vs_parallel_mse_pct"]) for row in selected],
            dtype=np.float64,
        )
        wins = sum(bool(row["composed_win"]) for row in selected)
        mean_relative = float(np.mean(relative))
        rows.append(
            {
                "dataset": dataset,
                "unit_size": unit_size,
                "seeds": len(selected),
                "composed_wins": wins,
                "mean_composed_vs_parallel_mse_pct": mean_relative,
                "std_composed_vs_parallel_mse_pct": float(np.std(relative)),
                "min_composed_vs_parallel_mse_pct": float(np.min(relative)),
                "max_composed_vs_parallel_mse_pct": float(np.max(relative)),
                "composition_support": wins >= 2 and mean_relative <= -0.5,
            }
        )
    return rows


def gate_decision(
    run_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    nonfinite = [
        row
        for row in run_rows
        if not np.isfinite(float(row["test_mse"]))
        or not np.isfinite(float(row["best_val_mse"]))
    ]
    severe_mismatch = [
        row
        for row in run_rows
        if float(row["test_val_ratio"]) > 3.0
        and float(row["test_mse"]) - float(row["best_val_mse"]) > 1.0
    ]
    if nonfinite or len(severe_mismatch) > max(2, len(run_rows) // 4):
        reasons.append(
            f"nonfinite_runs={len(nonfinite)}, severe_val_test_mismatch={len(severe_mismatch)}"
        )
        return "diagnostic_invalid_for_direction_rejection", reasons

    supported = [row for row in summary_rows if row["composition_support"]]
    dataset_safe = {
        dataset: any(
            row["dataset"] == dataset
            and float(row["mean_composed_vs_parallel_mse_pct"]) <= 0.25
            for row in summary_rows
        )
        for dataset in DATASETS
    }
    reasons.append(f"supported_settings={len(supported)}/{len(summary_rows)}")
    reasons.append(
        "dataset_non_degradation="
        + ",".join(f"{dataset}:{dataset_safe[dataset]}" for dataset in DATASETS)
    )
    if len(supported) >= 4 and all(dataset_safe.values()):
        return "partial_pass_prefix_causal_composition", reasons
    return "no_transition_control_explains", reasons


def fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt_value(row[field]) for field in fields) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    run_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    decision, reasons = gate_decision(run_rows, summary_rows)
    memory_sources = sorted({str(row.get("memory_source", "coeff")) for row in run_rows})
    hidden_only = memory_sources == ["hidden"]
    diagnostic_id = "B13-FUCO-B2" if hidden_only else "B13-FUCO-B1"
    memory_label = "A6 encoder hidden memory" if hidden_only else "A6 coefficient memory"
    mismatches = [row for row in run_rows if float(row["test_val_ratio"]) > 2.0]
    max_prefix = max(float(row["prefix_max_abs"]) for row in run_rows)
    parameter_sets = {
        (str(row["dataset"]), int(row["unit_size"])): {
            int(candidate["trainable_parameters"])
            for candidate in run_rows
            if candidate["dataset"] == row["dataset"]
            and candidate["unit_size"] == row["unit_size"]
        }
        for row in run_rows
    }
    lines = [
        "# Phase5 StageB B13-FUCO-B Prefix-Causal Composition Probe",
        "",
        "## 阶段记录",
        "",
        "| 字段 | 内容 |",
        "| --- | --- |",
        "| `candidate_id` | `B13-FUCO` |",
        f"| `diagnostic_id` | `{diagnostic_id}` |",
        "| `current_step` | Step 2/3：parameter-matched composition control |",
        "| `scope` | frozen A6 memory；trainable diagnostic probes；not end-to-end model performance |",
        f"| `memory_source` | `{','.join(memory_sources)}` |",
        f"| `decision` | `{decision}` |",
        "",
        "## Arms",
        "",
        f"- `parallel_no_transition`: every unit reads the same {memory_label} and continuous coordinate independently;",
        "- `prefix_causal_composed`: the same GRUCell additionally receives the previous latent unit state;",
        "- both arms use identical modules and parameter count; no predicted values are fed back.",
        "",
        "## Summary",
        "",
        markdown_table(
            summary_rows,
            [
                "dataset",
                "unit_size",
                "seeds",
                "composed_wins",
                "mean_composed_vs_parallel_mse_pct",
                "std_composed_vs_parallel_mse_pct",
                "min_composed_vs_parallel_mse_pct",
                "max_composed_vs_parallel_mse_pct",
                "composition_support",
            ],
        ),
        "",
        "## Gate Reading",
        "",
        f"[Decision] `{decision}`.",
        "",
    ]
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            f"[Fact] Maximum prefix-consistency absolute error across runs is `{max_prefix:.6e}`.",
            f"[Fact] Runs with test/validation MSE ratio above `2.0`: `{len(mismatches)}/{len(run_rows)}`.",
            "[Fact] Trainable parameter sets by dataset/unit size: "
            + "; ".join(
                f"{dataset}-U{unit_size}={sorted(values)}"
                for (dataset, unit_size), values in sorted(parameter_sets.items())
            )
            + ".",
            "",
        ]
    )
    if decision == "partial_pass_prefix_causal_composition":
        lines.extend(
            [
                "[Strong Evidence] Prefix-causal latent composition beats the exact no-transition control across the required dataset/granularity settings. B13 may enter Step 4-6 narrative and end-to-end method design, but the probe gain is not an end-to-end paper result.",
                "",
                "[Next] Perform a targeted prior-art audit and design the smallest end-to-end future-unit operator with the same composed-vs-no-transition control.",
            ]
        )
    elif decision == "no_transition_control_explains":
        if hidden_only:
            lines.extend(
                [
                    "[Decision] Moving the intervention before the A6 coefficient bottleneck does not make the GRU-based transition beat its parameter-matched no-transition control. Close the current prefix-causal GRU composition candidate.",
                    "",
                    "[Rollback] Return B13 to Step 2 and distinguish future-region-specific states from other non-recurrent future-stage generation mechanisms. Do not continue GRU/head tuning.",
                ]
            )
        else:
            lines.extend(
                [
                    "[Decision] Large-unit gradient pressure exists, but the parameter-matched no-transition probe explains the predictive value. Do not implement prefix-causal composition as a paper-core candidate.",
                    "",
                    "[Rollback] Repair the intervention point once with pre-coefficient hidden memory before deciding the current GRU-based composition candidate.",
                ]
            )
    else:
        lines.extend(
            [
                "[Failure Attribution] Probe optimization or numeric pathology prevents a direction-level conclusion. Repair the diagnostic before deciding B13.",
            ]
        )
    lines.extend(["", "## Failure Attribution Boundary", ""])
    if hidden_only:
        lines.extend(
            [
                "- `hypothesis_false`: not established for all future-unit architectures; a failed B2 closes only the current GRU-based prefix-causal composition candidate;",
                "- `intervention_point_wrong`: the post-coefficient bottleneck confound is removed, although a frozen hidden-memory probe is still not end-to-end adaptation;",
                "- `readout_or_head_design_wrong`: remains possible in principle, but B2 is the pre-registered final repair and does not authorize head sweeps;",
            ]
        )
    else:
        lines.extend(
            [
                "- `hypothesis_false`: not established by a frozen-coefficient probe alone;",
                "- `intervention_point_wrong`: possible because the A6 coefficient may already discard information needed by unit composition;",
                "- `readout_or_head_design_wrong`: possible if GRUCell/shared decoder is too weak;",
            ]
        )
    lines.extend(
        [
            "- `optimization_or_numeric_pathology`: tracked by non-finite loss and val/test mismatch;",
            "- `capacity_control_explains`: directly tested by exact parameter matching between the two arms.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.extract_batch_size <= 0:
        raise ValueError("epoch and batch sizes must be positive")
    if args.state_dim <= 0:
        raise ValueError("state_dim must be positive")
    for unit_size in args.unit_sizes:
        if unit_size <= 0 or PRED_LEN % unit_size != 0:
            raise ValueError(f"unit_size must divide {PRED_LEN}: {unit_size}")
    for row_limit in (args.max_train_rows, args.max_val_rows, args.max_test_rows):
        if row_limit <= 0:
            raise ValueError("row limits must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B13 parameter-matched future-unit composition probe."
    )
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--unit-sizes", nargs="+", type=int, default=list(DEFAULT_UNIT_SIZES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--memory-source", choices=["coeff", "hidden"], default="coeff")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--max-train-rows", type=int, default=8192)
    parser.add_argument("--max-val-rows", type=int, default=2048)
    parser.add_argument("--max-test-rows", type=int, default=2048)
    parser.add_argument("--data-seed", type=int, default=2021)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    validate_args(args)
    return args


def main() -> None:
    args = parse_args()
    if args.report_only:
        run_rows = read_csv(args.analysis_root / "b13_future_unit_probe_runs.csv")
        summary_rows = read_csv(args.analysis_root / "b13_future_unit_probe_summary.csv")
        write_report(
            args.analysis_root / "b13_future_unit_composition_report.md",
            run_rows,
            summary_rows,
        )
        return
    torch.manual_seed(args.data_seed)
    np.random.seed(args.data_seed)
    all_run_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        official_args = build_args(args, dataset)
        a6_model = load_a6_model(official_args, checkpoint_path(args.checkpoint_root, dataset))
        a6_model.to(torch.device(args.device))
        train = collect_probe_data(
            args,
            a6_model,
            official_args,
            "train",
            args.max_train_rows,
        )
        val = collect_probe_data(args, a6_model, official_args, "val", args.max_val_rows)
        test = collect_probe_data(args, a6_model, official_args, "test", args.max_test_rows)
        train, val, test = standardize_memory(train, val, test)
        del a6_model

        for unit_size in args.unit_sizes:
            for seed in args.seeds:
                for arm in ARMS:
                    all_run_rows.append(
                        train_probe(
                            args,
                            dataset,
                            unit_size,
                            arm,
                            seed,
                            train,
                            val,
                            test,
                        )
                    )

    comparison_rows = paired_comparisons(all_run_rows)
    summary_rows = summarize_comparisons(comparison_rows)
    args.analysis_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.analysis_root / "b13_future_unit_probe_runs.csv", all_run_rows)
    write_csv(
        args.analysis_root / "b13_future_unit_probe_comparisons.csv",
        comparison_rows,
    )
    write_csv(args.analysis_root / "b13_future_unit_probe_summary.csv", summary_rows)
    write_report(
        args.analysis_root / "b13_future_unit_composition_report.md",
        all_run_rows,
        summary_rows,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one dataset unit from the official AMD or SimpleTM release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any


HORIZONS = (96, 192, 336, 720)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch: expected={expected} actual={actual} path={path}"
        )


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return text.replace(old, new, 1)


def patch_simpletm(source: Path) -> dict[str, str]:
    exp_path = source / "experiments/exp_long_term_forecasting.py"
    run_path = source / "run.py"

    exp_text = exp_path.read_text(encoding="utf-8")
    exp_text = replace_once(
        exp_text,
        (
            "        train_data, train_loader = self._get_data(flag='train')\n"
            "        vali_data, vali_loader = self._get_data(flag='val')\n"
            "        test_data, test_loader = self._get_data(flag='test')\n"
        ),
        (
            "        train_data, train_loader = self._get_data(flag='train')\n"
            "        vali_data, vali_loader = self._get_data(flag='val')\n"
        ),
        "SimpleTM remove epoch-level test loader",
    )
    exp_text = replace_once(
        exp_text,
        (
            "            test_loss = self.vali(test_data, test_loader, criterion)\n\n"
            "            print(\"Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} "
            "Vali Loss: {3:.7f} Test Loss: {4:.7f}\".format(\n"
            "                epoch + 1, train_steps, train_loss, vali_loss, "
            "test_loss))\n"
        ),
        (
            "            print(\"Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} "
            "Vali Loss: {3:.7f}\".format(\n"
            "                epoch + 1, train_steps, train_loss, vali_loss))\n"
        ),
        "SimpleTM remove epoch-level test evaluation",
    )
    exp_path.write_text(exp_text, encoding="utf-8")

    run_text = run_path.read_text(encoding="utf-8")
    run_text = replace_once(
        run_text,
        "import argparse\n",
        "import argparse\nimport os\n",
        "SimpleTM add smoke-only environment gate import",
    )
    run_text = replace_once(
        run_text,
        (
            "            print('>>>>>>>testing : "
            "{}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))\n"
            "            exp.test(setting)\n"
        ),
        (
            "            if os.environ.get('FATST_SKIP_FINAL_TEST') == '1':\n"
            "                print('>>>>>>>formal test skipped by resource-smoke "
            "gate<<<<<<<<')\n"
            "            else:\n"
            "                print('>>>>>>>testing : "
            "{}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))\n"
            "                exp.test(setting)\n"
        ),
        "SimpleTM add smoke-only final-test gate",
    )
    run_path.write_text(run_text, encoding="utf-8")
    return {
        "patched_exp_sha256": sha256_file(exp_path),
        "patched_run_sha256": sha256_file(run_path),
    }


def patch_amd_smoke(source: Path, script_rel: str) -> dict[str, str]:
    main_path = source / "main.py"
    main_text = main_path.read_text(encoding="utf-8")
    main_text = replace_once(
        main_text,
        "    # start testing\n    model.eval()\n",
        (
            "    if os.environ.get(\"FATST_SKIP_FINAL_TEST\") == \"1\":\n"
            "        print(\"Final Test skipped by resource-smoke gate\")\n"
            "        return\n\n"
            "    # start testing\n"
            "    model.eval()\n"
        ),
        "AMD add smoke-only final-test gate",
    )
    main_path.write_text(main_text, encoding="utf-8")

    script_path = source / script_rel
    script_text = script_path.read_text(encoding="utf-8")
    script_text = replace_once(
        script_text,
        "for pred_len in 96 192 336 720",
        "for pred_len in 720",
        "AMD restrict smoke horizon",
    )
    script_text, replacements = re.subn(
        r"--train_epochs\s+(?:10|20)", "--train_epochs 1", script_text
    )
    if replacements != 1:
        raise RuntimeError(
            f"AMD smoke epoch patch expected one match, found {replacements}"
        )
    script_path.write_text(script_text, encoding="utf-8")
    return {
        "patched_main_sha256": sha256_file(main_path),
        "patched_script_sha256": sha256_file(script_path),
    }


def create_data_links(
    workspace: Path, data_root: Path, relative_path: str, baseline: str, dataset: str
) -> None:
    source_data = data_root / relative_path
    if not source_data.is_file():
        raise FileNotFoundError(source_data)

    if baseline == "AMD":
        target_names = {
            "ETTh1": "ETTh1.csv",
            "ETTh2": "ETTh2.csv",
            "ETTm1": "ETTm1.csv",
            "ETTm2": "ETTm2.csv",
            "Weather": "weather.csv",
            "ECL": "electricity.csv",
            "Solar": "solar_AL.txt",
        }
        link = workspace / "data" / target_names[dataset]
    else:
        target_names = {
            "ETTh1": "ETT-small/ETTh1.csv",
            "ETTh2": "ETT-small/ETTh2.csv",
            "ETTm1": "ETT-small/ETTm1.csv",
            "ETTm2": "ETT-small/ETTm2.csv",
            "Weather": "weather/weather.csv",
            "ECL": "electricity/electricity.csv",
            "Solar": "solar/solar_AL.txt",
        }
        link = workspace / "source" / "dataset" / target_names[dataset]
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(source_data)


def run_command(
    command: list[str], cwd: Path, env: dict[str, str], log_path: Path
) -> str:
    rendered = shlex.join(command)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"COMMAND {rendered}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            output.append(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {rendered}")
    return "".join(output)


def parse_simpletm_commands(script_path: Path) -> list[list[str]]:
    text = script_path.read_text(encoding="utf-8").replace(
        "$model_name", "SimpleTM"
    )
    text = re.sub(r"\\[ \t]*\r?\n", " ", text)
    commands: list[list[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().rstrip("\\").rstrip()
        if line.startswith("python -u run.py"):
            commands.append(shlex.split(line))
    if len(commands) != 4:
        raise RuntimeError(
            f"expected four SimpleTM horizon commands, found {len(commands)}"
        )
    horizons = tuple(int(value_after(command, "--pred_len")) for command in commands)
    if horizons != HORIZONS:
        raise RuntimeError(f"unexpected SimpleTM horizon order: {horizons}")
    return commands


def value_after(command: list[str], key: str) -> str:
    return command[command.index(key) + 1]


def append_override(command: list[str], key: str, value: str) -> None:
    command.extend([key, value])


def checkpoint_rows_simpletm(
    source: Path,
    dataset: str,
    horizon: int,
    repeat_count: int,
    metrics_text: str,
) -> list[dict[str, Any]]:
    pairs = re.findall(
        r"mse:([0-9eE+\-.]+), mae:([0-9eE+\-.]+)", metrics_text
    )
    if len(pairs) != repeat_count:
        raise RuntimeError(
            f"SimpleTM {dataset} H{horizon}: expected {repeat_count} metrics, "
            f"found {len(pairs)}"
        )
    candidates = []
    for checkpoint in (source / "checkpoints").glob("*/checkpoint.pth"):
        name = checkpoint.parent.name
        tokens = name.split("_")
        if len(tokens) >= 4 and tokens[0] == dataset and tokens[3] == str(horizon):
            candidates.append(checkpoint)
    candidates.sort(key=lambda path: int(path.parent.name.rsplit("_", 1)[-1]))
    if len(candidates) != repeat_count:
        raise RuntimeError(
            f"SimpleTM {dataset} H{horizon}: expected {repeat_count} checkpoints, "
            f"found {len(candidates)}"
        )
    rows = []
    for repeat, ((mse, mae), checkpoint) in enumerate(zip(pairs, candidates)):
        rows.append(
            {
                "baseline": "SimpleTM",
                "dataset": dataset,
                "horizon": horizon,
                "repeat": repeat,
                "seed_contract": "fix_seed_2025_advancing_rng_across_itr",
                "mse": float(mse),
                "mae": float(mae),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "test_role": "official_test_once_after_validation_selected_checkpoint",
            }
        )
    return rows


def run_simpletm(
    config: dict[str, Any], args: argparse.Namespace, source: Path, log_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = config["baselines"]["SimpleTM"]
    patch_hashes = patch_simpletm(source)
    if patch_hashes != spec["expected_runtime_patch_hashes"]:
        raise RuntimeError(
            "SimpleTM runtime patch hash mismatch: "
            f"expected={spec['expected_runtime_patch_hashes']} actual={patch_hashes}"
        )
    script_rel = spec["dataset_scripts"][args.dataset]["path"]
    commands = parse_simpletm_commands(source / script_rel)
    if args.mode == "resource-smoke":
        commands = [command for command in commands if value_after(command, "--pred_len") == "720"]
    rows: list[dict[str, Any]] = []
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONHASHSEED"] = "2025"
    env["FATST_SKIP_FINAL_TEST"] = "1" if args.mode == "resource-smoke" else "0"
    for command in commands:
        horizon = int(value_after(command, "--pred_len"))
        repeat_count = int(value_after(command, "--itr"))
        append_override(command, "--num_workers", "0")
        if args.mode == "resource-smoke":
            append_override(command, "--train_epochs", "1")
            append_override(command, "--itr", "1")
            repeat_count = 1
        before = ""
        result_path = source / "result_long_term_forecast.txt"
        if result_path.exists():
            before = result_path.read_text(encoding="utf-8")
        run_command(command, source, env, log_path)
        after = result_path.read_text(encoding="utf-8") if result_path.exists() else ""
        delta = after[len(before) :]
        if args.mode == "run":
            rows.extend(
                checkpoint_rows_simpletm(
                    source, args.dataset, horizon, repeat_count, delta
                )
            )
        elif re.search(r"mse:[0-9]", delta):
            raise RuntimeError("SimpleTM resource smoke unexpectedly accessed formal test")
    return rows, patch_hashes


def run_amd(
    config: dict[str, Any], args: argparse.Namespace, source: Path, log_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = config["baselines"]["AMD"]
    script_rel = spec["dataset_scripts"][args.dataset]["path"]
    patch_hashes: dict[str, Any] = {}
    if args.mode == "resource-smoke":
        patch_hashes = patch_amd_smoke(source, script_rel)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONHASHSEED"] = "2024"
    env["FATST_SKIP_FINAL_TEST"] = "1" if args.mode == "resource-smoke" else "0"
    output = run_command(["bash", script_rel], source, env, log_path)
    if args.mode == "resource-smoke":
        if "test MSE:" in output:
            raise RuntimeError("AMD resource smoke unexpectedly accessed formal test")
        checkpoints = list((source / "checkpoints").glob("*/best.pt"))
        if len(checkpoints) != 1:
            raise RuntimeError(f"AMD smoke expected one checkpoint, found {len(checkpoints)}")
        return [], patch_hashes

    pairs = re.findall(
        r"test MSE: ([0-9eE+\-.]+), test MAE: ([0-9eE+\-.]+)", output
    )
    if len(pairs) != 4:
        raise RuntimeError(f"AMD expected four formal metrics, found {len(pairs)}")
    base_name = spec["dataset_scripts"][args.dataset]["checkpoint_name"]
    checkpoint_dirs = [base_name, f"{base_name}2", f"{base_name}3", f"{base_name}4"]
    rows = []
    for horizon, (mse, mae), checkpoint_dir in zip(
        HORIZONS, pairs, checkpoint_dirs
    ):
        checkpoint = source / "checkpoints" / checkpoint_dir / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        rows.append(
            {
                "baseline": "AMD",
                "dataset": args.dataset,
                "horizon": horizon,
                "repeat": 0,
                "seed_contract": "official_seed_2024",
                "mse": float(mse),
                "mae": float(mae),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "test_role": "official_test_once_after_official_last_checkpoint",
            }
        )
    return rows, patch_hashes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", choices=("AMD", "SimpleTM"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mode", choices=("resource-smoke", "run"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.dataset not in config["evaluation_contract"]["datasets"]:
        raise ValueError(f"unsupported dataset: {args.dataset}")
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing dataset unit: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)
    log_path = args.output_dir / "run.log"

    baseline_spec = config["baselines"][args.baseline]
    for relative_path, expected in baseline_spec["source_hashes"].items():
        verify_hash(args.source_root / relative_path, expected, relative_path)
    dataset_spec = config["datasets"][args.dataset]
    verify_hash(
        args.data_root / dataset_spec["relative_path"],
        dataset_spec["sha256"],
        f"{args.dataset} dataset",
    )

    workspace = args.output_dir / "workspace"
    source = workspace / "source"
    shutil.copytree(args.source_root, source, ignore=shutil.ignore_patterns(".git"))
    create_data_links(
        workspace,
        args.data_root,
        dataset_spec["relative_path"],
        args.baseline,
        args.dataset,
    )

    if args.baseline == "AMD":
        rows, patch_hashes = run_amd(config, args, source, log_path)
    else:
        rows, patch_hashes = run_simpletm(config, args, source, log_path)

    failure_pattern = re.compile(
        r"Traceback|CUDA out of memory|too many open files|(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)",
        re.IGNORECASE | re.MULTILINE,
    )
    log_text = log_path.read_text(encoding="utf-8")
    match = failure_pattern.search(log_text)
    if match:
        raise RuntimeError(f"numeric/runtime failure token found: {match.group(0)}")

    write_csv(args.output_dir / "metrics.csv", rows)
    checkpoint_files = sorted(source.glob("checkpoints/**/checkpoint.pth"))
    checkpoint_files.extend(sorted(source.glob("checkpoints/**/best.pt")))
    expected_checkpoints = 1 if args.mode == "resource-smoke" else len(rows)
    if len(checkpoint_files) != expected_checkpoints:
        raise RuntimeError(
            f"checkpoint count mismatch: expected={expected_checkpoints} "
            f"actual={len(checkpoint_files)}"
        )
    completion = {
        "schema_version": 1,
        "baseline": args.baseline,
        "dataset": args.dataset,
        "mode": args.mode,
        "gpu": args.gpu,
        "source_commit": baseline_spec["source_commit"],
        "source_repository": baseline_spec["source_repository"],
        "source_license": baseline_spec["license_status"],
        "official_script": baseline_spec["dataset_scripts"][args.dataset]["path"],
        "official_script_sha256": baseline_spec["dataset_scripts"][args.dataset]["sha256"],
        "runtime_patch_hashes": patch_hashes,
        "formal_metric_rows": len(rows),
        "checkpoint_count": len(checkpoint_files),
        "test_access_count": len(rows),
        "test_access_boundary": (
            "none_resource_smoke"
            if args.mode == "resource-smoke"
            else "once_per_validation_selected_or_official_last_checkpoint"
        ),
        "complete": True,
    }
    with (args.output_dir / "complete.json").open("w", encoding="utf-8") as handle:
        json.dump(completion, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"dataset_unit_complete baseline={args.baseline} dataset={args.dataset} "
        f"mode={args.mode} metrics={len(rows)} checkpoints={len(checkpoint_files)}"
    )


if __name__ == "__main__":
    main()

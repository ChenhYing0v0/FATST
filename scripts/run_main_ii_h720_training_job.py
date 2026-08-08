#!/usr/bin/env python3
"""Prepare and run source-native H720 jobs for Main II.

The adapter changes only test hygiene, bounded-smoke controls, prediction export,
and the preregistered PatchTST/DLinear Solar loader. Official H720 optimization
arguments are extracted by executing the released shell script with a no-op
``python`` capture shim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Solar")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def git_head(source_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()


def verify_source(
    source_root: Path, baseline: str, spec: dict[str, Any]
) -> dict[str, str]:
    head = git_head(source_root)
    if head != spec["commit"]:
        raise RuntimeError(f"{baseline} commit mismatch: {head} != {spec['commit']}")
    code_root = source_root / spec["source_subdir"]
    observed: dict[str, str] = {}
    for relative, expected in spec["key_source_sha256"].items():
        path = code_root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{baseline} source hash mismatch: {relative}")
        observed[relative] = actual
    for dataset, relative in spec["scripts"].items():
        actual = sha256_file(code_root / relative)
        if actual != spec["script_sha256"][dataset]:
            raise RuntimeError(f"{baseline} script hash mismatch: {dataset}")
        observed[relative] = actual
    return observed


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"runtime patch mismatch for {label}: count={text.count(old)}")
    return text.replace(old, new, 1)


def replace_first(
    text: str, old: str, new: str, label: str, expected_count: int
) -> str:
    if text.count(old) != expected_count:
        raise RuntimeError(
            f"runtime patch mismatch for {label}: count={text.count(old)}"
        )
    return text.replace(old, new, 1)


def patch_eval_loops(text: str) -> str:
    train_loop = "            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):\n"
    train_replacement = train_loop + (
        "                fatst_limit = int(os.environ.get('FATST_MAX_TRAIN_BATCHES', '0'))\n"
        "                if fatst_limit and i >= fatst_limit:\n"
        "                    break\n"
    )
    text = replace_once(text, train_loop, train_replacement, "bounded train loop")
    vali_loop = "            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):\n"
    vali_replacement = vali_loop + (
        "                fatst_limit = int(os.environ.get('FATST_MAX_EVAL_BATCHES', '0'))\n"
        "                if fatst_limit and i >= fatst_limit:\n"
        "                    break\n"
    )
    return replace_once(text, vali_loop, vali_replacement, "bounded validation loop")


def patch_exp_file(path: Path, kind: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    text = replace_first(
        text,
        "        test_data, test_loader = self._get_data(flag='test')\n",
        "        # FATST Main II: no test loader is constructed during training.\n",
        f"{kind} epoch test loader",
        2,
    )
    if kind in {"itransformer", "patchtst"}:
        old = (
            "            test_loss = self.vali(test_data, test_loader, criterion)\n\n"
            "            print(\"Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}\".format(\n"
            "                epoch + 1, train_steps, train_loss, vali_loss, test_loss))\n"
        )
        new = (
            "            print(\"Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}\".format(\n"
            "                epoch + 1, train_steps, train_loss, vali_loss))\n"
        )
    else:
        old = (
            "                test_loss = self.vali(test_data, test_loader, criterion)\n\n"
            "                print(\"Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}\".format(\n"
            "                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))\n"
        )
        new = (
            "                print(\"Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}\".format(\n"
            "                    epoch + 1, train_steps, train_loss, vali_loss))\n"
        )
    text = replace_once(text, old, new, f"{kind} epoch test metric")
    text = patch_eval_loops(text)
    changes = ["remove_training_test_loader", "remove_epoch_test_metric", "bounded_smoke_loops"]
    text = replace_once(
        text,
        "os.path.join('./checkpoints/' + setting, 'checkpoint.pth')",
        "os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')",
        f"{kind} formal checkpoint root",
    )
    changes.append("formal_test_uses_args_checkpoints")
    if kind in {"patchtst", "dlinear"}:
        text = replace_once(
            text,
            "        # np.save(folder_path + 'true.npy', trues)\n",
            "        np.save(folder_path + 'true.npy', trues)\n",
            f"{kind} target export",
        )
        changes.append("enable_true_tensor_export")
    path.write_text(text, encoding="utf-8")
    return changes


def patch_entrypoint(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "            exp.test(setting)\n",
        "            # FATST Main II: formal test is a separate hash-audited phase.\n",
        "automatic final test",
    )
    path.write_text(text, encoding="utf-8")


def solar_class_from_itransformer(itransformer_root: Path) -> str:
    source = (
        itransformer_root / "data_provider" / "data_loader.py"
    ).read_text(encoding="utf-8")
    start = source.index("class Dataset_Solar(Dataset):")
    end = source.index("class Dataset_Pred(Dataset):", start)
    return source[start:end]


def patch_solar_loader(workspace: Path, itransformer_root: Path) -> list[str]:
    loader_path = workspace / "data_provider" / "data_loader.py"
    factory_path = workspace / "data_provider" / "data_factory.py"
    loader = loader_path.read_text(encoding="utf-8")
    solar_class = solar_class_from_itransformer(itransformer_root)
    loader = replace_once(
        loader,
        "class Dataset_Pred(Dataset):",
        solar_class + "class Dataset_Pred(Dataset):",
        "Solar loader insertion",
    )
    loader_path.write_text(loader, encoding="utf-8")

    factory = factory_path.read_text(encoding="utf-8")
    factory = replace_once(
        factory,
        "Dataset_Custom, Dataset_Pred",
        "Dataset_Custom, Dataset_Pred, Dataset_Solar",
        "Solar loader import",
    )
    factory = replace_once(
        factory,
        "    'custom': Dataset_Custom,\n",
        "    'custom': Dataset_Custom,\n    'Solar': Dataset_Solar,\n",
        "Solar data dictionary",
    )
    factory_path.write_text(factory, encoding="utf-8")
    return ["iTransformer_Dataset_Solar_copy", "Solar_data_factory_registration"]


def prepare_workspace(args: argparse.Namespace, config: dict[str, Any]) -> None:
    spec = config["training_baselines"][args.baseline]
    source_hashes = verify_source(args.source_root, args.baseline, spec)
    if args.workspace.exists():
        raise FileExistsError(args.workspace)
    shutil.copytree(
        args.source_root / spec["source_subdir"],
        args.workspace,
        ignore=shutil.ignore_patterns(".git", "checkpoints", "results", "test_results", "logs"),
    )
    kind = spec["runner_kind"]
    exp_relative = (
        "experiments/exp_long_term_forecasting.py"
        if kind == "itransformer"
        else "exp/exp_main.py"
    )
    changes = patch_exp_file(args.workspace / exp_relative, kind)
    patch_entrypoint(args.workspace / spec["entrypoint"])
    changes.append("disable_automatic_final_test")
    tools_path = args.workspace / "utils" / "tools.py"
    tools_text = tools_path.read_text(encoding="utf-8")
    tools_text = replace_once(
        tools_text, "np.Inf", "np.inf", f"{kind} NumPy 2 compatibility"
    )
    tools_path.write_text(tools_text, encoding="utf-8")
    changes.append("numpy2_np_Inf_to_np_inf")
    if kind in {"patchtst", "dlinear"}:
        if args.itransformer_source_root is None:
            raise ValueError("--itransformer-source-root is required for Solar patch")
        changes.extend(
            patch_solar_loader(args.workspace, args.itransformer_source_root)
        )
    patched_hashes = {
        relative: sha256_file(args.workspace / relative)
        for relative in [
            spec["entrypoint"],
            exp_relative,
            "data_provider/data_loader.py",
            "data_provider/data_factory.py",
            "utils/tools.py",
        ]
    }
    manifest = {
        "baseline": args.baseline,
        "source_root": str(args.source_root),
        "source_commit": spec["commit"],
        "source_hashes": source_hashes,
        "runtime_changes": changes,
        "patched_hashes": patched_hashes,
        "training_test_access": "disabled",
        "formal_test_path": "is_training=0_only_after_checkpoint_freeze",
    }
    (args.workspace / "fatst_runtime_patch_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"main_ii_prepare=pass baseline={args.baseline} workspace={args.workspace}")


def extract_h720_command(workspace: Path, script_relative: str) -> list[str]:
    script = workspace / script_relative
    with tempfile.TemporaryDirectory(prefix="fatst-main-ii-command-") as temp_dir:
        temp = Path(temp_dir)
        capture = temp / "commands.jsonl"
        wrapper = temp / "python"
        wrapper.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['FATST_COMMAND_CAPTURE'], 'a', encoding='utf-8') as f:\n"
            "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n",
            encoding="utf-8",
        )
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
        env = dict(os.environ)
        env["PATH"] = str(temp) + os.pathsep + env["PATH"]
        env["FATST_COMMAND_CAPTURE"] = str(capture)
        subprocess.run(["bash", str(script)], cwd=workspace, env=env, check=True)
        commands = [json.loads(line) for line in capture.read_text().splitlines()]
    selected: list[list[str]] = []
    for command in commands:
        if command and command[0] == "-u":
            command = command[1:]
        if "--pred_len" in command:
            pred_len = command[command.index("--pred_len") + 1]
            if pred_len == "720":
                selected.append(command)
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one H720 command in {script_relative}, found {len(selected)}"
        )
    return selected[0]


def set_option(command: list[str], name: str, value: str) -> None:
    if name in command:
        command[command.index(name) + 1] = value
    else:
        command.extend([name, value])


def effective_command(
    config: dict[str, Any], baseline: str, dataset: str, workspace: Path,
    data_root: Path, checkpoint_dir: Path, mode: str,
) -> list[str]:
    spec = config["training_baselines"][baseline]
    command = extract_h720_command(workspace, spec["scripts"][dataset])
    dataset_spec = config["datasets"][dataset]
    data_path = data_root / dataset_spec["relative_path"]
    if mode != "dry-run":
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        if sha256_file(data_path) != dataset_spec["sha256"]:
            raise RuntimeError(f"dataset hash mismatch: {dataset}")
    set_option(command, "--root_path", str(data_path.parent) + os.sep)
    set_option(command, "--data_path", data_path.name)
    set_option(command, "--num_workers", "0")
    set_option(command, "--checkpoints", str(checkpoint_dir / "checkpoints"))
    set_option(command, "--is_training", "0" if mode == "formal-test" else "1")
    if dataset == "Solar" and spec["runner_kind"] in {"patchtst", "dlinear"}:
        set_option(command, "--data", "Solar")
        set_option(command, "--enc_in", "137")
        set_option(command, "--model_id", "Solar_336_720")
    if mode == "resource-smoke":
        set_option(command, "--train_epochs", "1")
    return command


def find_single_checkpoint(output_dir: Path) -> Path:
    candidates = sorted((output_dir / "checkpoints").glob("**/checkpoint.pth"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one checkpoint, found {len(candidates)}")
    return candidates[0]


def run_job(args: argparse.Namespace, config: dict[str, Any]) -> None:
    spec = config["training_baselines"][args.baseline]
    patch_manifest = args.workspace / "fatst_runtime_patch_manifest.json"
    if not patch_manifest.is_file():
        raise FileNotFoundError(patch_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = (
        args.checkpoint_dir if args.mode == "formal-test" else args.output_dir
    )
    if args.mode == "formal-test" and checkpoint_dir is None:
        raise ValueError("formal-test requires --checkpoint-dir")
    command = effective_command(
        config, args.baseline, args.dataset, args.workspace, args.data_root,
        checkpoint_dir, args.mode,
    )
    record: dict[str, Any] = {
        "baseline": args.baseline,
        "dataset": args.dataset,
        "mode": args.mode,
        "source_commit": spec["commit"],
        "script": spec["scripts"][args.dataset],
        "script_sha256": spec["script_sha256"][args.dataset],
        "command": [sys.executable, *command],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training_test_access": "disabled" if args.mode != "formal-test" else "formal_once",
    }
    (args.output_dir / "effective_command.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.mode == "dry-run":
        print(
            f"main_ii_job_dry_run=pass baseline={args.baseline} "
            f"dataset={args.dataset} pred_len=720"
        )
        return

    env = dict(os.environ)
    if args.mode == "resource-smoke":
        runtime = config["runtime_contract"]
        env["FATST_MAX_TRAIN_BATCHES"] = str(runtime["smoke_max_train_batches"])
        env["FATST_MAX_EVAL_BATCHES"] = str(runtime["smoke_max_eval_batches"])
    log_path = args.output_dir / "run.log"
    if args.mode == "formal-test":
        checkpoint = find_single_checkpoint(checkpoint_dir)
        before = sha256_file(checkpoint)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, *command], cwd=args.workspace, env=env,
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    checkpoint = find_single_checkpoint(checkpoint_dir)
    checkpoint_hash = sha256_file(checkpoint)
    if args.mode == "formal-test":
        if checkpoint_hash != before:
            raise RuntimeError("formal test mutated checkpoint")
        model_id = command[command.index("--model_id") + 1]
        result_dirs = sorted(
            (args.workspace / "results").glob(f"{model_id}*"),
            key=lambda path: path.stat().st_mtime,
        )
        if not result_dirs:
            raise RuntimeError("formal test did not create a results directory")
        result_dir = result_dirs[-1]
        for name in ("pred.npy", "true.npy"):
            source = result_dir / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, args.output_dir / name)
    artifact = {
        **record,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "log_sha256": sha256_file(log_path),
    }
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "DONE").write_text("pass\n", encoding="utf-8")
    print(
        f"main_ii_job=pass mode={args.mode} baseline={args.baseline} "
        f"dataset={args.dataset} checkpoint={checkpoint_hash}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--action", choices=["prepare", "run"], required=True)
    parser.add_argument("--baseline", choices=["iTransformer", "PatchTST", "DLinear"], required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--itransformer-source-root", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--mode", choices=["dry-run", "resource-smoke", "formal-training", "formal-test"]
    )
    args = parser.parse_args()
    if args.action == "prepare" and args.source_root is None:
        parser.error("prepare requires --source-root")
    if args.action == "run" and any(
        value is None for value in (args.dataset, args.data_root, args.output_dir, args.mode)
    ):
        parser.error("run requires --dataset, --data-root, --output-dir, and --mode")
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.action == "prepare":
        prepare_workspace(args, config)
    else:
        run_job(args, config)


if __name__ == "__main__":
    main()

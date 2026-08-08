# Main II H720-prefix execution code explanation

## 1. Functional boundary

This execution path implements the frozen `source_native_h720_trained_one_model_all_horizons_system_benchmark`. It does not change ISCF-BSCA model code and does not turn heterogeneous external baselines into matched mechanism controls.

The path has four modules:

1. `configs/iscf_bsca_main_ii_h720_execution.json` freezes source commits, key source/script hashes, seven dataset hashes, the 21-job order, resource limits and Solar adaptation role.
2. `scripts/run_main_ii_h720_training_job.py` verifies an exact upstream checkout, copies a repo-external execution workspace, applies narrow runtime patches, extracts the released H720 command, and runs one smoke/training/test unit.
3. `scripts/evaluate_main_ii_h720_prefix_arrays.py` memory-maps one upstream H720 prediction/target pair,按 origin chunks 转换为 canonical `[origin,time,channel]`，并从同一 tensor 的 exact views 重算 H96/H192/H336/H720 MSE/MAE。
4. `scripts/check_main_ii_h720_prelaunch.py` and `scripts/remote/run_main_ii_h720_training.sh` enforce the 21-training/70-evaluation matrix and the local → smoke → training → test ordering.
5. `scripts/collect_main_ii_existing_iscf_prefix.py` verifies and reuses the seven completed ISCF H1--H720 full-crop formal audits without new test access; `scripts/evaluate_main_ii_timealign_checkpoint.py` streams one frozen official TimeAlign H720 checkpoint directly from its native test loader without retaining full arrays.
6. `scripts/evaluate_main_ii_qdf_checkpoint.py` 从 frozen QDF `config.yaml` 重建 ML3 experiment 并逐 batch 评估；`scripts/evaluate_main_ii_amd_simpletm_checkpoint.py` 从 Main I `metrics.csv`、official command 与 exact checkpoint 重建 AMD/SimpleTM test path，并支持 SimpleTM 三个 native repeats。
7. `scripts/check_main_ii_reused_artifacts.py` 将 56-row master manifest 解析为 42 个 non-ISCF reused jobs，并在 test 前逐 checkpoint 重算 hash；`scripts/remote/run_main_ii_reused_formal_tests.sh` 只在 21/21 新训练完成后，按单 GPU 顺序执行这些 tests，避免和 Tier B 抢占显存。
8. `scripts/analyze_main_ii_h720_prefix_results.py` 合并 7 个既有 ISCF evaluations、42 个 reused-baseline tests 与 21 个 newly trained tests，硬检查 70 checkpoints/280 raw rows/224 aggregate cells，并将 35 个 local H720 anchors 与 frozen Main I 做 $10^{-8}$ continuity audit；iTransformer/PatchTST/DLinear 的 21 个 H720 values 仅记录相对 published three-run means 的 signed deviations。

## 2. Runtime source patch

The runtime adapter deliberately changes only protocol hygiene and compatibility:

- removes construction and evaluation of the official test loader inside `train()`;
- preserves the native validation loss used by `EarlyStopping`;
- disables the automatic test call after training;
- replaces the upstream `testing` log marker with an explicit deferred-test marker so log audits do not confuse a disabled call with actual test access;
- makes formal `is_training=0` evaluation load from `args.checkpoints`;
- enables `true.npy` export where upstream commented it out;
- adds environment-bounded train/validation loops only for resource smoke;
- replaces `np.Inf` with `np.inf` for NumPy 2 compatibility.

PatchTST and DLinear Solar additionally copy the exact audited `Dataset_Solar` semantics from iTransformer and register `data='Solar'`. Their optimization command is the released ECL H720 profile with `data_path`, `enc_in=137` and `model_id` adapted as preregistered. These two cells remain labeled `source_informed_not_official`.

DLinear's native `data_factory` passes an additional `train_only` keyword to every dataset class. The DLinear Solar runtime class therefore accepts and ignores extra keyword arguments; this is a constructor-compatibility patch and does not alter data values, splits, scaling or optimization.

## 3. Command extraction and artifacts

Official shell scripts are executed with a temporary no-op `python` capture shim. All commands are captured after native shell-variable expansion, and exactly one command with `--pred_len 720` must exist. The adapter then changes only dataset absolute path, `num_workers=0`, isolated checkpoint root, execution phase and preregistered Solar fields.

Each unit writes:

- `effective_command.json`: exact argv, source/script identity and test role;
- `run.log`: stdout/stderr;
- `artifact_manifest.json`: checkpoint and log SHA256;
- `DONE`: completion token;
- formal test upstream 临时写入 `pred.npy`、`true.npy`；prefix evaluator 成功保留 four-row metrics、canonical tensor hashes 与 shape 后，仅精确删除这两个临时 arrays，以满足 remote 220 GiB hard limit。

Checkpoint SHA256 is computed before and after formal test; mutation is a hard failure.

ISCF reuse collector 还要求：selected manifest 恰好覆盖 seven dense datasets、actual checkpoint hash 等于 before/after-test hashes、dense CSV 完整覆盖 H1--H720、invariant JSON 的 H96/H192/H336/H720 `full_prefix_max_abs=0`，且 four-H mean 精确重建冻结 manifest。由于该既有 audit 未保留 arrays，新 Main II 记录保存 dense metric CSV、diagnostic NPZ、invariant JSON 与 checkpoint 的 hashes，而不伪造 tensor hash。

TimeAlign streaming evaluator 从冻结 `effective_config.json` 重建 exact official model/data loader，加载同一 H720 `checkpoint.pt`，逐 test batch 做一次 H720 forward 并同步累加 four prefixes。其 H720 MSE/MAE 必须在绝对误差 $10^{-8}$ 内复现 Main I 的 exact anchor；checkpoint pre/post hashes 必须相同。

## 4. Prefix metric semantics

For canonical tensors `prediction,target ∈ R^{N×720×C}`, horizon `H` uses `[:, :H, :]` from the same H720 forward tensor。Evaluator 以 memory-map 加载 arrays，逐 origin chunk 使用 float64 sums/counts 计算 global elementwise MSE/MAE，避免将 ECL 等完整张量复制到 host memory。每个 horizon 的 hash 初始化时包含 canonical shape 与 dtype，随后按 origin 顺序吸收 contiguous prefix bytes，因此保留的 hash 仍唯一标识完整 canonical prefix。每行同时记录 origin/channel counts 与 prediction/target hashes。这建立了 within-checkpoint prefix identity，但不会消除各 official repositories native test-loader 的 cross-system 差异。

`--remove-input-arrays-after-success` 只接受 basename 为 `pred.npy`/`true.npy` 的两个 `.npy` paths；必须先成功写入 `prefix_metrics.json` 才会 unlink，并随后将精确 removed paths 与 retention state 回写 JSON。Checkpoints、logs、effective configs 和 metric/hash audit 不删除。

AMD upstream 的 Main I metric 并非 global elementwise mean，而是对 test batches 的 MSE/MAE 做未按 batch size 加权的平均。为保证 Main II H720 与冻结 Main I H720 anchor 的连续性，AMD four-prefix primary columns 保留这一 official aggregation；同一 streaming pass 额外记录 `global_elementwise_mse/mae`，使该 source-native exception 可审计。其他 evaluators 的 primary columns 使用 global elementwise float64 accumulation。

## 5. Falsification and rollback

The execution path fails closed if a source/script/dataset hash differs, an official script yields zero or multiple H720 commands, training still references `flag='test'`, a checkpoint is missing or mutated, tensors are non-finite/non-H720, or any matrix count is incomplete. A Solar adapter failure blocks the complete table but does not justify deleting Solar or reporting a favorable subset.

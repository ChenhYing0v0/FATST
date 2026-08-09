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
8. `scripts/analyze_main_ii_h720_prefix_results.py` 合并 7 个既有 ISCF evaluations、42 个 reused-baseline tests 与 21 个 newly trained tests，硬检查 70 checkpoints/280 raw rows/224 aggregate cells；读取 frozen Main I anchor 时只接受 `96/192/336/720` standard rows，并显式忽略展示用 `Avg.` rows。随后将 35 个 local H720 anchors 与 frozen Main I 做 `max(1e-8, four float32 ULPs at anchor)` continuity audit；iTransformer/PatchTST/DLinear 的 21 个 H720 values 仅记录相对 published three-run means 的 signed deviations。Analyzer与table builder固定使用LF作为CSV line terminator，使Git归档字节与recorded SHA256保持一致。
9. `scripts/remote/run_main_ii_tier_c_chain.sh` 绑定 exact detached-worktree commit，等待 Tier B supervisor 退出并验证 21/21 `DONE` 后，单 GPU 顺序执行 new/reused Tier C tests 与 aggregate audit。任何非零退出都会中止后续阶段，因此不会在 incomplete/failed evidence 上生成 Main II table。

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
- formal test upstream 在 repo-external workspace 临时写入 `pred.npy`、`true.npy`；artifact manifest 记录其绝对路径，prefix evaluator 直接消费该唯一副本。成功保留 four-row metrics、canonical tensor hashes 与 shape 后，仅精确删除这两个临时 arrays，以满足 remote 220 GiB hard limit。禁止先复制一份到 unit output 再评估。

Checkpoint SHA256 is computed before and after formal test; mutation is a hard failure.

ISCF reuse collector 还要求：selected manifest 恰好覆盖 seven dense datasets、actual checkpoint hash 等于 before/after-test hashes、dense CSV 完整覆盖 H1--H720、invariant JSON 的 H96/H192/H336/H720 `full_prefix_max_abs=0`，且 four-H mean 精确重建冻结 manifest。由于该既有 audit 未保留 arrays，新 Main II 记录保存 dense metric CSV、diagnostic NPZ、invariant JSON 与 checkpoint 的 hashes，而不伪造 tensor hash。

TimeAlign evaluator 从冻结 `effective_config.json` 重建 exact official model/data loader，加载同一 H720 `checkpoint.pt`，逐 test batch 做一次 H720 forward 并同步 hash four prefixes。Primary metric 在内存中拼接 float32 arrays 后调用 official `train_repo.metric_rows()`，即 per-step float32 mean 再做 float64 cumulative mean；streaming float64 global means 作为 audit columns。其 H720 MSE/MAE 必须在 `max(1e-8, four float32 ULPs at anchor)` 内复现 Main I 的 exact anchor；checkpoint pre/post hashes 必须相同。

## 4. Prefix metric semantics

For canonical tensors `prediction,target ∈ R^{N×720×C}`, horizon `H` uses `[:, :H, :]` from the same H720 forward tensor。Evaluator 以 memory-map 加载 arrays，逐 origin chunk 使用 float64 sums/counts 计算 global elementwise MSE/MAE，避免将 ECL 等完整张量复制到 host memory。每个 horizon 的 hash 初始化时包含 canonical shape 与 dtype，随后按 origin 顺序吸收 contiguous prefix bytes，因此保留的 hash 仍唯一标识完整 canonical prefix。每行同时记录 origin/channel counts 与 prediction/target hashes。这建立了 within-checkpoint prefix identity，但不会消除各 official repositories native test-loader 的 cross-system 差异。

`--remove-input-arrays-after-success` 只接受 basename 为 `pred.npy`/`true.npy` 的两个 `.npy` paths；必须先成功写入 `prefix_metrics.json` 才会 unlink，并随后将精确 removed paths 与 retention state 回写 JSON。Checkpoints、logs、effective configs 和 metric/hash audit 不删除。

2026-08-09 首次 Tier C chain 在 14/21 new tests 后暴露 artifact collection defect：runner 虽删除 unit-output arrays，却保留 workspace arrays，并在 DLinear-ECL 再复制约 8.4 GB 时触发 220 GiB quota。该失败发生在完整 inference arrays 已写入之后，不属于 model/numeric failure。修复仅移除 duplication：manifest 指向 workspace 唯一副本，evaluator 成功后原位删除；prediction、loader、checkpoint 与 metric contract 均不变。

同日 reused evaluator 的第一个 TimeAlign ETTh1 checkpoint 暴露 numeric gate defect：streaming float64 reduction 相对冻结 Main I native vectorized reduction 的 MSE 差 $1.278\times10^{-8}$，约为该 float32 anchor 的 $0.43$ ULP，MAE 差 $4.28\times10^{-10}$；QDF ETTh2 在恢复 official seed 与 metric path 后仍有稳定的 two-ULP MAE re-forward drift。checkpoint 与 loader 均未变化，因此修复把固定 $10^{-8}$ 门限改为 `max(1e-8, four float32 ULPs at anchor)`，并逐 scalar 持久化实际 tolerance 与 delta。该修复不改变 prediction、primary metric、checkpoint、search、模型选择或三位小数论文值，只避免把 bounded float32/CUDA roundoff 误判为 protocol drift。

TimeAlign Solar 进一步显示，约 $1.34$ million elements/step 的官方 float32 vectorized reduction 与 streaming float64 reduction 可产生 $2.25\times10^{-6}$ MSE 差异，已不能归入 re-forward ULP。Evaluator 因而改为直接调用 frozen Main I 的 official `metric_rows()` 生成 primary columns，float64 streaming 值只作 audit；这修复 source metric semantics，不改变任何 prediction tensor 或模型选择。

AMD upstream 的 Main I metric 并非 global elementwise mean，而是在 GPU float32 tensor 上计算每个 batch 的 MSE/MAE，并用 `running=(running*i+batch)/(i+1)` 递归更新未按 batch size 加权的平均。为保证 Main II H720 与冻结 Main I H720 anchor 的连续性，AMD four-prefix primary columns 精确复刻这一 torch 运算与更新次序；同一 streaming pass 额外记录 `global_elementwise_mse/mae`，使该 source-native exception 可审计。除下述 QDF source-native exception 外，其余 evaluators 的 primary columns 使用 global elementwise float64 accumulation。最初用 NumPy float64 batch mean 再求和会在 ETTh1 H720 产生约 $1.19\times10^{-7}$ 的假 continuity mismatch；该 evaluator defect 已由 exact torch path 修复，不通过继续放宽门限掩盖。

QDF upstream 在 experiment construction 前同步设置 Python、NumPy、torch CPU/CUDA 的 `fix_seed`，随后把所有 CPU float32 batches `torch.cat` 后调用 `torch.mean`。Main II evaluator 因而先复刻相同 seed initialization，再在内存中保留同一次 H720 forward 的 CPU tensors，按 four prefixes 精确复刻该 native metric path，并把 streaming float64 global means 作为额外 audit columns。该修复使 ETTh2 MSE bitwise 复现 anchor，并将剩余 MAE 差异确定为稳定的 two-ULP checkpoint re-forward rounding；它由统一 four-ULP continuity rule 显式覆盖并记录，而不是隐藏或覆盖 Main I anchor。

## 5. Falsification and rollback

The execution path fails closed if a source/script/dataset hash differs, an official script yields zero or multiple H720 commands, training still references `flag='test'`, a checkpoint is missing or mutated, tensors are non-finite/non-H720, or any matrix count is incomplete. A Solar adapter failure blocks the complete table but does not justify deleting Solar or reporting a favorable subset.

## 6. Paper-table presentation layer

`scripts/build_iscf_bsca_main_ii_table.py`只消费已通过gate的224-cell aggregate CSV，
不会访问checkpoint、prediction tensor或official test。它先按
`system × dataset × horizon`验证完整identity，再计算每个dataset的four-horizon
arithmetic mean。Ranking在共同三位小数上进行：distinct displayed minimum为best，
第二个distinct value为second，ties保留。

2026-08-09 presentation alignment把Main II与Main I统一为：

- dataset order=`ETTm1, ETTm2, ETTh1, ETTh2, Weather, ECL, Solar`；
- best=`red + bold`，second=`blue + underline`；
- `iTransformer (2024b)`、`tabcolsep=1.2pt`；
- required packages=`booktabs, multirow, graphicx, xcolor`。

这次变更只重排已存在的dataset blocks并改变LaTeX style wrappers。224个aggregate
cells、全部MSE/MAE、Avg.计算、24/56 best、27/56 second及claim boundary均未变化。
生成摘要显式记录`presentation_template=Main_I_TimeAlign_Table_6_style`，便于后续
检测视觉契约漂移。

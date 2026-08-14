# ISCF-BSCA Core-Ablation：设计冻结与 Prelaunch Gate

日期：2026-08-14
候选版本：`ISCF-BSCA-v1-core-ablation-20260814`
当前步骤：Step 6–7B（设计冻结、实现与本地 prelaunch）

## 1. 实验问题与证据角色

本实验回答 Section 6 的核心归因问题：在相同 encoder、数据、训练预算、checkpoint selector 与 official-test scorecard 下，`ISCF-BSCA-v1` 的 Scope-Wise Forecasting Loss、Target-Adaptive Allocation、scope-specific projection，以及 multi-scope design 是否分别提供可复现的性能贡献。

该矩阵属于 `matched_mechanism_attribution`，不是 Main I/II 的 tuned best-setting 竞争表。Full 锚点固定为 exact `ISCF-BSCA-v1`，不得替换为 `ISCF-BSCA-MAIN-v1`，也不得因 main-result HPO 结果改变 ablation hyperparameters。

## 2. 冻结矩阵

- datasets：`ETTm1`、`ETTm2`、`ETTh1`、`ETTh2`、`Weather`；
- horizons：`{96, 192, 336, 720}`；
- metrics：MSE、MAE；
- seed：2021；
- checkpoint selector：四个 validation horizons 的 mean MSE；
- test role：`primary-mechanism-effectiveness-and-paper-benchmark`；
- 表格规模：5 variants × 5 datasets × 4 horizons = 100 cells；
- Full 20 cells 复用，4 个 controls 的 80 cells 重新 end-to-end joint training。

五个冻结变体如下：

1. `Full ISCF-BSCA`：exact `ISCF-BSCA-v1`；
2. `w/o BSCA`：仅保留 Uniform-Prefix Forecasting Loss；
3. `w/o Target-Adaptive Allocation`：将 learned target-wise allocation 替换为 equal non-adaptive fusion；
4. `Shared Scope Projection`：使用 capacity-matched `siff-q1-wide-control`；
5. `Fixed Scope (s=144)`：固定使用 preregistered scope 144，`s=144` 不表示搜索所得最优值。

历史 `ISCF-EQUAL` 的 loss path 与新 `w/o BSCA` 不同，因此明确排除复用。所有 formal controls 均从同一 initialization class 端到端训练；frozen replacement、warm-start 与 cross-swap 不进入方向级结论。

## 3. Artifact 与 checkpoint gate

远程新结果根目录固定为：

`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_core_ablation_20260814`

正式 test 前必须满足：

1. 20/20 新训练 runs 均存在 checkpoint、training log、effective config、initialization contract、validation metrics 与 trained invariants；
2. 所有 validation invariants 为 finite/pass，且没有提前生成 test artifacts；
3. 每个 dataset 的 matched controls 具有相同 encoder initialization hash；
4. 20 个新 checkpoint hash 全部唯一；
5. 生成 immutable training manifest，并在 formal test 启动前重新核对所有 checkpoint hashes；
6. formal test 后 checkpoint hash 不得变化。

Full 的 5 个 checkpoint 从 frozen reference root 复用，不重新训练。其 checkpoint hash 与 test artifacts 在最终 100-cell manifest 中再次审计。

## 4. Success/failure gates

每个 control 分别与 Full 比较，并同时要求：

- Full macro MSE 优于 control；
- Full macro MAE 优于 control；
- Full 在至少 3/5 个 dataset mean MSE 上获胜；
- Full 在至少 3/4 个 horizon-aggregated MSE 上获胜。

只有四个 controls 全部通过，才能将完整 BSCA contribution chain 提升为 `passed_core_candidate_matched_attribution`。任一 control 失败时仍完整报告全部 100 cells，并仅限制对应机制 claim；多项失败则回滚到 Step 4 mechanism design。数值异常仅判定为 exact control implementation/pathology，不直接否定研究方向。

## 5. 资源与调度

- 远程 GPU：3090 × 3；
- 新训练：20 runs；
- formal test：20 runs；
- 单作业预计峰值显存不超过 8 GiB；
- 新增存储上限预算：6 GiB；
- workload-aware 排序：优先分散 `Weather` 和 `ETTm1`，随后填充较快 datasets；
- rollback：单 run artifact/hash 失败时只修复或重训该 run；formal test 不得越过 manifest gate。

## 6. 可复现入口

- protocol：`configs/iscf_bsca_core_ablation_protocol.json`；
- runner：`scripts/remote/run_iscf_bsca_core_ablation.sh`；
- prelaunch checker：`scripts/check_iscf_bsca_core_ablation_prelaunch.py`；
- training manifest checker：`scripts/check_iscf_bsca_core_ablation_training_artifacts.py`；
- result analyzer：`scripts/analyze_iscf_bsca_core_ablation.py`；
- table builder：`scripts/build_iscf_bsca_core_ablation_table.py`。

## 7. 当前决策

本文件冻结后，必须先通过本地 prelaunch checks 与四种新 control path 的 remote resource smoke。二者通过后才能启动 20-run training。training manifest gate 通过后，仅执行一次授权的完整 formal test，并据完整 100-cell scorecard 形成 Ablation Table。

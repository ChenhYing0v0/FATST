# Main II H720 checkpoint × horizon-specific loader 重算与 prelaunch gate

## 1. 当前结论

[Fact] 旧 Main II 对每个 external baseline 只构造一次 `pred_len=720` 的 test loader，然后在同一批 H720 origins 上裁出 H96/H192/H336 prefix。该口径保证同一轨迹的 prefix identity，但没有复现 horizon-specific 脚本在每个 H 下的 test sample 数量。

[Strong Evidence] 常见 LTSF dataset 的可用 window 数量随 `pred_len` 改变，因此 H720 loader 通常比 H96/H192/H336 loader 保留更少的 test origins。旧表短 horizon 与 Main I fixed-H test surface 不完全相同，可能造成 one-model-for-all 短 horizon 结果异常偏好。用户提出的疑问成立，旧 Main II 需要完整重算，不能只修补个别 cells。

## 2. 冻结后的计算定义

对 baseline $b$、dataset $d$ 和 horizon $H$：

1. 读取该 baseline 在 horizon-specific 工作流中训练得到的同一份 H720 checkpoint，并在测试前后校验 checkpoint SHA256；
2. 从该 baseline 的 official H-specific script 重建 `pred_len=H` 的 test loader，保留其 dataset split、`seq_len`、batch size、scaler、feature mode、target、frequency 与 `drop_last`；
3. 将 loader 给出的 history 输入 H720 model；official API 若需要 decoder context，只输入 `label_len` 已知历史和全零 future slots，future label 不进入模型；
4. 取 H720 output 的前 H steps，与该 fixed-H loader 的 H-step label 比较；
5. 在 official loader 实际保留的全部 origins、steps、channels 上计算 MSE/MAE。

这一定义不要求不同 H 共享完全相同 origins；它要求每个 H 与对应 horizon-specific official evaluation 使用相同 test surface。

## 3. 矩阵与 checkpoint inventory

- external baselines：TimeAlign、QDF、AMD、SimpleTM、iTransformer、PatchTST、DLinear；
- datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather、ECL、Solar；
- horizons：96、192、336、720；
- checkpoint objects：63，其中 SimpleTM 为 7 datasets × 3 repetitions，其余 baseline 为 7 datasets × 1；
- external formal evaluations：63 × 4 = 252；
- external aggregate cells：7 × 7 × 4 = 196；
- ISCF-BSCA 使用当前已冻结 dataset-level profile 的 28-cell fixed-H formal scorecard，不重复访问 test；
- final Main II cells：8 systems × 7 datasets × 4 horizons = 224。

## 4. official loader audit

冻结的 test `drop_last` 为：TimeAlign=false、QDF=true、AMD=false、SimpleTM=true、iTransformer=true、PatchTST=true、DLinear=false。每个 formal job 都必须从实际 DataLoader 再读取并核对该值；只在文档中声明而未运行时验证不算通过。

PatchTST/DLinear 的 Solar 保留此前已冻结的 source-informed Solar loader/profile；QDF Solar 保留由 audited ECL form 构造并已用于 Main I 的脚本。它们必须继续以对应 source role 披露，不能称为 upstream 原生 Solar script。

## 5. input-only 与 leakage gate

所有 evaluator 写出：`model_horizon=720`、`loader_horizon=H`、`loader_drop_last`、`origin_count`、checkpoint hash、prediction/target digest、`input_only_inference=true` 与 `future_label_used_as_model_input=false`。任何 baseline 若必须把 future label 传入预测分支才能运行，该 job 直接失败，不允许以该结果填表。

已知 `label_len` decoder context 属于 forecast origin 之前的历史输入，不属于 future label leakage。所有 future decoder slots 必须为零；对忽略 decoder input 的 direct forecaster也保留这一统一调用约束。

## 6. launch、success/failure 与 rollback

### Prelaunch pass 条件

- 63/63 checkpoint objects 存在，SHA256 全部唯一且与先前 artifact manifest 一致；
- 252/252 job identities 完整；
- 远程 GPU 0/1/2 可用并记录 launch-time `nvidia-smi`；
- 用户目录保持在 220G hard limit 内；
- evaluator 通过 `py_compile`，runner 通过 `bash -n`，protocol 通过 JSON parse。

### Formal complete gate

- 252/252 jobs 完成，不能选择性丢弃不利结果；
- SimpleTM 逐 cell 对 3 repetitions 做算术平均，其余 baseline 单 checkpoint；
- H720 与同 checkpoint 的旧 H720 formal value 在冻结 numeric tolerance 内连续；
- 同一 checkpoint 的 origin count 随 H 增大不得增加；
- 所有 MSE/MAE finite，最终 196 external cells + 28 ISCF cells 完整。

### Rollback

loader/runtime 报错时，只允许修复 evaluator bridge 并记录修复；不得换 checkpoint 或改 model hyperparameters。`drop_last`、origin count 或 H720 continuity 失败时阻断 aggregation。若矩阵不完整，旧 Main II 继续保持 active，不产生 partial paper-table replacement。

## 7. 当前 gate

`authorized_prelaunch`。用户在 2026-08-13 显式授权全 baseline 重算与新 Main II。下一步是完成 local static verification、commit/push、远程 dry-run、最小 continuity smoke，然后在 3 GPUs 上执行完整 252-job matrix。

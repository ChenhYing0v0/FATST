# Main II H720-prefix benchmark 设计与 prelaunch gate（2026-08-08）

## 1. 当前结论

用户已将 Main I 暂时冻结，并把 Main II 改为 **H720-trained one-model-all-horizons system benchmark**：每个 external baseline 在每个 dataset 只使用一个按官方 H720 script 训练的模型；H96、H192 与 H336 由同一 H720 prediction tensor 裁剪前 $H$ steps 获得。该设计直接测试 fixed-H systems 在不为每个 horizon 单独训练模型时的多 horizon 服务能力。

本轮已完成 Main I freeze、官方 source audit、逐 dataset checkpoint/source manifest 与 machine-readable protocol。2026-08-08 用户已显式授权 Tier A local protocol/source patch、Tier B remote H720 training 与 Tier C formal prefix test；2026-08-09 三层 gate 均已完成，最终形成70 checkpoint evaluations、280 raw rows与224 aggregate cells。Canonical completion report位于`formal_results_20260809/result_and_table_audit.md`。

## 2. Main I freeze

Main I 当前版本冻结为 14 models × 7 dense datasets × four horizons，共 392 个 standard rows；含 four-H Avg. 后为 490 rows。ISCF-BSCA 在统一三位小数排名口径下为 29/56 best、19/56 second。冻结文件、输入 evidence 及 SHA256 见：

- `../main_i_final_amd_simpletm_20260808/main_i_freeze_manifest.json`；
- `../main_i_final_amd_simpletm_20260808/result_and_table_audit.md`。

Main II 只能读取冻结的 Main I H720 anchors，不得回写或替换 Main I。任何 Main I 数值、模型顺序、dataset surface、source role 或 caption 变化都需要用户显式 unfreeze 后原子重建。

## 3. 论文角色与 claim boundary

Main II v1 包含 ISCF-BSCA-MAIN-v1 及七个 external baselines：TimeAlign、QDF、AMD、SimpleTM、iTransformer、PatchTST、DLinear。主表使用当前 Main I 共同完整的七个 dense datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather、ECL、Solar。

该表能支持的结论是：在“每个 dataset 只有一个模型，并由同一 H720 trajectory 服务全部 four horizons”的 system setting 下，各模型的完整 MSE/MAE 表现如何。它不能独立支持 ISCF/BSCA mechanism attribution，因为 source repositories、lookback、objective、optimizer、selector、seed、parameter budget 以及 test-loader `drop_last` 均不完全 matched。正式机制归因仍由 five-dataset end-to-end ablation 与 decoder transfer 承担。

Exchange 暂不进入 Main II v1。原因不是选择性删除，而是冻结 Main I 目前只有 ISCF-BSCA、TimeAlign、QDF 的 Exchange companion；AMD、SimpleTM、iTransformer、PatchTST、DLinear 没有完整、已审计的 Main I Exchange H720 anchors。若后续要做 eight-dataset Main II，必须先显式扩展并原子重建 Main I Exchange surface。

## 4. Exact prediction/evaluation contract

对任一 external checkpoint，formal evaluator 必须只建立一次 H720 official-test loader；先按各repository已审计的output layout做必要的permutation，再统一保存为`[origin, time, channel]` aligned tensors：

$$
\widehat{\mathbf Y}\in\mathbb R^{N_{720}\times720\times C},
\qquad
\mathbf Y\in\mathbb R^{N_{720}\times720\times C}.
$$

对 $H\in\{96,192,336,720\}$，统一计算

$$
\operatorname{MSE}_H
=
\operatorname{mean}\left[
(\widehat{\mathbf Y}_{:,1:H,:}-\mathbf Y_{:,1:H,:})^2
\right],
$$

$$
\operatorname{MAE}_H
=
\operatorname{mean}\left[
\left|\widehat{\mathbf Y}_{:,1:H,:}-\mathbf Y_{:,1:H,:}\right|
\right].
$$

因此同一 checkpoint 的四个结果使用完全相同的 $N_{720}$ forecast origins，H96/H192/H336 是 H720 prediction 的 exact prefixes，CHPC 在该服务协议下由 construction 保证。禁止重新建立 H96/H192/H336 loader、重新调用 horizon-specific head、按 horizon 选择 checkpoint 或重复训练。

Primary table 保留每个 repository 的 native H720 loader，以便 H720 continuity audit。由于 PatchTST/SimpleTM 等 upstream 的 `drop_last` 与 AMD/DLinear 不同，这一表必须标记 `source-native system benchmark`。若之后需要严格相同 origins 的 sensitivity analysis，只能作为单独、预注册的 secondary block；它不能替换 Main I H720 continuity check。

Metric aggregation 同样保留 source-native continuity：ISCF、SimpleTM、iTransformer、PatchTST、DLinear 从同一 H720 tensor 做 float64 global-elementwise accumulation；TimeAlign 精确保留官方 NumPy float32 per-step mean 加 float64 cumulative mean，AMD 精确保留官方 GPU-float32 recursive unweighted batch mean，QDF 精确保留官方 concatenated CPU-torch-float32 global mean，并为这些 exceptions 同时保存 float64 global-elementwise audit columns。该差异必须随表披露，不能用于 matched mechanism attribution。

## 5. H720 与 Main I 的连续性

“理论上 H720 应与 Main I 一致”被拆成两种可审计情况：

1. **Local exact anchors**：ISCF-BSCA、TimeAlign、QDF、AMD、SimpleTM 复用 Main I 的 exact H720 checkpoints。相同 preprocessing、native H720 loader、metric 与 repetition aggregation 下，MSE/MAE 必须在数值容差 $max(10^{-8},4\operatorname{ULP}_{\mathrm{float32}}(m_{\mathrm{anchor}}))$ 内复现。该容差只覆盖同一 float32 checkpoint re-forward 与 native/streaming reduction 的 bounded rounding，当前 benchmark 上约为 $10^{-7}$ 或更严格；checkpoint、loader、preprocessing 或 metric-contract 漂移仍是 hard failure。SimpleTM 必须评估并平均全部三个 H720 native repetitions，禁止挑选一个有利 repeat。
2. **Published references**：iTransformer、PatchTST、DLinear 的 Main I H720 是 TimeAlign Table 6 的 published three-run mean，而 Main II phase 1 是官方源码 single-seed reproduction。两者不存在 checkpoint identity，因此不能要求 bitwise/numeric equality；必须报告 local-minus-published deviation，但不得据此修改已冻结 Main I。

这一拆分避免把“同一 checkpoint 的复测一致性”和“single-seed 对 published mean 的接近程度”混为一谈。

## 6. Source audit

| Baseline | Official repository / audited commit | H720 source coverage on 7 datasets | Current checkpoint state |
| --- | --- | --- | --- |
| TimeAlign | decisionintelligence/TimeAlign `ab2dff5...` | 7/7 official presets | 7/7 reusable |
| QDF | frozen official source `eb0693a...` | six released + Solar source-informed | 7/7 reusable |
| AMD | TROUBADOUR000/AMD `000d377...` | 7/7 official scripts | 7/7 reusable |
| SimpleTM | vsingh-group/SimpleTM `3c77d82...` | 7/7 official scripts | 21/21 H720 repetition checkpoints reusable |
| iTransformer | thuml/iTransformer `c2426e6...` | 7/7 official scripts | 7 retrain required |
| PatchTST | yuqinie98/PatchTST `204c21e...` | 6/7; no Solar script/loader | 6 retrain + Solar source patch required |
| DLinear | cure-lab/LTSF-Linear `0c11366...` | 6/7; no Solar script/loader | 6 retrain + Solar source patch required |

iTransformer、PatchTST、DLinear 的 source commits 是 2026-08-08 对官方仓库 HEAD 的 audit snapshot；在 remote launch 前仍需冻结完整 executed-file hashes。iTransformer 使用各 released `multivariate_forecasting` script 的 H720 command。PatchTST 与 DLinear 使用 released `L=336` H720 commands。两者的 Solar patch 预注册为 ECL H720 optimization profile + official iTransformer-style Solar split/loader semantics，并显式标为 `source_informed_not_official`。

逐 system × dataset 的 56-row manifest 位于 `checkpoint_and_source_manifest.csv`。其中 35 个 local-anchor rows 共包含 49 个 frozen checkpoint objects；iTransformer/PatchTST/DLinear 的 21 个 rows 当前为 `retrain_required` 或 `source_patch_required`。

## 7. Minimal sufficient matrix

- External baseline aggregate cells：7 baselines × 7 datasets × 4 horizons = 196；
- 含 ISCF-BSCA：8 systems × 7 datasets × 4 horizons = 224 cells；
- 每cell同时报告 MSE/MAE，共448个 scalars；
- 新训练：iTransformer/PatchTST/DLinear 各7个 H720 checkpoints，共21 jobs；
- reused checkpoint objects：ISCF 7 + TimeAlign 7 + QDF 7 + AMD 7 + SimpleTM 21 = 49；
- formal prefix evaluations：70 checkpoints × four H = 280 raw rows。

70 个 checkpoint evaluations 中，7 个 ISCF checkpoints 已有同一 H720 trajectory 的完整 dense-prefix formal audit，可在 checkpoint 与 test-artifact hashes 不变时复用；其余 63 个 checkpoint evaluations 新执行。禁止仅为了复制已有数值而重复访问 ISCF test。

当前 single-seed-first 原则保持不变。Optional 3-seed 只允许在完整 primary matrix 后、时间允许时按完整 experiment block 非选择性扩展，不允许根据 phase-1 结果决定只补有利模型、dataset 或 cells。

## 8. Resource、调度与 storage

Pre-smoke 粗估 21 个新训练 jobs 需要 60--180 GPU-hours；该区间只用于容量规划，必须由 bounded resource smoke 重校准。Tier B 启动时 remote `/home/yingch` 已占用约 181 GiB，因此 Main II 后续新增 storage budget 冻结为不超过剩余约 39 GiB，并始终受 220 GiB hard limit 约束。

Tier C 采用逐 checkpoint 的 storage-safe streaming/ephemeral-array audit。每个 H720 forward batch 在内存中转换为 NTC，并从同一 tensor 切取 H96/H192/H336/H720；MSE/MAE 使用 float64 global accumulators，且为每个 prediction/target prefix 永久保留 shape、origin count、channel count 与 deterministic SHA256。若 upstream 只能先写 `pred.npy`/`true.npy`，仅在 `prefix_metrics.json` 与 hashes 成功落盘后删除这两个精确临时输入；checkpoints、effective configs、logs、metrics、hashes 与 H720 continuity evidence永久保留。该策略只改变 artifact retention，不改变预测、裁剪、样本或指标定义。

三张 3090 使用 dataset-major longest-processing-time queue。首波计划为 iTransformer-ECL、PatchTST-ECL、iTransformer-Weather；随后优先 Solar、ETTm1，再填充 ETT datasets。每个新 baseline 先完成 7 个 H720 no-test smokes，训练 smoke 不产生 official-test evidence。正式训练期间只做 validation checkpoint selection，test 只在 21/21 checkpoint manifest 冻结后按单独授权执行。

## 9. Gates 与 rollback

### Local patch gate

- exact source commits、licenses、executed-file hashes完整；
- prediction/target export与prefix metric unit tests通过；
- 56-row manifest、21-job training dry-run、70-job prefix-test dry-run完整；
- PatchTST/DLinear Solar loader、profile来源与非官方标签冻结；
- training loop不存在 epoch-level test access。

### Remote training gate

- 21/21 checkpoints、effective configs、logs、unique hashes完整；
- no-test resource smoke通过且GPU/storage预算重校准；
- 无 OOM、NaN/Inf、Traceback、source/dataset hash mismatch；
- checkpoint只由 validation early stopping 或原生 official-last rule确定。

### Formal test gate

- 70/70 checkpoint evaluations、280/280 horizon rows完整；
- 224/224 aggregate cells均含MSE/MAE并保留negative results；
- pre/post-test checkpoint hashes不变；
- 5个 local-anchor systems 的 H720 continuity全部通过；
- 3个 published-reference systems 的 deviation完整披露；
- prefix identity逐 tensor 检查通过。

任一 local H720 exact anchor 失败时，立即停止 table publication，依次审计 checkpoint、preprocessing、test loader、metric、repeat aggregation；不得覆盖 Main I。Solar patch 失败只说明 exact source-informed cell未闭合，但会阻止整个 Main II table成为paper evidence，不能只删除Solar继续发布。性能不佳时完整报告，不得按 horizon、metric、cell、seed或dataset重新调参或删列。

## 10. 当前 authorization cursor

- Main I freeze：完成；
- existing-artifact/source audit：完成；
- Main II matrix design：完成；
- local protocol patch：**已完成**；
- remote H720 training：**21/21 H720 checkpoints已完成**；
- formal prefix test：**70/70 checkpoint evaluations与224/224 aggregate cells已完成并通过**；
- Exchange extension / optional common-origin sensitivity / 3-seed：**未授权**。

Tier A implementation/local dry-run → Tier B no-test resource smoke → Tier B formal training → freeze checkpoint manifest → Tier C formal prefix test → complete audit/table build 已全部闭合。不得把已完成的 Tier A/B/C 授权扩张到 Exchange、common-origin sensitivity、optional 3-seed 或 Main I mutation。

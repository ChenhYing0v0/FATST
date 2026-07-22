# SC-ISCF-PSA-D1 代码说明

## 1. Functional boundary

PSA-D1新增的不是model module，而是一个control-only execution/analysis surface：

- `configs/stage_c_iscf_psa_d1.json`：冻结one-arm five-run matrix、references与H2/H3 gates；
- `scripts/check_stage_c_iscf_psa_d1.py`：检查source、authorization和EQUAL objective；
- `scripts/remote/run_stage_c_iscf_psa_d1.sh`：给existing validation runner注入D1 config/output/profile；
- `scripts/remote/run_stage_c_iscf_psa_d1_diagnostics.sh`：v0.1 evaluator-only validation replay与checkpoint
  nonmutation；
- `scripts/analyze_stage_c_iscf_psa_d1.py`：联合new EQUAL与three read-only references做machine attribution。

ISCF forward、five scope arms、direct policy、loss实现和checkpoint selector均未修改。

## 2. Training dataflow

Runner调用existing `train_repo.py`，使用：

- input history `[B,C,720]`；
- five ISCF arm forecasts `[B,C,720,S=5]`；
- direct policy `[B,C,720,S=5]`；
- fused forecast `[B,C,720]`。

`objective_mode=equal_skill`时：

$$
\mathcal L=\mathcal L_{\mathrm{fused}}+
\mathcal L_{\mathrm{equal\ skill}},
\qquad
\mathcal L_{\mathrm{route}}=0.
$$

因此D1只重新训练原始ISCF-EQUAL，不注入ARMERR、coalition、shuffle、entropy或temperature机制。

## 3. Runner composition

`run_stage_c_iscf_psa_d1.sh`只设置：

- output root=`stage_c_iscf_psa_d1`；
- config=`stage_c_iscf_psa_d1.json`；
- protocol profile=`stage_c_iscf_psa_d1_control_v0`；
- run label=`PSA_D1`；
- `RESOURCE_SMOKE_JOBS=1`，使preflight只执行launch-order第一项Weather。

随后复用`run_stage_c_iscf_sps_step7b.sh`的production training path。该path读取config的single arm与five-dataset
`launch_order`，所以dry-run恰好输出5 jobs。`EVALUATION_SPLIT`固定为`val`；config若未授权remote或意外授权test，normal
launch固定exit 3。

Shared runner新增`RESOURCE_SMOKE_JOBS`，default仍为2，所以historical SPS/FRSC/RSCC wrapper行为不变；D1 wrapper显式
覆盖为1。该变量只控制smoke job count，不进入normal five-run scheduling。

## 4. Step7A checker

Checker验证：

1. datasets/seeds/arm/matrix没有漂移；
2. user authorization只覆盖Step7A与five validation runs，test/seeds/method保持false；
3. `PCC.py`、`train_repo.py`、evaluator相对RSCC launch commit `020eea3`无semantic diff；
4. random tensors上的EQUAL route loss与weighted route loss严格为0；
5. total loss逐值等于fused + weighted skill；
6. arms与policy gradients finite，five scope arms均收到nonzero gradient。

Remote resource smoke仍需在真实model/data/GPU上检查artifact、OOM与five-scope gradients。

## 5. Analyzer artifact mapping

Analyzer读取four arms × five datasets：

- new `iscf_equal_contemporaneous`：D1 output root；
- historical `iscf_equal_historical`：FCC EQUAL checkpoint，exact policy probe来自SCC-D0 replay；
- `iscf_equal_armerr`：RSCC Step7B；
- `iscf_rscc_shuffled`：RSCC Step7B。

每个run读取four standard-horizon validation metrics。effective matrix为20 runs/80 cells；five comparisons共100
comparison cells。

## 6. Analyzer outputs and statistics

### `run_audit.csv`

定义artifact completeness、expected/observed objective、initialization hash、checkpoint SHA256、evaluation split、test
usage与trained invariant。`dataset_initialization_paired`要求同一dataset的four arms hash完全相同。

### `validation_metrics.csv`

每行定义`arm,dataset,horizon,mse,mae,evaluation_split`，只允许H96/H192/H336/H720 validation。

### `comparison_cells.csv` / `comparison_summary.csv`

Cell gain定义为$100(r-c)/r$。Summary报告20-cell macro MSE/MAE gain、positive cells、5-dataset wins和4-horizon
wins。Dataset/horizon aggregation各只计数一次。

### `function_drift.csv`

使用source-row-aligned probes：

- `fused_relative_l1`：new/old fused difference除以historical fused absolute mean；
- `policy_mean_l1`：`[256,720,5]` policy mean absolute difference；
- `arms_relative_l1`：`[256,5,720]` arm difference除以historical arm absolute mean。

H2 function-match thresholds预先冻结为fused `<=0.01`、policy `<=0.02`。

### `training_health.csv`

只审计five new runs：epochs、minimum five-scope gradient、maximum route weight、maximum weighted route loss与finite
training/validation losses。route量必须逐run为0。

### `decision.json`

计算control mean gain

$$
G_C=\tfrac12(G_{\mathrm{ARMERR},E_h}+G_{\mathrm{SHUFFLED},E_h})
$$

与recovery ratio $R=G_{E_n,E_h}/G_C$，再按冻结gates输出：

- `contemporaneous_run_drift_explains`；
- `joint_training_route_regularization_supported_as_carrier_clue`；
- `h2_h3_unresolved`；
- protocol/numeric失败时`diagnostic_invalid_for_attribution`。

## 7. Code-theory consistency

理论目标是隔离“route objective during joint training”与“同轮重训本身”。代码只新增no-route contemporaneous EQUAL，
其余arms保持read-only，因此实现了最小缺失control。

限制：single seed validation只能做attribution clue，不能建立paper effectiveness；即使H2通过，也只说明某类training
regularization值得回Step4研究，不说明ARMERR或SHUFFLED具备scope semantics或novelty。任何结果均不自动授权test、
confirmation seeds或method promotion。

## 8. v0.1 evaluator-only repair

首次ETTh1 training完成后，evaluator在读取future bins时发现D1 config缺少`diagnostic_protocol`，于任何probe output前
抛出`KeyError`。v0.1只增加existing evaluator需要的training contract和eight future bins；training surface与decision
gates不变。

Dedicated diagnostic runner只对已训练run调用validation evaluator。输入checkpoint的参数shape与forward tensors不变；
runner在每次forward前后计算SHA256，只有hash完全相等才接受artifact。该repair不允许重训或test access。

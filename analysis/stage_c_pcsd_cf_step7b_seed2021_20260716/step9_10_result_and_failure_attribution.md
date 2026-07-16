# PCSD-CF Step7B Seed2021 Step9/10 Result

## Current Decision

- `current_step`: Step9 artifact evaluation完成；Step10 candidate decision完成
- `candidate`: `SC1-PCSD-CF-v1`
- `formal_gate`: `method_pass=false`
- `formal_decision`: `direct_credit_problem_supported_sc2_step2_4_only`
- `research_decision`: `pcsd_cf_representation_signal_retained_but_joint_training_blocked`
- `rollback`: SC1回Step4检查training-aware architecture contract；SC2只进入Step2-4 problem/source audit
- `held`: seed2022/2023、test、SC2 implementation、joint factorial

PCSD-CF当前实现不能作为paper-core method通过：DIRECT相对A6在5/5 datasets均失败，macro dense-H1..720 MSE
AUC退化`1.5833%`。但失败不应归因为parameter capacity、random partition或数值异常；最强的新证据是ordinary
fused task loss使joint DIRECT checkpoint中的scope arms系统性under-trained。因此当前只否定“PCSD-CF + plain
fused L1”这一完整实现，不否定output-coupling spectrum问题或shared coupling-field方向。

## What Was Tested And Why

矩阵为12 arms × 5 datasets × seed2021，共60个from-scratch E2E runs。所有run使用相同dataset-aware natural
profile、full-H720 pointwise L1、best-validation-H720 checkpoint，并在validation上计算H1..720全部prefix；
test从未读取。A6/M0检查function-preserving control，dense nonlinear head检查parameter capacity，random
partition检查canonical future geometry，fixed/equal/static检查arms与policy attribution。

primary metric `dense_mse_auc`定义为720个prefix MSE的算术平均：

$$
\mathrm{AUC}_{\mathrm{MSE}}=\frac1{720}\sum_{H=1}^{720}\mathrm{MSE}(\hat Y_{1:H},Y_{1:H}).
$$

相对收益均定义为$100(1-\mathrm{candidate}/\mathrm{reference})$，正值代表DIRECT更好。

## Artifact And Protocol Audit

- remote runner于`2026-07-16T17:26:52+08:00`正常结束；60/60 metrics、trained invariants与validation
  diagnostics齐全；日志无Traceback、OOM、NaN或error；
- 本地重新读取原始artifacts得到60/60 `status=ok`，与remote gate逐值一致；
- paired Encoder initialization、paired PCSD initialization、A6/M0 exact initialization全部通过；
- A6/M0训练后dense AUC最大绝对差仅`2.5711e-8`，排除runner、morphism和optimizer protocol偏差；
- raw evidence存于gitignored `raw/`，aggregate见`run_summary.csv`、`direct_comparisons.csv`与`gate.json`。

## Formal Effectiveness Result

| Reference | DIRECT macro gain | Dataset wins | Formal gate |
| --- | ---: | ---: | --- |
| A6 | -1.5833% | 0/5 | fail |
| PCSD equal | -0.0294% | 3/5 | fail |
| PCSD static | -0.6266% | 3/5 | fail |
| dense matched | +2.3492% | 5/5 | pass |
| random partition | +0.4499% | 3/5 | pass |

[Fact] DIRECT没有超过A6/equal/static，所以architecture method gate失败。[Strong Evidence] DIRECT却在5/5超过
parameter-matched dense nonlinear control，并超过random partition，说明结果不是“参数更多自然更好”，canonical
coupling geometry也有小但可复现的specificity signal。

### Dataset-level attribution

| Dataset | A6 AUC | DIRECT AUC | DIRECT vs A6 | Best fixed scope | Best fixed vs A6 | Same-run row/bin oracle | DIRECT-arm degradation median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 1.170411 | 1.179541 | -0.780% | 720 | -0.115% | -0.941% | 167.45% |
| ETTh2 | 0.428406 | 0.439894 | -2.682% | 360 | -0.335% | +14.847% | 24.47% |
| ETTm1 | 0.666559 | 0.682161 | -2.341% | 360 | +1.962% | -0.079% | 132.10% |
| ETTm2 | 0.194697 | 0.198405 | -1.904% | 720 | -1.481% | +4.143% | 326.05% |
| Weather | 0.497109 | 0.498152 | -0.210% | 720 | +0.079% | +1.752% | 319.77% |

`same-run row/bin oracle`对同一个DIRECT checkpoint的五arms逐validation row、逐future bin取最小MSE；它允许
不同sample/bin选不同arm，但不是可部署结果。其正值只有3/5，且macro `3.9444%`主要由ETTh2驱动，因此不能
单独证明一个cross-dataset target-specific router可学。

## Deep Failure Attribution

### 1. Joint arm credit starvation is the primary pathology

对每个dataset × scope，把DIRECT checkpoint中该scope arm的`arm_row_bin_mse`与相同scope独立from-scratch
fixed run的`fused_row_bin_mse`比较：25/25 pairs全部退化，最低`9.6655%`、中位数`89.9495%`、最高
`392.5403%`。退化对local scopes尤其严重，scope720也退化`15.76-29.72%`（ETTh1为`17.79%`）。

[Strong Evidence] plain fused loss并没有把五个arms训练到其独立可达skill。soft fusion下branch gradient被policy
weight缩放；global-heavy routing使低权重arms更弱，弱arms又进一步失去policy weight，形成self-reinforcing
credit starvation。`same_run_arm_skill=true`只说明best arm超过persistence，不能排除相对matched fixed training的
严重under-training；原formal proxy过弱。

### 2. Policy is numerically non-collapsed but target adaptation is weak

formal entropy minimum为`0.5478`、maximum usage为`0.7555`，所以没有one-hot numeric collapse。但五datasets的
scope720平均usage为`0.5007-0.7555`，不同future bins之间mean usage的maximum pairwise L1只有
`0.0051-0.0440`；相对地instance-level mean L1 deviation为`0.0851-0.2276`。

[Inference] router主要学到history-conditioned、近target-invariant的global-heavy mixture，没有实现论文所需的
history × natural target coupling allocation。该结论是bin-aggregated diagnostic，不等于逐target logits严格常数。

### 3. Horizon curves locate the damage

`horizon_gain_curves.svg/png`与`horizon_bin_gain.csv`显示DIRECT相对A6：ETTh2、ETTm1、ETTm2八个future bins
全部为负；ETTh1仅前两bins为正；Weather仅中后部三个bins略正。DIRECT相对dense control多数horizons为正，
说明structured field本身优于generic nonlinear capacity；相对equal/static的收益则随dataset和future region改变，
没有统一胜出。

## Diagnostic Failure Attribution Rule

1. `hypothesis_false`: 未成立。D14-A1 coupling-scope crossing仍存在，canonical field也超过random/dense controls；
2. `intervention_point_wrong`: 部分可能。target policy进入forecast fusion点正确，但plain fused gradient没有维护arm skill；
3. `readout_or_head_design_wrong`: 仍未完全排除。最佳independent fixed scope仅2/5超过A6，shared-field local arms的
   absolute quality仍可能不足；
4. `optimization_or_numeric_pathology`: numeric pathology排除，但optimization/credit pathology被25/25 matched pairs支持；
5. `capacity_control_explains`: 排除；DIRECT相对dense matched为5/5、macro +2.3492%。

因此failure attribution=`design_fault_suspected_joint_credit_starvation`，不是`hypothesis_false`，也不能据此直接
拒绝PCSD architecture方向。

## Step10 Decision

- `SC1-PCSD-CF-v1`: 从`effectiveness-unready`改为`partial_representation_signal_training_blocked`；不跑confirmation；
- formal preregistered结果允许SC2 Step2-4，但不允许直接实现SC2；
- 研究层面把问题收紧为：如何在one-forward、one-stage、full-domain projective training中，同时维持各coupling
  scope的forecast skill并让router获得future-region-specific capability credit；
- 下一步阅读`analysis/stage_c_sc2_projective_coupling_credit_step24_20260716/source_theory_audit.md`；先做Step5
  theory feasibility，不进行remote training。

## Artifact Definitions

- `horizon_gain_by_reference.csv`: source=`metrics_by_target_horizon.csv`；逐H DIRECT/reference MSE与relative gain；
- `horizon_bin_gain.csv`: 在冻结八bin内先平均逐H MSE，再算relative gain；
- `fixed_scope_summary.csv`: 各fixed run dense AUC、A6 AUC、relative gain与dataset内best fixed标记；
- `mechanism_by_dataset.csv`: DIRECT validation diagnostics聚合的oracle、best-arm、separation、entropy、usage与
  target/instance variation；
- `same_run_oracle_by_bin.csv`: 逐row arm minimum、best mean arm及learned usage；
- `direct_arm_vs_fixed_training.csv`: DIRECT same-run arm与独立matched fixed run的row-bin MSE及退化百分比；
- `deep_dive_gate.json`: formal decision不变；新增post-hoc cross-dataset competitiveness与failure attribution。

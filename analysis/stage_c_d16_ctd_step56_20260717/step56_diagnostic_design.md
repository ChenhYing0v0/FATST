# SC-D16-CTD Step 5/6 Theory And Diagnostic Design

## 1. Current record

| Field | Value |
| --- | --- |
| `current_step` | Step 5 theory feasibility + Step 6 diagnostic design |
| `problem` | SIFF ETTm2 short-prefix pathology可能来自H720 checkpoint，而非全部training epochs |
| `existence_evidence` | SIFF-vs-PCSD H1 -669.49%，H720 +0.6013%；all finite |
| `idea` | 保存same-training trajectory的per-epoch dense risk，做checkpoint counterfactual |
| `theory_check` | dense AUC可由one full prediction精确计算；不改training gradients |
| `design` | ETTm2 × 4 matched arms × seed2021；20 epochs；3 checkpoint rules |
| `narrative_gate` | diagnostic only；weighted checkpoint由ElasTST覆盖，不是Contribution |
| `effectiveness_gate` | 不适用paper effectiveness；只判failure attribution |
| `artifacts` | `configs/stage_c_d16_ctd_step6.json` |
| `decision` | Step5/6 pass for Step7A local implementation only；remote/test=false |

## 2. 为什么不能直接关闭scale-field方向

SIFF-v1 formal architecture gate已经失败，且paired PCSD/SIFF arms使用相同exact harmonic-L1 fused objective。
这使“训练measure完全不匹配”的解释失效，exact Q2 field的negative evidence明显增强。

但ETTm2 H1出现7.69倍MSE，而H720略优、所有values finite。项目规则要求对>100%局部pathology先区分：

1. `readout_or_head_design_wrong`：所有epochs都出现short-prefix failure；
2. `optimization_or_checkpoint_pathology`：trajectory中有healthy epoch，但H720 checkpoint没有选中；
3. `capacity_control_explains`：constant/Q1在同一checkpoint rule下解释任何rescue。

当前artifact只保存最终best-H720 checkpoint，无法完成上述区分。

## 3. Theory feasibility

### 3.1 No new model forward contract

对epoch $e$的一次完整validation prediction$\hat y^{(e)}_{1:T}$，可通过cumulative sums同时计算：

$$
L_{\mathrm{MSE}}^{(e)}(H)
=\frac1H\sum_{t=1}^{H}(\hat y_t^{(e)}-y_t)^2,
$$

$$
L_{\mathrm{MAE}}^{(e)}(H)
=\frac1H\sum_{t=1}^{H}|\hat y_t^{(e)}-y_t|.
$$

因此每个epoch只需一次full validation forward，即可得到H1..720、dense AUC与五个horizon bins；requested
horizon不进入模型。

### 3.2 Counterfactual checkpoint rules

同一条training trajectory冻结三个selection functions：

1. `best_h720_mse`：复现当前协议；
2. `best_dense_mse_auc`：匹配primary screening metric；
3. `best_dense_mae_auc`：匹配harmonic-L1 fused training norm。

这些rules只读取validation，不读取test。它们比较同一trajectory，不重新训练、不改变gradient。

### 3.3 Why 20 epochs without early stopping

原协议patience为5，导致ETTm2各arms在6–10 epochs停止。若沿用H720 early stopping，就可能在观察dense
trajectory前再次截断。diagnostic固定运行20 epochs，但记录“按原patience本应停止的epoch”；前N epochs的
optimization与原协议一致，额外epochs只用于failure attribution，不作为最终训练schedule推荐。

## 4. Frozen arms

| Arm | Readout | Objective | Role |
| --- | --- | --- | --- |
| `PCSD_EQUAL_TRAJECTORY` | `pcsd-coupling-field` | `equal_skill` | matched carrier |
| `SIFF_EQUAL_TRAJECTORY` | `siff-coupling-field` | `equal_skill` | primary pathology |
| `SIFF_CONSTANT_EQUAL_TRAJECTORY` | `siff-constant-control` | `equal_skill` | no-order same-parameter control |
| `Q1_WIDE_EQUAL_TRAJECTORY` | `siff-q1-wide-control` | `equal_skill` | conditioned-width control |

选择EQUAL而非MCCA有两点原因：

1. MCCA-v1 exact hypothesis已经关闭，不得作为新diagnostic carrier；
2. EQUAL为PCSD/SIFF共享的最简单arm-skill supervision，可隔离readout与checkpoint。

四arms使用ETTm2 frozen profile、seed2021、batch32、learning rate $10^{-4}$与相同20-epoch optimizer schedule。
不进行dataset-specific或arm-specific tuning。

## 5. Artifact contract

每个epoch必须记录：

- epoch index、train losses与learning rate；
- H1/H48/H96/H192/H336/H720 MSE/MAE；
- dense MSE AUC、dense MAE AUC；
- H1–48、49–96、97–192、193–336、337–720 bin metrics；
- original-patience counter；
- checkpoint hash。

每条trajectory只保留三个selected state dicts，避免保存20个full checkpoints。最终对三个selected states重新
生成dense metrics与trained invariants。

## 6. Frozen gates

### 6.1 Protocol gate

- 4/4 trajectories；
- 每条20/20 epochs；
- uses_test=false；
- objective=`equal_skill`且fused target measure=`exact_dense_prefix_harmonic`；
- encoder initialization在四arms间paired；
- no gradient surgery、requested-H input或hyperparameter change；
- original best-H720 dense AUC相对历史same arm误差不超过0.5%。

### 6.2 Pathology gate

在各arm自己的`best_dense_mse_auc` checkpoint下：

- SIFF H1 MSE / PCSD H1 MSE $\le 2.0$，即不再出现>100% degradation；
- SIFF dense-MSE gain over PCSD $\ge 0$；
- SIFF dense-MSE gain over constant $\ge 0$；
- SIFF dense-MSE gain over Q1-wide $\ge 0$；
- SIFF H337–720 bin相对其自身best-H720 checkpoint退化不超过1%。

所有条件同时成立才称`checkpoint_pathology_supported`。这些thresholds在result前冻结，不得按trajectory调整。

## 7. Decisions

1. **all gates pass**：
   只说明原H720 checkpoint造成假失败；随后才允许five-dataset unchanged confirmation。SIFF不立即恢复为paper
   candidate，HR/checkpoint也不计Contribution。
2. **H1 recovers but architecture/controls fail**：
   修正未来checkpoint protocol，但SIFF exact design仍关闭。
3. **SIFF best-dense H1 ratio仍>2**：
   pathology贯穿trajectory；`readout_or_head_design_wrong`获得稳定证据，回Step2关闭scale-field方向。
4. **protocol/numeric fail**：
   diagnostic无效，回Step6修复工具，不作方向判断。

## 8. Step 6 decision

`diagnostic_design_pass_step7a_local_only`

下一步只实现per-epoch dense evaluator、three-state retention、local synthetic identity与dry-run。Step7A通过前不得
remote；该diagnostic永不访问test，也不得升级为Contribution 2。

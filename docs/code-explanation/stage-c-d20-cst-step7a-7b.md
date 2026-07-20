# Stage C D20 CST Step 7A/7B 代码说明

## 1. 实现边界

`SC-D20-CST-v1`是`diagnostic_only`，用于回答一个收紧后的问题：D19中出现正向证据的history-spectrum
information，能否迁移到strong A6 learned-basis coefficient operator，并超过同维fixed random projection。
它不是论文method，也不允许从一次正向结果直接升级为Contribution 1。

本次实现涉及：

- `baselines/timealign_official/models/TimeAlign.py`：production readout与fixed projection buffer；
- `baselines/timealign_official/train_repo.py`：CLI、初始化审计与训练期diagnostics；
- `scripts/evaluate_stage_c_pcsd_cf_checkpoint.py`：checkpoint non-mutation evaluation与D20 probe artifact；
- `scripts/analyze_stage_c_d20_cst.py`：四层Step 9分析；
- `scripts/remote/run_stage_c_d20_cst_v1.sh`：15-run remote matrix；
- `scripts/check_stage_c_d20_cst_step7a.py`与`check_stage_c_d20_cst_step7b.py`：local implementation gate和prelaunch gate。

## 2. Forward tensor path

A6 carrier先得到：

```text
normalized_history  [B, 720, C]
memory              [B, C, P, D]
hidden              [B, C, R]
base_coeff          [B, C, 256]
full_prediction     [B, 720, C]
```

D20在不改变Encoder与basis synthesis的前提下增加一条compact statistic path：

```text
normalized_history.transpose(1, 2)  [B, C, 720]
fixed_projection                     [720, 64]
summary                              [B, C, 64]
summary_coeff = Linear(64, 256)       [B, C, 256]
coeff = base_coeff + summary_coeff    [B, C, 256]
full_prediction = coeff @ basis.T     [B, C, 720] -> [B, 720, C]
prediction_h = full_prediction[:, :H] [B, H, C]
```

`A6_CST_SPEC`的projection由32组non-DC real Fourier cos/sin columns构成；`A6_CST_RANDOM`使用seed
20260719 Gaussian QR得到的64维orthogonal subspace。两臂新增参数完全相同，projection均为non-trainable buffer。

代码将`Linear(R+64,256)`按代数等价形式写成已有`learned_basis_coeff(hidden)`与新增
`history_statistic_coeff(summary)`之和。这样可以直接复用A6 base-head参数，并明确审计两条coefficient path。

## 3. Paired initialization与projectivity

每个dataset的三臂从相同initialization class构造。CST两臂复制同一次构造得到的Encoder、basis与A6 base-head
初值，新增summary head严格zero-init。因此训练开始前：

$$
F_{A6}(x)=F_{SPEC}(x)=F_{RANDOM}(x).
$$

zero-init只保证initial function parity，不代表保留了任何trained capacity。所有臂都先生成完整$T=720$
trajectory，再做prefix crop，所以任意$H$的输出是同一full prediction的prefix，而不是horizon-specific model。

## 4. Training、checkpoint与test路径

- 五个dataset profile分别保留其冻结的A6-natural配置，不进行dataset/horizon/cell级反向调参；
- 三臂均从scratch端到端训练，seed固定为2021；
- checkpoint由validation上`H96/H192/H336/H720` MSE均值选择；
- formal evaluation读取该checkpoint，在official test split报告完整5 datasets × 4 horizons的MSE/MAE；
- evaluator校验checkpoint hash在test前后不变，并额外保存summary、summary coefficient与prediction contribution，供internal-health诊断。

## 5. Step 9四层分析

analyzer严格分开：

1. `paper_facing_effectiveness`：SPEC相对A6的transfer；
2. `matched_mechanism_attribution`：SPEC相对同维RANDOM的frequency specificity；
3. `internal_mechanism_health`：summary contribution、deformation、梯度与SPEC/RANDOM差异；
4. `failure_attribution`：映射到transfer失败、generic capacity control解释、internal path无效或待confirmation。

SPEC必须同时通过对A6和对RANDOM的冻结gate，正向internal diagnostics不能挽救negative official-test gate。

## 6. Code-theory consistency

- Intended theory：若compact history-spectrum是D19 skip收益中可迁移且frequency-specific的有效信息，它应在相同
  A6 carrier和相同新增维度下同时超过A6与random subspace。
- Code realization：SPEC/RANDOM只改变fixed projection geometry，其余architecture、parameter count、training、
  checkpoint和test protocol一致。
- Proxy boundary：Fourier low modes只是“compact spectrum”的一个有限proxy；negative只能否定该$q=64$ transfer
  protocol，不能否定所有history statistic或structured decoder。
- Falsification：SPEC未稳定超过A6，说明transfer不足；SPEC不超过RANDOM，说明收益可被generic added path/capacity
  解释；出现collapse、non-finite或inactive path时，本次诊断不得用于方向级拒绝。

## 7. Local gates

Step 7A为`9/9 pass`：15 CLI cases、15 constructors、60 shape/prefix cases、10 summary-gradient cases；
初始输出差和prefix gap均为0。Step 7B prelaunch为`10/10 pass`：冻结hash、完整15-run/60-cell matrix、
formal-test authorization、shell syntax、dry-run、evaluator smoke与analyzer smoke全部通过。

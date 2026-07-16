# SC2-PCC-v1-TI Step7B 工具说明

## Modules

- `configs/stage_c_sc2_pcc_step7b.json`：45-run matrix、hash、reference、diagnostics与hard gates；
- `scripts/check_stage_c_sc2_pcc_step7b.py`：manifest/CLI/hash/initialization/authorization prelaunch gate；
- `scripts/remote/run_stage_c_sc2_pcc_step7b.sh`：resumable dataset-major fixed-GPU runner；
- `scripts/evaluate_stage_c_pcsd_cf_checkpoint.py`：复用PCSD validation arms/policy/prefix evaluator；
- `scripts/evaluate_stage_c_sc2_pcc_gradient.py`：best-val shared-field gradient audit；
- `scripts/analyze_stage_c_sc2_pcc_step7b.py`：45 runs + locked references的method/specificity analysis；
- `scripts/sync_stage_c_sc2_pcc_step7b_results.sh`：排除checkpoint的大文件同步与本地重分析。

## Manifest Contract

`jobs_seed2021.csv`每行定义：

- `job_index`：dataset-major全局次序；
- `dataset/objective_mode`：唯一run identity；
- `profile/patch_num/d_model/d_ff`：frozen natural carrier fields；
- `seed`：2021；
- `evaluation_split/checkpoint_policy/test_used`：`val/best-val-h720/false`。

prelaunch checker把每行转换成production `train_repo.py` argv并实际调用parser，防止runner字符串与Python contract漂移。

## Gradient Diagnostic Tensor Flow

best-val checkpoint使用first sequential train row：

```text
x [1,720,C]
 -> PCSD one forward
 -> raw arms [1,C,5,720]
 -> five dense-prefix-measure L1 scope losses
 -> grad(loss_s, shared field parameters)
 -> five vectors g_s [N_shared]
 -> 10 pairwise cosine values
```

`shared field parameters`仅含`mode_weight/mode_bias/identity_synthesis/nonlinear_synthesis/temporal_bias`，不含policy
MLP。输出字段：

- `scope_losses/scope_gradient_norms`：five scope标量与gradient L2 norm；
- `pairwise_cosines`：每个scope pair及cosine；
- `pairwise_cosine_mean/min/max`：10 pairs聚合；
- `all_finite/all_scope_gradients_nonzero/pass`：numeric gate；
- `gradient_surgery_applied=false`：明确该诊断不改变training。

## Analyzer Statistics

- `dense_mse_auc`：H1..720 validation prefix MSE算术平均；
- `pcc_gain_fraction = 1 - PCC_AUC/reference_AUC`，先逐dataset计算再macro等权平均；
- `plain/pcc degradation = joint-arm MSE / independently-trained fixed-scope MSE - 1`；
- `relative_degradation_reduction = (plain degradation - PCC degradation) / |plain degradation|`；
- `arm_pair_improved`：PCC degradation严格小于plain；
- `minimum_pairwise_probe_nrmse`：任意two arms的probe RMSE除以all-arm RMS后的最小值；
- `pairwise_nrmse_retention`：PCC minimum NRMSE / plain minimum NRMSE，hard gate取five datasets最小值；
- `policy_normalized_entropy`：policy row/bin entropy除以$\log 5$后平均；
- `policy_usage_max`：row/bin平均scope usage的最大值。

`run_summary.csv`记录每个new/reference run及protocol状态；`pcc_comparisons.csv`记录PCC相对11个references/controls；
`arm_recovery.csv`记录25 scope pairs；`mechanism_by_dataset.csv`记录diversity、policy、oracle与gradient diagnostics；
`gate.json`执行冻结decision map。

## Code-Theory Consistency

- Intended theory：nested-risk transport必须同时改善最终forecast与same-run arm training，且不能由pointwise prior解释；
- Code realization：full PCC与A6/plain/pointwise/prior、25 arm pairs、diversity与policy均进入同一gate；
- Fairness：所有new modes同architecture、seed、data order与profile；old references只读且不重训；
- Boundary：shared-gradient cosine只解释失败发生在shared field的方向关系，不能单独建立或否定PCC；
- Falsification：pointwise/prior解释收益则PCC降级；arms恢复但A6不胜则回SC1 readout；arms不恢复则回intervention/
  shared-gradient Step4；numeric/protocol failure只返回Step5或修复，不作方向拒绝。

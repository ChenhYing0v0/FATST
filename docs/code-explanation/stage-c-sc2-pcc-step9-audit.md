# SC2-PCC Step9 Deep-Dive 代码说明

## Functional Module

`scripts/analyze_stage_c_sc2_pcc_step7b_deep_dive.py`只读取已完成的validation artifacts，不修改模型、不访问
test split，也不重新选择checkpoint。它补充Step7B frozen analyzer没有展开的objective attribution、horizon profile与
specialization diagnostics。

## Artifact Flow

```text
run_summary.csv + gate.json
  -> objective_scoreboard.csv
  -> mechanism_control_summary.csv
  -> deep_dive_gate.json

raw/*/metrics_by_target_horizon.csv
  -> horizon_gain_by_reference.csv
  -> horizon_bin_gain.csv
  -> horizon_gain_curves.svg/png

pcc_transport_full/*/training_log.csv
  -> pcc_training_diagnostics.csv
```

## Statistic Definitions

- `macro_gain_over_*_percent`：先在每个dataset计算
  $100(1-\mathrm{candidate}/\mathrm{reference})$，再对five datasets等权平均；
- `wins_over_*`：candidate的dense H1..720 MSE AUC严格低于reference的数据集数；
- `horizon_bin_gain`：先在冻结horizon bin内平均逐$H$ MSE/MAE，再计算相对收益；
- `pairwise_nrmse_retention_fraction`：某objective的minimum pairwise arm-output NRMSE除以同dataset plain
  PCSD DIRECT值；低值表示scope outputs趋同；
- `same_run_oracle_headroom_percent`：逐validation row/bin选择最低arm MSE相对learned fused MSE的收益，只作
  complementarity upper-bound diagnostic；
- `credit_argmax_accuracy`：training credit target与policy最大scope的一致率，不作为performance metric；
- `equal_skill_fraction_of_pcc_a6_gain`：equal-skill相对A6的macro gain除以full PCC相对A6的macro gain，用于估计
  generic direct arm supervision能解释的收益比例；它不是严格可加的因果分解。

## Interpretation Boundary

该脚本不改变pre-registered `gate.json`，不会因post-hoc horizon曲线或oracle headroom推翻formal method decision。
`deep_dive_gate.json`只用于failure attribution：区分arm recovery、arm specialization、router alignment与generic-control
explanation。任何下一candidate仍需回到11-step Step4-6重新通过narrative/theory gate。

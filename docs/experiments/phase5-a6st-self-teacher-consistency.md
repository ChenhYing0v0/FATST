# Phase5-A6ST Self-Teacher Consistency Gate

本文档记录 A6S2 之后的 Step 4/5 设计。目标是判断能否把 `EMA-0.999` 的 positive control
signal 转化为 training-time raw-final stability mechanism，而不是把 generic EMA final weights
直接包装成 paper-core。

## 11-Step Position

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10 completed：ETTh2 positive，但 cross-dataset safety gate 未通过；回 Step 4/5 |
| `problem` | A6-LBF 已恢复 dense-capacity path，但 raw final checkpoint 仍弱；A6S2 显示 EMA-0.999 final weights 有明显 control gain |
| `existence_evidence` | A6S2 `lbf_r256_ema0999` 相对 A6-LBF-r256 平均 MSE `-1.46%`，相对 ETTh2 best control `+0.67%`，wins `1/4` |
| `idea` | 使用 EMA teacher 的 prefix predictions 作为 online consistency target，训练 raw model 学到 trajectory-averaged behavior |
| `theory_check` | 不能在 test-time 直接加载 EMA weights；最终 evaluation 必须是 raw official-last weights。self-teacher loss 只作为训练期稳定化约束 |
| `design` | ETTh2-only minimal gate：固定 A6-LBF-r256，测试 self-teacher decay/weight/warmup |
| `narrative_gate` | conditional pass：若表述为 generic EMA/KD 则失败；若表述为 official-last-compatible raw-checkpoint stabilization for prefix-native operator，则可作为 diagnostic method candidate |
| `effectiveness_gate` | raw official-last MSE 是否接近 A6S2 EMA-0.999 control；是否改善 A6-LBF-r256 与 ETTh2 best controls gap；是否不增大 validation drift |
| `artifacts` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a6st_self_teacher_gate`; `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a6st_cross_dataset_sanity` |
| `decision` | failed_as_universal_method_etth2_specific_positive |

## Mechanism

[Fact] A6ST 维护一个 EMA teacher model。每个 training step：

1. raw student 产生 prefix output；
2. EMA teacher 在 `torch.no_grad()` 下产生同一个 prefix output；
3. 训练 loss 加入 `L1(student_prefix, teacher_prefix)`；
4. `optimizer.step()` 后用 student weights 更新 EMA teacher；
5. test 时保存并评估 raw student weights，不使用 EMA teacher weights。

[Decision] 这不同于 A6S2 `ema_eval` control：A6S2 在 test-time 替换权重；A6ST 只在 train-time
提供 consistency target。若 A6ST 成功，论文叙事可以写成 official-last-compatible raw-checkpoint
stabilization，而不是 checkpoint selector trick。

## Minimal Gate

| Variant | Role | Self-teacher decay | Loss weight | Warmup epochs |
| --- | --- | ---: | ---: | ---: |
| `a6st_w005_d0999_wu1` | weak consistency | 0.999 | 0.05 | 1 |
| `a6st_w01_d0999_wu1` | default consistency | 0.999 | 0.10 | 1 |
| `a6st_w02_d0999_wu1` | strong consistency | 0.999 | 0.20 | 1 |
| `a6st_w01_d0999_wu3` | delayed consistency | 0.999 | 0.10 | 3 |
| `a6st_w01_d0995_wu1` | shorter teacher memory | 0.995 | 0.10 | 1 |

## Gate Decision Rule

[Pass Candidate] A6ST 至少应明显优于 A6-LBF-r256，并接近或超过 A6S2 `ema0999` control；
若只略优于 base 或明显弱于 EMA-0.999 control，则它只是 weak regularizer。

[Fail Condition] 若所有 self-teacher variants 都没有接近 A6S2 EMA-0.999 control，或造成 validation
drift 更大，则停止 stability route，回 Step 2/3 重审 Stage A 是否还能作为 paper-core。

## ETTh2 Gate Result

[Fact] A6ST ETTh2-only gate 已完成。最佳 variant 为 `a6st_w02_d0999_wu1`，即
`self_teacher_loss_weight=0.20`、`self_teacher_decay=0.999`、`warmup=1`。

[Strong Evidence] 该 variant 相对 A6-LBF-r256 平均 MSE `-1.91%`，相对 ETTh2 best stage control
仅差 `+0.21%`，wins `2/4`。它也优于 A6S2 `lbf_r256_ema0999` control，后者为 `+0.67%`
vs best control、wins `1/4`。

[Strong Evidence] A6ST 同时降低 raw-model validation drift：best variant 的 last-vs-best validation
drift 为 `+3.86%`，而 A6S2 EMA-0.999 control 的 raw validation drift 仍为 `+9.85%`。
这支持“训练 raw final checkpoint”而不是“test-time EMA 替换”的机制解释。

[Decision] A6ST 标记为 `partial_pass_etth2_raw_final_stabilized`。下一步必须做 cross-dataset
sanity gate，检查该机制是否伤害 ETTm1/Weather。若 cross-dataset 不损害，再进入 full matrix 或
method refinement。

Result artifacts:

- `analysis/phase5_timealign_hss_a6st_self_teacher_gate_20260704/phase5_timealign_hss_a6s_stability_gate_report.md`
- `analysis/phase5_timealign_hss_a6st_self_teacher_gate_20260704/phase5_timealign_hss_a6s_summary.csv`

## Cross-Dataset Sanity Result

[Fact] 使用 ETTh2 最佳 setting `a6st_w02_d0999_wu1` 在 ETTm1/Weather 完成 safety gate。
所有 run 仍为 `official-last` / without early stop，且最终评估 raw student weights。

[Fact] ETTm1/Weather 合并后相对 best stage controls 平均 MSE `+1.20%`，wins `0/8`；
相对 A6-LBF-r256 为 `+0.95%`。

[Fact] 分数据集看，ETTm1 为 `+1.49%` vs best controls、wins `0/4`；Weather 为 `+0.91%`
vs best controls、wins `0/4`。这说明负向不是单一 horizon 造成。

[Strong Evidence] 将 ETTh2 正向与 ETTm1/Weather safety result 合并后，A6ST best 在三数据集
12 个 horizon 上为 `+0.87%` vs best controls、wins `2/12`，相对 A6-LBF-r256 约持平。

[Decision] 当前 A6ST 不能作为 universal method candidate。它保留的价值是：train-time
self-teacher consistency 确实可以修复 ETTh2 raw-final drift，但 uniform consistency objective
对 ETTm1/Weather 造成系统性小幅负向。

[Rollback] 回 Step 4/5。下一步必须先解释 dataset-conditioned stability 需求，并提出
selective/adaptive stability objective 或新的 capacity-preserving unified head；不得直接把
`a6st_w02_d0999_wu1` 扩展为 full matrix。

Result artifacts:

- `analysis/phase5_timealign_hss_a6st_cross_dataset_sanity_20260704/phase5_timealign_hss_a6s_stability_gate_report.md`
- `analysis/phase5_timealign_hss_a6st_cross_dataset_sanity_20260704/phase5_timealign_hss_a6s_dataset_summary.csv`

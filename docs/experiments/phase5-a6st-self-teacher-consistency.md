# Phase5-A6ST Self-Teacher Consistency Gate

本文档记录 A6S2 之后的 Step 4/5 设计。目标是判断能否把 `EMA-0.999` 的 positive control
signal 转化为 training-time raw-final stability mechanism，而不是把 generic EMA final weights
直接包装成 paper-core。

## 11-Step Position

| Field | Content |
| --- | --- |
| `current_step` | Step 4/5 -> Step 6/7/8：narrative/code-theory gate 通过后进入最小实现与 ETTh2 gate |
| `problem` | A6-LBF 已恢复 dense-capacity path，但 raw final checkpoint 仍弱；A6S2 显示 EMA-0.999 final weights 有明显 control gain |
| `existence_evidence` | A6S2 `lbf_r256_ema0999` 相对 A6-LBF-r256 平均 MSE `-1.46%`，相对 ETTh2 best control `+0.67%`，wins `1/4` |
| `idea` | 使用 EMA teacher 的 prefix predictions 作为 online consistency target，训练 raw model 学到 trajectory-averaged behavior |
| `theory_check` | 不能在 test-time 直接加载 EMA weights；最终 evaluation 必须是 raw official-last weights。self-teacher loss 只作为训练期稳定化约束 |
| `design` | ETTh2-only minimal gate：固定 A6-LBF-r256，测试 self-teacher decay/weight/warmup |
| `narrative_gate` | conditional pass：若表述为 generic EMA/KD 则失败；若表述为 official-last-compatible raw-checkpoint stabilization for prefix-native operator，则可作为 diagnostic method candidate |
| `effectiveness_gate` | raw official-last MSE 是否接近 A6S2 EMA-0.999 control；是否改善 A6-LBF-r256 与 ETTh2 best controls gap；是否不增大 validation drift |
| `artifacts` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a6st_self_teacher_gate` |
| `decision` | ready_for_minimal_remote_gate |

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

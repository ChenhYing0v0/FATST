# Phase5-A7DG Disagreement-Gated Self-Teacher

本文档记录 A6ST cross-dataset safety failure 后的 Step 4/5 回滚设计。A7DG 不直接延续
A6ST full matrix，而是先测试 selective stability objective 是否能保留 ETTh2 的 raw-final
stabilization，同时避免伤害 ETTm1/Weather。

## 11-Step Position

| Field | Content |
| --- | --- |
| `current_step` | Step 9/10 completed：remote gate partial positive，但未通过 paper-core effectiveness gate |
| `problem` | Uniform self-teacher consistency 修复 ETTh2 raw-final drift，但在 ETTm1/Weather 系统性负向 |
| `existence_evidence` | A6ST ETTh2：`-1.91%` vs A6-LBF-r256、`+0.21%` vs best controls、wins `2/4`；ETTm1/Weather：`+0.95%` vs A6-LBF-r256、`+1.20%` vs best controls、wins `0/8` |
| `idea` | 只在 student 与 EMA teacher 的 disagreement 足够高时施加 consistency；低 disagreement 时退化接近 A6-LBF |
| `theory_check` | self-teacher loss 仍是 train-time detached target；gate 只调节该 loss 的有效强度，不替换 checkpoint，不引入 validation-best selector |
| `design` | 在 A6-LBF-r256 上测试 absolute-gated 与 ratio-gated self-teacher；数据集为 ETTh2/ETTm1/Weather |
| `narrative_gate` | conditional pass：若写成 dataset hand-tuned threshold 则失败；若写成 disagreement-triggered final-checkpoint stabilization，并要求 gate 在低 drift 数据集自动降权，则可进入 diagnostic method gate |
| `effectiveness_gate` | ETTh2 不应丢失 A6ST 的主要 gain；ETTm1/Weather 相对 A6-LBF-r256 不应系统性变差；三数据集 combined 应优于 A6ST uniform setting |
| `artifacts` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a7dg_selective_self_teacher_gate` |
| `decision` | partial_positive_not_paper_core |

## Mechanism

[Fact] A7DG 在 A6ST 的 raw self-teacher loss 之外新增一个 detached gate：

$$
L = L_{pred} + \lambda \cdot g \cdot L_{st},
$$

其中 $L_{st}$ 是 student prefix prediction 与 EMA teacher prefix prediction 的 L1。

Absolute gate：

$$
g = \sigma((L_{st} - \tau) / T)
$$

Ratio gate：

$$
g = \sigma((L_{st} / L_{pred} - \tau) / T)
$$

[Decision] Gate 使用 detached signal，不让模型通过改变 gate signal 本身规避 loss。默认
`self_teacher_gate_mode=none`，旧 A6ST 行为不变。

## Minimal Gate

| Variant | Role | Gate | Threshold | Temperature | Self-teacher |
| --- | --- | --- | ---: | ---: | --- |
| `a7dg_abs004_t001_w02_d0999_wu1` | absolute disagreement trigger | absolute | 0.04 | 0.01 | `w=0.20,d=0.999,wu=1` |
| `a7dg_ratio008_t002_w02_d0999_wu1` | normalized disagreement trigger | ratio | 0.08 | 0.02 | `w=0.20,d=0.999,wu=1` |
| `a7dg_ratio010_t002_w02_d0999_wu1` | stricter normalized trigger | ratio | 0.10 | 0.02 | `w=0.20,d=0.999,wu=1` |

## Gate Decision Rule

[Pass Candidate] A7DG 必须同时满足：

- ETTh2 保留 A6ST 的主要 positive signal，至少明显优于 A6-LBF-r256；
- ETTm1/Weather 相对 A6-LBF-r256 不再系统性变差；
- `training_log.csv` 中的 `train_self_teacher_gate` 显示低 drift 数据集确实被降权，而不是阈值无效。

[Fail Condition] 若 ETTh2 gain 消失，或 ETTm1/Weather 仍系统性负向，则 selective self-teacher route
停止，回 Step 4/5 寻找新的 capacity-preserving unified head，而不是继续调 threshold。

## Implementation Notes

- Training script: `baselines/timealign_official/train_repo.py`
- Remote wrapper: `scripts/remote/run_phase5_timealign_hss_a7dg_selective_self_teacher_gate.sh`
- Analyzer: `scripts/analyze_phase5_timealign_hss_a6s_stability_gate.py`

New training-log columns:

- `self_teacher_gate_mode`
- `self_teacher_gate_threshold`
- `self_teacher_gate_temperature`
- `train_self_teacher_gate`
- `train_weighted_self_teacher_l1`

## Local Verification

[Fact] 本地验证已通过：

- `python -m py_compile baselines/timealign_official/train_repo.py scripts/analyze_phase5_timealign_hss_a6s_stability_gate.py`
- `bash -n scripts/remote/run_phase5_timealign_hss_a7dg_selective_self_teacher_gate.sh`
- CPU smoke：`self_teacher_gate_mode=ratio`，`max_train_batches=1`，`max_eval_batches=1`

[Fact] Smoke 的 `training_log.csv` 写出：

- `train_self_teacher_l1=0.015721`
- `train_self_teacher_gate=0.081953`
- `train_weighted_self_teacher_l1=0.001288`

[Decision] A7DG 已进入 `ready_for_remote_gate`。下一步按 remote policy commit/push 后启动
ETTh2/ETTm1/Weather selective gate。

## Remote Gate Result

[Fact] A7DG remote gate 已完成，覆盖 ETTh2/ETTm1/Weather × 3 variants。最佳 variant 为
`a7dg_abs004_t001_w02_d0999_wu1`。

[Strong Evidence] 相对 uniform A6ST，A7DG best 平均 MSE `-0.40%`，`11/12` horizons 更好。
分数据集看，ETTh2 `-0.05%`、ETTm1 `-0.53%`、Weather `-0.62%`。

[Strong Evidence] Gate 的 dataset separation 成立：best variant 的 `train_self_teacher_gate`
在 ETTh2 为 `0.88`，ETTm1 为 `0.31`，Weather 为 `0.22`。这支持 selective stability
objective 的方向，而不是 uniform consistency。

[Fact] 但 A7DG best 相对 best controls 仍 `+0.46%`、wins `2/12`；ETTm1 相对 A6-LBF-r256
仍 `+0.51%`，Weather 仍 `+0.23%`。

[Decision] A7DG 标记为 `partial_positive_not_paper_core`。它保留为 selective stability evidence，
但当前 threshold-gated implementation 不足以成为 paper-core method。

[Rollback] 回 Step 4/5。下一步不能继续做简单 threshold sweep；需要提出理论边界更强的
adaptive/selective objective，或转向新的 capacity-preserving unified head。

Result artifacts:

- `analysis/phase5_timealign_hss_a7dg_selective_self_teacher_gate_20260704/phase5_timealign_hss_a6s_stability_gate_report.md`
- `analysis/phase5_timealign_hss_a7dg_selective_self_teacher_gate_20260704/phase5_timealign_hss_a7dg_vs_uniform_a6st.csv`

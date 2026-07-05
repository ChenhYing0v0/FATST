# Phase5-A8TAG Teacher-Advantage-Gated Self-Teacher

本文档记录 A7DG partial-positive 后的 Step 4/5 回滚设计。A8TAG 不是继续调
disagreement threshold，而是用 supervised teacher advantage 决定 EMA teacher 是否有资格
作为 consistency target。

## 11-Step Position

| Field | Content |
| --- | --- |
| `current_step` | Step 4/5 -> Step 6/7/8：narrative/code-theory gate 后进入本地验证与远程 gate |
| `problem` | A7DG 证明 selective stability 比 uniform self-teacher 更合理，但 threshold-gated implementation 仍弱于 best controls，且 ETTm1/Weather 仍弱于 A6-LBF |
| `existence_evidence` | A7DG best 相对 uniform A6ST `-0.40%`、`11/12` wins，且 gate 在 ETTh2/ETTm1/Weather 为 `0.88/0.31/0.22`；但相对 best controls `+0.46%`、wins `2/12` |
| `idea` | 只有当 EMA teacher 在当前 supervised prefix 上比 raw student 更接近 label 时，才施加 self-teacher consistency |
| `theory_check` | Gate 边界来自 empirical risk comparison：`student_pred_loss - teacher_pred_loss`，不是 dataset threshold；teacher 不优于 student 时不应蒸馏 |
| `design` | 在 A6-LBF-r256 上测试 teacher-advantage binary gate 与 relative-advantage gate；数据集为 ETTh2/ETTm1/Weather |
| `narrative_gate` | conditional pass：若 teacher advantage 能解释何时该蒸馏，A8TAG 有比 A7DG 更强的机制边界；若只是调 loss weight 或 teacher 多数无 advantage，则失败 |
| `effectiveness_gate` | 必须优于 A7DG 或至少修复 ETTm1/Weather 相对 A6-LBF 的剩余负向，同时保留 ETTh2 positive signal |
| `artifacts` | `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_a8tag_teacher_advantage_gate` |
| `decision` | ready_for_remote_gate |

## Mechanism

设 student prefix prediction loss 为 $L_s$，EMA teacher prefix prediction loss 为 $L_t$，
student-teacher consistency loss 为 $L_{st}$。

Binary teacher-advantage gate：

$$
g = \mathbb{1}[L_t < L_s]
$$

Relative teacher-advantage gate：

$$
g = \mathrm{clip}((L_s - L_t) / L_s, 0, 1)
$$

最终优化：

$$
L = L_s + \lambda \cdot g \cdot L_{st}
$$

[Decision] $g$ 使用 detached supervised loss comparison，不让模型通过修改 gate signal 规避 loss。
默认 `self_teacher_gate_mode=none` 时旧行为不变。

## Minimal Gate

| Variant | Gate | Loss weight | Decay | Warmup | Role |
| --- | --- | ---: | ---: | ---: | --- |
| `a8tag_advbin_w02_d0999_wu1` | teacher-advantage-binary | 0.20 | 0.999 | 1 | 与 A6ST/A7DG 权重对齐的 label-grounded gate |
| `a8tag_advbin_w05_d0999_wu1` | teacher-advantage-binary | 0.50 | 0.999 | 1 | 检查 binary gate 是否需要更强 consistency |
| `a8tag_advratio_w10_d0999_wu1` | teacher-advantage-ratio | 1.00 | 0.999 | 1 | 用连续 advantage ratio 代替 hard gate |

## Gate Decision Rule

[Pass Candidate] A8TAG 必须满足：

- `train_self_teacher_advantage_l1` 在正向 dataset/horizon 上为正，且 gate 与 advantage 一致；
- 相对 A7DG best 有进一步改善，或至少把 ETTm1/Weather 拉回接近 A6-LBF；
- ETTh2 不丢失 A6ST/A7DG 的主要 positive signal。

[Fail Condition] 若 teacher advantage 多数为负或接近零，说明 EMA teacher 本身不是可靠 target；
若 metrics 仍只表现为小幅 regularization gain，则停止 self-teacher route，回 Step 4/5 寻找新的
capacity-preserving unified head。

## Implementation Notes

- Training script: `baselines/timealign_official/train_repo.py`
- Remote wrapper: `scripts/remote/run_phase5_timealign_hss_a8tag_teacher_advantage_gate.sh`
- Analyzer: `scripts/analyze_phase5_timealign_hss_a6s_stability_gate.py`

New / reused training-log columns:

- `train_self_teacher_target_l1`
- `train_self_teacher_advantage_l1`
- `train_self_teacher_gate`
- `train_weighted_self_teacher_l1`

## Local Verification

[Fact] 本地验证已通过：

- `python -m py_compile baselines/timealign_official/train_repo.py scripts/analyze_phase5_timealign_hss_a6s_stability_gate.py`
- `bash -n scripts/remote/run_phase5_timealign_hss_a8tag_teacher_advantage_gate.sh`
- CPU smoke：`self_teacher_gate_mode=teacher-advantage-binary`，`max_train_batches=1`，`max_eval_batches=1`

[Fact] Smoke 的 `training_log.csv` 写出：

- `train_prediction_l1=0.496292`
- `train_self_teacher_target_l1=0.496732`
- `train_self_teacher_advantage_l1=-0.00044`
- `train_self_teacher_gate=0.0`
- `train_weighted_self_teacher_l1=0.0`

[Decision] Smoke 表明当 teacher 不优于 student 时，A8TAG 确实关闭 self-teacher loss。该候选进入
`ready_for_remote_gate`。

## Remote Gate Result

[Fact] Remote gate 已完成，覆盖 ETTh2/ETTm1/Weather × 3 variants。最佳 variant 是
`a8tag_advratio_w10_d0999_wu1`，12 个 dataset-horizon setting 的 mean MSE 为
`0.285779`，相对 best stage control `+0.91%`，wins `0/12`。

[Fact] A8TAG 最佳 variant 相对 A6-LBF-r256 为 `+0.03%`，相对 A7DG best 变差约
`+0.47%`。Binary gate 在 ETTm1/Weather 高激活但效果更差，ratio gate 接近关闭并基本退化
到 A6-LBF。

[Decision] A8TAG 标记为 `failed_as_core_candidate`。teacher-advantage 是一个清晰但被证伪的
gate：它不能解释 ETTh2 上 useful self-teacher 的来源，也不能提供跨数据集收益。下一步回
Step 4/5 回溯未执行候选，不继续做 teacher-advantage threshold 或 weight sweep。

[Artifacts] `analysis/phase5_timealign_hss_a8tag_teacher_advantage_gate_20260705/phase5_timealign_hss_a8tag_interpretation.md`

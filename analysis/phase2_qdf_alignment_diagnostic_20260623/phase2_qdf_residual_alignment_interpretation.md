# Phase2-E2 QDF-to-FATST Residual Alignment Interpretation

更新时间：2026-06-24

## 1. 结论

[Decision] Phase2-E2 不支持继续把 QDF learned `off_diag/all` precision 直接本地化为
FATST R.3 的 objective repair。

原因不是 artifact 不完整。当前 gate 已满足：

- prediction artifacts complete: `True`;
- QDF matrices complete: `True`;
- ready for alignment decision: `True`;
- missing artifacts: `0`。

关键反证是：QDF learned matrices 没有对 R.3 specialist gaps 施加更高 pressure。

| Matrix family | Mean ratio | Specialist ratio | Non-specialist ratio | Gap / non-gap |
| --- | ---: | ---: | ---: | ---: |
| `prefix_risk` | `1.460518` | `1.528278` | `1.426638` | `1.071244` |
| `static_train_target_offdiag` | `0.156607` | `0.168521` | `0.150650` | `1.118626` |
| `qdf_off_diag_precision` | `0.531174` | `0.481565` | `0.555978` | `0.866158` |
| `qdf_all_precision` | `0.545758` | `0.502381` | `0.567446` | `0.885338` |
| `qdf_diag_precision` | `0.998441` | `0.999521` | `0.997902` | `1.001622` |

[Interpretation] 若 learned QDF precision 是 FATST R.3 specialist-gap repair 的好目标，它应至少
对 gap settings 给出更强或更可区分的 objective pressure。实际相反：`qdf_off_diag_precision`
和 `qdf_all_precision` 的 gap / non-gap ratio 都小于 `1`，说明它们在当前 R.3 residual 上没有
把 specialist gaps 标为更需要修复的方向。

## 2. Specialist Gap Settings

R.3 相对 fixed head 的 MSE gap settings 是：

- `ETTh2 / 720`;
- `ETTm1 / 96`;
- `ETTm1 / 720`;
- `Weather / 96`。

这些 gap 横跨 short horizon 和 long horizon。它们不像一个简单 late-horizon tail problem；
更像 prefix-consistent shared trajectory 与 horizon-specialist prediction 之间的 tradeoff。

## 3. 对 QDF 借鉴点的重新界定

[Fact] QDF upstream reproduction 仍然说明 `diag` 不是最优，future-step interaction 在 native QDF
体系内有效。

[Counter-Evidence] 但 Phase2-E1 和 Phase2-E2 连续否定了两种本地 objective route：

1. static train-target covariance/offdiag proxy：训练 gate 不通过；
2. QDF learned precision direct transfer：residual alignment 不支持 specialist-gap repair。

[Decision] QDF 可以继续作为“future steps 不应等权、residual interaction 可能重要”的背景证据，
但不应再作为下一步 FATST objective 的直接设计来源。

## 4. 11-Step Loop Decision

- `current_step`: Step 9-10。
- `problem`: R.3 仍有 specialist gaps，尤其是 short/long horizon 两端。
- `existence_evidence`: Phase1 R.3 vs fixed head 显示 `4/12` settings 仍输 fixed；
  Phase2-E2 residual artifacts 完整确认这些 gaps 存在。
- `idea`: 用 QDF learned `off_diag/all` precision 作为 local objective design evidence。
- `theory_check`: 若 QDF learned precision 能解释 FATST residual，它应对 R.3 hard settings 有更强
  alignment。
- `design`: 在同一批 R.3 residual 上比较 `identity/prefix_risk/static/QDF` matrix losses。
- `gate`: artifact complete，且 QDF learned matrices 对 specialist gaps 有正向区分。
- `artifacts`: `phase2_qdf_residual_alignment_losses.csv`,
  `phase2_qdf_residual_alignment_summary.json`。
- `decision`: fail；回滚到 Step 2-3，重新定义问题为 prefix-consistency 与 specialist-performance
  tradeoff，而不是继续 objective-matrix route。

## 5. 下一步实验方向

[Decision] 下一步进入 Phase3-A：Prefix-Consistent Carrier vs Horizon Specialist Tradeoff Diagnostic。

核心问题：

> R.3 的 prefix-consistent shared trajectory 是否天然牺牲了某些 horizon-specialist settings？

推荐先做诊断，不先上 MoE：

1. 使用已回传的 R.3 `predictions_test.npz`。
2. 对 `h96/h192/h336` 与 `h720` prefix 计算 residual difference、error covariance、step-region
   energy shift。
3. 重点比较四个 specialist gaps 与八个 non-gaps：
   - gap settings 是否集中在 short prefix 或 long tail；
   - h96 residual 是否与 h720 prefix residual 方向冲突；
   - prefix consistency 是否压制了 short-horizon calibration。
4. 若 tradeoff 成立，再设计最小 horizon-regime residual calibration：
   - 保留 R.3 base prediction；
   - 只加 low-rank residual calibration；
   - gate 以四个 specialist gaps 为主；
   - 不能大幅牺牲 non-gap 和 mean MSE。

[Rollback] 若 Phase3-A 不能证明 tradeoff 存在，则停止 R.3 repair route，回到更早的 base
architecture / external baseline selection。

# Phase2-E1 Off-Diagonal Block Quadratic Interpretation

更新时间：2026-06-23

## 1. 结论

[Decision] `PatchEncoderOffdiagBlockQuadratic` 不通过 Phase2-E1 gate，不能作为当前
FATST paper-core candidate。

核心原因不是训练崩溃，也不是 prefix consistency 被破坏，而是相对 R.3 的收益不存在：

- MSE wins vs R.3: `1/12`;
- MAE wins vs R.3: `0/12`;
- mean relative MSE vs R.3: `+0.0464%`;
- specialist gap wins vs R.3: `1`;
- H720 stability region wins vs R.3: `0`;
- max prefix mismatch MSE: `5.4052272504883855e-14`。

[Fact] 该候选仍明显优于更弱的 baselines：

- MSE wins vs uniform target-set: `10/12`;
- mean relative MSE vs uniform target-set: `-0.9871%`;
- MSE wins vs FixedHead: `8/12`;
- mean relative MSE vs FixedHead: `-0.3870%`。

[Interpretation] 这说明 `offdiag_block_quadratic` 是一个安全但无效的 R.3 增量。它没有
否定“future steps 不应等权”或“future-step interaction 有价值”，但否定了当前这个
static train-target block precision proxy 足以超过 R.3 的假设。

## 2. 与 QDF evidence 的关系

[Fact] QDF upstream gate 中 `all` 相对 `diag` 有 `11/12` MSE wins，mean relative MSE
为 `-1.08%`；同时 `off_diag` 是 `10/12` 个 setting 的 best meta type。

[Inference] QDF 的可借鉴点仍然成立：diagonal-only weighting 不足，future-step residual
interaction 值得研究。但 Phase2-E1 的失败说明，本地化时只用 train-split target covariance
构造一个 frozen off-diagonal block penalty，不能复现 QDF 的有效机制。

可能差异包括：

1. QDF 学到的是 error/residual-aware objective matrix，而 Phase2-E1 用的是 target-only
   static covariance proxy。
2. QDF 的 matrix 与模型训练动态共同形成，Phase2-E1 的 matrix 在训练前固定。
3. Phase2-E1 只把 off-diagonal signal 当作附加正则项，主 objective 仍由 R.3 prefix-risk
   pressure 主导。

## 3. 11-step loop 判断

- `current_step`: Step 9-10，远程结果评估与候选判定。
- `problem`: R.3 的 diagonal/prefix objective 仍缺少 explicit future-step residual
  interaction。
- `existence_evidence`: QDF upstream controls 支持该问题存在；Phase2-E1 本地候选未能转化为
  R.3 上的收益。
- `idea`: static off-diagonal block quadratic objective。
- `theory_check`: 该 proxy 只在 target covariance 层面近似 residual interaction，理论上弱于
  QDF learned objective。
- `design`: 保留 R.3 base loss，增加 block residual projection squared penalty。
- `gate`: mean MSE vs R.3 `< -0.3%`、MSE wins `>=7/12`、specialist gap wins
  `>=2/4`、prefix consistency pass。
- `artifacts`: 本目录下的 `phase2_offdiag_block_quadratic_*` CSV/JSON/report files。
- `decision`: fail；回滚到 Step 5-6，重新验证 QDF matrix 与 FATST residual/error 的对齐性。

## 4. 下一步实验方向

[Decision] 下一步不应做 `offdiag_quadratic_weight`、`block_size`、`ridge_eps` 的大 sweep。
这种调参只能验证当前 proxy 的局部数值敏感性，不能回答 QDF 为什么有效。

建议进入 Phase2-E2：QDF-to-FATST residual/loss alignment diagnostic。

目标：

1. 在同一批 FATST R.3 residual 上比较三类 objective matrix：
   `identity/prefix-risk`、`static train-target offdiag`、`QDF learned off_diag/all`。
2. 检查 QDF learned matrix 是否对 R.3 的真实 error direction 给出更合理的 loss ranking 或
   gradient pressure。
3. 若 learned matrix 与 R.3 residual 对齐，再实现 learnable/validation-informed local
   objective；若不对齐，停止 objective route，回到 base architecture 或 baseline story。

[Blocker] 当前本地 R.3 gate artifact 只确认存在 `metrics_by_target_horizon.csv`，没有发现
`predictions_test.npz`。因此 Phase2-E2 若要做 residual-level offline diagnostic，需要先补充
保存 prediction/true artifacts，或者补跑一个只保存预测的 R.3 diagnostic run。

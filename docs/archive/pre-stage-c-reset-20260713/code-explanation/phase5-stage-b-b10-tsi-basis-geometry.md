# Phase5 StageB B10-TSI-A Basis Geometry Diagnostic Code Explanation

## 诊断位置

| 字段 | 内容 |
| --- | --- |
| `script` | `scripts/analyze_phase5_stage_b_b10_tsi_basis_geometry.py` |
| `candidate_id` | `B10-TCO` |
| `diagnostic_id` | `B10-TSI-A` |
| `current_step` | StageB Step 3：problem-existence diagnostic |
| `input` | clean A6 checkpoint 中的 `learned_temporal_basis: [720, 256]` |
| `output` | `analysis/phase5_stage_b_b10_tsi_basis_geometry_20260708/` |
| `scope` | checkpoint-only geometry audit；不训练新模型，不读取 labels，不评估 MSE |

## Reader Path

本诊断回答一个窄问题：

> A6 的 `learned_temporal_basis` 是否已经把 future-stage / target-position 信息表达得足够充分，以至于
> B10 再引入 target-set-conditioned interface 只是额外 capacity？

这不是 B10 方法实现，也不是效果实验。它只决定 B10 是否可以继续进入更深的 target-set interface diagnostic。

## Tensor And Statistic Definitions

脚本读取：

```text
learned_temporal_basis = B              # [720, 256]
```

并按当前 benchmark horizons 拆成四个 future segments：

```text
early = B[0:96]                         # [96, 256]
mid   = B[96:192]                       # [96, 256]
late  = B[192:336]                      # [144, 256]
tail  = B[336:720]                      # [384, 256]
```

### `segment_effective_rank`

对每个 segment basis 子矩阵做 SVD：

```text
B_s = U_s S_s V_s^T
```

`effective_rank_95` 是累计奇异值平方能量达到 `95%` 时需要的 rank。它衡量该 segment 的 temporal basis
几何复杂度。tail segment 更长，rank 更高是预期现象；关键不是绝对 rank，而是不同 segment 是否只依赖很少
共同方向。

### `atom_stage_share`

对每个 atom/column `k`，计算其在四个 stage 上的能量占比：

```text
energy_s,k = sum_{t in stage_s} B[t,k]^2
share_s,k = energy_s,k / sum_s energy_s,k
```

脚本输出：

- `normalized_entropy`: 四个 `share_s,k` 的归一化 entropy，越接近 `1` 表示 atom 越跨 stage；
- `max_stage_share`: 最大 stage 能量占比，越接近 `1` 表示 atom 越局部化；
- `top64_atom_stage_specialized_rate_0p70`: top64 高能 atom 中 `max_stage_share >= 0.70` 的比例。

### `row_space_overlap`

对每个 segment 的 row space 做比较。因为不同 segment 的时间长度不同，不能直接比较 time-axis rows；
脚本比较的是它们在 coefficient 维度 `$K=256$` 上的 row subspace：

```text
B_s = U_s S_s V_s^T
Q_s = V_s[:rank]^T                       # [256, rank]
overlap(s1, s2) = mean(svd(Q_s1^T Q_s2)^2)
```

该值越高，表示不同 stage 使用相似的 coefficient axes；越低，表示 basis 已经形成较分离的
stage-specific coefficient geometry。

## Current Result

当前 B10-TSI-A 的主要结果是：

| Dataset | top64 atom entropy | stage-specialized rate | rank32 row-space overlap |
| --- | ---: | ---: | ---: |
| ETTh2 | `0.8108` | `0.0156` | `0.1324` |
| ETTm1 | `0.8764` | `0.0000` | `0.1510` |
| Weather | `0.8658` | `0.0000` | `0.1368` |

[Fact] 高能 temporal atoms 并没有强烈局部化到单一 stage。

[Fact] 但不同 stage 的 coefficient row spaces overlap 不高，说明 basis 已经携带明显的 stage-differentiated
coefficient geometry。

## Code-Theory Consistency

[Intended theory] 如果 basis 完全不含 stage 信息，则 B10 可以直接叙事为“补充 future-stage information”。

[Code realization] B10-TSI-A 检查实际 checkpoint 中的 basis geometry，而不是只从 architecture 形式推断。

[Observed boundary] 结果不支持“basis stage-blind”叙事。A6 的 basis 已经有 stage geometry；真正仍未解决的是：

```text
hidden = encoder(history)
coeff = learned_basis_coeff(hidden)       # [B, C, 256], target-set-blind
y_s = B_s @ coeff                         # every segment reads a different row subspace
```

也就是说，同一个 `coeff` 必须同时服务多个 stage row subspaces，而 requested target set 没有进入
`history -> coeff/state` 生成路径。

## Next Diagnostic Constraint

B10 的下一步不能回到 B9 式 post-hoc stage modulation。下一步应推进 `B10-TSI-B`：

1. 在真实 forward batch 中读取 `coeff: [B, C, 256]`；
2. 计算每个 stage row subspace 对 `coeff` 的 projection / contribution energy；
3. 判断不同 requested target set 是否需要不同的 history readout 或 coefficient state；
4. 必须设置 no-target-set capacity control，避免重复 B9 被 no-stage control 阻断的问题。

只有当 B10-TSI-B 显示 target-set-aware history readout 的问题真实存在，B10 才能进入 Step 4-6 method design。

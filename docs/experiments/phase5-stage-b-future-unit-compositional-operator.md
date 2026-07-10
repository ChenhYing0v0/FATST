# Phase5 StageB B13 Future-Unit Compositional Operator

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B13-FUCO` |
| `current_step` | Step 2/3：future-unit problem definition and intervention-point diagnostic |
| `problem` | A6-LBF-r256 用一个 global coefficient state 与 full-horizon temporal basis 表示同一条 future trajectory；multi-horizon 能力主要来自 prefix restriction，而不是可组合、可提前停止的 future generation |
| `existence_evidence` | B9 显示 canonical stages 对共享 coefficient 的 gradients 低相似；B11 显示 future basis subspace geometry 连续变化；B12 显示过小 local unit/rank 会形成容量瓶颈，但高-rank local factorization 可接近 A6 |
| `idea` | 将 future 表示为 benchmark-independent latent future units；requested horizon 决定生成多少 units，unit states 通过 prefix-causal composition 形成，而不是对 full-horizon representation 做 clipping |
| `theory_check` | 若较大的非 benchmark unit sizes 上仍存在稳定的 shared-state gradient pressure 与 adjacent/far geometry，则 future-unit generation 有 Step 2/3 问题基础；若信号只在 canonical boundaries 或小 units 上成立，则该问题可能是 segmentation artifact |
| `design` | `B13-FUCO-A` 已确认 large-unit granularity robustness；`B13-FUCO-B1` coefficient-memory control 未通过；当前执行一次 pre-coefficient hidden-memory repair probe，不实现 model |
| `narrative_gate` | pending targeted literature audit and Diagnostic A/B evidence |
| `effectiveness_gate` | not applicable before Step 4-6 |
| `artifacts` | `analysis/phase5_stage_b_b13_future_unit_granularity_20260710/`; `analysis/phase5_stage_b_b13_future_unit_composition_20260710/`; `scripts/analyze_phase5_stage_b_b13_future_unit_granularity.py`; `scripts/analyze_phase5_stage_b_b13_future_unit_composition_probe.py` |
| `decision` | `partial_pass`：problem evidence 通过，但 coefficient-memory transition 被 no-transition control 解释；等待 hidden-memory intervention repair |

## Step 2：Problem Definition

Clean A6 的 primary prediction path 是：

```text
history -> hidden [B,C,R]
        -> coeff [B,C,K]
basis[:H] [H,K] @ coeff
        -> prediction [B,H,C]
```

A6 是 prefix-compatible unified trajectory operator，但内部不存在显式的 future-state evolution：

- sample/channel-specific information 被压缩为一个 global `coeff[b,c]`；
- future position 仅通过 `learned_temporal_basis[t]` 进入 readout；
- requested horizon `H` 只限制所使用的 basis rows；
- short/long forecasts 不对应不同数量的 latent future computation units。

B13-FUCO 的问题不是“怎样切分 A6 basis”，而是：

> 一个 unified model 是否应通过组合不同数量的 latent future units 生成不同长度的 forecast，
> 而不是从同一个 full-horizon trajectory representation 中返回 prefix？

## Candidate Mechanism Boundary

目标机制暂定为：

```text
memory = Encoder(history)
state_m = FutureUnitGenerator(memory, state_<m, coordinate_m)
segment_m = SharedUnitDecoder(state_m, local_coordinate)
prediction_H = concatenate(segment_0, ..., segment_{ceil(H/U)-1})[:H]
```

必要约束：

1. unit index/coordinate 不能是 `{96,192,336,720}` 的 hard stage id；
2. later units 不能改写 earlier units，保证 prefix invariance；
3. unit composition 发生在 latent state path，不反馈已预测数值，避免 autoregressive error accumulation；
4. requested horizon 决定实例化多少 units，而不是只裁剪已经生成的 full-horizon state；
5. method 不能写成 `A6 + residual correction`；
6. composed-unit candidate 必须超过 parameter-matched no-transition / independent-unit controls。

## Why B13 Is Not B9 Or B12

### B9-FSN-SCF

B9 使用 canonical stages `[0,96), [96,192), [192,336), [336,720)`，从同一个 base coefficient
通过 hard stage tokens 生成 coefficient field。不同 stage states 没有 generative dependency，且 no-stage
control 完全解释了收益。

B13 的 units 必须与 benchmark horizons 解耦，并形成 prefix-causal latent composition；因此 B9 只能作为
shared-state pressure 的历史证据，不能作为 B13 implementation template。

### B12-STBO

B12 用一个 Linear 同时生成所有 tile coefficients，tile basis 是 static shared/bank/independent 参数，
requested horizon 只决定保留多少 tile outputs。它检验的是 blockwise low-rank factorization，不是 future-unit
generation。

B12 的有效经验是：`L48-R16/R32` 等过小 local capacity 会伤害 A6 carrier，而 `L360-R256` 可以接近 A6。
因此 B13 的 problem diagnostic 不再使用 `24/48/96` 作为主 unit sizes。

## Step 3 Diagnostic A：Large-Unit Granularity Stability

### Unit Sizes

Diagnostic A 使用能够整除 `pred_len=720`、但不等于 benchmark horizons 的较大 unit sizes：

| Unit size | Unit count | Role |
| ---: | ---: | --- |
| `120` | `6` | main；保留足够 units 做 adjacent/far 检验 |
| `144` | `5` | main；与 benchmark horizons 解耦 |
| `180` | `4` | main；中等 coarse composition |
| `240` | `3` | main；较大信息承载单元 |
| `360` | `2` | coarse control；只检查 first/last pressure，不进入主 gate |

选择原则：

- unit size 足够大，避免重复 B12 的 small-unit capacity confound；
- main sizes 仍保留 `3-6` 个 units，能够区分 adjacent 与 far pairs；
- 不使用 canonical horizon boundaries；
- `360` 只有两个 units，不能提供充分的 locality/continuity 证据，只作为 coarse control。

### Gradient Pressure

对每个 unit $m$：

$$
\mathcal{L}_m = \operatorname{MSE}(\hat y_{mU:(m+1)U}, y_{mU:(m+1)U}),
\qquad
g_m = \frac{\partial \mathcal{L}_m}{\partial \mathrm{coeff}}.
$$

统计：

- all-pair mean/min gradient cosine；
- adjacent-pair 与 far-pair cosine；
- first-last cosine；
- negative pair rate；
- max/min gradient norm ratio；
- normalized shared-gradient alignment efficiency：

$$
\eta_{\mathrm{shared}}
=
\frac{\left\|\sum_m g_m\right\|_2^2}
{M\sum_m \left\|g_m\right\|_2^2}.
$$

当所有 unit gradients 完全同向且等范数时，$\eta_{\mathrm{shared}}=1$；当 gradients 近似正交时，
$\eta_{\mathrm{shared}}\approx 1/M$。它用于描述 shared state 的方向复用程度，不直接证明某个新架构有效。

### Basis Geometry Control

对每个 A6 basis unit `basis[mU:(m+1)U]` 计算 rank-32 row subspace，统计：

- adjacent/far subspace overlap；
- distance vs overlap Spearman；
- pairwise gradient cosine 与 basis overlap 的 Spearman。

该 control 用于防止把 A6 basis rows 自身的近正交 geometry 误写为“future-unit generator 必然需要”。
Diagnostic A 即使通过，也只能形成 problem evidence；若 gradient pattern 高度由 basis overlap 解释，下一步必须
在 Diagnostic B 中加入 no-transition / geometry-matched controls。

### Bootstrap

对 batch-level statistics 做 deterministic bootstrap，默认 `1000` iterations，报告：

- mean pairwise cosine 的 `p05/p50/p95`；
- first-last cosine 的 `p05/p50/p95`；
- adjacent-minus-far cosine gap 的 `p05/p50/p95`；
- shared alignment efficiency 的 `p05/p50/p95`。

## Diagnostic A Gate

Main unit size 的单个 dataset/size 记为 robust support，当：

1. bootstrap `p95(mean_pairwise_cosine) < 0.50`；
2. bootstrap `p95(first_last_cosine) < 0.35`。

若至少两个 datasets 各自在 `4` 个 main sizes 中有至少 `3` 个 robust support，则 Diagnostic A 判为：

```text
partial_pass_large_unit_granularity_robust
```

这只允许进入 Diagnostic B，不允许进入 model implementation。

若信号只在一个 dataset、单个 unit size 或 `U=360` coarse control 上成立，则判为：

```text
granularity_or_dataset_specific
```

若较大 units 上普遍变为高 cosine，则判为：

```text
large_unit_problem_not_supported
```

并回滚到 Step 2，停止 future-unit generation architecture。

## Failure Attribution Boundary

- `hypothesis_false`：只有较大 unit sizes 上跨数据集信号消失，且 basis/control 不能解释时才考虑；
- `basis_geometry_confounded`：gradient pressure 与 basis overlap 高度一致；不能否定 future generation；
- `granularity_specific`：只在某个 unit size 成立；不能进入 method；
- `capacity_control_explains`：留给 Diagnostic B 的 parameter-matched controls；
- `direction_level_rejection`：Diagnostic A 不具备单独否定所有 future-unit architecture 的权限。

## Next Gate If Diagnostic A Passes

Diagnostic B 才比较：

```text
shared global state
independent unit states
prefix-causal composed unit states
no-transition capacity control
```

所有 arms 必须 parameter-matched，并使用不依赖 benchmark horizon IDs 的 unit coordinates。只有 composed
unit states 稳定超过 no-transition/independent controls，B13 才能进入 Step 4-6 narrative/method design。

## Diagnostic A Result

`B13-FUCO-A` 已在 clean A6 checkpoint 上完成。完整 artifacts：

- `analysis/phase5_stage_b_b13_future_unit_granularity_20260710/`；
- `analysis/phase5_stage_b_b13_future_unit_granularity_20260710/b13_future_unit_granularity_report.md`。

Gate 结果：

| Dataset | Main sizes passed | Mean cosine range | First-last range |
| --- | ---: | ---: | ---: |
| ETTh2 | `4/4` | `0.122-0.150` | `0.023-0.059` |
| ETTm1 | `4/4` | `0.152-0.230` | `0.060-0.187` |
| Weather | `4/4` | `0.064-0.084` | `0.026-0.034` |

[Strong Evidence] `120/144/180/240` 四个较大、非 benchmark unit sizes 在三个 datasets 上全部通过
pre-registered bootstrap gate。所有 `12/12` main settings 的 adjacent gradient cosine 均高于 far
cosine，而 A6 basis adjacent/far overlap 基本持平；只有 `1/12` settings 的 gradient-vs-basis pair
Spearman 达到 `0.75`。

[Decision] `partial_pass_large_unit_granularity_robust`。B9 的 shared-state pressure 不是 canonical
horizon boundary 或 small-unit artifact。该结果允许进入 Diagnostic B，但仍不允许实现 paper-core model。

## Step 3 Diagnostic B：Prefix-Causal Composition Control

### Question

Diagnostic B 回答比 gradient conflict 更接近机制的问题：

> 在相同 frozen history memory、相同参数量、相同 unit decoder 与相同 continuous coordinate path 下，
> 让 unit state 读取 previous latent unit，是否稳定优于每个 unit 独立从 history 生成 state？

### Frozen Memory

固定 clean A6 encoder 与 `learned_basis_coeff`，从 train/val/test splits 提取：

```text
memory = coeff: [N,C,256]
target_norm: [N,C,720]
```

`target_norm` 使用每个 sample/channel 的 history mean/std 做 A6-compatible normalization。encoder 不更新；
probe 结果不能写成 end-to-end forecasting performance。选择 `coeff` 而不是高维 encoder hidden，是因为
Diagnostic B 要直接测试“从同一个 A6 global coefficient 生成 independent 或 composed future-unit states”；
同时避免 ETTh2/ETTm1/Weather 不同 readout dimension 带来的参数不公平。

### Unit Sizes

Diagnostic B 使用：

- `U=180`：四个 units，Diagnostic A 的 balanced primary setting；
- `U=240`：三个较大的 units，验证 composition advantage 是否不依赖单一 granularity。

暂不使用 `U=120`，避免 Diagnostic B 规模扩大为新的 sweep；不使用 `U=360`，因为两 units 不足以充分
检验 progressive composition。

### Parameter-Matched Arms

两个 arms 使用完全相同的 modules 与 trainable parameter count：

```text
base = InputProject(coeff)                          # [N,D]
coord_m = CoordinateMLP(center_m / 720)            # [D]
```

`parallel_no_transition`：

```text
state_m = GRUCell(base + coord_m, base)
segment_m = SharedDecoder(state_m)
```

所有 units 独立读取相同 base state；没有 unit-to-unit transition。

`prefix_causal_composed`：

```text
state_-1 = base
state_m = GRUCell(base + coord_m, state_{m-1})
segment_m = SharedDecoder(state_m)
```

两者只差 recurrent state source。它们共享：

- `InputProject`；
- continuous `CoordinateMLP`；
- `GRUCell`；
- `SharedDecoder`；
- latent dimension、optimizer、sampling、loss 与 checkpoint selection。

这使 `parallel_no_transition` 同时成为 independent-unit/no-transition capacity control。

### Optimization Protocol

- frozen A6 encoder；
- latent dimension `64`；
- train/val/test rows 上限 `8192/2048/2048` per dataset；
- seeds `2021/2022/2023`；
- `20` epochs，AdamW；
- balanced full-trajectory normalized MSE；
- best validation checkpoint 只用于 diagnostic optimization control，不作为论文 protocol claim；
- 报告 overall 与 per-unit test MSE、train/val/test gap、parameter count 与 prefix consistency。

### Diagnostic B Gate

对每个 dataset/unit-size setting，将相同 seed 的 two arms 配对。composition support 要求：

1. `prefix_causal_composed` mean test MSE 低于 `parallel_no_transition`；
2. 至少 `2/3` seeds 获胜；
3. mean relative MSE improvement 至少 `0.5%`。

整体 gate：六个 dataset/size settings 中至少四个达到 composition support，且三个 datasets 各至少一个
size 不退化超过 `0.25%`，则判为：

```text
partial_pass_prefix_causal_composition
```

该结果才允许 B13 进入 Step 4-6 narrative/method design。若 no-transition 持平或更好，则判为：

```text
no_transition_control_explains
```

并回滚到 Step 2：large-unit gradient pressure 存在，但不需要 compositional state transition。

若出现 non-finite loss，或超过四分之一 runs 同时满足 `test/val ratio > 3` 且绝对 gap `>1`，则判为：

```text
diagnostic_invalid_for_direction_rejection
```

只修复 probe，不否定 future-unit direction。

不能用固定 `normalized MSE > 10` 作为 pathology gate。这里的 future target 使用 history-window std
归一化，遇到 distribution shift 时没有统一的绝对 MSE 上界；numeric validity 必须结合 finite check 与
val/test mismatch 判断。

## Diagnostic B1 Result：Coefficient-Memory Intervention

`B13-FUCO-B1` 已在 frozen A6 coefficient memory 上完成 `36` 个 probe runs。完整 artifacts：

- `analysis/phase5_stage_b_b13_future_unit_composition_20260710/`；
- `analysis/phase5_stage_b_b13_future_unit_composition_20260710/b13_future_unit_composition_report.md`。

| Dataset | Unit size | Composed wins | Mean composed vs parallel MSE | Support |
| --- | ---: | ---: | ---: | --- |
| ETTh2 | `180` | `1/3` | `+11.3264%` | no |
| ETTh2 | `240` | `0/3` | `+19.9064%` | no |
| ETTm1 | `180` | `0/3` | `+4.0635%` | no |
| ETTm1 | `240` | `3/3` | `-3.9800%` | yes |
| Weather | `180` | `2/3` | `-3.2406%` | yes |
| Weather | `240` | `2/3` | `-4.6688%` | yes |

[Fact] 只有 `3/6` settings 达到 composition support，低于预注册的 `4/6`；ETTh2 两个 sizes 均超过
`+0.25%` non-degradation boundary。所有 arms 在相同 unit size 下参数量完全一致，最大 prefix error 为
`0`，`36/36` runs 均未触发 val/test numeric pathology。

[Decision] `no_transition_control_explains`。当前结果阻断的是：

```text
A6 global coefficient -> GRU future-unit transition -> shared decoder
```

不能据此关闭整个 future-unit generation direction。A6 `coeff [B,C,256]` 已是针对 full trajectory
训练的 global readout bottleneck；要求它在冻结后再支持 progressive transition，可能把 intervention 放得
太晚。另一方面，ETTm1-U240 与 Weather 两个 sizes 的正结果说明 transition 并非在所有数据上都无价值，
但这种 dataset-dependent mixed result 不满足 paper-core gate。

Failure attribution：

- `capacity_control_explains`：对当前 coefficient-memory mechanism 成立；
- `optimization_or_numeric_pathology`：不成立；
- `intervention_point_wrong`：仍有实质可能，因为 composition 发生在 full-trajectory coefficient 之后；
- `hypothesis_false`：尚未成立；
- `readout_or_head_design_wrong`：GRUCell/shared decoder 仍可能过弱，但本阶段不允许继续做 head sweep。

## Diagnostic B2：Pre-Coefficient Hidden-Memory Repair

为只修复一次 intervention point，而不把 B13 变成 architecture tuning，B2 保持完全相同的
parameter-matched arms、unit sizes、seeds、state dimension、loss 与 gate，仅将 frozen memory 改为：

```text
history -> A6 encoder -> hidden [B,C,R] -> diagnostic probe
```

即在 `learned_basis_coeff` 之前读取 history representation。由于 datasets 的 `R` 不同，两个 arms 均以
同一个 dataset-specific `InputProject(R,64)` 开始；同一 dataset/unit-size 内的参数量仍必须 exact match，
但不跨 datasets 比较 raw parameter count。

Remote protocol：

- GPU：在 `529_Lab-3090` 启动前由 `nvidia-smi` 选择空闲且安全的 GPU；
- unit sizes：`180/240`；seeds：`2021/2022/2023`；epochs：`20`；
- train/val/test row caps：`4096/1024/1024`，作为 resource-bounded diagnostic；
- checkpoint selection 与 B1 相同，只允许作为 probe optimization control；
- runner：`scripts/remote/run_phase5_stage_b_b13_hidden_memory_probe.sh`；
- sync：`scripts/sync_phase5_stage_b_b13_hidden_memory_probe_results.sh`。

B2 仍使用 B1 的 pre-registered gate，不因 B1 结果调整阈值。若 hidden-memory B2 通过，则 B13 只进入
Step 4-6 narrative/literature audit；若仍由 no-transition control 解释，则关闭当前
`GRU-based prefix-causal composition` candidate，并回到 Step 2 重新判断真正需要的是
future-region-specific state 还是其它非 recurrent future-stage generator。B2 失败不能自动否定所有
future-unit architecture，但不允许继续围绕 GRU/head 叠加调参。

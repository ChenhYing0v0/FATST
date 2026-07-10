# Phase5 StageB B13-FUCO-B2 Hidden-Memory Deep Analysis

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B13-FUCO` |
| `diagnostic_id` | `B13-FUCO-B2` |
| `current_step` | Step 3 problem/mechanism diagnostic completed；return to Step 2 |
| `problem` | prefix-causal latent composition 是否在 pre-coefficient A6 hidden memory 上稳定超过 exact parameter-matched no-transition unit generation |
| `narrative_gate` | failed for current GRU-based composition |
| `effectiveness_gate` | not applicable；probe is not an end-to-end model |
| `decision` | `no_transition_control_explains` |
| `rollback` | close current GRU-based prefix-causal composition；return StageB to Step 2 future-region-specific generation search |

## 1. Protocol Integrity

[Fact] Remote B2 完成 `36` runs 与 `18` paired comparisons：

- 每个 dataset/unit-size/seed 都恰有两个 arms；
- 所有 paired `parameter_delta=0`；
- 最大 `prefix_max_abs=0`；
- train/val/test metrics 全部 finite；
- `0/36` runs 触发预注册的 severe val-test mismatch gate；
- clean A6 checkpoint checksum 与本地一致。

不同 dataset 的 A6 hidden width 不同：ETTh2 `1536`、ETTm1 `256`、Weather `6144`。因此 raw trainable
parameter count 跨 datasets 不相等，但同一 dataset/unit-size pair 内严格相等：

| Dataset | U180 params | U240 params |
| --- | ---: | ---: |
| ETTh2 | `139316` | `143216` |
| ETTm1 | `57396` | `61296` |
| Weather | `434228` | `438128` |

[Boundary] 该 protocol 只支持 arms 内相对比较，不能比较不同 datasets 的 absolute normalized MSE 或
parameter efficiency，也不能写成 end-to-end forecasting gain。

## 2. Aggregate Gate

负值表示 composed 更好：

| Dataset | Unit size | Wins | Mean relative MSE | Std | Support |
| --- | ---: | ---: | ---: | ---: | --- |
| ETTh2 | `180` | `1/3` | `+5.1639%` | `20.0428%` | no |
| ETTh2 | `240` | `1/3` | `+5.3589%` | `19.0149%` | no |
| ETTm1 | `180` | `3/3` | `-2.3454%` | `0.7474%` | yes |
| ETTm1 | `240` | `3/3` | `-16.0953%` | `2.7937%` | yes |
| Weather | `180` | `2/3` | `-1.8434%` | `4.0261%` | yes |
| Weather | `240` | `3/3` | `-6.4462%` | `1.5320%` | yes |

`4/6` settings 达到 composition support，但整体 gate 还要求每个 dataset 至少一个 size 不退化超过
`+0.25%`。ETTh2 两个 sizes 均平均退化约 `+5%`，所以 decision 必须是：

```text
no_transition_control_explains
```

[Strong Evidence] 这不是 marginal threshold miss。ETTh2 两个 sizes 都只有 `1/3` seeds 获胜，seed-level
范围分别为 `[-23.08%, +21.31%]` 与 `[-21.23%, +22.13%]`；其高 variance 与两个 size 的同向负均值表明
current recurrent transition 缺少跨 seed stability。

## 3. Per-Unit Mechanism Check

对每个 setting，先对三 seeds 的 per-unit relative MSE 求均值。`later mean` 不含 unit 0：

| Dataset | U | Unit 0 | Later units mean | Last unit |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | `180` | `-11.6593%` | `+9.7837%` | `+13.3852%` |
| ETTh2 | `240` | `-5.4278%` | `+9.8996%` | `+11.5961%` |
| ETTm1 | `180` | `+0.8531%` | `-3.2989%` | `-2.1641%` |
| ETTm1 | `240` | `-17.7450%` | `-12.2488%` | `+7.4951%` |
| Weather | `180` | `-3.5646%` | `-1.5450%` | `-4.9009%` |
| Weather | `240` | `-12.6430%` | `-5.0934%` | `-4.2947%` |

该表比 aggregate setting gate 更直接检验 narrative：

1. `unit 0` 没有 previous-unit information。两个 arms 对 unit 0 的 computation topology 相同；差异只能
   来自 later-unit joint loss 改变了 shared weights，而不能来自 prefix-causal composition。
2. ETTm1-U240 与 Weather-U240 的最大收益恰好出现在 unit 0；这更像 shared-parameter optimization 或
   implicit regularization，而不是 accumulated future context。
3. ETTm1-U240 的最后一个 unit 平均退化 `+7.50%`，与 aggregate `-16.10%` 的强正向结果相反。
4. ETTh2 的 unit 0 平均改善，但 later units 明显恶化，符合 recurrent transition 随深度放大不稳定性的
   pattern。
5. 六个 settings 都没有表现出“composition depth 越深、relative benefit 越强”的稳定趋势。

[Decision] ETTm1/Weather 的 aggregate positive results 不能被解释为“previous latent future unit 给 later
unit 提供了有效 compositional context”。当前机制的核心 causal narrative 未通过。

## 4. B1 与 B2 的描述性比较

| Dataset | U | B1 coefficient memory | B2 hidden memory |
| --- | ---: | ---: | ---: |
| ETTh2 | `180` | `+11.3264%` | `+5.1639%` |
| ETTh2 | `240` | `+19.9064%` | `+5.3589%` |
| ETTm1 | `180` | `+4.0635%` | `-2.3454%` |
| ETTm1 | `240` | `-3.9800%` | `-16.0953%` |
| Weather | `180` | `-3.2406%` | `-1.8434%` |
| Weather | `240` | `-4.6688%` | `-6.4462%` |

[Caution] B1 使用 `8192/2048/2048` rows，B2 为 resource-bounded `4096/1024/1024`；因此该表只能说明
hidden-memory repair 改变了结果 pattern，不能将 B1 到 B2 的差值因果归因于 intervention point。

[Moderate Evidence] U240 在 ETTm1 与 Weather 上均比 U180 更强，支持用户提出的“大 unit 承载更多明显
信息”作为后续 design prior；但 ETTh2 两个 sizes 都失败，所以 large unit 不能救回 current GRU mechanism。

## 5. Optimization And Numeric Reading

ETTh2 best epochs 全部在 `1-2`，ETTm1 为 `4-20`，Weather 为 `13-20`。ETTh2 recurrent probe 对初始化
明显敏感。Weather/ETTh2 的 history-window normalization 会产生很大的 train/validation target MSE；该
quantity 没有统一 absolute upper bound，因此不能用固定 MSE threshold 宣布 numeric invalid。

[Fact] 当前预注册 validity checks 全部通过；所以本结果不是
`diagnostic_invalid_for_direction_rejection`。但 ETTh2 seed variance 是 mechanism stability 的负证据，
不能被“数值没发散”掩盖。

## 6. Failure Attribution

- `capacity_control_explains`：[Strong Evidence] exact parameter-matched no-transition control 在 ETTh2 获胜，
  且 overall gate 未通过；
- `intervention_point_wrong`：B2 已移除 post-coefficient bottleneck 这一主要 confound；frozen hidden 仍不是
  end-to-end adaptation，但没有证据支持为此升级到昂贵 implementation；
- `readout_or_head_design_wrong`：理论上可能，但 B2 是预注册的唯一 repair，不能继续 GRU/head sweep；
- `optimization_or_numeric_pathology`：未触发 invalid gate；ETTh2 optimization sensitivity 作为当前机制的
  stability failure 记录；
- `hypothesis_false`：只对“previous-unit recurrent transition 是必要机制”成立；不对所有 future-unit
  generation architecture 成立。

## 7. Final Decision And Rollback

[Decision] 关闭当前：

```text
history/A6 memory
  -> shared base
  -> GRU prefix-causal future-unit transition
  -> shared segment decoder
```

不得进入 Step 4-6 narrative design、end-to-end implementation 或新一轮 GRU/head tuning。

[Rollback] StageB 回到 Step 2。保留的问题不是“怎样让 recurrence 更强”，而是：

> 一个不依赖 full-horizon clipping 的 native future-unit generator，是否需要对不同 large future regions
> 形成不同的 history retrieval/state，而这些 region-specific states 是否可以在没有 recurrent transition
> 的情况下成立？

下一步优先诊断 `future-unit-specific history retrieval`，而不是再次测试 late coefficient modulation、hard
stage id、residual correction 或 small-unit tiling。任何新 architecture 仍需包含 parameter-matched
constant/no-unit controls，并继续以 `U=180/240` 为主 granularity。

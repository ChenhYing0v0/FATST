# SC1-SIFF-v2-EQ-ATTR Step 9 四层诊断

## 1. 当前节点

| Field | Content |
| --- | --- |
| `current_step` | Step 9 complete；Step 10 exact-v1 closure recommended |
| `candidate_version` | `SC1-SIFF-v2-EQ-ATTR-v1` |
| `problem` | `SIFF_EQUAL` 相对旧 A6 有正收益，但该收益是否来自 ordered scale coordinate 尚未被 matched controls 证明 |
| `existence_evidence` | Phase A 50/50 runs、200/200 official-test cells 完整；此前 fair audit 中 SIFF_EQUAL vs A6 为正 |
| `idea` | 用 10-arm matrix 分离 measure objective、equal-skill objective、SIFF architecture、ordering、partition 与 independent-scope interaction |
| `theory_check` | 七项 hard comparisons 必须逐项通过；internal health 只验证机制路径是否工作，不能挽救 negative effectiveness |
| `design` | 5 datasets × 10 arms × seed2021；validation four-H 选 checkpoint；official test H96/H192/H336/H720 判定机制 |
| `narrative_gate` | conditional pass before launch；结果后未通过完整 contribution chain |
| `effectiveness_gate` | main comparisons 2/3 pass；`SIFF_EQUAL > A6_MEASURE` fail |
| `matched_attribution_gate` | EQUAL-context controls 3/4 pass；`SIFF_EQUAL > INDEPENDENT_EQUAL` fail |
| `internal_health_gate` | 7/7 pass |
| `artifacts` | `primary/`、`deep_diagnostics/`、remote output root |
| `decision` | exact v1 不进入 confirmation；seeds2022/2023 保持 false；回 Step 4 重构 method/claim |

## 2. 本次到底测试了什么

本次不是再次询问“SIFF 是否比旧 A6 好”，而是询问更严格的完整论文主张：

> 在相同 fixed-past、统一 $T=720$ generation、相同 dataset profile 与 matched initialization 下，
> EQUAL-trained ordered continuous scale field 是否同时优于简单 objective control、parent architecture 以及
> constant/permuted/wide/independent EQUAL-context controls？

其中：

- `A6_MEASURE - A6_FULL` 测量 harmonic measure objective 本身的收益；
- `SIFF_EQUAL - SIFF_MEASURE` 测量 equal-skill 是否修复 SIFF 的 arm credit starvation；
- `SIFF_EQUAL - PCSD_EQUAL` 测量 SIFF architecture 在相同 objective 下的增量；
- constant/permuted/Q1-wide/independent controls 分别排除 generic capacity、任意 arm identity、单一宽 field 与
  independent-arm ensemble 解释。

## 3. Artifact 与 protocol audit

[Fact] Remote analyzer 与本地同步证据一致：

| Audit | Result |
| --- | ---: |
| expected / completed runs | 50 / 50 |
| expected / available standard-horizon test cells | 200 / 200 |
| run status `ok` | 50 / 50 |
| test invariant `pass` | 50 / 50 |
| official-test split recorded | 50 / 50 |
| diagnostic NPZ | 50 / 50 |
| training logs / effective configs / initialization contracts | 50 / 50 |
| checkpoint hash in invariant matches remote run audit | 50 / 50 |
| per-dataset encoder initialization hash classes across 10 arms | 1 / dataset |
| non-finite standard metrics | 0 |

Checkpoint 本体没有同步进 Git workspace；remote analyzer 已在运行时将 `checkpoint.pt` 内容 SHA-256 与
`test_audit_invariants.json` 逐 run 比对。本地同步后再次检查 invariant hash 与 remote `run_audit.csv` 一致。

因此，本结果不存在 partial-matrix、test-selected checkpoint、cross-arm initialization mismatch 或 numeric failure。

## 4. 统计定义

- `macro_gain_percent`：对 5 datasets × 4 horizons 的 cell-wise relative gain
  $100(1-M_{candidate}/M_{reference})$ 取算术平均；正值代表 candidate 更好。
- `dataset_wins`：每个 dataset 内四个 horizons 的 relative gain 平均为正的数据集数。
- `horizon_wins`：每个 horizon 跨五个 datasets 的 relative gain 平均为正的 horizon 数。
- `cell_wins`：20 个 dataset-horizon cells 中 candidate metric 更小的数量。
- hard MSE gate：macro gain 至少 `0.3%`、dataset wins 至少 `3/5`、horizon wins 至少 `3/4`、cell wins
  至少 `11/20`。三项 main comparisons 另要求 MAE macro gain 非负。
- `oracle_gain`：每个 row-bin 事后选最低 MSE arm，相对 learned fused forecast 的 gain；只表示尚未被 fusion
  利用的 headroom。
- `pairwise_arm_nrmse`：probe arms 两两预测 RMSE 的归一化均值；用于排除完全相同输出。
- `policy_entropy`：五 arm policy entropy 除以 $\log 5$；1 接近 uniform，0 接近 one-hot。
- `nonconstant_component_rms_ratio`：固定 learned policy 后，移除 SIFF nonconstant scale component 引起的
  forecast RMS change，相对 fused forecast RMS 的比例。
- `ordered_vs_constant_probe_nrmse`：ordered 与 constant control 的 probe fused forecasts 差异。

## 5. Layer 1：paper-facing effectiveness

| Comparison | MSE macro | cells | datasets | horizons | MAE macro | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SIFF_EQUAL vs A6_FULL | +1.6436% | 17/20 | 4/5 | 4/4 | +0.9084% | pass |
| SIFF_EQUAL vs A6_MEASURE | -0.2366% | 10/20 | 2/5 | 1/4 | -0.3961% | **fail** |
| SIFF_EQUAL vs PCSD_EQUAL | +0.5906% | 14/20 | 3/5 | 4/4 | +0.5069% | pass |

[Strong Evidence] SIFF_EQUAL 确实优于旧 `A6_FULL` 和同 objective 的 `PCSD_EQUAL`，但它没有超过更简单的
`A6_MEASURE`。因此，旧的 `+1.6436%` baseline gain 不能归因于 SIFF architecture；measure-aligned objective
单独在 A6 上已经获得 `+1.8762%` MSE、`+1.2982%` MAE，并且是 20/20 cells、5/5 datasets、4/4 horizons
全胜。

`SIFF_EQUAL vs A6_MEASURE` 的负值并非单一 cell 造成：

| Dataset | MSE gain |
| --- | ---: |
| Weather | -0.2730% |
| ETTm1 | +0.4858% |
| ETTm2 | -1.6799% |
| ETTh1 | -1.1113% |
| ETTh2 | +1.3954% |

按 horizon 聚合仅 H336 为正：H96 `-0.6855%`、H192 `-0.1134%`、H336 `+0.1967%`、H720
`-0.3442%`。这不是“只损害长预测”或“只损害短预测”，而是跨 horizon 的总体不足。

Layer 1 decision=`fail (2/3 main comparisons)`。

## 6. Objective × architecture decomposition

| Effect | MSE macro | MAE macro | Interpretation |
| --- | ---: | ---: | --- |
| A6_MEASURE vs A6_FULL | +1.8762% | +1.2982% | measure objective 本身强且稳定 |
| PCSD_MEASURE vs A6_MEASURE | -0.8658% | -0.6263% | PCSD field 在 measure context 下引入损失 |
| SIFF_MEASURE vs PCSD_MEASURE | -1.5259% | -1.2449% | ordered SIFF 在普通 measure 下进一步变差 |
| PCSD_EQUAL vs PCSD_MEASURE | +0.0177% | -0.2880% | equal-skill 对 PCSD 几乎无 MSE 增益 |
| SIFF_EQUAL vs SIFF_MEASURE | +2.0991% | +1.4489% | equal-skill 明显修复 SIFF |
| SIFF_EQUAL vs PCSD_EQUAL | +0.5906% | +0.5069% | 修复后 SIFF 超过 parent |

[Strong Evidence] equal-skill 不是 generic optimizer improvement，而是与 SIFF architecture 存在明显 interaction。
在 `SIFF_MEASURE` 中，五个 arms 的 loss CV macro 达 `111.85%`，而 `SIFF_EQUAL` 降至 `3.45%`；前者的
row-bin oracle macro 为 `-18.79%`，后者为 `+6.39%`。也就是说，equal-skill 的确完成了预期的“修复同一
SIFF run 内 arm skill starvation”。

但该修复主要把 SIFF 从一个不健康的 measure-only state 拉回正常区间，并未让最终模型超过简单的
`A6_MEASURE`。这是“训练机制工作，但 paper-facing method 不够强”，两者不能混为一谈。

## 7. Layer 2：matched mechanism attribution

| Comparison | MSE macro | cells | datasets | horizons | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| ordered vs constant EQUAL | +0.9393% | 19/20 | 5/5 | 4/4 | pass |
| ordered vs permuted EQUAL | +0.3959% | 14/20 | 3/5 | 4/4 | pass |
| ordered vs Q1-wide EQUAL | +1.1619% | 18/20 | 4/5 | 4/4 | pass |
| ordered vs independent EQUAL | +0.2580% | 12/20 | 3/5 | 4/4 | **fail** |

[Strong Evidence] ordered coordinate 不是无效装饰：它稳定超过 constant、permuted 与 Q1-wide controls，说明
scale variation、正确 ordering 与 multi-scale partition 都携带可利用信息。

[Strong Evidence] 但完整的 ordered cross-scale coupling necessity 尚未成立。independent-scope control 已解释绝大多数
收益；ordered 仅领先 `0.2580%`，低于预注册 `0.3%` 门槛。其 dataset 分布为 Weather `+0.8092%`、ETTm1
`+0.8310%`、ETTm2 `-0.2960%`、ETTh1 `+0.6186%`、ETTh2 `-0.6725%`。

更关键的是，该 comparison 在 validation 为 `-0.3783%`，到 test 才变为 `+0.2580%`，存在 split reversal。
这不否定 test-primary 规则，但说明这个小正值不适合在单 seed Phase A 中越过已冻结门槛。

Layer 2 decision=`fail (3/4 EQUAL-context controls)`。

## 8. Layer 3：internal mechanism health

| Metric | Macro / range | Gate |
| --- | ---: | --- |
| all finite | 5/5 datasets | pass |
| prefix projectivity max gap | 0 | pass |
| oracle-positive datasets | 5/5；macro +6.3937% | pass |
| pairwise arm NRMSE | 0.1587 | pass |
| policy normalized entropy | 0.8122 | pass |
| nonconstant component RMS ratio | 0.1475 | pass |
| ordered-vs-constant probe NRMSE | 0.1234 | pass |

Layer 3 decision=`7/7 pass`。因此不能把 Layer 1/2 failure 写成 arms 全部退化、ordered component 未进入输出、
policy one-hot collapse、projectivity error 或 numerical pathology。

### 8.1 仍需保留的内部限定

1. ETTh1/ETTh2 policy entropy 分别为 `0.958/0.969`，接近 uniform；Weather 为 `0.518`，明显更 selective。
   macro gate 通过，但 adaptive routing 强度具有明显 dataset dependence。
2. nonconstant component ratio 在 ETTh1/Weather/ETTm1 为 `0.241/0.215/0.203`，在 ETTm2/ETTh2 只有
   `0.046/0.032`。component 被使用，但并非所有数据集都同等依赖 ordered variation。
3. `SIFF_INDEPENDENT_EQUAL` 的 arm loss CV 更低（`0.90%`），oracle headroom 更高（`8.17%`），learned fusion
   相对 best fixed arm 的优势也更强。它不是一个退化 control，而是强且合理的替代 architecture。
4. SIFF_EQUAL 的 policy 随 future bin 有变化，但不是简单单调的“越远使用越大 scale”。这支持 learned
   horizon signature 的存在，却不足以证明当前 ordered coordinate 是唯一或最优实现。

## 9. Layer 4：failure attribution

### `hypothesis_false`

[Strong Evidence] exact v1 的联合假设不成立：当前 ordered EQUAL SIFF 没有同时超过 `A6_MEASURE` 与
`SIFF_INDEPENDENT_EQUAL`。这关闭的是
`ordered scale field + current shared mode construction + equal-skill objective` 的完整 candidate，不是关闭
“multi-horizon coupling granularity 是否值得研究”这一问题。

### `intervention_point_wrong`

[Fact] 暂无直接证据。nonconstant component 与 ordered-vs-constant forecast contrast 均显著非零，说明介入点确实
影响最终 forecast。不能因性能 gate 失败就反推 intervention point 必然错误。

### `readout_or_head_design_wrong`

[Hypothesis] 当前 shared field 可能承担了不必要的 cross-scale coupling 约束。independent control 几乎打平，且
内部 fusion 更稳，提示未来若保留 coupling 主线，应重新审视“共享连续 scale field 是否必须”，而不是继续微调
当前 rank、policy 或 scale set。

### `optimization_or_numeric_pathology`

[Fact] 不支持。50/50 runs finite、best-four-H validation checkpoint 正常、test 不改 checkpoint、projectivity gap
为 0。`SIFF_EQUAL vs A6_MEASURE` 在 validation/test 均为负（`-0.4581%/-0.2366%`），也不是明显的
validation-to-test false failure。

### `capacity_control_explains`

[Strong Evidence] 是当前主因：

1. `A6_MEASURE` 以更简单架构获得比 SIFF_EQUAL 更高的 baseline gain；
2. independent-scope EQUAL control 解释了 ordered SIFF 的绝大部分 architecture gain；
3. 这两个 controls 分别阻断 performance necessity 与 ordered-coupling specificity。

## 10. Step 9 结论与 rollback

Machine decision=`close_exact_candidate_effectiveness_fail`；人工四层复核与之相符。

1. `SC1-SIFF-v2-EQ-ATTR-v1` 不升级为 paper-core candidate；
2. seeds2022/2023 confirmation 不授权，不用额外 seeds 挽救预注册 Phase-A failure；
3. 不继续调整当前 SIFF rank、scale set、policy entropy loss 或 dataset-specific profile；
4. 保留三条正证据作为下一轮 Step 4 输入：
   - multi-scale order 相对 constant/permuted/Q1-wide 有稳定信号；
   - equal-skill 能修复 SIFF-specific arm starvation；
   - D14 coupling-crossing problem evidence 仍成立；
5. 回 Step 4 的核心问题改为：为什么简单 `A6_MEASURE` 已获得主要收益，而 multi-arm decoder 只能在内部形成
   healthy diversity，却没有把 conditional headroom 转换成超过 simple shared operator 的 fused forecast？

该 rollback 不应立刻叠加新 loss/router。下一候选必须先给出相对 `A6_MEASURE` 与 independent-scope control 的
必要性，并重新通过 narrative/design gate。

## 11. Artifacts

- primary analyzer：`analysis/stage_c_siff_equal_attribution_step9_20260718/primary/`；
- deep internal diagnostics：`analysis/stage_c_siff_equal_attribution_step9_20260718/deep_diagnostics/`；
- local raw evidence（gitignored）：`analysis/stage_c_siff_equal_attribution_step9_20260718/raw/`；
- remote root：`/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_equal_attribution_v2`；
- frozen source commit：`c4c4730be09f4c6471653018a39b6a9cba365bee`。

## 12. Post-Step9 portfolio decision（2026-07-18）

用户在完整获知上述negative attribution gate后，决定仍将该exact model保留为本阶段
`frozen_performance_near_candidate`，因为它是当前正式公平矩阵中最接近论文可发表performance的模型。

该决定不修改本报告的实验判断：“不升级为`passed_core_candidate`”仍成立；变化仅在candidate portfolio层面：

1. v1的source/config/profile/checkpoint/results全部冻结，不允许事后修改；
2. v1成为当前best candidate、reproducibility target与下一轮redesign的mandatory parent；
3. attribution status仍是`performance_partial_pass_attribution_blocked`，seeds2022/2023不自动授权；
4. 任意policy、objective、scale field或architecture修改均创建新的`test_informed` candidate。

不可变清单见`configs/stage_c_siff_equal_attribution_v1_candidate_freeze.json`；后续improvement audit见
`analysis/stage_c_siff_candidate_step4_source_audit_20260718/source_informed_improvement_audit.md`。

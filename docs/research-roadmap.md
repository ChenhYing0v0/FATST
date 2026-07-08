# Research Roadmap

本文档是当前可重启的主研究路径。旧阶段细节已归档或保存在 `analysis/`，本文件只保留会影响后续
决策的结论。

## Current State

| Field | Content |
| --- | --- |
| `paper_target` | 高水平 SCI 期刊时间序列预测论文 |
| `working_title` | Horizon-Agnostic Supervision Scheduling for Unified Multi-Horizon Forecasting |
| `current_stage` | Phase5：A6-LBF-r256 clean operator validated；StageB B10 native design gate or rollback |
| `current_11_step` | StageB Step 4-6: decide native trainable target-query memory readout, or rollback B10 to Step 2/3 |
| `active_carrier` | `A6-LBF-r256` pure learned-basis forecast operator |
| `active_ledger` | `docs/stage-ledgers/phase5-timealign-interface.md` |

## Long Research Loop Rule

每个新 StageB 候选必须记录：

1. Research and analyze existing work.
2. Propose the specific problem.
3. Evaluate whether the problem is real.
4. Propose the core idea.
5. Evaluate theoretical feasibility.
6. Design method and experiment plan.
7. Implement.
8. Run remote training when needed.
9. Evaluate artifacts.
10. Decide paper-story and performance pass/fail.
11. If failed, choose explicit rollback point.

Narrative gate 属于 Step 4-6；effectiveness gate 属于 Step 9-10。Diagnostic-only 实验不能因为
metric 正向就直接升级为 method。

## StageA Final Decision

[Decision] StageA 结果已固定：`A6-LBF-r256` 是当前论文的重要创新点与后续 StageB 起点。

[Claim] A6-LBF-r256 用 learned temporal basis + hidden coefficients 形成 prefix-native unified forecast
operator。在当前实验集合上，一个 unified model 整体优于 fixed-horizon per-horizon TimeAlign。

### Evidence

相对 fixed-horizon per-horizon TimeAlign official-last（clean A6 rerun）：

| Dataset | MSE wins | Mean MSE change |
| --- | ---: | ---: |
| ETTh2 | 4/4 | `-10.53%` |
| ETTm1 | 3/4 | `-1.64%` |
| Weather | 2/4 | `-0.22%` |
| Overall | 9/12 | `-4.13%` |

相对 official unified TimeAlign（clean A6 rerun）：

| Dataset | MSE wins | Mean MSE change |
| --- | ---: | ---: |
| ETTh2 | 4/4 | `-2.78%` |
| ETTm1 | 3/4 | `-1.20%` |
| Weather | 4/4 | `-1.26%` |
| Overall | 11/12 | `-1.75%` |

[Validation] Clean rerun after removing the A6 future reconstruction/alignment branch is at
`analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/`. It preserves the accepted result and is close to the
historical A6 artifact: overall mean MSE change `+0.20%`, `6/12` MSE wins. Effective clean A6 training uses
`w_recon=0.0`, `w_align=0.0`, `readout_mode=learned-basis-forecast-operator`, `basis_rank=256`, and
`pred_loss_mode=multi-prefix`.

### Contribution Boundary

A6-LBF-r256 可以作为：

- unified multi-horizon forecasting 的核心 architecture contribution；
- 后续 StageB 的 clean carrier；
- 与 fixed-horizon TimeAlign 的主对照证据。

不能写成：

- teacher/self-teacher 或 EMA 稳定化方法；
- target-query / QBR / nested / residual adapter 的混合路线；
- 依赖 best-val/early-stop 的 protocol trick。

## StageB Redesign Entry

[Decision] StageB 从 A6-LBF-r256 出发重新设计，不沿用 pre-cleanup B0 作为正式路线。

StageB 应回答的新问题：

> 在 A6-LBF-r256 这个已成立的 unified carrier 上，future-aware supervision / reliability-aware
> allocation 是否能进一步提升 unified model 的跨数据集稳定性和论文机制深度？

StageB 设计前必须先写新的 experiment note，至少包含：

- `current_step`;
- `problem`;
- `existence_evidence`;
- `idea`;
- `theory_check`;
- `design`;
- `narrative_gate`;
- `effectiveness_gate`;
- `artifacts`;
- `decision`;
- rollback point。

### B1 Reliability Diagnostic Decision

[Decision] B1-RED full diagnostic completed at
`analysis/phase5_stage_b_reliability_diagnostic_20260706/`.

[Strong Evidence] A6-LBF-r256 的 future units 存在 material heterogeneity：
48-step unit max/min MSE gap 为 ETTh2 `195.16%`、ETTm1 `99.09%`、Weather `248.48%`。

[Counter-Evidence] 该 heterogeneity 与 forecast step/distance 高度绑定：
`Spearman(step, MSE)` 为 ETTh2 `0.99`、ETTm1 `0.99`、Weather `1.00`。train-only proxies
对 raw MSE 的相关性不能直接作为 reliability evidence，因为它可能只是共同随 future distance 增长。

[Decision] B1-RED 状态为 `partial_pass_distance_confounded`。B2 reliability-aware supervision
allocation 不通过 narrative gate，不能进入 implementation。StageB 回到 Step 2/3：若继续，必须重新定义
一个能越过 step-distance confounder 的 reliability problem；若不能，则保持 A6-LBF-r256 作为独立
unified forecasting contribution。

### B3 Problem Redefinition Decision

[Decision] StageB Step 2/3 redefinition completed at
`analysis/phase5_stage_b_step23_redefinition_20260706/`.

[Rejected] Raw future-unit reliability 和 B2-RAS 不能继续。原因是 B1 已证明 raw hard/easy units 与
forecast distance 几乎同序，直接加权会退化为 horizon-distance weighting。

[Proposed] 更强的问题候选是 `B3-DSR`: distance-normalized seasonal residual reliability。它不再预测
raw unit MSE，而是先移除 step-distance trend，再测试 train-only `seasonal_residual` 是否解释
structural residual difficulty。

[Evidence] B1 中 `seasonal_residual` 对 detrended MSE 的 Spearman 相关在 ETTh2/ETTm1/Weather
分别为 `0.35/0.59/0.81`，而对 raw MSE 的相关为 `-0.74/-0.79/-0.10`。这说明该 proxy 不只是
随 step 增长的 distance proxy。

[Decision] B3 只通过 problem-candidate gate，尚未通过 Step 4-6 narrative gate。下一步只能做
diagnostic robustness：detrending 形式、block size、bootstrap stability 和 leakage-free train-only
path。若 B3 失败，StageB 应关闭或转入更大的 label-autocorrelation objective problem。

### B3 Diagnostic Decision

[Decision] B3 distance-normalized seasonal residual diagnostic completed at
`analysis/phase5_stage_b_distance_normalized_seasonal_residual_20260706/`.

[Moderate Evidence] `seasonal_residual` 对 `linear_step_residual` 的 Spearman 在所有 dataset/block
size 上均为正：ETTh2 `0.31-0.38`、ETTm1 `0.43-0.62`、Weather `0.81-0.83`。

[Counter-Evidence] 该信号没有通过更严格的 robustness gate。ETTh2/ETTm1 的 `rank_step_residual`
出现负相关；Weather 的 `prefix_normalized_residual` 在 block `24/96` 为负；bootstrap sign stability
在多个 rank/prefix label 上低于阈值。`nan` rank residual 表示对应 block size 下 unit MSE rank 被
monotonic step trend 完全解释，不能作为正证据。

[Decision] B3 状态为 `partial_pass_needs_stronger_proxy_or_method_boundary`。当前不得实现
reliability-aware loss weighting。下一步只能二选一：继续寻找更强 train-only structural proxy，或关闭
StageB 并转向更大的 label-autocorrelation objective problem。

### TimeAlign Dependency Diagnostic Decision

[Decision] StageB dependency audit completed at
`analysis/phase5_stage_b_timealign_dependency_audit_20260706/`.

[Evidence] Under the same inherited TimeAlign align/recon setting, A6-LBF-r256 beats official unified TimeAlign
on `11/12` settings with mean MSE change `-1.94%`. This supports an A6-LBF head/operator contribution.

[Risk] The current objective still uses inherited `w_recon * recon_loss + w_align * align_loss`. Last-epoch weighted
alignment share is ETTh2 `0.19`, ETTm1 `0.08`, Weather `0.12`; therefore full architecture independence is not
established.

[Decision] This audit produced `partial_dependency_risk_confirmed` and required a causal no-align/no-recon ablation
before any B5 basis-aware alignment design. That ablation has now returned and supersedes the audit-level decision
below.

### TimeAlign Dependency Ablation Decision

[Decision] B4 no-align/no-recon dependency ablation completed at
`analysis/phase5_stage_b_timealign_dependency_ablation_20260706/`.

[Fact] The returned 12-run matrix compares four A6-LBF-r256 arms:
`current_align_recon`, `no_align_recon`, `align_no_recon`, and `no_align_no_recon` on ETTh2/ETTm1/Weather with
horizons 96/192/336/720.

[Strong Evidence] Removing both inherited auxiliary losses does not collapse A6-LBF-r256. `no_align_no_recon`
changes mean MSE by only `+0.07%` versus current and wins `7/12` horizon settings. `align_no_recon` is slightly
better on mean MSE (`-0.04%`) and wins `8/12`, but the effect size is too small to justify a new align method by
itself.

[Mechanism Note] `no_align_recon` and `no_align_no_recon` are metric-identical in the returned artifacts. This is
consistent with the current code path: when `w_align=0`, reconstruction alone mostly trains the future branch rather
than the history-derived forecast operator.

[Decision] B4 status is `dependency_ablation_pass_for_head_contribution_but_not_for_b5`. The result strengthens the
paper boundary for Contribution 1: A6-LBF is not merely an inherited TimeAlign alignment artifact. It simultaneously
weakens B5 basis-aware future alignment as the next paper-core method, because the diagnostic did not show a material
dependence on inherited alignment.

[Code Decision] The active A6-LBF implementation now removes the future reconstruction/alignment branch and sets
`w_recon=w_align=0.0` for `readout_mode=learned-basis-forecast-operator`. Official TimeAlign keeps its inherited
future branch for baseline reproduction. This makes A6-LBF a cleaner research carrier: the only active mechanism is
history encoder -> learned coefficients -> prefix-native temporal basis.

### B6 Prefix-Native Objective Entry

[Decision] StageB rolls back to Step 2/3 and opens `B6-PLO`: prefix-native label/basis objective diagnostic.

[Problem] A6-LBF-r256 already changes the forecast operator into learned-basis coefficient space, and its active code
path now removes inherited generic TimeAlign auxiliary terms. The next credible StageB question is whether the
remaining time-domain prefix point objective should explicitly match the prefix-native label autocorrelation /
learned-basis structure.

[Updated After Code Cleanup] After removing inherited auxiliary terms from A6-LBF, the B6 question becomes sharper:
the architecture is now cleanly basis-native, while the objective is still time-domain prefix L1. B6 asks whether the
target labels/residuals have a stable basis-space structure that should guide supervision.

[Required Diagnostic Before Implementation] B6 must first test train-only label autocorrelation, learned-basis
projection coverage, and coefficient-space residual structure. It may use `current_align_recon` and
`no_align_no_recon` as controls, but no new objective or alignment method should be implemented until Step 4-6
narrative gate is written and passed.

### B6 Prefix-Native Objective Diagnostic Decision

[Decision] B6 offline diagnostic completed at
`analysis/phase5_stage_b_prefix_native_objective_diagnostic_20260706/`.

[Fact] Train labels are compressible, but mostly by generic low-frequency structure. PCA top32 versus DCT top32 is:
ETTh2 `0.917/0.889`, ETTm1 `0.939/0.930`, Weather `0.832/0.831`.

[Counter-Evidence] A6 learned temporal basis does not provide top32 advantage over DCT. Label coverage is ETTh2
`0.675`, ETTm1 `0.690`, Weather `0.251`; residual coverage is ETTh2 `0.287`, ETTm1 `0.110`, Weather `0.081`.
All are weaker than or close to DCT at top32.

[Decision] Status is `diagnostic_not_enough_pause_b6`. Do not implement a prefix-native label/basis objective now.
The result would not provide a clean distinction from generic frequency-domain auxiliary losses such as FreDF/TransDF.

### Clean A6 Rerun Decision

[Decision] Clean A6-LBF-r256 rerun completed at
`analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/` with status `clean_a6_validated`.

[Fact] The active implementation has no A6 future reconstruction/alignment branch and trains with
`effective_w_recon=0.0`, `effective_w_align=0.0`.

[Strong Evidence] The clean rerun still beats fixed-horizon TimeAlign by overall mean MSE `-4.13%` with `9/12`
MSE wins, and official unified TimeAlign by `-1.75%` with `11/12` MSE wins. Relative to historical A6-LBF-r256 it
changes mean MSE by only `+0.20%`.

[Decision] StageA clean carrier is validated. StageB remains paused; do not implement B5 or B6. If research
continues beyond Contribution 1, return to Step 2/3 and define a problem that is neither generic frequency
supervision nor step-distance-confounded reliability weighting.

### B7 Unified Prefix Optimization Diagnostic

[Decision] B7-UPO diagnostic completed at
`analysis/phase5_stage_b_unified_prefix_optimization_20260707/`.

[Problem] A6-LBF-r256 已经提供 unified forecast operator，但当前 `multi-prefix` objective 把
`h96/h192/h336/h720` prefix losses 简单平均。由于每个 prefix loss 又对 `1..H` 求均值，short future
steps 被多个 prefix 重复覆盖，long-tail steps 只被 long horizon 覆盖。

[Fact] 当前 horizons `[96,192,336,720]` 下，`0-96` segment 的平均 scalar supervision weight 是
`336-720` tail segment 的 `14.39x`；`96-192` 是 `6.89x`；`192-336` 是 `3.14x`。

[Moderate Evidence] Segment-level comparison shows A6 gains vs fixed TimeAlign shrink in the under-weighted tail:
overall early `0-96` relative MSE is `-3.57%`, while tail `336-720` is only `-0.16%`. ETTh2 and ETTm1 support this
reading; Weather is a counterexample.

[Decision] B7-UPO status is `prefix_imbalance_problem_candidate`, not method-ready. It is stronger than reviving B6
because it directly deepens unified prediction training mechanics and does not become a generic frequency auxiliary
loss. However, it must pass a stronger gradient/task diagnostic before any implementation.

[Next Required Action] Run `B7-GTD`: compute per-prefix gradient cosine similarities, gradient norms, and conflict
pairs on A6 shared parameters for small train batches. If conflict/imbalance is stable and aligns with segment-tail
weakness, then enter Step 4-6 method design. If not, pause StageB again.

### B8 Future-Query Aligned Architecture Direction

[Decision] `B7-UPO` 有价值，但当前降级为 small objective contribution candidate。随后 StageB 曾提出
`B8-FQA` 作为 architecture-level candidate。

[Motivation] StageA 主要改变 decoder/head。第二个主创新点应改变 feeding A6 learned-basis forecast operator
的 representation interface。

[Problem] A6-LBF-r256 为每个 channel 计算一个 sample-specific coefficient vector：

```text
coeff = learned_basis_coeff(hidden)
y[t, c] = learned_temporal_basis[t] @ coeff[c] + bias[t]
```

因此 `coeff[c]` 对 future position 不变。Future positions 由全局 basis rows 区分，但没有
sample-specific target-position representation。

[Idea] 引入 future-position query/placeholder tokens，使其 attend 到 history tokens，并在 learned-basis
operator 前生成 future-segment-specific coefficient modulation。modulation gate 零初始化，使初始 forward
与 clean A6-LBF-r256 完全等价。

[Literature] 本轮不是只参考 Zotero/本地 notes；已外部核验 TimeAlign、ElasTST、TimePerceiver 的 arXiv 或官方
repository 资料。TimeAlign 提供 future-aligned representation 的问题动机；ElasTST 证明 future
placeholders 与 structured masks 可服务 horizon-invariant varied-horizon forecasting；TimePerceiver 使用
target-position-aware decoder queries。B8 只把这些作为机制证据，并将其改写到 A6 的 basis-coefficient
interface。

[Diagnostic] `B8-OCD` 已完成，见
`analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707/b8_ocd_report.md`。诊断固定 clean A6
prediction 与 checkpoint 中的 `learned_temporal_basis`，比较 learned basis 与 DCT control 的 global/segment
residual correction。

[Fact] learned basis 的 segment-specific correction 确实相对 global correction 有明显 headroom。Rank 64 下
segment-minus-global reduction 为 ETTh2 `16.85%`、ETTm1 `28.19%`、Weather `22.01%`。

[Counter-Evidence] DCT control 的绝对 residual reduction 更强。Rank 64 segment reduction 中，learned basis
为 ETTh2 `79.05%`、ETTm1 `72.77%`、Weather `61.91%`，DCT control 为 ETTh2 `87.61%`、ETTm1
`91.85%`、Weather `91.18%`。

[Decision] `B8-FQA` 状态改为 `rejected_by_ocd_control`。叙事逻辑仍然通顺，但当前 evidence 不能证明它是
A6 learned-basis coefficient interface 特有的强 architecture problem；不要实现 B8。

[Rollback] StageB 回到 Step 2/3，继续寻找 architecture-level 第二主创新问题。B7-UPO 仍仅作为 small
objective contribution candidate 保留。

### B9 Native Future-Stage Operator Problem Candidate

[Decision] 用户明确排除 residual-style architecture 作为 paper-core method。StageB 的新候选是
`B9-FSN`: native future-stage-aware operator。

[Problem] A6-LBF-r256 的 primary path 使用一个 `coeff[b,c]` 服务所有 future stages：

```text
coeff = learned_basis_coeff(hidden)
y[t,c] = learned_temporal_basis[t] @ coeff[b,c] + bias[t]
```

这不是 residual correction 问题，而是 primary prediction path 问题：不同 future stages 是否对同一个
coefficient 施加不一致的训练方向？

[External Literature] 本轮不是只参考 Zotero/本地 notes；已外部核验 TimePerceiver、ElasTST、MQ-RNN 与
Temporal Fusion Transformer。TimePerceiver 支持 target timestamp queries；ElasTST 支持 future
placeholders/masks；MQ-RNN 明确使用 horizon-specific contexts；TFT 显式处理 known future inputs 和
multi-horizon architecture。

[Diagnostic] `B9-SGC` 已完成，见
`analysis/phase5_stage_b_b9_stage_gradient_diagnostic_20260707/b9_stage_gradient_report.md`。诊断不拟合
residual，不设计 correction module；它计算四个 future stage losses 对同一个 A6 `coeff` 的梯度方向。

[Fact] 三个 dataset 的 stage-gradient cosine 都很低：

| Dataset | Mean pairwise cosine | Early-tail cosine | Negative pair rate |
| --- | ---: | ---: | ---: |
| ETTh2 | `0.072` | `0.041` | `0.083` |
| ETTm1 | `0.171` | `0.112` | `0.042` |
| Weather | `0.048` | `0.014` | `0.083` |

[Step 4-6 Design] `B9-FSN-SCF` 即 Stage-Native Coefficient Field。它保留 A6 的
`learned_temporal_basis` 作为 unified operator 坐标系，但把 coefficient 从
`coeff: [B,C,K]` 扩展为 `coeff_field: [B,C,S,K]`，其中 `S=4` 对应 `[0,96)`,
`[96,192)`, `[192,336)`, `[336,720)`。

核心路径是：

```text
coeff_base = learned_basis_coeff(hidden)
coeff_s = StageCoefficientField(coeff_base, hidden, stage_token_s)
y[t in stage_s,c] = learned_temporal_basis[t] @ coeff_s[b,c] + bias[t]
```

[Theory Check] 若 `stage_gate_s=0`，所有 `coeff_s=coeff_base`，prediction 退回 clean A6；训练后
stage loss 的梯度主要作用于对应 stage coefficient field，同时仍共享 encoder、base coefficient 和 temporal
basis。这解决的是 primary-path stage pressure routing，不是 output residual repair。

[Narrative Gate] Step 4-6 已通过：问题清楚、机制在 tensor path 上可解释、与 StageA 的 learned-basis
operator 连续、与 TimePerceiver/ElasTST/MQ-RNN/TFT/SRP++ 有明确边界，且可 function-preserving 初始化。

[Step 7 Implementation] 最小 `B9-FSN-SCF` 与 `B9-no-stage-control` 已实现。新增 readout modes：
`stage-native-coefficient-field` 与 `stage-native-coefficient-field-no-stage`。本地 fallback/prefix 检查为：

```text
max_abs_b9_vs_a6_h720 = 0.0
max_abs_no_stage_vs_a6_h720 = 0.0
max_abs_b9_h96_vs_h720_prefix = 0.0
max_abs_no_stage_h96_vs_h720_prefix = 0.0
```

ETTh2 CPU smoke 已通过 B9 与 no-stage 两个 paths，且 `model_diagnostics.json` 已导出 stage gate 与参数量诊断。

[Step 8-10 Result] Remote small gate 已完成并同步到
`analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/`。

| Comparison | Overall MSE wins | Mean relative MSE |
| --- | ---: | ---: |
| `b9_fsn_scf` vs `a6_clean` | `12/12` | `-0.13%` |
| `b9_no_stage` vs `a6_clean` | `12/12` | `-0.13%` |
| `b9_fsn_scf` vs `b9_no_stage` | `2/12` | `+0.0036%` |

[Decision] `B9-FSN-SCF` 状态为 `blocked_by_no_stage_control`。它不能作为 paper-core method 继续推进，
因为相对 A6 的微弱收益被同参数 no-stage control 完全解释。不得启动 full matrix，也不得将结果解释为
future-stage-aware routing 的正证据。

[Rollback] 回到 Step 4 重新设计 native-stage mechanism；若找不到能压过 no-stage control 的机制约束，则回到
StageB Step 2/3 重新寻找第二主创新点。

### B10 Target-Set Conditioned Operator Redefinition

[Decision] B9-FSN-SCF 被 no-stage control 阻断后，StageB 不再继续简单 stage-token coefficient modulation。
新的问题候选是 `B10-TCO`: Target-Set Conditioned Operator。

[Problem] A6-LBF-r256 更准确地说是 prefix-compatible learned-basis trajectory operator：

```text
f_A6(history) -> y_{1:720}
return y_{1:H}
```

它支持多 horizon evaluation，但 requested horizon / target set $J$ 没有进入 computation graph。短 horizon
预测仍是从同一条 720-step trajectory 上 prefix slicing。

[Reframing] A6 的 learned-basis head 不是普通 dense Linear head，而是 factorized temporal-coordinate operator：

```text
coeff = learned_basis_coeff(hidden)
y[t,c] = learned_temporal_basis[t] @ coeff[b,c] + bias[t]
```

其中 `learned_temporal_basis[:, k]` 可理解为 shared temporal atom，`coeff[b,c,k]` 是 sample/channel-wise
coordinate。但这个结构仍没有表达“本次请求的 target set 是什么”。

[External Evidence] 本轮方向落地不只参考本地 notes。外部网络重新核验了 TimePerceiver、ElasTST、
MQ-RNN 与 Temporal Fusion Transformer：target timestamp queries、future placeholders/masks、
horizon-specific contexts 与 known future inputs 都支持 target/future-side information 进入 prediction graph。
B10 只吸收 target-set conditioning 的机制证据，不复制完整外部架构。

[Idea] B10 应研究：

```text
f_B10(history, J) -> y_J
```

其中 $J$ 是 requested target set，例如 `{1..96}`, `{1..192}`, `{1..336}`, `{1..720}`。B10 默认采用
`prefix-invariant target-set computation`：requested target set 进入 forward graph，但更长 target set
中的后续 positions 不允许改写已有 prefix outputs。

[Existing Evidence] 当前 artifacts 支持该问题值得诊断：

- A6 vs fixed-horizon specialist 虽整体 `9/12` wins、mean MSE `-4.13%`，但 Weather 只有 `2/4` wins，
  ETTm1-720 仍输给 fixed specialist；
- B7-UPO 显示当前 multi-prefix slicing/objective 下 tail region gain 仅 `-0.16%`；
- B9-SCF 显示把 stage token 塞进 coefficient 会被 no-stage capacity control 解释。
- B10-TSI-A 显示 A6 basis 不是 stage-blind：top64 atoms 的 entropy 为
  `0.8108/0.8764/0.8658`，但 rank32 stage row-space overlap 只有
  `0.1324/0.1510/0.1368`。
- B10-TSI-B 显示真实 `coeff` 同时激活多个低同向性 stage row subspaces：rank64 projection share 为
  `0.3882/0.4950/0.2764`，projection cosine 为 `0.3759/0.4702/0.1639`，output entropy 为
  `0.7969/0.8958/0.9042`。

[Updated Boundary] B10 不能叙事为“给 basis 补 stage 信息”。更准确的问题是：

> `learned_temporal_basis` 已经形成 stage-differentiated coefficient geometry，但
> `learned_basis_coeff(hidden)` 仍只生成一个 target-set-blind coefficient/state；requested target set
> 没有进入 `history -> coeff/state` 生成路径。

[B10-TSI-C Result] frozen-coeff linear target-set readout 未能超过 no-target-set capacity controls：

- ETTh2: target vs pooled-4H `-185.5316%`;
- ETTm1: target vs pooled-4H `+0.2812%`;
- Weather: target vs pooled-4H `-26.5683%`.

[Failure Attribution] 该结果不能否定 target-set-aware 方向。ETTh2/Weather 的明显发散说明当前 diagnostic
存在 `readout_or_head_design_wrong` 或 `optimization_or_numeric_pathology` 风险；它只否定
`frozen coeff -> Linear_s(coeff) -> basis_s projection` 这个过晚、过线性的 readout 设计。

[B10-TSI-D Result] failure-attribution diagnostic 已完成。它比较 `coeff_late`、`memory_pool`、
`memory_plus_coeff` 三个 feature sources，并加入 rank-truncated row-space target、pooled control、
wrong-target control 和 shrinkage target-set readout。

主 rank64 stabilized target vs pooled control：

- `coeff_late`: `-12.3695%`;
- `memory_pool`: `-40.7499%`;
- `memory_plus_coeff`: `-44.4687%`。

rank16 稳定性对照仍为负：

- `coeff_late`: `-5.1506%`;
- `memory_pool`: `-32.0345%`;
- `memory_plus_coeff`: `-36.3672%`。

[Decision] `B10-TCO` 当前状态为 `offline_readout_route_blocked_but_direction_not_rejected`。
这阻断 frozen/offline ridge readout route，不能阻断更大的 native target-query architecture 方向。

[Next Required Action] 不再继续用 frozen offline oracle 消耗。下一步只能二选一：

1. 写 Step 4-6 native trainable target-query memory readout 的 narrative/method gate，并保留 no-target
   query implementation control；
2. 若该 narrative gate 不能解释 B10-TSI-C/D 的 readout pathology，则 B10 rollback 到 StageB Step 2/3。

## Active Implementation

| File | Role |
| --- | --- |
| `baselines/timealign_official/models/TimeAlign.py` | clean official + A6-LBF model |
| `baselines/timealign_official/train_repo.py` | clean training/evaluation adapter |
| `scripts/remote/run_phase5_a6_lbf_r256_main.sh` | clean A6-LBF-r256 remote runner |
| `docs/code-explanation/phase5-clean-timealign-a6-lbf.md` | implementation explanation |
| `scripts/analyze_phase5_stage_b_reliability_diagnostic.py` | StageB B1 diagnostic analyzer |
| `scripts/analyze_phase5_stage_b_b3_dsr_diagnostic.py` | StageB B3 diagnostic analyzer |
| `scripts/analyze_phase5_stage_b_timealign_dependency_audit.py` | StageB TimeAlign dependency audit analyzer |
| `scripts/analyze_phase5_stage_b_timealign_dependency_ablation.py` | returned no-align/no-recon dependency ablation analyzer |
| `scripts/analyze_phase5_stage_b_b6_prefix_objective_diagnostic.py` | StageB B6 prefix-native objective diagnostic analyzer |
| `scripts/analyze_phase5_a6_clean_operator_rerun.py` | clean A6 validation analyzer |
| `scripts/analyze_phase5_stage_b_unified_prefix_optimization.py` | B7 unified prefix optimization diagnostic analyzer |
| `scripts/analyze_phase5_stage_b_b8_ocd_coefficient_oracle.py` | B8-OCD coefficient-space oracle diagnostic analyzer |
| `scripts/analyze_phase5_stage_b_b9_stage_gradient_diagnostic.py` | B9-SGC native future-stage gradient diagnostic analyzer |
| `scripts/analyze_phase5_stage_b_b9_fsn_scf_small_gate.py` | B9-FSN-SCF small gate analyzer |
| `scripts/remote/run_phase5_stage_b_timealign_dependency_ablation.sh` | completed no-align/no-recon ablation runner |
| `docs/experiments/phase5-stage-b-distance-normalized-seasonal-residual-diagnostic.md` | StageB B3 diagnostic protocol |
| `docs/experiments/phase5-stage-b-timealign-dependency-and-basis-align-diagnostic.md` | StageB dependency/basis-align protocol |
| `docs/experiments/phase5-stage-b-prefix-native-label-objective-diagnostic.md` | StageB B6 prefix-native objective diagnostic protocol |
| `docs/experiments/phase5-stage-b-unified-prefix-optimization-diagnostic.md` | StageB B7 unified prefix optimization diagnostic protocol |
| `docs/experiments/phase5-stage-b-future-query-aligned-basis-architecture.md` | StageB B8 future-query aligned basis architecture protocol |
| `docs/experiments/phase5-stage-b-native-future-stage-operator.md` | StageB B9 native future-stage operator protocol |
| `docs/experiments/phase5-stage-b-target-set-conditioned-operator.md` | StageB B10 target-set-conditioned operator protocol |
| `docs/code-explanation/phase5-stage-b-b6-prefix-objective-diagnostic.md` | B6 diagnostic analyzer explanation |
| `docs/code-explanation/phase5-clean-a6-rerun-analysis.md` | clean A6 validation analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b7-unified-prefix-optimization.md` | B7 diagnostic analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b8-ocd-coefficient-oracle.md` | B8-OCD analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b9-stage-gradient-diagnostic.md` | B9-SGC analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b9-fsn-scf.md` | B9-FSN-SCF implementation explanation |
| `docs/code-explanation/phase5-stage-b-b10-tsi-basis-geometry.md` | B10-TSI-A basis geometry analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b10-tsi-coeff-usage.md` | B10-TSI-B coefficient usage analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b10-tsi-target-set-oracle.md` | B10-TSI-C oracle/control analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b10-tsi-failure-attribution.md` | B10-TSI-D failure attribution analyzer explanation |
| `scripts/analyze_phase5_stage_b_b10_tsi_basis_geometry.py` | B10-TSI-A basis geometry analyzer |
| `scripts/analyze_phase5_stage_b_b10_tsi_coeff_usage.py` | B10-TSI-B coefficient usage analyzer |
| `scripts/analyze_phase5_stage_b_b10_tsi_target_set_oracle.py` | B10-TSI-C target-set oracle/control analyzer |
| `scripts/analyze_phase5_stage_b_b10_tsi_failure_attribution.py` | B10-TSI-D failure attribution analyzer |
| `analysis/phase5_stage_b_b10_tsi_basis_geometry_20260708/b10_tsi_basis_geometry_report.md` | B10-TSI-A basis geometry report |
| `analysis/phase5_stage_b_b10_tsi_coeff_usage_20260708/b10_tsi_coeff_usage_report.md` | B10-TSI-B coefficient usage report |
| `analysis/phase5_stage_b_b10_tsi_target_set_oracle_20260708/b10_tsi_target_set_oracle_report.md` | B10-TSI-C oracle/control report |
| `analysis/phase5_stage_b_b10_tsi_failure_attribution_20260708/b10_tsi_failure_attribution_report.md` | B10-TSI-D rank64 failure attribution report |
| `analysis/phase5_stage_b_b10_tsi_failure_attribution_rank16_20260708/b10_tsi_failure_attribution_report.md` | B10-TSI-D rank16 stability control |
| `scripts/remote/run_phase5_stage_b_b9_fsn_scf_small_gate.sh` | B9-FSN-SCF remote small gate runner |
| `scripts/sync_phase5_stage_b_b9_fsn_scf_small_gate_results.sh` | B9-FSN-SCF result sync/analyze wrapper |

## Archive Map

| Path | Content |
| --- | --- |
| `docs/archive/phase5-stage-a/` | old StageA design/code-explanation docs |
| `analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/` | accepted A6-LBF evidence |
| `analysis/phase5_stage_b_reliability_diagnostic_20260706/` | B1 distance-confounded reliability diagnostic |
| `analysis/phase5_stage_b_step23_redefinition_20260706/` | B3 problem redefinition audit |
| `analysis/phase5_stage_b_distance_normalized_seasonal_residual_20260706/` | B3 partial/not method-ready diagnostic |
| `analysis/phase5_stage_b_timealign_dependency_audit_20260706/` | TimeAlign dependency risk audit |
| `analysis/phase5_stage_b_timealign_dependency_ablation_20260706/` | B4 dependency ablation result |
| `analysis/phase5_stage_b_prefix_native_objective_diagnostic_20260706/` | B6 negative diagnostic |
| `analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/` | clean A6 validation result |
| `analysis/phase5_stage_b_unified_prefix_optimization_20260707/` | B7 unified prefix optimization diagnostic |
| `analysis/phase5_stage_b_future_query_aligned_architecture_research_20260707/` | B8 future-query aligned architecture direction research |
| `analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707/` | B8-OCD negative coefficient oracle diagnostic |
| `analysis/phase5_stage_b_b9_stage_gradient_diagnostic_20260707/` | B9-SGC positive problem-candidate diagnostic |
| `analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/` | B9-FSN-SCF remote launch record and future small-gate analysis |
| `analysis/phase5_stage_a_architecture_exhaustion_audit_20260705/` | old route-level audit before A6-LBF was promoted |

## Current Prohibitions

- 不再启动 A5/A6-QBR/A6S/A6ST/A7DG/A8TAG 旧路线；
- 不把 old B0 ablation 直接作为 StageB method；
- 不恢复 teacher/self-teacher/EMA/diagnostic export 旧代码，除非新的 StageB narrative gate 明确需要；
- 不在 remote 端手工修改代码；先 commit/push，再 remote `git pull`。

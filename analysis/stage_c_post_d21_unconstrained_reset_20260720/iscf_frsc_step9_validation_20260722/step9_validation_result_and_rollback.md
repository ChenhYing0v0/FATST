# SC-ISCF-FRSC-v0 Step9 Validation Result and Rollback

## 1. Decision record

| Field | Record |
| --- | --- |
| `current_step` | Step9 validation audit complete；Step10 continuation decision |
| `problem` | full-rank scope conditioning能否保留ISCF carrier，同时产生超过identity与generic conditioning controls的useful specialization |
| `existence_evidence` | frozen D1.1 canonical alpha .55相对identity `+0.7997%` MSE；same-alpha random显著负向；best global `+0.8677%` |
| `idea` | $Q_s=P_s+(1-\alpha)(I-P_s)$；candidate scope alpha .55 |
| `theory_check` | minimum eigenvalue .45；invertible；no zero-gradient direction；no requested-H/loss/router |
| `design` | four new arms × five datasets × seed2021；5 frozen identity references；H96/192/336/720 validation MSE/MAE |
| `narrative_gate` | conditional pass only if candidate beats identity, same-alpha global, best-tuned global and random-binding controls |
| `effectiveness_gate` | failed on validation continuation surface；not a formal-test or final paper-facing rejection |
| `artifacts` | `remote_analysis/*.csv/json`、remote launch records、25/25 run audits |
| `decision` | `frsc_v0_validation_continuation_not_supported_rollback_step4`；formal test false；retain ISCF architecture prior |

## 2. Artifact and protocol integrity

[Fact] remote training在`2026-07-22T11:30:46+08:00`结束，总时长约49分26秒。20/20 new runs均生成checkpoint、
training log、four-horizon metrics、effective config、initialization contract、model diagnostics、validation diagnostics与trained
invariants。联合5个historical identity references后，analyzer审计25/25 runs和100/100 validation rows。

| Check | Result |
| --- | ---: |
| new run artifacts | 20/20 |
| effective run audits | 25/25 |
| validation rows | 100/100 |
| new trained invariants | 20/20 pass |
| actual paired encoder/readout initialization | 5/5 datasets；每dataset各1个encoder hash与1个PCSD hash |
| all finite / readout contract / full rank | 20/20 |
| full-prefix maximum gap | 0 |
| log Traceback/OOM/NaN/Inf | 0 |
| evaluation split | validation only |
| `uses_test_split` | false |

因此`optimization_or_numeric_pathology=false`，也没有frozen-replacement fairness问题：四个new arms均为from-scratch
joint training；identity只作为先前同协议、同seed、同profile的exact parent reference。

## 3. Primary effectiveness: candidate versus identity

gain定义为$100(1-\mathrm{candidate}/\mathrm{reference})$，正值表示candidate更优。

| Metric | Macro gain | Cell wins | Dataset wins | Horizon wins | Worst dataset degradation | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MSE | -1.2745% | 7/20 | 2/5 | 0/4 | 4.6493% | fail |
| MAE | -0.4184% | 10/20 | 2/5 | 0/4 | 1.7055% | fail |

预注册MSE要求macro至少`+0.3%`、12/20 cells、3/5 datasets、3/4 horizons且worst dataset degradation不超过
3%；MAE要求macro为正。所有primary guards均失败。

### Dataset and horizon attribution

| Dataset | MSE gain vs identity | MAE gain vs identity |
| --- | ---: | ---: |
| Weather | +0.1202% | -0.2543% |
| ETTm1 | +0.3494% | +0.4010% |
| ETTh1 | -1.4854% | -0.6371% |
| ETTh2 | -4.6493% | -1.7055% |
| ETTm2 | -0.7076% | +0.1036% |

MSE在H96/H192/H336/H720分别为`-1.7525%/-1.4795%/-1.1776%/-0.6886%`；损失随horizon变长而减小，
但没有任何聚合horizon转正。因此FRSC-v0不能支撑unified-horizon effectiveness claim，且不能把问题解释为单一短或长
horizon异常。

## 4. Matched mechanism attribution

| Comparison | MSE gain | MAE gain | MSE cells/datasets/horizons | Decision |
| --- | ---: | ---: | --- | --- |
| scope-a055 vs global-a055 | +0.7215% | +0.3953% | 19/20；5/5；4/4 | same-alpha scope effect supported |
| scope-a055 vs global-a045 | +0.0703% | +0.0599% | 14/20；4/5；4/4 | below preregistered +0.1% MSE gate |
| scope-a055 vs random-a055 | +0.1781% | -0.0330% | 11/20；4/5；2/4 | canonical binding unsupported |

[Strong Evidence] scope topology在相同conditioning strength下不是恒等或完全无效；它稳定超过global-a055。但
[Fact] best-tuned global-a045几乎消除该优势，说明observed gain尚可由generic conditioning-strength/geometry interaction解释。
canonical-vs-random同时未达到`+0.3%` MSE gate且MAE为负，因此不能建立canonical temporal-scope binding贡献。

random control只用于归因：其失败阻止canonical claim，但不得方向级拒绝ISCF architecture。

## 5. Internal mechanism health

candidate aggregate statistics：

| Statistic | Value | Interpretation |
| --- | ---: | --- |
| all finite / projection contract | true | execution healthy |
| pairwise normalized RMS | 0.1741 | arms未collapse；identity为0.1694 |
| minimum pairwise normalized RMS | 0.0821 | all arm pairs可辨 |
| conditioning-delta/raw RMS | 0.2448 | intervention material，不是near-identity |
| policy normalized entropy | 0.7943 | policy未collapse |
| scope winner count | 4/5 | 多个arms在future bins中具有skill |
| oracle headroom | 3.9158% | 低于identity 5.0381% |

internal health总gate只因oracle-headroom preservation guard失败：candidate比identity低`1.1223` percentage points，而
容许差为`0.5`。这不等价于numeric pathology，也不单独证明fusion更差；它表示当前conditioning没有保留parent的arm-level
skill-spread diagnostic。结合primary performance negative，内部活性不能挽救candidate。

## 6. Failure attribution and rollback

- `hypothesis_false`：未在方向级成立。same-alpha scope-vs-global为稳定positive，且本次仅为single-seed validation。
- `capacity_control_explains`：在exact-candidate continuation层成立。identity明显更强，best-tuned same-capacity global control
  几乎解释scope gain。
- `intervention_point_wrong` / `readout_or_head_design_wrong`：仍是合理可能性；本结果只说明固定alpha .55的当前FRSC
  conditioning不能把scope effect转成carrier gain。
- `optimization_or_numeric_pathology`：排除。

Decision=`frsc_v0_validation_continuation_not_supported_rollback_step4`。不执行formal test、不补seed、不做per-dataset/
per-horizon alpha tuning，也不叠加loss/router。关闭的是exact FRSC-v0 development candidate，不是ISCF architecture prior。
下一步回到Step4，从“如何让scope arms形成不同且有用的feature/forecast roles”重新设计完整mechanism；任何新候选必须重新
通过narrative/design gate后才能实现。

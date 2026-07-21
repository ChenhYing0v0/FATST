# ISCF-v0 Function-Level Audit Result and Step4 Handoff

## 1. Long-stage record

| Field | Record |
| --- | --- |
| `current_step` | ISCF-v0 carrier freeze complete；Step2/3 existing-artifact function audit complete；return narrow Step4 |
| `problem` | independent coupling scopes是否具有稳定、可学习且不是人为ordered的function relation，从而支持一个新的paper problem？ |
| `existence_evidence` | ISCF-v0由FCC independent arm原样冻结；post-hoc derived existing-test comparison vs A6_FULL MSE/MAE=`+1.3584%/+0.9144%`，5/5 datasets、4/4 horizons、3/3 seeds正向 |
| `idea` | 不训练新模型，只复用15个checkpoints已保存的arm forecasts、targets、row-bin errors和policy usage，审计aligned low dimension、common/private residual、complementarity、topology与scale order |
| `theory_check` | function-space比parameter SVD更贴近预测函数；但probe只有256 rows且来自已访问test tensors，结果只能作为test-informed problem evidence |
| `design` | 5 datasets × 3 seeds；`probe_arms [256,5,720]`；64次independent circular-shift null；四个冻结gates |
| `narrative_gate` | not evaluated；generic low-rank/shared-private/multi-scale mixing已有强prior，不足以直接形成contribution |
| `effectiveness_gate` | not applicable；没有new method、training或new test access |
| `artifacts` | `summary.json`、five CSV summaries、frozen config/protocol/analyzer |
| `decision` | `function_relation_unresolved_requires_narrow_step4_audit` |

## 2. Carrier freeze

[Decision] 用户于`2026-07-21`明确把FCC的`SIFF_INDEPENDENT_EQUAL`固定为后续research carrier，paper-facing
identity为：

```text
ISCF-v0 — Independent-Scope Coupling Field
```

冻结边界：

- code identity仍为`readout_mode=siff-independent-scope-control`，保证15个existing checkpoints可复现；
- $Q=5$、`scale_basis=I_5`、scopes=`{1,48,144,360,720}`；
- policy=`direct`，objective=`equal_skill`；
- ranks为ETTh1/ETTh2/ETTm1/ETTm2/Weather=`109/116/116/106/116`；
- encoder profiles、from-scratch joint training、four-horizon validation selector和full-crop evaluation均不变；
- status=`strong_research_carrier_frozen_paper_core_pending`；
- 它不继承SIFF-v2失败的ordered-field claim，改名不等于paper-core promotion。

机器contract：`configs/stage_c_iscf_v0_carrier.json`。

## 3. Artifact and split audit

| Item | Result |
| --- | --- |
| run units | 15/15：5 datasets × seeds2021/2022/2023 |
| source | existing `pcsd_test_audit_diagnostics.npz` |
| probe arms | 每run `[256,5,720]` |
| full row-bin tensors | arm MSE、fused MSE、policy usage完整 |
| finite | all arrays finite |
| new checkpoint training | false |
| dataset loader/test-label access | false；只读取既有NPZ |
| checkpoint mutation | false |
| role | `diagnostic_only_test_informed_reuse` |

本轮没有`component_history_modes`，因此不声称mode/parameter subspace evidence。frozen-output关系也不能替代
future end-to-end joint-training effectiveness。

## 4. Frozen gate result

| Gate | Result | Decision |
| --- | ---: | --- |
| aligned low dimension | observed EV2超过shift-null：`0/15` | fail |
| common + private residual | common超过null `15/15`；median private=`0.0680` | pass |
| scope complementarity | median oracle headroom=`8.5813%`；median unique best scopes=`3` | pass |
| cross-seed topology stability | median Spearman≥0.5：`4/5` datasets | pass |

Machine decision：

```text
function_relation_unresolved_requires_narrow_step4_audit
```

三项通过不覆盖low-dimensional exact hypothesis的失败，所以不能直接实现`learned low-rank scope matrix`。

## 5. Aligned low-dimensional relation

| Statistic | Result |
| --- | ---: |
| median centered scope EV2 | `0.6281` |
| median shift-null EV2 p95 | `0.7223` |
| observed > null | `0/15` |
| median centered effective rank | machine CSV retained |

[Strong Evidence] 五个scope deviations不是一个稳定的two-factor aligned field。independent shifts保留各arm marginal
distribution但破坏coordinate alignment后，EV2反而更高；因此不能把ISCF-v0压缩成`Q=2` learned mixing并预期保留
其function structure。

[Decision] 关闭“根据当前checkpoints直接设计低秩learned relation matrix”的立即路线。这不是对所有learned
scope relation的方向级拒绝，只拒绝由本轮evidence支持的two-factor compression叙事。

## 6. Common/private structure and heterogeneity

| Dataset | Common residual | Private residual | Pairwise prediction NRMSE | Residual corr. |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | `0.8850` | `0.1150` | `0.4378` | `0.8516` |
| ETTh2 | `0.9204` | `0.0796` | `0.0910` | `0.8996` |
| ETTm1 | `0.9787` | `0.0213` | `0.1668` | `0.9735` |
| ETTm2 | `0.9290` | `0.0710` | `0.0854` | `0.9097` |
| Weather | `0.9935` | `0.0065` | `0.1781` | `0.9912` |

15/15 runs的common residual energy均超过shift null；median residual correlation为`0.9151`。但private energy高度
dataset-dependent：ETTh1约11.5%，Weather仅0.65%，ETTm1约2.13%。全局15个runs中只有9个private energy至少5%。

[Interpretation] scopes共享大量forecast difficulty，同时在部分datasets保留不可忽略的private behavior。这支持继续问
“何处共享、何处保持scope-specific”，但不支持一个固定比例、固定低秩或universal common/private architecture。

[Self-critique] shared residual可能来自相同target difficulty、shared encoder/synthesis和共同训练objective，而不是一个
可被新module利用的causal common component。现有frozen functions只能提供problem clue，不能证明增加shared path会提升
end-to-end performance。

## 7. Complementarity, fusion and scope usage

| Statistic | Result |
| --- | ---: |
| median oracle headroom over fused | `8.5813%` |
| median fused gain over best fixed arm | `+0.4041%` |
| fused gain positive | `9/15` runs |
| median unique best scopes across 8 bins | `3` |
| median policy normalized entropy | `0.8118` |

oracle headroom表明scope forecasts具有互补误差，但learned fusion只在9/15 runs超过best fixed arm。dataset median fused
gain为ETTh1 `+5.5438%`、ETTh2 `+0.4041%`、ETTm1 `-0.0506%`、ETTm2 `+1.7849%`、Weather
`-0.6457%`。因此不能从oracle反推一个新router；policy/selection仍是未解决的实现问题。

八个future bins × 15 runs共120个best-scope slots中，scale720占72、scale360占32，二者合计`86.67%`；其余
scale1/48/144合计16。五个scopes不等价地贡献，但该统计来自test-informed probes，禁止据此直接删scope或按dataset
选择arm set。

## 8. Topology and ordered-scale semantics

| Dataset | Median cross-seed topology Spearman | Stable≥0.5 |
| --- | ---: | --- |
| ETTh1 | `0.9636` | yes |
| ETTh2 | `0.7697` | yes |
| ETTm1 | `0.7939` | yes |
| ETTm2 | `0.3091` | no |
| Weather | `0.5758` | yes |

pairwise function topology在4/5 datasets跨seed稳定，说明independent fields不是完全任意的parameter noise。但
scale-distance Spearman的run median仅`0.2121`，15 runs中11个为正、只有7个至少0.3；Weather median为负。

[Decision] canonical scale order只留下weak/non-universal footprint，不能恢复SIFF-v2 ordered-field claim。下一Step4若继续，
必须研究non-ordered、dataset-heterogeneous scope relationship，而不是把log-scale重新写回architecture。

## 9. Latest primary-source boundary

检索日期：`2026-07-21`。query scope：`multi-scale forecasting experts`、`shared/private multi-task relation`、
`data-driven expert specialization`、`multi-resolution expert collaboration`。来源使用conference proceedings、OpenReview、
AAAI与arXiv primary pages。Zotero semantic index返回item IDs但local metadata endpoint connection refused，故本轮不能可靠
标记FSA subset presence；repo canonical notes已包含TimeMixer/Pathformer/Moirai-MoE，其余列为external discovery。

| Primary work | Coverage | ISCF boundary |
| --- | --- | --- |
| [TimeMixer, ICLR 2024](https://openreview.net/pdf?id=7oLshfEIC2) | multiscale decomposition与Future-Multipredictor-Mixing | multiple future predictors或complementarity本身不新 |
| [Pathformer, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2be6705de7412adf107900add727a795-Abstract-Conference.html) | input-side multi-scale modeling与adaptive pathways | generic adaptive multi-scale selection不新 |
| [Cross-Stitch Networks, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.html) | end-to-end learned shared/task-specific representation mixing | generic common/private mixing不新 |
| [Factorial Multi-Task Learning, ICML 2013](https://proceedings.mlr.press/v28/gupta13a.html) | task relatedness、shared low-dimensional subspaces、partial/no sharing | learned task-relation/low-rank sharing不新；本轮low-rank evidence还失败 |
| [Moirai-MoE, ICML 2025](https://proceedings.mlr.press/v267/liu25an.html) | 反对human frequency specialization，采用data-driven token experts | generic data-driven experts与反heuristic叙事不新 |
| [M2FMoE, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39362) | multi-resolution frequency experts、shared splitter与inter-expert collaboration | multi-resolution expert collaboration已有最新覆盖 |

[Decision] 不能把下一贡献写成“独立多尺度专家”“shared-private mixing”“learned relation matrix”或“adaptive fusion”。
只有完整链`future-output coupling scope -> empirically identified non-ordered relation -> native jointly trained operator ->
matched independent/ordered/capacity controls`才可能形成contribution-level novelty；当前尚未完成这条链。

## 10. Failure attribution and next cursor

- **What failed**：aligned two-factor/low-dimensional relation 0/15；canonical order weak；fusion gain不是15/15稳定；
- **What did not fail**：ISCF-v0 carrier performance、function diversity、oracle complementarity和4/5 dataset topology stability；
- **What remains untested**：validation-only mode activations、shared/private structure的可干预性、jointly trained
  non-ordered relation operator是否超过ISCF-v0；
- **Direction level**：`diagnostic_only_unresolved`。frozen test probes不能通过或拒绝future method direction；
- **Rollback**：回Step4 source-informed problem formulation，不进入Step7，不启动remote training/test。

下一节点限定为：

```text
ISCF Step4: non-ordered common/scope-specific relation problem audit
```

在任何method implementation前必须回答：

1. common residual是否在validation activations/Jacobians中仍存在，而不是仅由shared target difficulty产生；
2. dataset-heterogeneous private energy能否用一个统一operator contract表达，禁止per-dataset architecture selection；
3. 新operator如何区别于Cross-Stitch、generic multi-task relation learning与multi-scale expert mixing；
4. 必须保留ISCF-v0、ordered SIFF-v2、common-only/Q1和matched capacity controls；
5. 不预设第二loss/router，不使用requested H补充无信息condition。

## 11. Artifacts

- `configs/stage_c_iscf_v0_carrier.json`；
- `scripts/analyze_stage_c_iscf_v0_function_audit.py`；
- `protocol.md`；
- `summary.json`；
- `run_function_metrics.csv`；
- `dataset_function_summary.csv`；
- `pairwise_scope_metrics.csv`；
- `bin_scope_specialization.csv`；
- `seed_topology_stability.csv`。

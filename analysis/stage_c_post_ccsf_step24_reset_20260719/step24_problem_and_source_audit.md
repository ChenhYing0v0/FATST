# post-CCSF Step 2/4 问题重构与 source-informed audit

## 1. 当前研究记录

| Field | Content |
| --- | --- |
| `current_step` | Contribution 1 Step 2/4；Contribution 2 Step 2 |
| `problem` | fixed-past unified multi-horizon generation中，除“预测完整future再prefix crop”之外，是否还存在可归因、可发表的native decoder contract？ |
| `existence_evidence` | D14-A证明output-sharing scope crossing真实；D6证明local/global support随prefix变化；但D11、PCSD、PCC、SIFF与CCSF均未把这些现象转化为可归因的paper-core mechanism |
| `idea` | 先证明requested horizon与exact projectivity之间的约束，再把剩余问题收紧为future-coordinate interaction是否存在prefix-safe增量价值 |
| `theory_check` | exact projectivity下，requested horizon不能改变共享prefix；任何真正的$H$-adaptive prediction要么冗余，要么破坏projectivity |
| `design` | external primary-source audit + `SC-D17-PFC` dual-carrier validation-fit → test-evaluation diagnostic；不实现method、不启动remote training |
| `narrative_gate` | no active method candidate；仅冻结problem diagnostic |
| `effectiveness_gate` | not applicable；D17不产生paper-facing effectiveness claim |
| `artifacts` | 本文、`configs/stage_c_d17_projective_future_context_diagnostic.json`、D17结果目录 |
| `decision` | 关闭“exact-projective且requested-$H$ adaptive”的问题表述；保留future-coordinate operator作为唯一待验证的decoder problem family |

检索日期为2026-07-19。按项目规则，本次以外部primary sources为主；Zotero只作为seed library，未用其覆盖情况
判断novelty。

## 2. 先处理一个理论矛盾

设固定past为$x$，requested horizon为$H\leq T$，统一模型输出

$$
F_H(x)\in\mathbb R^H.
$$

exact prefix projectivity要求对任意$H\leq K\leq T$：

$$
F_H(x)=P_HF_K(x),
$$

其中$P_H$只保留前$H$个坐标。令

$$
G(x)=F_T(x),
$$

立即得到

$$
F_H(x)=P_HG(x).
$$

因此：

1. [Fact] 在固定$x$与deterministic point forecast下，exact-projective family完全由一个full-domain operator
   $G$决定；
2. [Fact] requested horizon $H$不能改变任何共享prefix坐标，否则违反上式；
3. [Strong Evidence] 把$H$显式输入decoder、router或arm，只可能被网络忽略，或产生horizon-specific shared-prefix
   prediction并破坏projectivity；
4. [Inference] 当前论文不能同时把“shared-prefix严格不变”与“prediction随requested horizon自适应”都写成核心
   机制。二者在该设定下不是两个互补优点，而是相互排斥的contract。

这不是说multi-horizon问题不存在。它说明fixed-past设定下真正可学习的变量是history $x$与future coordinate
$\tau$，而不是deployment request $H$。multi-horizon价值只能落在：

- 同一$G(x)$如何对不同future coordinates组织生成；
- 不同prefix risk如何训练和评价同一个$G$；
- 如何在不访问prefix外部信息时保留统一推理与截断一致性。

## 3. 历史证据重新归位

| Evidence | 已证明 | 未证明 / 不得外推 |
| --- | --- | --- |
| D14-A dual-carrier, 3 seeds | point/block/global output sharing的最优粒度随dataset/horizon变化；sample-specific oracle约6.8%–8.6% | 一个learned scale router能够利用该headroom |
| D6 | local b144与global DCT在short/long prefix上存在稳定crossing | 该crossing来自multi-horizon gradient conflict |
| D11 | strict short/long directional gradient conflict为0/5；support-specific仅2/5 | 不能把“horizon conflict”作为Contribution 1基础 |
| PCSD-CF | shared coupling field可构造多scope arms；exact v1 arms失衡、fused forecast失败 | coupling-scope problem为假 |
| PCC | generic/equal-skill controls解释大部分收益 | nested-prefix credit transport是有效training contribution |
| SIFF-v2 parent | 性能接近paper level，arms健康、ordered field有部分specificity | ordered scale field与adaptive routing已形成完整归因 |
| CCSF + D2–D4 | contrast具有semantic sensitivity；region competence更可辨识 | contrast、teacher、covariance或temperature能实现net mixture utility |

[Strong Evidence] 过去的主要误区不是“没有找到更强router”，而是把真实的output heterogeneity直接等同于
“可由fixed-past识别的best-arm selection”。D14的oracle是target-conditional oracle；它不保证inference时存在
足够信息选择该arm。CCSF的失败正好切断了这一步外推。

## 4. 外部工作带来的claim边界

| Primary source | 已覆盖的核心问题 | 对当前项目的约束 |
| --- | --- | --- |
| ElasTST, NeurIPS 2024, https://proceedings.neurips.cc/paper_files/paper/2024/hash/d7aa002885ccbe68cf6880da583761b2-Abstract-Conference.html | varied-horizon inference invariance、structured attention、multi-scale patch、horizon reweighting | projectivity、普通multi-scale与prefix reweighting不能单独成为两项创新 |
| TimeMixer, ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html | past multiscale mixing与Future-Multipredictor-Mixing | “多个scale predictor再融合”已有强直接先例 |
| ProNet, Information Sciences 2024, https://arxiv.org/abs/2310.19322 | AR/NAR progressive generation与output-space prediction dependency | “利用future-output dependency”本身不是新claim |
| Implicit Forecaster, NeurIPS 2025, https://openreview.net/forum?id=gqoeQPhQcE | 用frequency/amplitude/phase constituent waves形成global forecast | global future-wave decoder已被直接覆盖 |
| TimePerceiver, NeurIPS 2025, https://proceedings.neurips.cc/paper_files/paper/2025/hash/c6c682ba9bd8839104f2a82901da4109-Abstract-Conference.html | target timestamp query、encoder-decoder与training co-design | future query本身不是足够的新意 |
| QDF, ICLR 2026, https://openreview.net/forum?id=vpO8n9AqEG | label autocorrelation的off-diagonal quadratic loss与future-step heterogeneous weights | generic output-covariance loss与step weighting均有直接prior |
| Loss Shaping, ICML 2024, https://openreview.net/forum?id=9CCoVyFuEp | per-step constraints与primal-dual loss shaping | worst-horizon/minimax式training也不是空白区域 |

本次检索没有发现一个与下述完整chain完全等价的工作：

`fixed-past full-domain draft -> prefix-causal future-context operator -> exact crop invariance -> direct parallel output`

但这只是[Medium-confidence absence finding]，不是novelty证明。ProNet、AR forecasters、trajectory refinement与
causal sequence models都构成邻近prior。若后续进入Step 4，必须把区别写在完整contract上，而不能声称首次建模
future dependency。

## 5. 三类问题候选的去留

### 5.1 Requested-$H$ adaptive operator：关闭

若保持exact projectivity，则$H$对共享prefix没有作用；若让$H$有作用，则shared prefix不再一致。该方向既存在
理论冲突，也受到ElasTST与varied-horizon prior的直接压力，不再作为当前论文核心。

### 5.2 Prefix-risk / horizon-measure objective：降为protocol与control

A6_MEASURE已经实现对标准prefix集合的统一validation selection与measure-aligned训练；ElasTST、QDF与Loss
Shaping又分别覆盖horizon reweighting、future-step coupling和per-step constrained optimization。除非新的decoder
产生一个此前不存在的training mismatch，否则现在单独设计更复杂loss只会成为第三次“objective先于mechanism”
的尝试。Contribution 2继续停在Step 2。

### 5.3 Prefix-safe future-coordinate operator：保留为problem family

设一个强parent先并行产生full-domain draft

$$
\tilde y_{1:T}=G_0(x).
$$

再考虑满足三角信息约束的operator：

$$
\hat y_\tau=\Phi_\theta\left(x,\tau,\tilde y_{1:\tau}\right).
$$

因为第$\tau$个输出只读取$\tilde y_{1:\tau}$，任意prefix crop不会改变已生成坐标。这条路线不同于：

- requested-$H$ conditioning：$\Phi$不知道$H$；
- autoregressive rollout：$\tilde y_{1:\tau}$是一次并行draft，不是逐步反馈的observed/generated target；
- global refinement：不允许$\tau$读取$\tilde y_{\tau+1:T}$；
- scale-arm routing：没有best-arm selection或competence teacher。

[Hypothesis] 强parent的forecast draft中，局部shape、phase与transition信息可能对当前坐标的residual correction有
增量价值，而global basis readout或pointwise head未显式利用这一future-domain context。

[Self-critique] 对deterministic squared-error point forecasting，Bayes-optimal conditional mean本来就是$x$的函数。
因此future draft不会增加information-theoretic information；它最多改变function factorization、inductive bias与
optimization geometry。若一个高容量pointwise control同样解释收益，这个方向就不值得升为论文机制。

## 6. SC-D17-PFC：problem diagnostic

### 6.1 要测试什么

`SC-D17-PFC`（Projective Future-Context diagnostic）问：

> 在不读取target、requested horizon或prefix外坐标的前提下，frozen forecast draft的ordered causal context是否能在
> held-out rows上提供超越pointwise calibration与same-capacity shuffled-context control的residual correction？

它不是模型消融，也不是paper-facing test。输入使用现有authorized official-test probe artifacts，因此明确标记
`test_informed / diagnostic_only`。A6/A6_MEASURE artifacts没有保存raw probe predictions，因此D17-v1不能假称为
A6 carrier；它使用具有完整probe的`SIFF_EQUAL`与`PCSD_EQUAL`两个不同decoder carrier，并把结论明确限制为
dual-carrier conditional evidence。根据Frozen Component Replacement Fairness，失败不能单独形成方向级否决。

### 6.2 数据与arms

每个carrier、dataset读取：

- `probe_fused [256,720]`：`SIFF_EQUAL`或`PCSD_EQUAL` parent forecast；
- validation/test `probe_targets [256,720]`：validation labels只拟合diagnostic residual；test labels只评分；
- validation-to-test transfer：只在validation probe拟合固定ridge correction，并在独立test probe评估；
  test labels不参与拟合、feature选择或超参数选择。

四个比较层：

1. `parent`：对应carrier的原fused draft；
2. `pointwise_wide`：只看当前$\tilde y_\tau$与固定full-domain coordinate
   $u_\tau=(\tau-1)/(T-1)$的高容量Fourier/polynomial features；crop时不重新归一化；
3. `causal_ordered`：在pointwise features上加入固定lags
   $\{1,2,4,8,16,32,64,128\}$的ordered draft context；
4. `causal_row_shuffled`：相同feature dimension，但context来自另一row，破坏sample-specific ordered relation；
5. `symmetric_ordered`：同时使用past与future draft坐标，仅作non-projective upper control。

所有correction使用同一fixed-alpha standardized ridge；不按dataset/horizon选择超参数。metric报告
$H\in\{96,192,336,720\}$的MSE。

### 6.3 统计定义

- `gain_over_parent_percent = 100(1-MSE_model/MSE_parent)`；
- `causal_gain_over_pointwise_percent = 100(1-MSE_causal/MSE_pointwise)`；
- `causal_gain_over_shuffled_percent = 100(1-MSE_causal/MSE_shuffled)`；
- `symmetric_gain_over_pointwise_percent`衡量允许prefix外信息后可获得的upper-control增量；
- `prefix_invariance_max_abs_gap`：分别在full draft与crop draft重算causal features时，共享prefix上的最大绝对差；
  必须接近machine precision。

### 6.4 预冻结gate

D17只有以下六项全部通过，才把future-context problem升到Step 4：

1. causal相对pointwise macro MSE gain $\geq0.5\%$；
2. causal相对row-shuffled macro MSE gain $\geq0.3\%$；
3. 两个carriers的macro gain均为正；
4. causal相对pointwise在至少4/5 datasets、8/10 carrier-dataset cells为正；
5. causal相对pointwise在至少3/4 standard horizons为正；
6. `prefix_invariance_max_abs_gap <= 1e-10`。

`symmetric_ordered`不参与pass gate：它若明显更强，只说明non-projective full-domain context有用，不能证明当前
paper contract。

### 6.5 首版协议作废记录

首次执行曾在同一test probe的256个flattened `sample × channel` rows上直接two-fold。pointwise correction相对
parent异常提高约21.27%；检查evaluator后确认，同一样本的不同channels以及强重叠时间窗口可能跨fold。该protocol
不足以证明跨split generalization，全部表面gate与数值作废，标记
`diagnostic_protocol_invalid_for_problem_promotion`。

新协议不再在test上fit：remote只从原checkpoint导出validation probes，随后validation fit、existing test probe
evaluate。作废说明见`d17_projective_future_context/INVALID_PROTOCOL.md`。

## 7. 决策与rollback

当前没有新的Contribution 1或Contribution 2被提出，更没有训练授权。

- 若D17全通过：只把问题状态改为`problem_supported`，再进入Step 4检索、定义native triangular operator与
  function/capacity controls；Contribution 2仍需从该operator的真实training mismatch中产生；
- 若causal不超过pointwise：在两个frozen carriers上归为`conditional capacity_control_explains`，方向保持
  unresolved；不得仅凭frozen diagnostic关闭future-context family；
- 若causal超过pointwise但不超过shuffled：说明额外features有效、ordered sample-specific context不成立；
- 若symmetric强而causal弱：说明可利用结构依赖prefix外信息，与exact projectivity冲突，不能作为当前主线；
- 若D17在两个carrier均失败：只能说明现有frozen drafts不支持该correction factorization，状态标记
  `conditional_negative / direction_unresolved`；结合source pressure再决定是否值得做一次matched E2E problem
  probe，不得直接写成future-context方向已被否定；
- 无论结果如何，不再在SIFF/CCSF上追加router、teacher、temperature或scale sweeps。若连dual-carrier signal也
  不存在，需要诚实评估fixed-past unified multi-horizon是否还能支撑“两项native contribution”的整篇论文，而
  不是继续为既定标题找模块。

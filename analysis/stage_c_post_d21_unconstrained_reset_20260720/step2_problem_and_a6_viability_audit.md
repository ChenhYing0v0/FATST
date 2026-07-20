# Stage C Post-D21 Unconstrained Reset：A6 论文承载力与新问题路径审计

## 1. 当前节点

| Field | Content |
| --- | --- |
| `current_step` | Contributions 1/2 joint Step 2；D22-HFA proposed |
| `problem` | 放宽 exact projectivity、requested-H 禁用等人为约束后，fixed-past deterministic multi-horizon forecasting 还存在哪一种可被数据支持的必要自由度？ |
| `existence_evidence` | A6/MEASURE 正收益；D18 specialist advantage weak；D14 oracle/crossing large；D21 past × region interaction split-unstable |
| `idea` | 先分离 Bayes-level horizon invariance、finite-capacity interference 与 target-coordinate evidence access，再决定 architecture slot |
| `theory_check` | separable pointwise MSE 下，同一 future coordinate 的 Bayes conditional mean 不依赖 requested horizon；H-adaptation只能来自有限容量、非分离风险、可用context、计算预算或任务定义变化 |
| `design` | D22-HFA 先做 theorem/source/evidence audit，再决定是否执行小型 target-coordinate information-access diagnostic |
| `narrative_gate` | pending；不得从“放宽约束”直接跳到 horizon embedding/router implementation |
| `effectiveness_gate` | not applicable before problem gate |
| `artifacts` | 本报告；D18/D21/SIFF/A6既有审计 |
| `decision` | A6保留为strong carrier/control，不单独升级为paper主体；新方法training未授权 |

## 2. A6-LBF 是否足以独立承载论文

### 2.1 可以保留的价值

1. A6-LBF 已经是稳定、干净的 unified carrier，并相对历史 TimeAlign controls 有真实性能收益；
2. history经系数映射生成full trajectory，再做prefix crop，tensor contract简洁且便于严格控制；
3. A6_MEASURE在当前公平矩阵中是最强简单控制之一，相对A6_FULL的收益跨20/20 standard cells为正；
4. 它适合作为后续新operator必须超过的强基线，也适合作为function-preserving或matched-capacity control。

### 2.2 不能独立承担高水平论文主体的原因

1. learned basis并非独占机制：N-BEATS/N-HiTS、BasisFormer、Implicit Forecaster与FlowState已覆盖basis、
   hierarchical interpolation、wave/functional decoding等相邻链条；
2. exact projectivity本身是full-trajectory prefix crop的函数性质，不足以单独构成architecture novelty；
3. uniform-horizon诱导的harmonic measure及weighted checkpoint已被ElasTST直接覆盖；
4. 当前A6正结果主要相对TimeAlign lineage，尚未建立相对最新varied-horizon/target-query decoder的完整优势；
5. 若把A6与MEASURE硬拆成两个创新，会把已有primitive或prior-covered objective包装成独立贡献，完整novelty chain不够稳固。

因此结论是：`A6-LBF = strong carrier + possible component + mandatory control`，不是当前可直接冻结的
standalone paper core。若未来新问题要求projective base trajectory，A6可以被包含；若新问题要求不同接口，也允许
替换carrier，不再以“必须兼容A6”为设计约束。

## 3. 放宽自由度后仍需保留的理论边界

令$x$为同一fixed past，$Y_t$为同一future coordinate，requested horizon为$H\geq t$。在pointwise squared loss下，

$$
f_t^*(x,H)=\arg\min_a\mathbb E[(Y_t-a)^2\mid X=x,H]
=\mathbb E[Y_t\mid X=x],
$$

只要$H$只是用户请求而不携带额外信息，Bayes-optimal shared-prefix prediction不应依赖$H$。因此取消exact
projectivity只增加finite-model freedom，不自动创造新的统计信息。

真正可能要求不同预测的来源必须至少属于以下之一：

1. `finite_capacity_or_optimization_tradeoff`：同一参数集无法同时逼近不同lead-time映射；
2. `target_coordinate_information_access`：不同future coordinate需要从history读取不同证据；
3. `nonseparable_or_decision_risk`：评价并非逐点MSE之和，而是trajectory、tail、peak或utility risk；
4. `future_context_changes`：不同请求附带不同known-future covariates或可用信息；
5. `compute_or_resolution_contract`：不同horizon要求不同预算、采样率或输出分辨率；
6. `probabilistic_joint_forecast`：目标是联合分布/依赖结构，而不只是conditional mean。

当前论文仍以deterministic MSE/MAE为primary，因此D22优先审计前两项；后四项属于可能的任务转向，不能静默混入。

## 4. 外部边界刷新

检索日期：`2026-07-20`。来源以外部primary sources为主，Zotero只作seed。主题包括
`variable/dynamic horizon forecasting`、`target timestamp query decoder`、`functional/basis decoder`、
`multi-horizon attention`与`horizon reweighting`。

| Work | 已覆盖边界 | 对本项目的约束 |
| --- | --- | --- |
| [ElasTST, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d7aa002885ccbe68cf6880da583761b2-Abstract-Conference.html) | varied-horizon invariance、structured masks、multi-scale patches、horizon reweighting | invariance/reweighting不能单独claim |
| [MQTransformer](https://arxiv.org/abs/2009.14799) | context-dependent multi-horizon encoder-decoder attention | generic future-query/history-attention不能单独claim |
| [N-HiTS](https://ojs.aaai.org/index.php/AAAI/article/view/25854) | hierarchical interpolation与multi-rate synthesis | generic local/global basis synthesis不能单独claim |
| [Implicit Forecaster, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html) | frequency/amplitude/phase trajectory decoder | generic structured decoding已有强直接prior |
| [TimePerceiver, 2025](https://arxiv.org/abs/2512.22550) | target-timestamp queries读取history latent | target-coordinate query必须有新的problem/mechanism coupling |
| [FlowState, 2025](https://openreview.net/forum?id=R50AT6nAsM) | functional basis、continuous time与dynamic horizon | arbitrary-horizon/functional basis本身不足 |

这不意味着不能使用query、basis或continuous coordinate；它们可以成为primitive，但新贡献必须证明完整的
`problem -> necessary freedom -> operator -> training -> effect`链条。

## 5. 新工作路径：SC-D22 Horizon Freedom Audit

### D22-A：Bayes 与任务边界

冻结上述Bayes invariance结论，明确requested horizon什么时候只是request、什么时候带来新信息或新utility。
任何新候选必须写清自己属于六类来源中的哪一类，不允许用“multi-horizon更复杂”代替problem evidence。

### D22-B：复用证据判断 finite-capacity frontier

重新联合分析D18 specialists、A6_MEASURE、dense-horizon/checkpoint artifacts：

- own-horizon specialist是否形成稳定Pareto improvement；
- shared model是否在某些lead-time段系统性让步；
- observed tradeoff是否超过seed/split波动与simple measure control。

D18当前只提供weak evidence（macro `+0.1659%`、7/15 cells），所以该分支默认低优先级；没有新证据不得直接实现
horizon embedding、hypernetwork或soft-projectivity sweep。

### D22-C：小型 target-coordinate information-access diagnostic

只把ordered patch memory当作诊断载体，不把它写成论文主线。问题是：

> 在matched capacity与validation-fit→test transfer下，future coordinate与history token/patch的joint access，
> 是否提供超越global compressed state、pooled memory、order-shuffled memory与target-shuffled query的稳定增量？

该诊断首先使用neutral/raw-history primary carrier，A6只作sensitivity；冻结component replacement结果不能拒绝E2E
方向。只有至少跨3/5 datasets、standard horizons与test transfer均有material specificity，才返回Step4。

### 若D22-C通过

Contribution 1才进入source-informed Step4，暂定问题为`lead-time-conditioned evidence operator`。其论文主语是
future coordinate如何组织预测计算，不是ordered memory。TimePerceiver/MQTransformer/FlowState为mandatory
controls。

Contribution 2保持开放，不预先指定loss。它只能由首个E2E operator的真实训练瓶颈产生：例如coverage collapse、
coordinate imbalance、optimization conflict或calibration error。没有诊断证据，不再先造credit objective。

### 若D22-C失败

关闭当前deterministic-MSE fixed-past architecture search。随后只允许二选一：

1. 把现有结果收束为projective unified forecasting的理论/empirical paper；
2. 显式改变任务到probabilistic、decision-aware、known-future-context、compute/resolution adaptive之一，并作为新主线重新审计。

## 6. 截稿导向的执行顺序

1. `1–2 days`：完成D22-A/B现有artifact审计，不训练新模型；
2. `1–2 days`：若B仍不足，只实现D22-C最小offline/neutral diagnostic与matched controls；
3. `1 day`：根据problem gate决定是否授权唯一一个Step4 method candidate；
4. `2–4 days`：只实现一个minimal E2E candidate，seed2021五dataset完整test gate；
5. pass后才补seeds、held-out datasets与Contribution 2；fail则执行上述paper/task pivot，不再连续开D23/D24 rescue。

## 7. 决策

`a6_standalone_paper_core_not_supported / unconstrained_problem_first_reset / d22_hfa_next`。

放宽约束是必要修正，但不是授权任意architecture搜索。新的自由度必须先证明它改变了可预测信息、有限容量frontier、
风险、context或部署contract；否则它只会增加参数和叙事复杂度。

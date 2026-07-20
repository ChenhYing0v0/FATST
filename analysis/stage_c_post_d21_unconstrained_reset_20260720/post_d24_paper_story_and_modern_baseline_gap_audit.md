# Post-D24 Step2/4：Paper-Story Consolidation 与 Modern Native-Baseline Gap Audit

## 1. Decision

| Field | Content |
| --- | --- |
| `current_step` | Post-D24 Step2/4 consolidation complete；return SC-MNB Step1-3 |
| `problem` | Bayes boundary、finite-frontier negative、target-access positive与capacity explanation能否形成完整SCI方法论文链？ |
| `existence_evidence` | D18、D22-C、D23、D24完整证据链 |
| `idea` | 将requested-horizon semantics、target-coordinate access与trajectory capacity分成三个不同命题 |
| `theory_check` | pure request $H$不改变同一coordinate的Bayes mean；coordinate $t$仍可要求不同history computation |
| `design` | evidence/source/baseline audit only；不实现method、不训练、不访问test |
| `narrative_gate` | problem boundary pass；method-paper narrative fail/incomplete |
| `effectiveness_gate` | not applicable；当前没有active method |
| `artifacts` | 本报告及D18、D22-C、D23、D24 reports |
| `decision` | `scientifically_coherent_problem_boundary_but_method_narrative_incomplete` |
| `next_step` | `SC-MNB` modern native-baseline reproduction protocol Step1-3；execution未授权 |

[Decision] 当前证据链可以形成一个严谨的科学结论，但不能形成高水平SCI方法论文的完整正向叙事。它证明了
哪些动机不成立、哪些计算自由度真实存在、以及为什么弱query operator无法兑现该自由度；它尚未给出一个同时
保持strong trajectory capacity并获得split-stable coordinate-access增益的paper-core method。

因此：

1. 不启动D25 architecture；
2. 不把negative/control chain包装成两项contributions；
3. 不恢复H embedding/router、FCMI rescue、D24 nonlinear rescue或第二loss；
4. 下一步先回答更基础的carrier viability问题：A6/MEASURE相对modern native baselines是否仍具有可发表的
   performance、unified-weight与cost位置。

该动作仍在`deterministic-MSE fixed-past architecture search`内，不是probabilistic、decision-aware或
known-future-context task pivot。

## 2. Evidence-chain consistency

### 2.1 四段证据回答的是不同命题

| Evidence | Supported conclusion | Cannot conclude |
| --- | --- | --- |
| D18 frontier negative | 当前A6上没有稳定cross-horizon specialist Pareto frontier | 任意finite model都没有capacity tradeoff |
| D22-C target-access positive | matched small operator中，future coordinate-specific history access有稳定增量 | requested $H$携带predictive information；raw cross-attention是新方法 |
| D23 capacity explanation | weak query family的收益不足以替代strong trajectory function class | target access问题为假；DENSE本身是method |
| D24 coarse-deformation negative | frozen strong forecast后没有chronologically stable linear coarse correction | 所有nonlinear native coordinate interaction都无效 |

D22-A与D22-C不矛盾。对同一fixed past，requested horizon $H$只是请求；但不同future coordinate
$t_1\ne t_2$对应不同预测函数

$$
f_{t}^{*}(x)=\mathbb E[Y_t\mid X=x].
$$

$f_t^*$不依赖“用户还请求到多远”，不等于所有$t$必须用相同finite computation读取history。D22-C支持的是后者的
计算组织差异，不是H semantics。

### 2.2 当前最强可辩护主张

[Strong Evidence] 在当前五dataset fixed-past deterministic-MSE setting中：

1. pure requested-H conditioning缺少Bayes和finite-frontier必要性；
2. target-coordinate access在matched small model中有稳定价值；
3. trajectory-wide function class/capacity对强性能至关重要；
4. target access、ordered binding、trajectory capacity必须由独立controls分开归因。

这可作为method design principle或benchmark/control contribution的依据，但没有正向method effectiveness时，
不能成为本文的完整paper core。

## 3. SCI narrative gate

### 3.1 为什么 problem boundary 通过

- 理论边界明确且可复核；
- D18、D22-C、D23、D24分别有matched controls、完整negative cells与failure attribution；
- positive与negative结果可以由不同命题一致解释；
- 结论直接约束H router、target query、capacity control与trajectory synthesis的实验设计。

### 3.2 为什么 method-paper narrative 不通过

1. 当前没有active method或paper-facing positive effectiveness；
2. A6的learned basis、prefix crop与harmonic measure分别存在强prior，不能standalone claim；
3. D22-C operator是11,553-param problem diagnostic，没有与modern native models作paper-facing比较；
4. D23只证明FCMI在弱query family内部有效，A6 comparison为0/20；
5. D24没有发现可作为低成本successor的稳定deformation surface；
6. 单独发表当前negative chain需要更多backbones、datasets与专门benchmark设计，成本并不低于继续寻找正向方法，
   且偏离当前项目的method-paper目标。

结论不是停止deterministic architecture search，而是先阻止在未知carrier competitiveness上继续堆叠局部module。

## 4. Latest primary-source audit

检索日期：`2026-07-20`。检索主题包括`varied-horizon forecasting`、`future-horizon query`、
`generalized target-position decoder`、`selective patch representation`、`basis forecasting`与
`time-series benchmarking confounds`。来源使用会议proceedings、OpenReview、arXiv与official repositories。
Zotero MCP两次返回`connection refused`，故下表`Zotero presence=unknown`；该缺口不用于novelty判断。

| Work | Primary/official evidence | Direct boundary | Zotero presence |
| --- | --- | --- | --- |
| ElasTST, NeurIPS 2024 | [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d7aa002885ccbe68cf6880da583761b2-Abstract-Conference.html), [official code](https://github.com/microsoft/ProbTS/tree/elastst) | one model、varied horizons、invariance、multi-scale patches、horizon reweighting | unknown |
| CATS, NeurIPS 2024 | [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/cf66f995883298c4db2f0dcba28fb211-Paper-Conference.pdf), [official code](https://github.com/dongbeank/cats) | future horizons作为queries读取past patches | unknown |
| TimePerceiver, NeurIPS 2025 | [paper](https://arxiv.org/abs/2512.22550), [official code](https://github.com/efficient-learning-lab/TimePerceiver) | arbitrary input/target positions、target-timestamp query、decoder/training co-design | unknown |
| SRSNet, NeurIPS 2025 Spotlight | [paper](https://arxiv.org/abs/2510.14510), [official code](https://github.com/decisionintelligence/SRSNet) | selective patching、dynamic reassembly、adaptive fusion | unknown |
| BasisFormer, NeurIPS 2023 | [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e150e6d0a1e5214740c39c6e4503ba7a-Abstract-Conference.html) | learned history/future bases、cross-attention coefficients、basis consolidation | unknown |
| Current Benchmarking Hinders Real Progress, ICML 2026 position | [OpenReview](https://openreview.net/forum?id=gtwbLmO7Wb) | architecture comparison必须隔离design和implementation confounds | unknown |

[Strong Evidence] `varied horizon`、`future query`、`selective patches`、`learned basis`及其简单组合均已有直接prior。
新candidate只有在完整`problem -> capacity-preserving computation -> contained controls -> split-stable effect`
链上才可能成立，不能靠重新命名primitive建立novelty。

## 5. Minimal modern native-baseline gap

### 5.1 Blocking P0 matrix

| Baseline | Why mandatory | Native contract | Required datasets/horizons |
| --- | --- | --- | --- |
| `ElasTST` | 与one-model varied-horizon claim最直接对齐 | official `elastst` branch；同一weights评估多H | five datasets；96/192/336/720 |
| `CATS` | D22-C future-query primitive的最近paper-facing control | official per-script cross-attention model | five datasets；四H |
| `TimePerceiver` | 最新target-position decoder/training co-design control | official generalized forecasting scripts | five datasets；四H |
| `SRSNet` | 当前项目指定重点baseline；代表modern selective patch representation | official NeurIPS 2025 scripts | five datasets；四H |
| `A6_FULL/A6_MEASURE` | same-run internal carrier与measure attribution | repo-native frozen protocol | five datasets；四H及dense diagnostic |

`CATS`与`TimePerceiver`不能互相替代：前者是最接近D22-C primitive的LTSF query baseline，后者改变了
input-target formulation与training contract。`SRSNet`回答strong modern representation是否已经超过A6；
`ElasTST`回答single-model varied-horizon位置。

### 5.2 非P0项目

- `iTransformer/TimeMixer/PatchTST`：可由native baseline suites补充，但不是本轮claim-specific最小集合；
- `BasisFormer/Implicit Forecaster`：只有新candidate使用basis/wave synthesis时才升级为formal mechanism controls；
- `TimesFM/Timer-XL/Sundial/Time-MoE`等foundation models：pretraining、zero-shot与supervised from-scratch
  protocol不同，应放独立table；不阻塞当前carrier viability gate；
- `TQNet/MQTransformer`：保留为prior-art边界；除非active method重用其context/covariate contract，否则不是P0运行项。

## 6. Reproduction and evaluation roles

外部baseline必须在各自official repository中source-faithful复现，不把代码复制进本项目method tree，也不从
`R_2026_FSA`导入结果。

结果至少拆成三张表：

1. `unified-weight varied-horizon`：同一checkpoint/weights评估四H，报告MSE、MAE、参数、训练次数、inference
   calls与prefix/invariance contract；
2. `native fixed-H performance`：CATS、SRSNet等按official per-H contract训练，作为accuracy context，不冒充
   single-model baseline；
3. `foundation zero-shot/pretrained`：若后续需要，单独报告，不与from-scratch supervised结果混成同一公平gate。

共同治理：

- 使用官方数据split与native checkpoint selection；
- 记录source commit、environment、effective config、dataset identity与每个cell；
- test不选择config、input length、epoch或baseline版本；
- 不强迫外部model使用A6 natural profile；source-faithful与matched-control角色分开；
- 完整报告五datasets × 四horizons及negative cells；
- carrier gate不能用published table直接替代local native reproduction。

## 7. Carrier viability gate and consequences

`SC-MNB`只冻结Step1-3 protocol方向，当前`implementation/training/test=false`。

未来prelaunch必须预先定义：

1. source commits与license；
2. native commands、environments与dataset-path mapping；
3. checkpoint selection和multi-H weight identity；
4. MSE/MAE normalization/split equivalence audit；
5. resource smoke与完整matrix；
6. machine-readable completeness gate。

结果解释：

- 若A6/MEASURE相对ElasTST、TimePerceiver、CATS、SRSNet具备competitive accuracy，并在single-weight、
  parameters或compute上有清楚优势，则允许返回Step4寻找`capacity-preserving coordinate access`新operator；
- 若A6被modern baselines广泛、稳定超过，则A6只保留historical control，后续不得继续在其接口上堆方法；
- 若结果因protocol不等价无法判断，则decision=`unresolved`，先修复reproduction fairness，不能选择有利table。

## 8. Final decision

`scientifically_coherent_problem_boundary_but_method_narrative_incomplete /
modern_native_baseline_gap_blocking / sc_mnb_step1_3_next`。

当前rollback point仍是Step2/4；没有active method，没有D25，没有remote training或official-test授权。

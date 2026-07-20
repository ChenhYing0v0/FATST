# Stage C Post-D21 / D22 Restart Handoff

## 0. 使用方式

本文件是新会话的唯一首读入口。它只保存当前有效状态、约束、证据边界和下一动作；详细历史仍由
`paper-mainline`、`research-roadmap`、Stage C ledger与`analysis/`承担。

新会话必须按以下顺序读取：

1. `AGENTS.md`；
2. 本 handoff；
3. `analysis/stage_c_post_d21_unconstrained_reset_20260720/step2_problem_and_a6_viability_audit.md`；
4. `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md`；
5. `docs/paper-mainline.md`；
6. `docs/research-roadmap.md`。

若上述文件与更旧的聊天、archive或历史段落冲突，以本文件和三份主线文档顶部的最新cursor为准。

## 1. 当前权威状态

| Field | Content |
| --- | --- |
| `project` | R_2026_FATST |
| `stage` | StageC-UVHF |
| `handoff_date` | 2026-07-20 |
| `source_parent_commit` | `73755fd` |
| `current_step` | SC-D22-HFA Step 2 proposed；先执行D22-A/B |
| `active_problem` | relaxed constraints后，哪一种horizon freedom具有真实统计或有限容量必要性？ |
| `active_method` | none |
| `method_training_authorized` | false |
| `remote_training_authorized` | false |
| `next_action` | 完成D22-A Bayes/task boundary与D22-B existing-artifact finite-capacity frontier audit |
| `conditional_next` | 只有A/B不能回答且design gate通过时，才设计D22-C small diagnostic |
| `rollback` | D22失败则停止当前deterministic-MSE fixed-past architecture search |

当前工作树存在两个与本次handoff无关的untracked目录，必须原样保留，不得在新会话中清理、归档或提交：

- `SRP-7C55/`；
- `analysis/phase5_timealign_hss_h1c_capacity_preserving_gate_20260701_partial/`。

## 2. 本轮正式转向

### 2.1 已撤销为硬约束

以下项目不再是新architecture必须满足的先验条件：

- exact projectivity；
- requested horizon禁止进入模型；
- 必须生成full-$T=720$ trajectory再prefix crop；
- 必须兼容A6 decoder/interface；
- 必须预先形成`decoder + training loss`两个contributions。

它们仍可作为设计选择或matched controls，但必须由问题、理论和实验决定。

### 2.2 仍然有效的理论护栏

在fixed past、pointwise separable MSE且requested horizon不携带额外信息时，同一future coordinate的Bayes最优
conditional mean不依赖requested horizon。故“允许输入H”不等于“H-adaptation有统计必要性”。

新自由度必须明确来自以下之一：

1. finite-capacity/optimization tradeoff；
2. target-coordinate-specific history evidence access；
3. nonseparable/decision risk；
4. different known-future context；
5. compute/resolution contract；
6. probabilistic joint target。

当前任务仍以deterministic MSE/MAE为primary，D22优先审计1和2；其余四类属于显式task pivot。

## 3. A6、SIFF与历史路线的当前角色

| Item | Current Role | Boundary |
| --- | --- | --- |
| `A6-LBF-natural-baseline` | strong carrier / mandatory control / possible component | 不足以standalone承载高水平论文；新设计无需强制兼容 |
| `A6_MEASURE` | strong training control | harmonic horizon measure受ElasTST直接prior覆盖，不能单独claim |
| `SIFF-v2-EQ-ATTR-v1` | frozen historical performance-near candidate | internal 7/7，但未超过A6_MEASURE且independent control阻塞归因；不继续调参 |
| `D14 crossing/oracle` | historical clue | oracle不等于past-identifiable benefit；D21已证明interaction split-unstable |
| `D17-D21` | closed evidence | 不做representation/readout/seed rescue |
| `CTD` | paused by user | 新会话不得自动恢复 |
| `NIFRO/IARL/New-idea.md` | deferred next-paper idea | 不占当前paper slot |

## 4. 必须保留的关键结果

1. A6-LBF相对TimeAlign lineage有真实性能收益，但modern varied-horizon baseline comparison仍不完整；
2. `A6_MEASURE > A6_FULL`约`+1.8762%` MSE，20/20 standard cells正向；
3. D18 specialists相对A6_MEASURE仅`+0.1659%`、7/15 cells，finite-capacity horizon frontier证据弱；
4. SIFF_EQUAL相对A6_FULL `+1.6436%`，但相对A6_MEASURE `-0.2366%`；
5. D21 oracle headroom为`7.64%/10.41%`，但interaction相对additive仅`+0.0347%/-0.0069%`，且validation→test不稳定；
6. 因此当前没有active paper-core method，也没有已经成立的two-contribution chain。

## 5. D22-HFA 的执行顺序

### D22-A：Bayes/task boundary

- 写清requested H是纯request还是带来新information/risk/context/compute contract；
- 审计放宽约束后的primary-source prior；
- 禁止从“自由度更大”直接推出method necessity。

### D22-B：existing-artifact finite-capacity frontier

优先复用D18 specialists、A6_MEASURE、dense-horizon与checkpoint artifacts，回答：

- specialists是否形成稳定own-H/Pareto advantage；
- shared model是否在特定lead-time segment系统性让步；
- tradeoff是否超过measure control、seed和split波动。

本步骤不训练新模型、不访问新test选择candidate、不恢复CTD。

### D22-C：仅在A/B不足时的小诊断

问题限定为future coordinate与history token/patch的joint access是否稳定超越：

- global compressed state；
- pooled memory；
- order-shuffled memory；
- target-shuffled query；
- matched-capacity generic control。

ordered patch memory只作诊断载体，不作论文主语。neutral/raw-history为primary，A6为sensitivity。frozen
component replacement只能形成conditional evidence，不能方向级拒绝E2E method。

只有D22 problem gate通过，Contribution 1才允许回Step4 source-informed design，暂定问题族为
`lead-time-conditioned evidence operator`。Contribution 2必须来自首个E2E operator暴露的真实训练瓶颈，不得预设。

## 6. 研究与实验治理

- 外部调研默认广泛web search并优先primary sources；Zotero只作seed；
- paper-facing/formal mechanism evaluation使用official test的`{96,192,336,720}`完整矩阵；
- validation只用于checkpoint、普通超参数、debug与解释性diagnostic；
- official test已是`test_informed benchmark decision surface`，不得声称untouched；
- final confirmation应增加冻结后未参与设计的新datasets；
- paper-core比较默认matched end-to-end joint training；frozen replacement只作`diagnostic_only`；
- 一个problem diagnostic失败后必须明确failure attribution，不得直接堆叠下一机制；
- 远端实验前必须commit/push、`nvidia-smi`检查并使用`529_Lab-3090`的`moe`环境；
- 远端训练输出默认写入`/home/yingch/exp_outputs/r-2026-fatst`。

当前five-dataset profiles保持：ETTh1、ETTh2、ETTm1、ETTm2、Weather。dataset之间允许不同自然profile；
同一dataset的机制比较必须共享profile，params差异不参与profile选择。

## 7. 新会话第一轮的完成定义

新会话第一轮只完成以下结果，不应直接进入model implementation：

1. D22-A/B analysis report；
2. evidence table与每个统计量定义；
3. `finite_capacity_frontier_supported / not_supported / unresolved`决定；
4. 是否需要D22-C；
5. 若需要，给出D22-C Step3/6前的problem、controls、gate与rollback设计；
6. 同步paper-mainline、roadmap、ledger并提交推送。

## 8. 禁止无损重启时发生的漂移

- 不把A6重新写成已经成立的论文主体；
- 不把SIFF内部健康度写成paper-core pass；
- 不恢复EVS、CCSF、PCC、PCSD、JAPO、D19或D20的参数/seed rescue；
- 不因为放宽约束就直接实现explicit H embedding；
- 不把ordered patch memory升格为论文主线；
- 不在Contribution 1 problem gate之前设计Contribution 2；
- 不导入`R_2026_FSA`代码、配置或artifact，除非用户明确批准具体来源与范围。

## 9. Restart Prompt

本文件末尾Prompt与交付给用户的Prompt应保持一致；新会话以工作区当前文件为准，不依赖旧聊天记忆。

```text
请在 /Users/river/PaperResearch/Project/R_2026_FATST 中继续 Stage C 研究。

首先严格阅读并遵守仓库 AGENTS.md，然后按顺序阅读：
1. docs/stage-ledgers/stage-c-post-d21-d22-restart-handoff-20260720.md
2. analysis/stage_c_post_d21_unconstrained_reset_20260720/step2_problem_and_a6_viability_audit.md
3. docs/stage-ledgers/stage-c-unified-forecasting-redesign.md
4. docs/paper-mainline.md
5. docs/research-roadmap.md

当前权威状态是：Post-D21 unconstrained reset，SC-D22-HFA Step 2。exact projectivity、requested-H禁用、full-T prefix crop、A6 interface compatibility以及预设decoder+loss双贡献均不再是硬约束；但在fixed past、pointwise MSE且H不携带额外信息时，同一future coordinate的Bayes conditional mean不依赖requested horizon，因此不得直接实现H embedding/router。

A6-LBF仅是strong carrier/control/possible component，不足以standalone承载论文；SIFF-v2是冻结历史候选；D17-D21均已关闭，不做seed、readout、representation或参数rescue；CTD保持paused；当前没有active method，model/remote training均未授权。

请先执行SC-D22-HFA的D22-A/B：
- 完成Bayes/task-boundary与最新外部primary-source audit；
- 复用D18 specialists、A6_MEASURE、dense-horizon/checkpoint artifacts，审计是否存在稳定finite-capacity horizon frontier；
- 明确定义统计量、controls、validation/test角色和failure attribution；
- 给出finite_capacity_frontier_supported / not_supported / unresolved决定。

只有A/B无法回答且证据合理时，才设计D22-C的小型target-coordinate information-access diagnostic。ordered patch memory只能作诊断载体，不是论文主线；neutral/raw-history为primary，A6为sensitivity；frozen replacement不得用于方向级拒绝。不要实现method，不要启动remote training，除非先完成problem/narrative/design gate并在文档中获得授权。

完成后同步更新analysis report、docs/paper-mainline.md、docs/research-roadmap.md和Stage C ledger，执行最小诚实验证，并按AGENTS.md提交、推送。请从专业时序预测研究员角度进行审计，不要为了凑两个contributions而预先设计第二个loss/router。
```

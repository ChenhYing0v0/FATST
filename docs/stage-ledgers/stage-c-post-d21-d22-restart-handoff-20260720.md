# Stage C Post-D21 / D22 Restart Handoff

## 0. 使用方式

本文件是新会话的唯一首读入口。它只保存当前有效状态、约束、证据边界和下一动作；详细历史仍由
`paper-mainline`、`research-roadmap`、Stage C ledger与`analysis/`承担。

新会话必须按以下顺序读取：

1. `AGENTS.md`；
2. 本 handoff；
3. `analysis/stage_c_post_d21_unconstrained_reset_20260720/step2_problem_and_a6_viability_audit.md`；
4. `analysis/stage_c_post_d21_unconstrained_reset_20260720/d22_ab_bayes_frontier_audit.md`；
5. `docs/stage-ledgers/stage-c-unified-forecasting-redesign.md`；
6. `docs/paper-mainline.md`；
7. `docs/research-roadmap.md`。

若上述文件与更旧的聊天、archive或历史段落冲突，以本文件和三份主线文档顶部的最新cursor为准。

## 1. 当前权威状态

| Field | Content |
| --- | --- |
| `project` | R_2026_FATST |
| `stage` | StageC-UVHF |
| `handoff_date` | 2026-07-20 |
| `source_parent_commit` | `1319a4a` |
| `current_step` | SC-D22-HFA D22-C Step3/7A；static/prelaunch pass |
| `active_problem` | finite-capacity frontier不支持后，target-coordinate raw-history access是否具有split-stable必要性？ |
| `active_method` | none |
| `method_training_authorized` | false |
| `remote_training_authorized` | true for frozen D22-C diagnostic after commit/push and GPU preflight |
| `next_action` | commit/push冻结protocol；3090 preflight后启动five-dataset seed2021完整problem gate |
| `conditional_next` | 只有D22-C problem diagnostic通过，才返回Step4设计lead-time-conditioned evidence operator |
| `rollback` | 有效失败关闭D22-C exact v1并回joint Step2/3；按用户决定不自动pivot task |

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
7. D22-A/B dense复核后，SPEC96 own-H为`+1.2748%`、5/5 datasets，但SPEC192/SPEC336为负，
   0/15 arm-dataset Pareto dominance；decision=`finite_capacity_frontier_not_supported`；
8. A6_MEASURE相对A6_FULL在五个lead-time bins全部5/5正向；H96只保留局部optimization clue；
9. D22-C static/prelaunch已通过；仅冻结的neutral/raw-history diagnostic remote/test获授权，paper method仍未授权。

## 5. D22-HFA 的执行顺序

### D22-A：Bayes/task boundary（completed）

- 写清requested H是纯request还是带来新information/risk/context/compute contract；
- 审计放宽约束后的primary-source prior；
- 禁止从“自由度更大”直接推出method necessity。

### D22-B：existing-artifact finite-capacity frontier（completed）

优先复用D18 specialists、A6_MEASURE、dense-horizon与checkpoint artifacts，回答：

- specialists是否形成稳定own-H/Pareto advantage；
- shared model是否在特定lead-time segment系统性让步；
- tradeoff是否超过measure control、seed和split波动。

本步骤没有训练新模型、没有访问新test选择candidate、没有恢复CTD。完整结果见
`analysis/stage_c_post_d21_unconstrained_reset_20260720/d22_ab_bayes_frontier_audit.md`。

### D22-C：static/prelaunch passed；diagnostic authorized

问题限定为future coordinate与history token/patch的joint access是否稳定超越：

- global compressed state；
- pooled memory；
- order-shuffled memory；
- target-shuffled query；
- matched-capacity generic control。

ordered patch memory只作诊断载体，不作论文主语。neutral/raw-history为primary，A6为sensitivity。frozen
component replacement只能形成conditional evidence，不能方向级拒绝E2E method。

六臂neutral/raw-history runner与machine aggregator已实现；所有arms共享完全相同trainable parameter set、
seed2021 initialization、optimizer、window selection与validation selector。local synthetic smoke已完成六臂
forward/backward、checkpoint、validation/test和decision artifacts；parameter gap为0。

用户已授权继续该task边界下研究，因此冻结的five-dataset diagnostic remote/test在commit/push与GPU preflight后
可执行。只有D22 problem gate通过，Contribution 1才允许回Step4 source-informed design，暂定问题族为
`lead-time-conditioned evidence operator`。A6 sensitivity、paper method与Contribution 2仍未授权。

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

## 7. 当前执行定义

1. commit/push D22-C frozen diagnostic protocol；
2. 按AGENTS.md检查3090 GPU memory/process；
3. 运行seed2021 five datasets × six arms，validation只选择checkpoint，official test一次性完整审计；
4. 同步artifacts、执行machine gate与failure attribution；
5. 不恢复H embedding/router、D17-D21 rescue、CTD或Contribution 2预设。

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
3. analysis/stage_c_post_d21_unconstrained_reset_20260720/d22_ab_bayes_frontier_audit.md
4. docs/stage-ledgers/stage-c-unified-forecasting-redesign.md
5. docs/paper-mainline.md
6. docs/research-roadmap.md

当前权威状态是：SC-D22-HFA D22-C Step3/7A，D22-A/B已完成且D22-C static/prelaunch通过。exact projectivity、requested-H禁用、full-T prefix crop、A6 interface compatibility以及预设decoder+loss双贡献均不再是硬约束；但在fixed past、pointwise MSE且H不携带额外信息时，同一future coordinate的Bayes conditional mean不依赖requested horizon，因此不得直接实现H embedding/router。

A6-LBF仅是strong carrier/control/possible component，不足以standalone承载论文；SIFF-v2是冻结历史候选；D17-D21均已关闭，不做seed、readout、representation或参数rescue；CTD保持paused；当前没有active method。仅冻结的D22-C diagnostic remote/test获授权，paper method未授权。

已确认D18中SPEC96 own-H为`+1.2748%`、5/5 datasets，但SPEC192/SPEC336为负，三个specialists均没有standard-horizon Pareto dominance；A6_MEASURE相对A6_FULL在五个lead-time bins全部5/5正向。H96只保留局部optimization clue，不做seed或soft-projectivity rescue。

下一步提交并推送D22-C frozen protocol，完成3090 GPU preflight后运行seed2021 five-dataset × six-arm完整problem gate。ordered patch memory只能作诊断载体，不是论文主线；neutral/raw-history为primary，A6 sensitivity未授权；frozen replacement不得用于方向级拒绝。

完成后同步更新analysis report、docs/paper-mainline.md、docs/research-roadmap.md和Stage C ledger，执行最小诚实验证，并按AGENTS.md提交、推送。请从专业时序预测研究员角度进行审计，不要为了凑两个contributions而预先设计第二个loss/router。
```

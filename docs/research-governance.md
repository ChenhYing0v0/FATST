# FATST 研究路径保存体系

本文档定义项目级研究进度管理机制，目标是防止候选方案、阶段结论和论文主线之间发生上下文丢失。

## External Literature Default

- Zotero `FSA` collection是user-curated seed/reference library，不是文献完整性或时效性的source of truth。
- 外部调研默认先做broad web search，并以arXiv、OpenReview、正式会议论文集、official project page与
  official code等primary sources作为主要证据。
- Zotero未收录、seed notes未提及或本地PDF缺失，都不能作为方向新颖、文献空白或现有方法不存在的证据。
- 每次paper-core prior-art audit必须记录search date、topic/query scope、source type、full text/code状态，
  以及该工作是Zotero seed还是external discovery。
- 对coverage快速变化的方向，在method冻结与投稿前分别执行freshness search；未完成全文/代码核查时降低
  claim confidence。

## Prior-Art Overlap Interpretation

- mechanism primitive或operator family重叠不再自动判定`novelty fail`；external audit首先用于收紧claim、
  识别必须对比的baseline和mandatory controls；
- novelty按完整链条评估：`problem -> task-specific constraint -> mechanism composition -> implementation path
  -> empirical claim`。已有primitive若服务于不同问题，并形成不同的数学约束、tensor path或训练/评估协议，
  可以构成contribution-level novelty；
- 必须区分`component novelty`与`contribution novelty`：单个组件未必新，但面向multi-horizon unified
  forecasting的projectivity、prefix consistency、support geometry与生成路径组合可以是新的贡献边界；
- 只有prior work实质覆盖相同问题、相同约束、相同主要实现路径和相同claim时，overlap才可单独阻断
  narrative gate；
- 反向约束同样成立：仅改名、仅迁移到forecasting、仅有实现细节差异，不足以证明创新；必须说明
  task-specific coupling为什么必要，并用matched controls验证。

## 文档分层

| Layer | 文件 | 职责 | 不应包含 |
| --- | --- | --- | --- |
| Paper Mainline | `docs/paper-mainline.md` | 维护论文级问题定义、核心贡献、主方法叙事、主实验安排和转向规则 | 阶段内所有失败路线、详细 metric 表、日志级分析 |
| Stage Ledger | `docs/stage-ledgers/<stage>.md` | 维护单个研究阶段的 active question、candidate queue、pending tasks、阶段内 decision cursor 和 paper-mainline sync 状态 | 长篇实验报告、完整 CSV 解读、训练日志 |
| Roadmap Index | `docs/research-roadmap.md` | 维护跨阶段 11-step 决策索引、rollback 记录和阶段切换记录 | 每个候选的完整细节、所有并行 idea 的长描述 |
| Experiment Reports | `analysis/<run>/...md` | 保存一次实验或诊断的完整结果分析、表格、图、failure analysis | 论文总纲判断、未执行候选管理 |
| Code Explanation | `docs/code-explanation/*.md` | 解释新增或修改代码的 tensor flow、配置、脚本和 artifact 语义 | 阶段路线决策 |

## Stage Ledger 是强制入口

每次用户要求“继续推进研究”“按计划继续”“设计下一步实验”“远程实验已完成请分析”时，必须先打开当前
active `Stage Ledger`，再决定下一步。

继续研究前必须检查：

1. `decision_cursor`：当前处于哪个 11-step step，上一轮 gate 的结论是什么。
2. `candidate_queue`：是否存在已提出但未执行、未拒绝、未降级的并行候选。
3. `pending_tasks`：是否有尚未完成的分析、同步、实验或文档更新。
4. `latest_evidence`：最近一次实验只否定了哪个具体 hypothesis，不能扩大为主线失败。
5. `paper_mainline_sync`：本阶段结论是否已经达到影响论文总纲的阈值。

如果没有读取 active `Stage Ledger`，不得直接提出新的 paper-core 实验。

## Candidate Queue 规则

阶段内每个 idea 必须进入 `candidate_queue`，状态只能使用以下枚举：

- `proposed`：已提出，但尚未完成 narrative gate。
- `narrative_ready`：Step 4-6 已通过 narrative gate，可进入实现或实验设计。
- `running`：已启动本地或远程实验。
- `analysis_pending`：实验完成但尚未形成 decision。
- `passed_core_candidate`：通过 effectiveness gate，可考虑进入 paper-mainline。
- `partial_pass`：有正向信号，但还不足以作为 paper-core。
- `failed_as_core_candidate`：不适合作为 paper-core，但可作为 diagnostic/control evidence。
- `diagnostic_only`：只用于诊断，不可直接升级为 paper-core。
- `control_only`：只用于对照，不可直接升级为 paper-core。
- `deferred`：暂缓，必须说明恢复条件。
- `rejected_by_narrative_gate`：叙事、贡献边界或理论可行性不足，不进入实验。
- `superseded`：被更强或更干净的候选替代。

每个 candidate 至少记录：

- `id`：稳定短名。
- `status`：上述枚举之一。
- `hypothesis`：它验证的具体 hypothesis。
- `narrative_gate`：是否具备 SCI 级叙事潜力；diagnostic/control 可写 `not_required`。
- `effectiveness_gate`：需要什么实验结果才通过。
- `blocking_or_next_action`：下一步是实现、等待结果、分析、降级还是删除。
- `related_artifacts`：代码、脚本、analysis report 或 commit。

任何实验失败后，只能更新对应 candidate 的状态。除非 `candidate_queue` 中同一问题族的候选均已
失败或被 narrative gate 拒绝，否则不能把单个候选失败写成阶段主线失败。

## 阶段文档轻量化规则

Stage Ledger 只保存可执行决策，不保存完整实验分析。推荐规模：

- active stage ledger 控制在 200-350 行；
- 每次实验结果只写 5-10 行 summary，并链接 `analysis/` 中的完整报告；
- 表格只保存 candidate queue、experiment ledger、sync log 三类；
- 不复制大段 metric 表；只写方向性结论、关键 deltas、pass/fail 判断和 artifact 路径；
- 旧 candidate 超过两个阶段不再可能执行时，标为 `superseded` 或 `deferred`，不要无限保留在 active queue。

## Paper Mainline 同步规则

Stage Ledger 结论只有在满足以下任一条件时，才同步到 `docs/paper-mainline.md`：

1. 改变论文核心问题或 working title。
2. 改变预期贡献边界，例如新增、删除或合并贡献。
3. 产生 `passed_core_candidate` 或连续 `failed_as_core_candidate`，足以改变方法主线。
4. 改变主 baseline、active carrier、主实验矩阵或目标 claim。
5. 触发 11-step rollback 到 Step 2/3，说明问题定义或存在性证据需要重审。

不应同步到 paper-mainline 的内容：

- 单个 diagnostic/control 的局部 metric；
- 尚未通过 narrative gate 的 speculative idea；
- 单个数据集上的弱正向或弱负向；
- 只是 implementation bug、训练脚本修复或资源调度细节。

每次同步 paper-mainline 时，必须在对应 Stage Ledger 的 `paper_mainline_sync_log` 中记录：

- 日期；
- 同步原因；
- 修改的 paper-mainline section；
- 是否改变了贡献、方法、实验安排或只改变转向规则。

## Roadmap 同步规则

`docs/research-roadmap.md` 继续作为跨阶段索引，但不再承担所有阶段内 backlog。它只记录：

- 新阶段创建；
- 11-step rollback；
- gate decision；
- active Stage Ledger 路径；
- 关键 artifact 路径；
- 是否触发 paper-mainline 同步。

当一个阶段内部只是从 candidate A 切换到 candidate B，且不改变论文总纲时，优先更新 Stage
Ledger；Roadmap 只在该切换改变 11-step rollback 或阶段状态时更新。

## 继续研究的标准流程

每次继续研究时执行以下顺序：

1. 读取 `docs/paper-mainline.md` 的当前状态表，确认论文级约束。
2. 读取 active `docs/stage-ledgers/<stage>.md`，确认 `decision_cursor`、candidate queue 和
   pending tasks。
3. 如果用户说远程实验完成，先同步并分析 artifacts，把完整报告写入 `analysis/`。
4. 在 Stage Ledger 中更新对应 candidate 的 `effectiveness_gate` 和 `decision`。
5. 检查是否存在未执行候选；若有，优先做 candidate triage，而不是直接重构主线。
6. 只有达到 Paper Mainline 同步阈值时，才更新 `docs/paper-mainline.md`。
7. 若进入新候选实验，先确认 narrative gate，再实现、验证、commit/push、远程启动。

## 失败结论的边界

失败结论必须按最小可证伪单元表述：

- 一个 implementation 失败，只否定该 implementation。
- 一个 candidate 失败，只否定该 candidate 的 hypothesis。
- 一个 candidate family 失败，才否定该机制族。
- 只有多个机制族都失败，且 candidate queue 没有合理 remaining candidates，才重审 stage mainline。
- 只有 stage mainline 被重审后仍不成立，才改写 paper-mainline 的核心贡献或论文路线。

## Failure Attribution 规则

任何可能阻断 paper-core 方向的 diagnostic，必须先做 failure attribution，不能把诊断设计缺陷归纳为方向失败。

失败至少要拆成以下类别：

| 类别 | 含义 | 可否否定方向 |
| --- | --- | --- |
| `hypothesis_false` | 核心问题本身未被支持 | 只有覆盖了正确 intervention point 和稳定 control 后才可以 |
| `intervention_point_wrong` | 信息注入位置太晚、太早或不在真实机制路径上 | 不可以，只否定该设计 |
| `readout_or_head_design_wrong` | readout/head 形式太弱、过线性、数值病态或与机制不一致 | 不可以，只否定该 head/readout |
| `optimization_or_numeric_pathology` | 发散、ill-conditioned inversion、过拟合、val/test mismatch、超过 `100%` 的异常退化 | 不可以，必须先标记诊断不适合做方向结论 |
| `capacity_control_explains` | no-mechanism/no-condition control 解释收益 | 可以阻断当前 method，但仍需说明是否只否定该 intervention/readout |

若出现 `optimization_or_numeric_pathology`，报告中的 `decision` 必须使用
`diagnostic_invalid_for_direction_rejection`、`design_fault_suspected` 或等价状态，而不是
`rejected_direction`。只有在重新设计稳定诊断、覆盖正确 mechanism path，并再次输给 controls 后，才能否定更大方向。

每份 blocking diagnostic report 必须回答：

1. 哪个具体实现或诊断设计失败？
2. 是否存在明显数值/优化/overfit 病态？
3. 失败是否可能由 intervention point 或 readout/head 设计导致？
4. 哪些方向级问题仍未被测试？
5. 下一步是修正诊断、重设 intervention point，还是才允许 rollback？

## Paper-Facing Evaluation 与 Milestone Test Audit 规则

本项目将official test split固定为paper-core effectiveness与Step 9-10继续/回滚决策的主要性能证据。此前
“test只在最终论文表格中使用”的规则废止；但test也不得退化为逐次调参反馈。固定职责如下：

| Split | 固定职责 | 禁止事项 |
| --- | --- | --- |
| Train | parameter optimization与train-only diagnostics | 伪装成generalization evidence |
| Validation | 与论文horizon对齐的日常开发、early stopping、checkpoint与低成本机制筛选 | 把validation gain写成最终effectiveness |
| Test | 冻结candidate version的里程碑effectiveness gate、main result与formal ablation | 选epoch、选checkpoint、逐次机制搜索或逐dataset/horizon反向调参 |

### 默认 paper-facing scorecard

标准long-term forecasting任务默认使用$H\in\{96,192,336,720\}$。main result、formal ablation和常规
validation development screen必须逐dataset、逐horizon报告MSE与MAE。若任务或dataset不支持该集合，必须在
结果产生前冻结替代集合及原因。

machine-readable默认协议保存在`configs/paper_facing_evaluation_protocol.json`。候选protocol可以增加更严格
threshold或task-specific evidence，但不得静默删除该默认scorecard。

内部candidate排序默认使用paired relative gain：

$$
G(A,B;d,H)=100\left(1-\frac{L_A(d,H)}{L_B(d,H)}\right).
$$

正值表示candidate更好。报告至少包含完整cell table、所有dataset-horizon cells的equal-weight macro gain、
cell wins，以及先在每个dataset内跨horizon平均后得到的dataset wins。不同dataset的raw MSE/MAE不可直接混成
唯一总分；各candidate protocol仍需在结果返回前冻结具体gain与wins threshold。

新unified-horizon candidate的默认checkpoint score为validation上四个标准horizon MSE的算术平均：

$$
S_{\mathrm{val,std}}=\frac14\sum_{H\in\{96,192,336,720\}}L_{\mathrm{val}}(H).
$$

所有matched arms必须共享该规则。若要用baseline normalization、dense AUC或其他checkpoint score，必须在结果
前说明其与paper claim的关系并预注册，不能在看到结果后切换。

### Dense 与机制诊断的职责

H1..720 dense curve、dense-prefix AUC、horizon bins、per-epoch trajectory、gradient/router/arm statistics默认属于：

1. 机制为何成功或失败的`diagnostic_only`证据；
2. 证明unified-horizon连续行为的补充paper evidence；
3. 发现standard horizons掩盖的局部pathology。

它们默认不替代四个标准horizon的常规ranking gate。若某项dense metric要成为primary method gate，必须由论文
问题直接要求并在实验前冻结。历史dense结果保留，不因新规则被删除；重新评估时应标为retrospective diagnostic。

每次test audit必须在访问test前完成文本与machine-readable preregistration，至少冻结：

1. `candidate_version`、architecture/objective与source commit；
2. dataset profiles、seeds、checkpoint policy及checkpoint hashes；
3. 完整candidate/control matrix、全部horizons、primary/secondary metrics与pass thresholds；
4. `test_role`、用户授权日期、是否允许retraining以及访问次数；
5. 各种结果对应的继续、confirmation、rollback或candidate-version升级动作。

一个冻结candidate version默认只允许一次完整test audit。audit必须运行全部预注册dataset × arm × seed matrix，
不得只报告正向dataset、horizon或control。test可以决定该version是否继续，但不能选择checkpoint，也不能直接用于
per-dataset/per-horizon超参数优化。

观察test后发生的任何architecture、objective、loss coefficient、training schedule或control变化，都必须创建新的
candidate version，并标记`test_informed=true`及其具体来源。后续报告不得把official test重新描述为完全untouched。
若确需对同一version重跑test，必须由用户显式授权，并记录原因、变化范围及是否只是artifact repair。

test performance是effectiveness的primary gate，但不是mechanism attribution的替代品：paper-core pass仍需
validation/train diagnostics、matched capacity/random/equal controls与failure attribution。若validation与test排序
反转，必须标记`validation_test_reversal`并优先检查split representativeness、checkpoint rule与seed stability，
不得隐藏其中任一侧。

最终模型与论文中的formal ablations可以在同一次完整official test audit中展示，但必须先一次性冻结main/ablation
matrix。该结果用于论文报告，不授权根据某个test cell继续修改机制。若观察test后修改architecture、objective、
loss coefficient、training schedule或control definition，新版本必须标记`test_informed=true`并重新经过
narrative/design gate。

每份test audit报告必须记录：`test_access_date`、`user_authorization`、`candidate_version`、`checkpoint_hash`、
`checkpoint_retrained`、`test_role`、`matrix_complete`、test metrics、validation-test comparison及Step 9-10 decision。

## Frozen Component Replacement 公平性规则

joint training得到的Encoder与Decoder会共同塑造中间representation。若冻结一个与原Decoder共同训练的
Encoder，再只替换Decoder，则实验回答的是“新Decoder能否兼容A6-specific representation”，而不是“新
Encoder-Decoder架构端到端训练后是否有效”。因此：

1. frozen replacement只可作为`diagnostic_only`，用于information access、matched within-family attribution、
   debugging或counterfactual；结论必须写明`conditional on frozen representation`；
2. 当frozen component曾与control head联合训练时，replacement gap不得用于拒绝paper-core方向、判定method
   readiness，或强制转入capacity-preserving redesign；
3. paper-core effectiveness默认要求from-scratch end-to-end joint training，所有arms共享data split、dataset
   profile、objective、optimizer class、checkpoint selection与evaluation protocol；
4. warm-start、freeze/unfreeze、cross-swap或$2\times2$ encoder-decoder exchange只作为次级attribution，不作为
   primary method gate；
5. 若确需冻结实验，应尽可能使用对称controls，并分别报告representation compatibility、optimization与
   architecture effectiveness，禁止将三者合并为一个pass/fail。

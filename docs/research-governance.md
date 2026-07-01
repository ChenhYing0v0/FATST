# FATST 研究路径保存体系

本文档定义项目级研究进度管理机制，目标是防止候选方案、阶段结论和论文主线之间发生上下文丢失。

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

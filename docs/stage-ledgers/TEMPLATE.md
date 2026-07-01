# <Stage Name> Stage Ledger

本文档是阶段内执行账本，只保存当前研究阶段的候选队列、决策 cursor 和 artifact 链接。详细实验分析放入
`analysis/`，代码解释放入 `docs/code-explanation/`，论文总纲放入 `docs/paper-mainline.md`。

## Stage Scope

| Field | Content |
| --- | --- |
| `stage_id` |  |
| `paper_mainline_role` |  |
| `active_question` |  |
| `active_carrier` |  |
| `entry_evidence` |  |
| `stage_exit_condition` |  |
| `stage_rollback_condition` |  |

## Decision Cursor

| Field | Content |
| --- | --- |
| `current_11_step` |  |
| `current_candidate` |  |
| `latest_decision` |  |
| `next_required_action` |  |
| `rollback_point` |  |

## Candidate Queue

| ID | Status | Hypothesis | Narrative Gate | Effectiveness Gate | Blocking Or Next Action | Artifacts |
| --- | --- | --- | --- | --- | --- | --- |
|  | `proposed` |  |  |  |  |  |

## Experiment Ledger

| Experiment | Candidate | Role | Result Summary | Decision | Full Report |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

## Pending Tasks

| Task | Owner | Trigger | Status | Next Action |
| --- | --- | --- | --- | --- |
|  | Codex |  | `pending` |  |

## Paper Mainline Sync Log

| Date | Trigger | Paper Section | Change Type | Decision |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

## Notes For Next Continuation

- Before proposing a new paper-core method, check whether any candidate in `candidate_queue` is still
  `proposed`, `narrative_ready`, `analysis_pending`, or `partial_pass`.
- Do not convert a single candidate failure into stage failure unless the queue has no remaining viable candidate.

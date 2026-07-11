# Stage Ledgers

本目录保存阶段内执行账本。每个 active research stage 必须有一个 ledger，用于记录候选队列、
pending tasks、decision cursor 和 paper-mainline sync 状态。

## Active Ledger

- `stage-c-unified-forecasting-redesign.md`：当前 StageC unified forecasting decoder/training redesign。

## Historical Ledgers

- `phase5-timealign-interface.md`：Phase5 StageA/StageB 的完整候选与 rollback 记录；不再是 active cursor。
- `phase5-stageb-b13-restart-handoff-20260710.md`：StageB/C1 关闭时的 restart snapshot。

## Usage

- 继续研究、设计下一步实验或分析远程结果前，先读取 active ledger。
- 新阶段启动时，从 `TEMPLATE.md` 复制结构并登记到 `docs/paper-mainline.md` 与
  `docs/research-roadmap.md`。
- 完整实验分析不要写入 ledger；只写 summary、candidate status、next action 和 artifact path。

# Experiments

实验记录按以下读者路径组织：

```text
what we plan to test
why it matters
how data/artifacts are constructed
what each metric means
how results support or falsify the plan
what decision follows
```

不要只保存未解释的 metric list 或 gate label。

## 与 Stage Ledger 的关系

- `docs/stage-ledgers/` 保存阶段内 candidate queue、pending tasks 和 decision cursor。
- `docs/experiments/` 保存实验方案和 protocol。
- `analysis/` 保存实验完成后的详细结果分析。
- 单次实验完成后，完整分析写入 `analysis/`；Stage Ledger 只写 5-10 行 summary、candidate
  status 和 artifact path。
- 若实验结果改变 paper claim、贡献边界或主实验安排，再同步 `docs/paper-mainline.md`。

## 当前路线

- `phase4-horizon-supervision-scheduling-r3-reset.md`: 当前 active research
  route，锚定 `Horizon Supervision Scheduling for Unified Multi-Horizon
  Forecasting`，以 R.3 作为干净起点，强调 training/evaluation 解耦，training unit
  不按 evaluation horizons 定义。
- `phase4-horizon-decoupled-protocol.md`: R4.1 protocol，定义 supervision unit API、
  trace、最小触达文件、本地 smoke 和远程 gate。

## 历史 Phase4 记录

- `phase4-horizon-agnostic-supervision-reset.md`: 历史 diagnostic route；已被
  Phase4-R horizon-decoupled reset 取代。
- `phase4-component-balanced-objective-design.md`: 暂停的候选；只作为 HSS 的潜在扩展，
  不是当前第一实现。

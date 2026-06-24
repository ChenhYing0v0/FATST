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

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

- Phase5 StageA 已固定为 `A6-LBF-r256`。当前 active route 由
  `docs/stage-ledgers/phase5-timealign-interface.md` 和 `docs/research-roadmap.md`
  记录。
- 下一份新增实验文档应是 StageB 重新设计文档，命名为
  `phase5-stage-b-*.md`，并先完成 problem definition 与 narrative gate。

## 历史记录

- `phase4-horizon-agnostic-supervision-reset.md`: 历史 diagnostic route；已被
  Phase4-R horizon-decoupled reset 取代。
- `phase4-component-balanced-objective-design.md`: 暂停的候选；只作为 HSS 的潜在扩展，
  不是当前第一实现。
- `docs/archive/phase5-stage-a/experiments/`: Phase5 StageA 旧候选和 B0 预清理诊断文档。

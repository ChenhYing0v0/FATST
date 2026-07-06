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
- 当前 StageB 重新设计入口是
  `phase5-stage-b-reliability-aware-supervision-redesign.md`，先执行 B1 diagnostic-only
  problem-existence gate，再决定是否进入 method implementation。
- B1 已判定为 `partial_pass_distance_confounded`，B2-RAS 不进入实现。当前下一步是
  `phase5-stage-b-distance-normalized-seasonal-residual-diagnostic.md`，验证 `B3-DSR`
  是否能作为 non-distance-confounded reliability problem。
- B3 已判定为 `partial_pass_needs_stronger_proxy_or_method_boundary`。当前不得实现
  reliability-aware loss weighting；下一步必须先决定是否继续找更强 train-only structural proxy，
  或关闭 StageB 并转向 label-autocorrelation objective route。
- 当前 StageB 已转向 TimeAlign dependency / basis-aware alignment 诊断：
  `phase5-stage-b-timealign-dependency-and-basis-align-diagnostic.md`。现有 artifact audit 只能得到
  `partial_dependency_risk_confirmed`；下一步是远程 no-align/no-recon ablation，不是 method implementation。

## 历史记录

- `phase4-horizon-agnostic-supervision-reset.md`: 历史 diagnostic route；已被
  Phase4-R horizon-decoupled reset 取代。
- `phase4-component-balanced-objective-design.md`: 暂停的候选；只作为 HSS 的潜在扩展，
  不是当前第一实现。
- `docs/archive/phase5-stage-a/experiments/`: Phase5 StageA 旧候选和 B0 预清理诊断文档。

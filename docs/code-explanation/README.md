# Code Explanation

模型结构或非平凡代码更新后，在本目录同步写 code-facing explanation。

模型文档优先按 forward computation flow 组织，说明 tensor names、shapes、
operations，以及 changed tensor 进入下游模块的位置。

非模型代码按 functional module 组织，例如 training、data loading、metrics、
runner、diagnostics、remote scripts 或 analysis。

## 当前 Phase5 入口

- `phase5-clean-timealign-a6-lbf.md`: StageA 固化后的 clean official TimeAlign +
  A6-LBF-r256 实现说明。
- `docs/archive/phase5-stage-a/code-explanation/`: 旧 StageA 变体和诊断代码说明归档。

# Code Explanation

模型结构或非平凡代码更新后，在本目录同步写 code-facing explanation。

模型文档优先按 forward computation flow 组织，说明 tensor names、shapes、
operations，以及 changed tensor 进入下游模块的位置。

非模型代码按 functional module 组织，例如 training、data loading、metrics、
runner、diagnostics、remote scripts 或 analysis。

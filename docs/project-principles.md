# R_2026_FATST 基础原则

## 已确认事实

- [Fact] 本仓库是 `R_2026_FSA` 的后续研究仓库，但以干净起点推进。
- [Fact] 目标是围绕 time series forecasting 产出高水平 SCI 期刊论文。
- [Fact] 准备推进的方向包括 one model for multi-horizon、future-aware
  architecture、MoE。
- [Fact] Zotero 的 `FSA` 子集是当前文献种子集合。
- [Fact] 重点对比 baseline 是 SRSNet。
- [Fact] 本地可用 conda 环境是 `r2026-fsa`。
- [Fact] 远程 `529_Lab-3090` 可用 conda 环境是 `moe`。
- [Fact] 用户提到 `3090` 或 `529lab` 时，默认指 `529_Lab-3090`。
- [Fact] 远程 GPU 1、2 通常较稳定；GPU 0 存在进程被 kill 的风险。
- [Fact] GitHub repository is
  `git@github.com:ChenhYing0v0/FATST.git`.

## 清洁边界

- 不默认引入旧仓库 `R_2026_FSA` 的代码、配置、实验产物或项目记忆。
- 需要使用旧仓库证据时，先明确来源、用途和落点，并获得用户确认。
- 可以保持目录架构兼容，例如 `analysis`、`artifacts`、`baselines`、
  `docs`，但不继承旧实现细节。

## 研究推进原则

- 先建立可审计的最小实验链路，再扩展模型复杂度。
- 每个新 model component 都需要说明它服务的 mechanism claim。
- 每个实验结论都需要记录数据集、horizon、baseline、seed、环境和输出路径。
- 强结论必须来自可复查 artifact，而不是聊天上下文。

## Git 与远程同步原则

- 每次完成阶段性的代码更新后，先完成最小必要验证，再进行一次 focused
  `git commit` 和 `git push`。
- 每次准备在 `529_Lab-3090` 上运行远程实验前，本地必须先完成
  `git commit` 和 `git push`，确保远程实验绑定到可追踪代码版本。
- `529_Lab-3090` 的代码同步方式是进入远程项目目录后执行 `git pull`，
  不通过手动拷贝本地源码作为默认同步路径。
- commit message 默认使用 Conventional Commits。
- commit 只包含当前阶段相关文件；敏感文件、环境文件、大型数据和实验输出不得提交。

## 远程实验原则

- 每次跑实验前先查看 `nvidia-smi`。
- 优先选择显存占用低的 GPU。
- 默认优先考虑 GPU 1 或 GPU 2；避免 GPU 0，除非用户明确接受风险。
- 保留显存安全余量，不把 GPU 填满。
- 记录实际使用 GPU、启动命令、conda 环境、代码版本和输出目录。
- 后续正式 3090 实验默认写入 repo 外部输出目录，例如
  `/home/yingch/exp_outputs/r-2026-fatst/phase0`；仓库内 `artifacts/runs/...`
  只用于本地 smoke、小型临时验证或已经启动的历史 run。

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

## 长研究执行模板

后续围绕 decoder、future-aware、MoE 或新架构的长研究循环固定为：

1. 调研分析。
2. 提出待解决问题。
3. 评估问题是否值得研究，以及问题是否真实存在。
4. 提出 idea。
5. 评估 idea 的理论可行性。
6. 设计方案。
7. 实现方案。
8. 远程训练。
9. 评估结果。
10. 判断是否通过：同时看性能收益与论文故事是否成立。
11. 若不通过，评估应回退至哪一步，然后继续循环。

该模板不是报告格式装饰，而是阶段推进的决策边界。step 6 的设计方案必须包含
可证伪 hypothesis、数学数据流、实验协议和通过/回退条件。一个机制只有在性能证据和
paper narrative 都站得住时才继续叠加；若证据不足，应明确回退点，而不是继续堆模块。

每一轮长研究都必须记录 `current_step`、`problem`、`existence_evidence`、`idea`、
`theory_check`、`design`、`gate`、`artifacts` 和 `decision`。其中 `decision` 必须说明
是否通过；若不通过，必须说明回退到 11-step loop 的哪一步。后续不能在一个机制未通过时
直接叠加 future-aware、MoE 或其他复杂模块。

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

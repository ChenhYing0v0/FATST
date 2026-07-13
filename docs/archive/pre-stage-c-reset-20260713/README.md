# Pre-StageC Reset Archive

本目录保存 2026-07-13 research reset 前的历史 protocol、code explanation、stage ledger 与两份顶层主线
snapshot。它们用于审计“什么已经测试过、为什么关闭”，不再作为 active research instructions。

- `experiments/`: Phase0-Phase5 历史设计，以及已完成或被取代的 StageC calibration protocols；
- `code-explanation/`: 对应历史实现说明；
- `stage-ledgers/`: StageB handoff 与 candidate queue；
- `paper-mainline-stageb-snapshot.md`: reset 前 paper mainline；
- `research-roadmap-stageb-snapshot.md`: reset 前 11-step roadmap。

历史详细指标仍保存在 repo 根目录 `analysis/`。需要复活某条路线时，必须先在 active StageC ledger 中
写明 source、目的、failure-attribution 与新的 narrative gate，不能直接从 archive 启动 runner。

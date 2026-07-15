# SC1-D10 Raw Scale Identifiability Code Explanation

## Scope

D10只新增raw-data diagnostic worker、aggregate analyzer与remote/sync wrappers，不读取checkpoint、不修改model、
不训练forecast network。冻结协议见
`analysis/stage_c_sc1_d10_raw_scale_identifiability_20260715/d10_step23_diagnostic_design.md`。

## Worker Data Flow

1. 使用TimeAlign官方dataset class构造train/validation datasets，但自行选择chronological indices，避免train
   loader shuffle破坏fit/gap/holdout contract；
2. `history/future [W,720,C]`按history mean/std做instance normalization；
3. channel展开为rows，history乘DCT-II，future乘RGNB synthesis；
4. canonical、history-permuted与future-permuted families保持同一七组sizes；
5. 每组用fit-only mean/std whiten，并以paired orthonormal sketch压到16维；
6. 固定三组lambda计算closed-form ridge，在holdout与validation分别报告$R^2$；
7. 同时写出7×7 matrix cells与global/detail 2×2 cells。

## Analyzer Flow

1. 审计五dataset artifact completeness：每个dataset应有2646个matrix rows与216个binary rows；
2. 对每个dataset/family/sketch/lambda/split重建矩阵；
3. 计算binary directional selectivity、interaction、detail-only diagonal gain、best count与6! mapping p-value；
4. 先聚合9 replicates，再应用dataset replication、paired control与holdout/validation gates；
5. 另将7×7与2×2 cells按`dataset/family/split/row/column`聚合，保留跨9个
   `sketch_seed × ridge_lambda` replicates的mean $R^2$，供off-diagonal pattern审计；
6. decision严格三选一：detail-monotone、binary-only或no-aligned-scale。无论哪种结果，method/test/SC2均false。

## Aggregate Artifact Schema

- `dataset_matrix_cells.csv`：`future_group_index/history_group_index`分别指future RGNB group与history DCT
  band；`mean_r2`为同dataset、family、split、cell的replicate均值，`replicate_count`应为9；
- `dataset_binary_cells.csv`：两个index分别是global/detail二分后的future/history cell，其他聚合定义相同；
- `replicate_metrics.csv`：每行对应一个dataset、family、split、sketch seed与ridge lambda，存放该7×7/2×2
  matrix导出的五项统计；
- `dataset_metrics.csv`：将`replicate_metrics.csv`按dataset、family、split聚合；
- `gate.json`：记录invariants、每项冻结gate的布尔值与计数、最终decision及authorization boundary；
- `research_interpretation.md`：只报告canonical validation聚合和direction boundary，不替代详细结果报告。

## Code-Theory Consistency

代码中的history/future group sizes完全相同，但仍统一sketch为16维，保证所有probe cell具有相同parameter class。
binary metric比较同一future target下global/detail history inputs，避免future global天然更易预测造成伪interaction；
detail-monotone metric排除global row/column，避免D9发现的粗二分伪造details内部对角结构。

仍属于proxy的部分：random sketch只观测每个group的16维随机子空间，不能证明完整subspace predictability；三
sketch seeds与三fixed lambdas用于检查稳定性。linear ridge失败也不能否定所有nonlinear dependence，因此negative
decision只关闭“足以支持当前aligned-scale architecture叙事的linear predictive evidence”。

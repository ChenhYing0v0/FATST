# D17-v0 protocol invalidation

本目录中的`fold_metrics.csv`、`aggregate_metrics.csv`、`comparison_cells.csv`与`summary.json`来自首次
`same-test-probe two-fold row split`。该split在`probe_fused [256,720]`的flattened
`sample × channel` rows上直接对半切分，同一批次、同一时间邻域甚至同一样本的不同channels可能跨fold。

pointwise correction相对parent出现约21.27%的异常大收益，说明该协议不足以证明跨split generalization。尽管修复
fixed-domain coordinate后prefix invariance为0、其余signal gates表面通过，这些结果全部标记
`diagnostic_protocol_invalid_for_problem_promotion`，不得进入paper claim、candidate gate或stage decision。

替代协议为：只在独立validation probe拟合fixed ridge correction，再在既有authorized test probe评估；test labels
不参与拟合、feature选择或超参数选择。新结果写入独立的`d17_validation_to_test/`目录。

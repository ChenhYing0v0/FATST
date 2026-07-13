# Analysis

保存分析脚本、统计说明、可复查报告和结果解释。

新增 statistic、CSV column 或 figure quantity 时，需要说明来源 tensor/file、
计算方式和含义。

## Active StageC Entry

- `stage_c_natural_baseline_test_20260713/`: frozen 3-dataset × 3-seed × 8-horizon test reference；
- `stage_c_contribution_research_reset_20260713/`: PMFO/PIR Step 1-3 deep audit；
- `stage_c_d1_pmfo_pir_offline_v2_20260713/`: accepted D1 problem diagnostics；
- `stage_c_step46_pmfo_pir_theory_gate_20260713/`: external prior-art、mixed-radix proof与MIPR measure geometry。

其余目录是不可变历史 evidence store，不是 active candidate queue。继续研究应从
`docs/stage-ledgers/stage-c-unified-forecasting-redesign.md` 进入；不得因某个旧 analysis 目录存在就直接
恢复对应 method。

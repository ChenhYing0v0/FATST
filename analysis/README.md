# Analysis

保存分析脚本、统计说明、可复查报告和结果解释。

新增 statistic、CSV column 或 figure quantity 时，需要说明来源 tensor/file、
计算方式和含义。

## Active StageC Entry

- `stage_c_natural_baseline_test_20260713/`: frozen 3-dataset × 3-seed × 8-horizon test reference；
- `stage_c_d14a_output_coupling_granularity_20260715/`: neutral PCA64 + matched-factor RRR的D14-A0
  五数据集三fold结果；exact gate fail，方向级拒绝因intervention/DoF设计不足而无效，回Step 2-3；
- `stage_c_fixed_past_mainline_reset_20260715/`: 当前fixed-past主线复盘、external audit、CADMO/CPGA
  provisional design与D14 gate；
- `stage_c_post_d12_revision_surface_mainline_20260715/`: 已转入`New-idea.md`的future-paper历史复盘，非active。

其余目录是不可变历史 evidence store，不是 active candidate queue。继续研究应从
`docs/stage-ledgers/stage-c-unified-forecasting-redesign.md` 进入；不得因某个旧 analysis 目录存在就直接
恢复对应 method。

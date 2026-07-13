# StageC Five-Dataset Profile Extension Pipeline

## Scope

该pipeline只为ETTh1/ETTm2冻结dataset-aware natural profiles，不修改模型结构，也不读取test。active文件：

- `configs/stage_c_five_dataset_profile_extension.json`：grid、selection、stability和14-run budget；
- `scripts/analyze_stage_c_five_profile_extension.py`：三阶段完整性审计与validation-only决策；
- `scripts/remote/run_stage_c_five_profile_extension.sh`：3090三阶段续跑式训练；
- `scripts/sync_stage_c_five_profile_extension_results.sh`：轻量同步并在本地独立重算。

archive中的R2脚本只作为历史设计证据。本pipeline保留相同coarse-grid与CV gate，但使用独立protocol profile、
profile hash、run prefix和artifact root，避免历史入口重新成为active dependency。

## Training And Artifact Flow

每个run的输入是`batch_x: [B,720,C]`，TimeAlign encoder输出token memory，A6-LBF head产生
`prediction: [B,720,C]`。训练只监督H720，checkpoint由validation H720选择；训练结束后仍在validation split
裁剪并报告H48/96/144/192/288/336/512/720。profile只改变`patch_num,d_model,d_ff`，其余optimizer、loss、
rank256 basis head与stopping contract固定。

远端输出路径为：

```text
phase_root / SC0FIVE_R2{A|B|C}_profile / dataset / h720_full / seed / artifacts
```

Phase B复用Phase A中的medium run；Phase C复用seed2021 winner，只新增2022/2023。runner检测
`metrics_by_target_horizon.csv`后跳过已完成run，因此阶段中断后可安全续跑。

## Analyzer Definitions

- `macro_dense_regret`：对每个horizon，以该dataset所有候选的最低validation MSE为分母计算normalized
  regret，再对8 horizons取均值；
- `max_dense_regret`：上述8项regret的最大值；
- `mean_dense_mse_cv`：selected profile在3 seeds、每个horizon的MSE CV，再对8 horizons取均值；
- `max_dense_mse_cv`：8个horizon CV的最大值；
- `active_forward_parameters`：只作报告与审计，不进入selection或stability gate。

每个run还必须通过profile hash、protocol profile、`final_evaluation_split=val`、seed/dimension、finite training
log和8-horizon completeness检查。结果不完整时仍写diagnostics与`analysis_incomplete` summary，便于定位和续跑。

## Code-Theory Consistency

理论目标是建立不被candidate identity或test反馈污染的dataset-aware control。代码用固定grid、预注册tie-break、
validation-only artifacts和跨seed absolute stability实现该目标。它不证明所选profile是全局最优，也不以参数量
相近为公平性依据；它只冻结一个可重复、有限预算的自然起点。若后续D2表现差，不能回头为D2改变profile。

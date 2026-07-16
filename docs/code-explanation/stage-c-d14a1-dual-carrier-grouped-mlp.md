# StageC D14-A1 Dual-Carrier Grouped-MLP Code Explanation

## 1. Scope And Authorization

本实现是`diagnostic_only`，用于修复D14-A0不同scale几乎产生同一函数的问题，不是PCSD/CCRL paper method。
当前只授权`neutral_raw, seed2021`的validation-only远程screen；A6-natural、D14-B、test与paper-method training均
由串行gate拦截。

## 2. Model Forward Flow

### Neutral carrier

1. loader输入`batch_x: [B,720,C]`；
2. TimeAlign既有instance normalization得到`x_norm: [B,720,C]`；
3. `raw-history-identity`转置并增加token axis，得到
   `memory: [B,C,1,720]`；该路径没有learned encoder或global basis；
4. flatten得到`hidden: [B,C,R]`，neutral时`R=720`；
5. `GroupedMLPReadout`按future partition取$G=720/s$个group，每组独立执行
   `hidden @ input_weight[g] + hidden_bias[g] -> GELU -> @ output_weight[g] + output_bias[g]`；
6. group outputs先为`[B,C,G,s]`，再按`group_indices: [G,s]`scatter回
   `forecast_full: [B,C,720]`；
7. requested target length只执行`forecast_full[..., :H]`，最终返回`[B,H,C]`。

### A6-natural sensitivity carrier

`timealign-token-mlp`把normalized history变成`memory: [B,C,P,D]`，其中五profile的
`R=P*D`分别为768、1536或3072；flatten后进入同一个GroupedMLP contract。所有模块从头joint training，
没有frozen replacement或checkpoint warm-start。A6-LBF exact control只在neutral problem pass后与该carrier一起运行。

## 3. Sharing Topology And Parameter Shapes

对scale $s$，`GroupedMLPReadout`保存：

- `input_weight: [G,R,k_s]`；
- `hidden_bias: [G,k_s]`；
- `output_weight: [G,k_s,s]`；
- `output_bias: [G,s]`；
- `group_indices: [G,s]`，canonical或train-independent random partition。

不同groups没有共享decoder parameters；同group targets共享hidden features。local checker进一步对任意两target的
decoder gradients求support，验证“同group iff parameter overlap”。这比只比较nominal block label更直接地证明
intervention确实改变了计算图。

point arm固定`k_1=4`；其他`k_s`最小化与point arm的总decoder parameter gap。checker还使用
`GELU(u)-GELU(-u)=u`构造任意full-affine block map的exact witness，避免把linear expressivity缺失误认为
coupling effect。

## 4. Training And Artifact Flow

远程runner按`Weather -> ETTm1 -> ETTm2 -> ETTh1 -> ETTh2`的dataset-major顺序调度三GPU。neutral包含
5个canonical scales与3个random-partition controls，共40 runs；A6 sensitivity另加A6-LBF control，共45 runs。

所有candidate使用full-H720 pointwise L1训练、best-validation H720 MSE checkpoint、validation full-crop评估，
不读取test。每个grouped run随后由checkpoint evaluator生成：

- `validation_diagnostics.npz`：`row_bin_mse/mae [N,3]`、相同行的persistence losses、前1024行
  `probe_predictions/probe_targets [N_probe,720]`；一行对应一个window-channel；
- `trained_invariants.json`：carrier/readout/scale/partition、validation row数、冻结参数数、from-scratch与
  parameter-gap检查；
- 既有`training_log.csv`、`metrics_by_target_horizon.csv`、`effective_config.json`与environment记录。

Step7A还直接调用training CLI parser验证两套runner参数：neutral不得携带legacy encoder overrides；只有
A6-natural传入冻结的`patch_num/d_model/d_ff`。这可在访问dataset或启动GPU前发现carrier参数串线。

official `data_provider(..., "val")`默认shuffle validation。训练期aggregate validation mean不受row order影响，
但sample-wise oracle要求不同arms逐行对齐。因此checkpoint evaluator会基于同一个validation dataset重建
`shuffle=False`的sequential loader，并在`trained_invariants.json`记录
`row_order=dataset_sequential`。`REEVALUATE=1`可只覆盖row diagnostics而不重训checkpoint。

## 5. Analyzer Statistics

- `carrier_skill_relative_gain`：train-only selected fixed canonical scale相对persistence的validation MSE改善；
- `prediction_disagreement_median`：所有canonical arm pair的prediction RMSE，除以target去row mean后的RMS，
  用于确认不同scale确实学成不同函数；
- `crossing`：至少一对canonical scales在short/mid/long bins中出现大于0.1%的双向胜负；
- `canonical_oracle_relative_gain`：逐sample × bin选择最佳canonical scale，相对train-only selected fixed scale的
  MSE headroom；
- `oracle_vs_validation_best_fixed_gain`：相对validation上事后最佳fixed scale的严格oracle headroom；它不用于
  选method，只用于排除train-to-validation static scale-selection gap；
- `sample_oracle_vs_validation_bin_policy_gain`：sample × bin oracle相对“每个future bin固定一个scale”的增量，
  隔离instance-specific headroom；
- `canonical_vs_random_oracle_relative_gain`：canonical oracle相对matched random-partition oracle的改善；
- `severe_degradation`：最差canonical arm相对persistence恶化超过100%的numeric/optimization pathology标记。

neutral只有同时通过invariants、function separation、carrier skill、crossing、oracle与contiguity gates，才授权A6。
A6结果单独报告，不与neutral平均。

三seed confirmation由`analyze_stage_c_d14a1_multiseed.py`聚合：一个crossing pair、function separation、carrier
skill或contiguity必须在同dataset至少2/3 seeds复现；strict oracle、instance oracle和contiguity再做five-dataset
macro gate。A6-LBF performance只报告carrier compatibility，不参与scale problem gate。

## 6. Code-Theory Consistency

### Intended theory

有限样本/有限capacity下，future targets的最佳parameter-sharing scope可能随future region变化；真正的scale
diagnostic必须改变nonlinear sharing topology，同时保持基本forecast function class可比。

### Code realization

group ownership决定hidden bank sharing，random partition隔离temporal contiguity，full-affine containment与参数匹配
隔离基础capacity，neutral/A6串行双carrier隔离scale hypothesis和paper-carrier compatibility。

### Remaining proxy

fixed groups仍不是adaptive PCSD，单seed crossing也不是稳定性证据；neutral raw-history head和A6 profiles的优化难度
可能不同。A6-negative尤其可能来自原A6 encoder/global-basis共适配，不是scale hypothesis为假。

### Falsification boundary

只有neutral diagnostic有效且problem gate失败，才关闭当前PCSD/CCRL pair。neutral carrier无skill或arms没有function
separation时，结论为`diagnostic_invalid_for_direction_rejection`；A6-negative永远只记为
`carrier_interface_or_profile_incompatibility`或nonconfirming sensitivity。

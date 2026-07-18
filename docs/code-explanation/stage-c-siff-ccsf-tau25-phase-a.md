# CCSF tau0.25 formal Phase-A代码说明

## Forward与artifact路径

CCSF训练后的正式evaluator沿真实forward计算：

1. `batch_x [B,720,C] -> encode_history -> hidden [B,C,D]`；
2. SIFF field产生`arms [B,C,S=5,T=720]`和`base_logits [B,C,T,S]`；
3. CCSF由arms构造`contrast_descriptor [B,C,T,S,6]`；
4. shared scorer产生`correction_logits [B,C,T,S]`；
5. `policy=softmax(base_logits+correction_logits)`，得到full forecast，再以prefix crop评估H96/192/336/720。

`scripts/evaluate_stage_c_pcsd_cf_checkpoint.py`把arms denormalize后用于error audit，同时保持policy/logits/
descriptor在model计算域中保存。probe张量统一把`B,C`展平成`row`：arms为`[row,S,T]`，policy/logits为
`[row,T,S]`，descriptor为`[row,T,S,6]`。它不修改checkpoint，并在test前验证授权与best-validation checkpoint
合同。

## Runner

`scripts/remote/run_stage_c_siff_ccsf_tau25_phase_a.sh`从formal config生成dataset-major 50-job manifest。所有arms
共享dataset profile、seed、training objective的公共部分、20 epochs/patience5和四horizon validation selector；
差异只来自预注册readout/objective/control及independent matched rank。

runner先训练到`final_evaluation_split=val`，再哈希checkpoint，调用冻结evaluator读取test，之后再次哈希并要求
不变。`RESOURCE_SMOKE=1`只运行Weather/CCSF_RELCAL三train batches，不访问test；正式矩阵才执行test evaluator。

## Four-layer analyzer

`scripts/analyze_stage_c_siff_ccsf_tau25_phase_a.py`拒绝partial matrix。它读取50个run的200个标准cells，执行
Step6原始11个comparisons（其中10个hard、1个interaction term），并只对full `ccsf_relcal`读取内部张量。

最终JSON明确分开：`paper_facing_effectiveness`、`matched_mechanism_attribution`、
`internal_mechanism_health`和`failure_attribution`。Phase A通过时仍输出`confirmation_authorized=false`，要求人工
Step9 review后才能启动seeds2022/2023。

## Code-theory consistency

Intended theory是target-free arm contrast为projective scope policy提供相对competence信息，RELCAL只在训练时用
future labels塑造该path。代码中inference policy不接收target或requested horizon；完整T=720计算后crop保持
projectivity。base policy与final policy来自同一模型同一次forward，使best-arm accuracy gain是matched internal
attribution，而不是跨checkpoint比较。

仍为proxy的部分包括：best-arm accuracy不等价于fused MSE、probe rows不是全部test rows、RMS correction activity
不证明有用。真正falsifier是完整official-test matrix上的main comparisons/controls，内部指标只能解释成功或失败。

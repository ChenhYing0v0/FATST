# CCSF Step 6 设计审计代码说明

## 1. 作用边界

`scripts/check_stage_c_siff_ccsf_step6.py`不实现模型、训练或读取实验结果。它只把
`configs/stage_c_siff_ccsf_step6.json`中的Step 6 method/control contract转成可重复检查的静态artifact，防止
Step 7实现阶段静默删减arms、改变参数匹配、引入horizon semantics或提前授权remote/test。

## 2. 输入与输出

输入config定义candidate、datasets/seeds、CCSF descriptor/scorer、objective、10 arms、matched ranks、正式矩阵、
comparisons、hard gates、claim-control map及authorization。

脚本写出：

- `matrix_audit.csv`：每行一个协议结构检查；
- `parameter_audit.csv`：每个dataset的ordered/independent CCSF参数量与relative gap；
- `claim_control_matrix.csv`：每个conditional claim需要的comparison及其是否存在；
- `step6_gate.json`：五类gate汇总与Step7A/remote/test授权状态。

## 3. 模块级计算

### `parameter_count`

从config读取`history=32`、`target_coordinate=4`、`scale_coordinate=1`、`contrast=6`、`hidden=64`及
`output=1`，计算两层MLP：

$$
(32+4+1+6)\times64+64+64\times1+1=2881.
$$

该值只计算新增correction scorer，不把已有SIFF参数重复计入。

### `audit_matrix`

检查exact 10-arm set、comparison引用、10项hard comparisons、Phase A run/cell数量、shared-temperature pilot数量、
requested horizon/bin未进入model input及remote仍为false。`phase_a_test_cells=run_count×4 horizons`，不是读取
真实test结果。

### `audit_parameters`

对每个dataset，将相同2,881参数分别加到ordered parent与已冻结rank-matched independent parent。CSV列定义：

- `ordered_ccsf_parameters`：ordered parent总参数 + correction scorer；
- `independent_ccsf_parameters`：matched independent parent总参数 + 同一scorer；
- `relative_gap`：两者绝对差除以ordered CCSF参数；
- `threshold`：冻结上限0.005；
- `pass`：`relative_gap <= threshold`。

### `audit_claims`

对`paper_performance`、`contrast_architecture`、`relative_calibration`、`architecture_objective_interaction`和
`ordered_scale_field`逐项检查所需comparisons是否存在。`hard_control_count`只表示其中多少comparison进入hard
gate，不代表实验已经通过。

### `main`

汇总matrix、parameter、claim-control、narrative status与local-only authorization。只有五项全为true时，decision
才是`step6_pass_step7a_local_only`。输出中的`implementation_authorized=true`仅指Step7A本地实现；独立字段
`remote_authorized=false`和`formal_test_authorized=false`防止扩大授权。

## 4. Code-theory consistency

预期理论是：一个paper-core候选在实现前必须有可证伪的architecture/objective/interaction归因，并保持projective、
horizon-agnostic与capacity-matched边界。当前脚本只验证**设计合同完整**，不验证CCSF forward、projectivity、gradient、
training stability或performance；这些必须由Step7A construction tests与后续正式artifacts证伪。

若config删除必要arm、使independent参数gap超过0.5%、把requested horizon/bin设为输入、或提前授权remote，静态gate
应失败。这是本脚本的直接falsifier。

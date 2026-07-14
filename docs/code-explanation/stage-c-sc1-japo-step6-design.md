# SC1-JAPO Step 6 Design Checker

## Scope

`scripts/check_stage_c_japo_step6_design.py`读取冻结的Step6 config与five-dataset profile contract，使用synthetic
float64 tensors验证architecture、initialization、controls、gradient与prefix contracts。它不是production
`JAPOReadout`，不读取dataset、checkpoint、validation或test。

## Functional Modules

### Contract loading

- `sha256`核对`configs/stage_c_five_dataset_natural_profiles.json`，防止mechanism阶段静默改变dataset profiles；
- `arm_contract`固定A6与六个JAPO arms的唯一差异；
- `screen_gate_rows`把seed2021与three-seed decision rules写成CSV。

### Expert bank

`initialize_linear`为两个experts独立生成`branch_weight: [E,K,R]`与`branch_bias: [E,K]`。
`expert_coefficients`执行：

```text
history [B,C,R]
  -> latent [B,C,E,K]
  -> atom_basis [E,T,K]
  -> coefficients [B,C,T,E]
```

`atom_basis`使用`std=sqrt(E/K)`，使near-uniform mixture的理论initial variance与单个A6-style basis一致。

### Router

`project_context`执行linear、`tanh`与non-affine RMS normalization。`router_gate`根据arm选择：

- JOINT/PERM/RANDOM：`RMS(history_context * atom_context)`；
- HISTORY：history context复制到atoms；
- ATOM：atom context复制到batch/channel；
- UNIFORM：固定`0.5/0.5`。

所有learned variants最终产生`gate: [B,C,T,E]`，只在expert axis softmax。requested $H$不是函数参数。

### Prefix and gradient audit

`audit_profile`先计算full coefficients/output，再只对`atom.start < H`的atoms重算router与synthesis，记录
`prefix_projectivity_max_abs`。JOINT full output的synthetic loss反向传播到expert branch、atom basis、coefficient
bias、history/descriptor projections与gate weights；每个tensor必须finite且nonzero。

## Artifact Definitions

### `profile_design_checks.csv`

- `readout_dim`：profile中的`state_width=P*D`；
- `a6_readout_parameters`：$KR+K+TK+T$；
- `japo_expert_bank_parameters`：$E(KR+K+TK+T)$；
- `japo_router_parameters`：history/descriptor projections与expert gate vectors之和；
- `japo_to_a6_readout_ratio`：JAPO bank+router与A6 readout params之比，只报告；
- `expert_pair_max_abs_difference`：两个branch weights的max absolute gap，检测identical copy；
- `initial_gate_entropy`：expert probability entropy除以$\log E$；
- `initial_expert_usage_min/max`：全synthetic history-atoms的mean expert probabilities；
- `uniform_control_max_abs`：UNIFORM output与expert arithmetic mean的max gap；
- `minimum_control_functional_effect`：JOINT与五个same-bank controls之mean absolute difference最小值；
- `prefix_projectivity_max_abs`：active-only output与full-output crop之max gap；
- `all_joint_gradients_finite/nonzero`：九类active parameter tensors的gradient gate；
- `uniform_variance_ratio_theory`：$E\sum_e(1/E)^2=1$。

### Other artifacts

- `arm_contract.csv`：arm、expert bank、gate、descriptor和primary attribution role；
- `screen_gate_matrix.csv`：每个stage的macro/dataset requirement与decision；
- `design_gate.json`：hash、max/min statistics、boolean gates与Step7A authorization。

## Code-Theory Consistency

- intended theory：independent full-rank experts保留A6 containment，history-atom product解除fixed separability，
  expert-only softmax保持projectivity；
- code realization：direct tensor equations、paired bank reuse、active-only recomputation与autograd audit；
- remaining proxy：synthetic control differences不代表真实forecast gains，near-uniform initialization不保证learned
  specialization；
- falsification：production实现出现prefix gap、router parameter zero gradient、expert hash相同、descriptor changes
  同时改变expert/basis rows，或remote结果被UNIFORM/HISTORY/ATOM/PERM/RANDOM解释。

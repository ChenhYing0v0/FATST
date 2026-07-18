# Stage C SIFF_EQUAL Attribution v2 Code Explanation

## 1. 功能边界

该版本先冻结Step 6归因协议，再完成Step 7A production implementation。forecast forward公式未改变；新增代码负责
把论文问题转写成可审计的10-arm matrix、导出scale-component intervention artifact，并阻止缺失control、错误
objective、错误checkpoint selector或越权remote/test launch。

## 2. 文件与数据流

### `configs/stage_c_siff_equal_attribution_v2.json`

配置按以下结构组织：

1. `candidate`：版本、test-informed 状态与当前授权边界；
2. `dataset_contract`：五个冻结 dataset profiles 及 profile hash；
3. `training_contract`：seed、from-scratch joint training、checkpoint selector 与 standard horizons；
4. `arms`：10 个 architecture/objective/control contracts；
5. `comparisons`：diagnostic、claim-narrowing 与 hard-gate comparisons；
6. `evaluation_layers`：effectiveness、attribution、internal health 与 failure attribution；
7. `phase_a` / `confirmation`：run matrix、test cell 数与 conditional launch rule；
8. `authorization`：Step 7A、remote、test 与 confirmation 的独立布尔开关。

所有 arms 先生成完整 `T=720` forecast，任意 standard horizon 只做 prefix crop。checkpoint 由 validation
`H96/H192/H336/H720` MSE 平均选择，test labels 不进入 epoch selection。

### `scripts/check_stage_c_siff_equal_attribution_step6.py`

checker 的执行顺序为：

1. 读取归因 config 与冻结 dataset profile contract；
2. 检查 candidate identity、`test_informed` 与 profile hash；
3. 检查 10 个 arms 的 decoder、objective 与 SIFF mode；
4. 禁止 PCC/MCCA objective 混入当前 EQUAL-context matrix；
5. 重新计算 Phase A 和 confirmation 的 run/cell 数；
6. 检查七项 hard comparisons、四层 evaluation protocol 与 diagnostic boundary；
7. 检查 checkpoint selector、from-scratch joint training 与授权状态；
8. 写出 `step6_gate.json`。

checker 只验证“冻结协议是否完整且自洽”，不声称 model performance 已通过。

### `analysis/.../step6_gate.json`

主要字段：

- `checks_total` / `checks_passed`：Step 6 静态 gate 数；
- `passed`：是否允许进入 Step 7A；
- `profile_contract_hash`：实际读取的 dataset profile hash；
- `phase_a_runs` / `phase_a_test_cells`：首轮正式矩阵规模；
- `confirmation_runs` / `confirmation_test_cells`：条件确认规模；
- `next_step`：通过后唯一允许的研究步骤；
- `remote_authorized` / `test_authorized`：防止把 Step 6 pass 误写成远程实验授权。

## 3. Code-theory consistency

### Intended theory

在同一 EQUAL training context 下，通过 matched controls 分离 harmonic measure、equal-skill supervision、
ordered scale coordinate、partition structure 与 cross-arm interaction，判断 `SIFF_EQUAL` 的 test gain 是否具有
可归因的 architecture mechanism。

### Code realization

config 将每个理论问题映射成明确的 arm pair；checker 强制所有 hard comparisons、完整五数据集矩阵、统一
checkpoint rule 与四层 decision boundary 同时存在。

### Remaining proxy

Step 7A只证明所有路径可构造、可前向/反向并能生成所需artifact。trained ordered component是否足够大、policy是否
使用scale field、各arms是否保持差异，仍需Step 9 internal diagnostics验证。

### Falsification evidence

以下任一结果可证伪或收紧当前设计：

1. `SIFF_EQUAL` 未超过 A6/PCSD effectiveness controls；
2. 未超过任一 EQUAL-context specificity control；
3. arms collapse、policy degeneracy 或 ordered component 近零；
4. 结果依赖单一 dataset/horizon/cell；
5. confirmation seeds 无法复现 Phase A。

## 4. Step 7A 实现要求

Step 7A 必须补齐：

- 10-arm CLI/constructor coverage；
- paired initialization 与 matched parameter accounting；
- full-forecast / prefix-crop projectivity；
- trainable path 的 forward/backward finite 检查；
- scale-component artifact schema；
- 四层 analyzer dry-run；
- remote/test authorization 保持 false。

如果 production code 无法满足任一冻结 contract，应回 Step 6 重新评估设计，而不是在 runner 中静默降级。

## 5. Step 7A forward与artifact路径

### SIFF component path

`SIFFCouplingFieldReadout`中的实际shape流为：

1. Encoder output flatten：`hidden [B,C,R]`；
2. `component_history_modes`：
   `hidden [B,C,R] × mode_weight [Q,D,R,K] -> components [B,C,Q,D,K]`；
3. `scale_basis [S,Q] × components -> history_modes [B,C,S,D,K]`；
4. 每个scope执行pooling/synthesis，得到`arms [B,C,S,T]`；
5. `policy_weights [B,C,T,S]`融合为`full [B,C,T]`；
6. domain-only crop后返回`forecast [B,H,C]`。

新增`component_ablation_forecasts`固定第5步policy，逐个把
`components[:,:,q]`置零并重新走第3-5步，输出：

- `full [B,C,T]`；
- `ablated [B,C,Q,T]`。

checkpoint evaluator将full和ablated分别denormalize，再保存
`scale_component_contribution [row_channel,Q,T] = full - ablated`。该统计是non-additive intervention：
SIFF含nonlinear synthesis，因此不能把各component delta直接相加解释为forecast decomposition。生产评估只为
`siff-coupling-field + equal_skill` candidate导出该artifact，避免在不需要该统计的controls上重复高成本反事实前向。

### Analyzer path

`analyze_stage_c_siff_equal_attribution_v2.py`读取：

- `test_audit_metrics_by_target_horizon.csv`：四个standard horizons的MSE/MAE；
- `pcsd_test_audit_diagnostics.npz`：fused、arms、policy、probe和component tensors；
- `test_audit_invariants.json`：projectivity、protocol、checkpoint hash与test authorization。

它依次计算effectiveness、matched attribution、internal health与failure attribution；只有前三层全部通过时才把
`confirmation_authorized`设为true。

## 6. Step 7A gate

`check_stage_c_siff_equal_attribution_step7a.py`覆盖50 CLI jobs、35个unique constructors、10个objective gradient
paths、两类matched-rank controls、5个component cases、evaluator/analyzer smoke和remote authorization guard。

结果为13/13 categories pass；remote与formal test仍为false。详细证据见
`analysis/stage_c_siff_equal_attribution_step7a_20260718/step7a_implementation_gate_report.md`。

## 7. Step 7B prelaunch control

`check_stage_c_siff_equal_attribution_step7b.py`不修改模型。它读取committed Step7A gate与job manifest，验证
candidate identity、profile hash、50-run/200-cell matrix、four-horizon checkpoint、正式test授权、
`test_audit_authorized`返回值、confirmation hold以及remote runner executable/syntax。

remote runner在dry-run时只执行Step7B checker与evaluator/analyzer synthetic smoke，所有临时gate写入repo-external
output root。正式路径仅在`remote_training_authorized`和`formal_test_access_authorized`同时为true时开放；
resource smoke只要求remote authorization，且使用`final_evaluation_split=none`，不会读取test。

Step7B machine gate为9/9。该结果只授权seed2021 Phase A，不授权seeds2022/2023。

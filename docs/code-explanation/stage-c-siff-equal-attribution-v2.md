# Stage C SIFF_EQUAL Attribution v2 Code Explanation

## 1. 功能边界

本次更新冻结归因协议与 Step 6 checker，没有修改 model forward。代码产物负责把论文问题转写成可审计的
10-arm matrix，并阻止缺失 control、错误 objective、错误 checkpoint selector 或越权 remote/test launch。

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

Step 6 只证明设计可执行。ordered component 是否真正改变 forecast、policy 是否使用 scale field、各 arms 是否
保持差异，仍需 Step 7A artifact contract 与 Step 9 internal diagnostics 验证。

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

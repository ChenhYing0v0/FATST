# SC1-D4 Structured-Basis Mechanism Diagnostic Protocol

## Status

| Field | Value |
| --- | --- |
| `candidate` | `SC1-D4` |
| `role` | `diagnostic_only` |
| `current_step` | Step 4 |
| `question` | balanced interval优势是否超越standard structured bases，来自interval locality还是exact balancing？ |
| `suite` | 5 datasets × 3 checkpoints × 3 grouping seeds × 7 bases = 315 fits |
| `horizons` | 48, 96, 144, 192, 288, 336, 512, 720 |
| `test_used` | false |
| `forecast_model_updated` | false |
| `method_training_authorized` | false |

## 1. Tensor And Split Contract

完全复用D2/D3 frozen A6 memory：

$$
M\in\mathbb R^{B\times C\times P\times D},\quad
h\in\mathbb R^{BC\times PD},\quad
u\in\mathbb R^{BC\times720}.
$$

train前16 batches按sample ID做80/20 fit/inner-holdout；official validation前8 batches只作final evaluation；
test不加载。所有basis $Q\in\mathbb R^{720\times720}$均满足$QQ^T=I$，head预测
$\widehat\alpha$并以$\widehat u=\widehat\alpha Q$生成future。

所有arms固定random coefficient groups，sizes为
`[1,1,2,4,8,16,32,64,128,256,208]`。每组使用`PD -> GELU(32) -> n_l`，从而只改变basis geometry，
不重新测试D2已关闭的true depth grouping。

## 2. Basis Controls

| Family | Construction | Attribution |
| --- | --- | --- |
| `balanced_interval` | recursive midpoint Haar-like orthonormal basis | candidate |
| `identity` | time-point basis | basis transform是否必要 |
| `dct2` | orthonormal DCT-II | global smooth/frequency control |
| `pca_fit` | fit targets covariance eigenvectors；validation不可见 | data-adaptive whitening/energy control |
| `permuted_interval` | permute candidate time columns | preserve atom values/sparsity，destroy contiguity |
| `random_interval_tree` | random 25%-75% recursive interval splits | preserve local Haar family，remove exact balancing |
| `random_orthogonal` | QR Gaussian basis | replicate D3 anchor |

## 3. Metrics

每个arm报告八个prefix horizons的evaluation-space MSE/MAE。训练仍只使用full H720；prefix metrics仅裁剪同一
full forecast，故本轮测试shared full-domain readout的dense-horizon behavior，不声称variable-H training。

对fit targets的coefficient covariance $\Sigma_Q=Q\Sigma_uQ^T$记录：

$$
\rho_{off}=\frac{\|\Sigma_Q-\operatorname{diag}(\Sigma_Q)\|_F}{\|\Sigma_Q\|_F},
$$

以及top-16/64/144/256 sorted coefficient variance capture。对每个H，`active_atoms_H`统计basis rows中在
prefix $[1,H]$上有非零support的数量；global bases通常为720，interval bases可native partial synthesis。

primary performance estimand为八horizons平均log MSE contrast：

$$
\Delta(c)=\frac18\sum_H\log\frac{E_{c,H}}{E_{balanced,H}}.
$$

三个grouping seeds先在dataset/checkpoint内平均，再形成15个primary units。

## 4. Preregistered Gates

1. 315/315完整；test/freeze/validation/orthogonality invariants通过；
2. D3 replication：H720 balanced vs random-orthogonal reduction至少`0.5%`，至少3/5 datasets有2/3
   checkpoints为正；
3. standard global noninferiority：balanced相对identity、DCT-II、PCA-fit的八horizon macro reduction分别
   不低于`-0.25%`，各至少3/5 datasets不低于`-1%`，且至少6/8单horizon macro不低于`-0.25%`；
4. locality gate：balanced vs permuted-interval reduction至少`0.5%`，至少3/5 datasets有2/3 checkpoints为正；
5. balance-specificity gate：balanced vs random-interval-tree使用同一`0.5% + 3/5`标准；
6. locality与specificity comparisons的MAE不得低于`-0.25%`。

Decision：

| Condition | Decision |
| --- | --- |
| incomplete/pathology | `diagnostic_invalid_for_direction_rejection` |
| random replication fail | `d3_signal_not_replicated_reaudit_step2` |
| global noninferiority fail | `standard_structured_basis_explains_gain_return_step2` |
| locality fail | `locality_not_supported_coordinate_effect_only_step2` |
| locality pass、specificity fail | `interval_local_family_supported_refine_step4` |
| all pass | `balanced_interval_specificity_supported_step5` |

## 5. Failure Boundary

D4仍是frozen-memory probe，不是end-to-end decoder。PCA使用fit targets是预注册strong control，不是candidate
可用validation oracle。若balanced只在H720获益而短中horizons退化，不得用H720平均掩盖multi-horizon失败。
若random interval tree匹配balanced，只能收紧exact-balancing claim，不能否定interval-local generation。

任何pass只授权Step 5 proof/design，不授权remote method training、Encoder、MIPR或MoE。

# SC1-D11 Future-Component Responsibility：结果、归因与回滚

## Decision Summary

| Field | Decision |
| --- | --- |
| `current_step` | Contribution 1 Step 2/3 diagnostic closure |
| `formal_decision` | `transform_generic_pressure_sc2_only` |
| `strict_directional_conflict` | fail，`0/5 datasets` |
| `support_specific_component_conflict` | fail，`2/5 datasets`，未达到预注册`3/5` |
| `generic_component_pressure` | pass，`3/5 datasets` |
| `magnitude_imbalance` | fail，`2/5 datasets` |
| `method_authorization` | false；不得实现conflict-aware decoder、PCGrad、component loss或新router |
| `SC2_authorization` | false；只允许Step 1-3 source/problem/novelty re-audit |
| `rollback` | Contribution 1回Step 2并暂停；下一研究问题转向projective supervision coverage，而不是gradient conflict |

## 1. D11实际检验了什么

D11不是训练实验，也没有测试某个新模型。它读取五个dataset各三个A6 natural checkpoints，在固定train与
validation batches上比较short prefix measure和long prefix measure对同一shared forecast path提出的局部更新
要求。核心问题是：

> 一个shared unified forecasting model是否因为同时服务short与long prefixes，而被迫沿相反方向更新同一组
> 参数或同一future component？

若两个gradient的inner product为负，一个方向的一阶下降会使另一个目标上升，才称为directional conflict。
低但为正的cosine只表示两个目标关注点不同，不是冲突。

## 2. Protocol And Artifact Completeness

- datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- checkpoints：seeds 2021/2022/2023，共15个；
- splits：train、official validation各4 batches；test未读取；
- measures：short=`{48,96,144}`，long=`{336,512,720}`；
- losses：MSE primary，L1 replication；
- future coordinates：RGNB、DCT、三个fixed random orthogonal controls；
- total paths：`coeff_tensor`、`coeff_params`、`encoder_params`、`basis_params`、`all_params`；
- accepted rows：total gradient `1200`、component `1200`、component-group `8400`、reachability `120`；
- full-forward reconstruction max=`0`；gradient additivity relative max=`4.7041e-7`；orthogonality max=
  `2.6132e-14`；
- no forecast training、no forecast update、no test，全部invariants通过。

第一次remote run因为zero responsibility vector的cosine未定义而被判为invalid。修复没有改gate：zero vector不算
conflict，cosine只在双方非零的active pairs上计算，并新增zero-group count。accepted v2在commit `6c90b7b`上
重新生成了全部rows；本地analyzer复算与remote gate完全一致。

## 3. Primary Result：不存在strict short/long directional conflict

五个dataset、三个seeds、四个validation batches、五条total paths的MSE short/long gradient全部为正inner
product，即validation negative batch fraction在所有dataset/path上均为`0`。各dataset在任意path/batch上的最小
cosine仍为：

| Dataset | Minimum validation MSE cosine | Directional gate |
| --- | ---: | --- |
| ETTh1 | 0.203233 | fail |
| ETTh2 | 0.427311 | fail |
| ETTm1 | 0.299458 | fail |
| ETTm2 | 0.053682 | fail |
| Weather | 0.100635 | fail |

因此这里不是“冲突弱一点”，而是在本次checkpoint-local first-order evidence中，连strict negative sign都没有
出现。L1与train replication也没有挽救该假设。

## 4. Component Result：有responsibility redistribution，但不是统一的跨regime冲突

| Dataset | RGNB JS | RGNB component negative | RGNB max cancellation | DCT JS | Random median JS | Formal component gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ETTh1 | 0.072415 | 0.194444 | 0.432825 | 0.011074 | 0.001072 | fail，0/3 seeds complete |
| ETTh2 | 0.065007 | 0.246032 | 0.376942 | 0.016608 | 0.001278 | pass，3/3 |
| ETTm1 | 0.039251 | 0.238095 | 0.407898 | 0.003895 | 0.000835 | fail，0/3 |
| ETTm2 | 0.070848 | 0.214286 | 0.448666 | 0.014251 | 0.000801 | pass，2/3 |
| Weather | 0.030840 | 0.416667 | 0.306820 | 0.004007 | 0.001073 | fail，0/3 |

这里必须作两层收紧：

1. formal support-specific gate只有ETTh2、ETTm2通过，即`2/5`，低于预注册`3/5`；不能作为unified
   architecture problem；
2. 全部RGNB cases的same-component short-vs-long negative fraction都为`0`。表中的component negative来自
   **同一regime内部不同groups之间**的negative pairs，不是short与long要求同一component反向更新。

RGNB的JS明显高于DCT/random，主要来自projective support的nested coverage。short measure对group 5/6严格零
梯度，而long measure会更新它们。跨全部dataset/seeds/validation MSE平均的responsibility shares为：

| RGNB group | Group size | Short share | Long share |
| ---: | ---: | ---: | ---: |
| 0 | 16 | 0.551868 | 0.628484 |
| 1 | 16 | 0.107863 | 0.089623 |
| 2 | 32 | 0.166823 | 0.086978 |
| 3 | 64 | 0.136211 | 0.039724 |
| 4 | 128 | 0.037235 | 0.070642 |
| 5 | 256 | 0 | 0.064107 |
| 6 | 208 | 0 | 0.020441 |

这说明different prefix measures确实改变哪些projective components能获得监督，但证据指向的是
`coverage / update opportunity asymmetry`，不是`conflicting update direction`。此外，DCT cancellation在每个
dataset/seed上都高于RGNB；因此不能用within-regime cancellation把RGNB包装成特殊冲突结构。

## 5. Learned A6 Basis Reachability

| Dataset | Residual energy in learned span | Orthogonal complement |
| --- | ---: | ---: |
| ETTh1 | 0.730634 | 0.269366 |
| ETTh2 | 0.819042 | 0.180958 |
| ETTm1 | 0.849615 | 0.150385 |
| ETTm2 | 0.826202 | 0.173798 |
| Weather | 0.822036 | 0.177964 |

该结果只说明A6 learned temporal span覆盖大部分、但非全部validation residual energy。它不能单独证明basis
capacity是性能瓶颈，也不能把span complement直接升级为new decoder依据。

## 6. Failure Attribution

- [Fact] `optimization_or_numeric_pathology=false`：没有训练，所有reconstruction/additivity/orthogonality
  invariants通过；
- [Fact] `hyperparameter_failure=false`：D11没有新模型超参数，也没有用训练性能作gate；
- [Strong Evidence] `hypothesis_false`：作为跨dataset统一问题，strict short/long directional conflict为`0/5`；
- [Strong Evidence] support-specific conflict不足：formal gate仅`2/5`，且same-component cross-regime conflict为0；
- [Boundary] 这是problem hypothesis的失败，不是某个architecture implementation的失败，因为D11没有实现或训练
  architecture；
- [Uncertainty] D11是checkpoint-local first-order diagnostic，不排除训练早期某些step出现暂时冲突；但若一个
  paper-core机制依赖这种冲突，它仍缺少稳定、跨dataset的存在性证据。

## 7. Paper And Research Decision

Contribution 1不能转向“conflict-aware future-component decoder”。D6的local/long support crossing、RGNB
projectivity和balanced-interval geometry仍作为已确认scaffold保留，但它们目前没有新的active method carrier。

formal decision `transform_generic_pressure_sc2_only`也不是SC2通过。它只表示剩余问题更靠近training distribution：
当prefix lengths按某个measure采样时，nested projective groups获得的有效监督次数不同。外部primary-source边界已
显示：

- [Time-o1](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0cd62dea69635f4c5b569848267fe5a8-Abstract-Conference.html)
  已覆盖transformed-label alignment与forecast-step task overload；
- [Loss Shaping Constraints](https://arxiv.org/abs/2402.09373)已对forecast steps分别施加constraints并动态形成
  dual weights；
- [Do Current Multi-Task Optimization Methods in Deep Learning Even Help?](https://proceedings.neurips.cc/paper_files/paper/2022/file/580c4ec4738ff61d5862a122cdf139b6-Paper-Conference.pdf)
  表明evaluation weighting、sampling scheme与大量hyperparameter tuning会显著改变MTO结论；
- [Temporal horizons in forecasting](https://openreview.net/forum?id=BeudQIxT1R)给出long-horizon training与
  short-horizon generalization的不对称性及long-horizon loss-landscape roughness证据。

因此下一步只能回到Contribution 2 Step 1-3，审计`projective supervision coverage`是否能形成一个区别于
generic step weighting、importance sampling、GradNorm与transformed component loss的完整问题链。未经该audit，
`SC2-MIPR`仍held，不实现coverage normalization、new loss、PCGrad或joint factorial。

## 8. Self-Critique

- RGNB的两个zero groups部分来自其compact support定义，JS差异可能是geometry的预期结果，而不是数据驱动发现；
- fixed short/long measures只代表当前诊断，不覆盖所有continuous deployment measures；
- component responsibility在`coeff [B,C,256]`上精确可加，但parameter-space非线性训练轨迹没有被模拟；
- 下一步若只把activation probability倒数乘到loss上，极可能只是已有importance weighting的task-specific实例，
  不能因应用到multi-horizon forecasting就自动成为SCI-level contribution。

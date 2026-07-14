# SC1-D5 Conditioning-Locality Frontier: Research Interpretation

## 1. Preregistered Decision

原始analyzer decision为`local_family_headroom_not_supported_basis_component_only`，但按Diagnostic Failure
Attribution Rule，**该decision只能关闭预注册的`active_atoms_h48 <= 96 + covariance selector`设计，不能用于
方向级否决**。方向级状态标记为：

`diagnostic_invalid_for_direction_rejection / design_fault_suspected`。

原因是一个预注册但被selector排除的arm——`block_dct2_b144`——显示了稳定且具有multi-horizon含义的
crossed interaction。下一步不是实现method，而是在未使用的validation batches上做D6 confirmation。

## 2. Validity

- 585/585 fits、15/15 metadata与15/15 fit-only selections完整；
- test未使用，A6 forecast model未修改；
- PCA与block PCA只使用fit targets；
- all metrics finite，orthogonality invariants通过；
- primary selector在15/15 units选择`block_pca_fit_b96`，没有使用validation。

## 3. Primary Gate Failure

fit-only selected b96 PCA相对balanced仅`+0.0322%` MSE，只有2/5 datasets达到2/3 checkpoints为正；
相对global DCT/PCA分别为`-0.8284%/-1.4723%`，只关闭DCT gap的`3.76%`。因此预注册gate确实失败，
不能事后改阈值宣称D5 pass。

[Failure attribution] selector最小化covariance off-diagonal ratio，但D4/D5已经显示conditioning不只由该单一
统计决定；`<=96` cap又排除了仍比global basis稀疏5倍的b144。失败更接近diagnostic selection/design fault，
不是optimization pathology或capacity confound。

## 4. Pre-Registered Arm That Changes The Direction-Level Judgment

`block_dct2_b144`的八horizon macro：

| Control | MSE reduction | MAE reduction | Positive datasets |
| --- | ---: | ---: | ---: |
| balanced interval | +0.7111% | +0.2326% | 4/5 |
| global DCT-II | -0.1438% | -0.2688% | 1/5 |
| global fit PCA | -0.7832% | -0.6146% | 2/5 |

它在H48只激活144 atoms，而global DCT/PCA为720。更关键的是，相对global DCT：

- short horizons `{48,96,144}`平均约`+1.05%`；
- long horizons `{336,512,720}`平均约`-1.15%`；
- 13/15 primary units short为正，13/15 long为负，11/15同时发生crossing；
- macro从H48 `+1.0766%`、H96 `+1.3243%`、H144 `+0.7415%`，转为H336 `-1.0364%`、
  H512 `-1.2539%`、H720 `-1.1619%`。

[Strong Evidence] 这不是“某个local basis全局更好”，而是**future support scale与evaluation horizon存在
crossed interaction**：local block减少短prefix对global atoms的依赖，global smooth coordinates在长domain更好。
这比“balanced midpoint basis更准”更贴合multi-horizon unified forecasting的问题叙事。

## 5. Innovation Boundary

balanced interval basis仍可作为forecast-generation component；D5没有否定其组件级创新。但新的paper-core
问题不能是固定balanced basis，而应是：

> 一个horizon-agnostic operator如何在同一future function中同时保留local prefix synthesis与global
> long-domain coherence，使输出域限制自然获得合适support scale，而不把horizon ID作为semantic input？

本轮尚未提出operator。local/global dictionary、wavelet packet、lifting或hybrid basis均需要D6确认后重新走
Step 4 source-informed narrative audit。

## 6. Next Step

`SC1-D6`只复验`block_dct2_b144`相对global DCT/balanced的short-positive、long-negative interaction，使用
official validation batches 8-15；D5使用的是0-7。D6若pass，只授权返回Step 4；若fail，D5 crossing降为
exploratory split-specific evidence。

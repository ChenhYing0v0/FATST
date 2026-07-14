# SC1-D5 Conditioning-Locality Frontier Diagnostic Protocol

## Status

| Field | Value |
| --- | --- |
| `candidate` | `SC1-D5` |
| `role` | `diagnostic_only` |
| `current_step` | Step 2/3 complete；direction-level rejection invalid，D6 confirmation next |
| `problem` | local-support orthogonal family能否缩小DCT/PCA conditioning gap，同时保留small prefix active-set？ |
| `suite` | 5 datasets × 3 checkpoints × 3 grouping seeds × 13 bases = 585 fits |
| `test_used` | false |
| `forecast_model_updated` | false |
| `method_training_authorized` | false；primary gate fail，b144 crossing requires D6 |

## 1. Why This Diagnostic Matters

D4已经支持contiguous interval locality，却否定exact midpoint balancing的特异accuracy，并显示DCT/PCA整体
更准。D5不把block DCT或block PCA当作论文方法，而是检查两种性质是否存在可解的Pareto gap：

1. global DCT/PCA提供较好的coefficient decorrelation与energy compaction；
2. interval basis只需为短prefix生成少量相交atoms；
3. 若local structured family能同时接近两者，才值得设计新的horizon-agnostic generation operator。

## 2. Tensor, Split And Head Contract

完全复用D4的frozen A6 memory、fit/inner-holdout/official-validation分离、random coefficient groups与
`GroupedNonlinearHead`。每个head仍预测720个orthogonal coefficients，训练只使用H720；八个horizons由同一
forecast裁剪评估。PCA及所有block-PCA仅使用fit targets covariance，official validation不参与basis构造、
early stopping或family selection。

block basis为严格block diagonal orthogonal transform。对block $I_b$：

$$
Q=\operatorname{diag}(Q_{I_1},\ldots,Q_{I_m}),
$$

其中$Q_{I_b}$为local DCT-II或该block fit covariance的eigenbasis。block sizes预注册为
`16, 32, 48, 96, 144`；最后一block允许更短。controls为balanced interval、global DCT-II与global fit-only PCA。

## 3. Fit-Only Selection And Statistics

为了避免从validation挑选最优block size，每个dataset/checkpoint先在fit covariance上做确定性选择：

1. 只考虑`active_atoms_h48 <= 96`的local families；
2. 最小化coefficient covariance off-diagonal ratio；
3. ties依次使用更高top-16 variance capture与family name。

三个grouping seeds只估计相同basis下head optimization的平均表现，并在log-error space取平均。primary estimand仍为八horizon平均
$\log(E_{control}/E_{selected})$。输出的每个CSV定义如下：

- `d5_checkpoint_family_summary.csv`：每个dataset/checkpoint/family在三个group seeds上的absolute MSE/MAE；
- `d5_checkpoint_geometry_summary.csv`：fit covariance geometry与各H active atoms；
- `d5_selected_families.csv`：只由fit geometry得到的选择；
- `d5_selected_comparisons.csv`：selected local对balanced/DCT/PCA的逐checkpoint-horizon log effect；
- `d5_macro_comparisons.csv`：上述effects在15个primary units上的macro；
- `d5_local_family_comparisons.csv`：所有预注册local families对三controls的逐unit-horizon描述性effects；
- `d5_local_family_summary.csv`：上述family effects的macro与cross-dataset方向，仅用于检查selector是否掩盖反例；
- `d5_local_family_dataset_summary.csv` / `d5_local_family_horizon_summary.csv`：全部local families的
  dataset与horizon一致性审计；
- `d5_summary.json`：gate与decision。

## 4. Preregistered Gate

必须先满足585/585、15 metadata、no-test、no-forecast-update、fit-only PCA、orthogonality与finite checks。

`local headroom gate`：

- selected local相对balanced的八horizon macro MSE reduction至少`0.5%`；
- MAE不低于`-0.25%`；
- 至少3/5 datasets有2/3 checkpoints为正。

`global conditioning gate`：

- selected local相对DCT不低于`-0.5%`，相对PCA不低于`-1.0%`；
- 至少6/8 horizons相对DCT不低于`-1.0%`；
- 至少关闭balanced-to-DCT log-error gap的50%。

| Result | Decision |
| --- | --- |
| invalid/pathology | `diagnostic_invalid_for_direction_rejection` |
| local与global gates均pass | `conditioning_locality_gap_supported_return_step4` |
| 只local headroom pass | `local_conditioning_headroom_partial_repeat_step2` |
| local headroom fail | `local_family_headroom_not_supported_basis_component_only` |

任何pass只授权返回Step 4进行external source-informed method audit；不授权新decoder training、Encoder、MIPR
或MoE。

## 5. Failure Attribution Boundary

本轮若失败，只能说明预注册block-local orthogonal family没有显示足够headroom。它不能否定所有learned
lifting、wavelet packet或overcomplete local dictionary；但若连fit-only local PCA upper control都不能改善
balanced，则继续设计复杂local operator的优先级应显著下降，balanced basis只保留为generation component。

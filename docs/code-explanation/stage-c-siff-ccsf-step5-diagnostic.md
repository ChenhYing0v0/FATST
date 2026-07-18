# StageC SIFF CCSF Step5 diagnostic 代码说明

## 模块职责

`scripts/analyze_stage_c_siff_ccsf_theory.py`复用SIFF_EQUAL Step9保存的probe artifacts，检查不含target的arm
contrast是否能cross-fit预测relative arm competence，并比较旧PCC standardized teacher与relative-regret teacher的
几何。脚本不加载checkpoint、不训练forecast model、不生成可部署router。

## 输入与shape

每个dataset读取：

- `probe_arms [256,5,720]`；
- `probe_targets [256,720]`；
- `arm_row_bin_mse [N,8,5]`；
- `policy_row_bin_usage [N,8,5]`。

脚本先由probe重新计算前256 rows的八bin arm MSE并与保存数组做`allclose`，保证row/bin对齐。target只用于构造
offline best-arm label与held-out expected arm MSE；feature construction完全不读取target。

## Feature path

`feature_sets()`先计算五arm pointwise consensus，再得到centered contrast`[256,5,720]`。每个future bin对每个arm
计算mean、std、RMS和end-minus-start slope，形成`[256,8,20]` contrast summary，并拼接三个fixed coordinates。

该bin summary只服务Step5 existence diagnostic，不是production descriptor。理论报告已冻结production候选应改用
scope-native groups，避免benchmark-horizon feature。

## Cross-fit path

256 rows固定分为两半。`StandardScaler + multinomial LogisticRegression(C=1)`在一半的`128×8` cells拟合best-arm
label，另一半评估，然后反向。所有feature arms共用相同row folds。shuffled control只打乱contrast部分的row对应，
保留coordinate distribution。

输出：

- `contrast_predictability_by_fold.csv`：逐dataset/fold/feature的accuracy和allocation；
- `contrast_comparisons.csv`：primary相对coordinate、shuffled、existing policy的paired gain；
- `teacher_geometry.csv`：PCC standardization与relative-regret temperature grid的entropy/confidence statistics；
- `summary.json`：5项冻结gate与macro结果。

## Teacher geometry

`pcc_std`使用cross-arm standard deviation缩放centered MSE；`relative_regret`使用mean arm MSE缩放。二者都只在
offline array上计算softmax teacher。`confidence_dispersion_correlation`是每个dataset内`1-normalized entropy`与
`std(loss)/mean(loss)`的Pearson correlation，再在summary中做dataset macro平均。

## Code-theory consistency

理论假设是“arm prediction disagreement携带history-only policy缺少的relative competence信息”。脚本通过
target-free feature与shuffled/coordinate controls检验可辨识性，并未检验end-to-end optimization或fused forecast。

可证伪结果是contrast不超过coordinate/shuffled，或只在少数fold为正。本次5/5 gate通过，允许进入Step6；但probe
来自test-derived artifacts且没有full Encoder hidden，因此不能把classifier收益写成model performance，也不能用
temperature grid选择candidate超参数。

# Stage C SIFF-v3 TSAF Step7A Code Explanation

## 1. 模块目标

`SIFFCouplingFieldReadout`原本由`hidden [B,C,R]`与future coordinate共同产生scope policy。TSAF保留SIFF arm
generator，但把policy改为只由future coordinate和ordered log scale产生。该改动删除unsupported sample-wise
competence freedom，不删除forecast对history的依赖。

## 2. Forward flow

### SIFF arms

1. `hidden [B,C,R]`经`mode_weight [Q,D,R,K]`得到`components [B,C,Q,D,K]`；
2. `scale_basis [S,Q]`得到`modes [B,C,S,D,K]`；
3. 五个scope synthesis得到`arms [B,C,S,T]`。

### TSAF allocation

1. `coordinate_field [T,D] -> target_state [T,P]`；
2. `allocation_scale_features [S,2] -> scale_state [S,P]`；
3. broadcast sum与shared bias形成`joint_state [T,S,P]`；
4. GELU与shared scalar output形成`logits [T,S]`；
5. softmax后broadcast为`weights [B,C,T,S]`；
6. 转置arm axis后加权求和，得到`full [B,C,T]`，最后crop为`[B,H,C]`。

`target-scale-global`把`target_state`置零；`target-scale-field-permuted`只翻转scale feature与真实scope的对应。

## 3. 参数路径

TSAF mode删除direct policy的`history_projection`、`policy_hidden`和`policy_output`，新增：

- `target_allocation_projection: D -> P`；
- `scale_allocation_projection: 2 -> P`；
- `target_scale_allocation_bias [P]`；
- `target_scale_allocation_output: P -> 1`。

`tsaf_parameter_count`显式扣除direct policy并加入上述参数。scalar output zero-init使初始weights严格uniform。

## 4. Training adapter

CLI的`--pcsd-policy-mode`新增三个TSAF values。scope-credit training允许`direct`、`static-target`与TSAF modes；
`equal`和`fixed`仍不能承担learned policy training。`TimeAlign.Model`拒绝在非SIFF readout上使用TSAF mode，PCSD
base在standalone调用时也显式报错。

## 5. Local checker与artifact

`scripts/check_stage_c_siff_v3_tsaf_step7a.py`输出：

- `cases.csv`：26项shape、invariance、semantic、gradient、parameter和production cases；
- `summary.json`：case总数、通过数及remote/test=false边界。

checker使用synthetic tensors，不读取dataset或checkpoint。production constructor case通过真实`TimeAlign.Model`
验证`[1,720,2]`输出与H96 exact crop。

## 6. Code-theory consistency

- Intended theory：future coordinate决定共享scale allocation，history决定每个scope预测内容；
- Realized code：TSAF logits完全不读取`hidden`，SIFF arms仍读取`hidden`；
- Proxy：二次log-scale features和shared MLP只是平滑scale geometry的有限参数化；
- Falsification：TSAF不能超过categorical target-only、permuted/global controls与direct parent时，不能claim
  target-scale field有效；
- 未实现内容：runner matrix、remote smoke、training result与official test。

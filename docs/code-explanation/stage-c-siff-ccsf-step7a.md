# CCSF Step 7A 代码说明

## 1. 模型forward流程

### `layers/CCSF.py::CCSFCouplingFieldReadout`

该类继承`SIFFCouplingFieldReadout`，不修改Encoder或SIFF arm generator。输入`hidden [B,C,R]`后：

1. `arm_forecasts`生成`arms [B,C,S,T]`；
2. `_true_contrast_descriptor`沿scope轴计算consensus、normalized contrast与relative disagreement；
3. 对每个scope，用已有`group_indices`将normalized contrast重排为`[B,C,G,group_size]`，计算group mean/RMS/
   endpoint difference，再scatter回target coordinate；
4. `contrast_descriptor`根据readout mode返回true、all-zero或scope-axis cyclic permutation；
5. `correction_logits`复用parent的`history_projection [B,C,32]`，拼接target coordinate、scale coordinate与
   descriptor，按target chunks通过共享`43 -> 64 -> 1` scorer；
6. `policy_tensors`计算`softmax(base_logits + correction_logits)`；
7. `forward_with_ccsf_diagnostics`以policy凸融合arms，得到full forecast后才执行prefix crop。

scorer final layer为zero initialization，所以模型初始函数与相同seed的SIFF parent一致。zero-control保留全部scorer
参数，只把六维contrast输入置零；permuted-control使用固定`[1,2,3,4,0]`scope映射。

### `models/TimeAlign.py`

`CCSF_READOUT_CONFIG`把四个readout name映射到`scale_components`、`scale_basis_mode`与`correction_mode`。
所有CCSF modes进入既有`COUPLING_READOUTS`路径。训练请求details时，模型调用
`forward_with_ccsf_diagnostics`，对arms执行与既有PCSD/SIFF相同的raw-scale denormalization，并附加dimensionless
base policy、contrast与correction tensors。

## 2. Objective流程

### `contrast_scope_calibration_loss`

输入：

- `fused_forecast [B,T,C]`；
- `arm_forecasts [B,C,T,S]`；
- `policy [B,C,T,S]`；
- `target [B,T,C]`。

函数使用`prefix_measure`构造dense-prefix harmonic target weights。`fused_loss`是weighted fused L1，`skill_loss`
是uniform scope mean后的weighted arm L1。

`ccsf_relative_calibration`先用每个target的mean arm error归一化relative regret，经stop-gradient与temperature生成
teacher；`1-normalized entropy`作为confidence。`ccsf_standardized_calibration`保留旧cross-arm standardization作为
teacher geometry control。两者计算`KL(teacher || policy)`，总目标为：

$$
L=L_{fused}+L_{equal\text{-}skill}+0.1L_{cal}.
$$

返回的`CCSFObjectiveResult`包含loss decomposition、detached teacher/confidence、measure与用于training log的
diagnostics。

## 3. Training adapter

`train_repo.py`增加：

- 四个CCSF active readout names；
- 两个CCSF objective modes；
- `--ccsf-correction-hidden-dim`；
- `--ccsf-calibration-temperature`；
- `--ccsf-calibration-weight`。

v1固定hidden dimension=64、weight=0.1，并只接受预冻结temperature grid。训练branch根据objective family分别调用
PCC、MCCA或CCSF函数，避免用一个宽松signature掩盖不同数学定义。effective config和model diagnostics记录
correction mode、dimensions、parameter count、permutation、temperature与weight。

## 4. Step7A config、checker与remote guard

`configs/stage_c_siff_ccsf_step7a.json`将Step6的10个abstract arms绑定到实际readout/objective modes，并记录
Step6 config hash。`local_smoke_temperature=0.1`明确标记为非formal selection。

`scripts/check_stage_c_siff_ccsf_step7a.py`生成50-job manifest并检查：construction、paired initialization、parent
containment、parameters、contrast controls、projectivity、objective algebra、10-arm gradients、two-step correction
optimization、diagnostic shapes与authorization。每个CSV列均直接来自config、model tensor、gradient或计算出的gap；
不读取dataset artifacts。

`scripts/remote/run_stage_c_siff_ccsf_v1.sh`在dry-run时只列出50个adapters。非dry-run先读authorization；当前必然
以exit code 3拒绝。即使未来手工把authorization改true，Step7A template仍以exit code 4拒绝，因为正式Step7B
runner尚未冻结。

## 5. Code-theory consistency

### Intended theory

CCSF需要让same-forward arm contrast进入projective fusion policy，同时让relative teacher只监督policy而不通过
label改变arm forecasts。

### 实现对应

- contrast由arms构造，scorer在fusion前执行；
- teacher由arm error构造后detach；
- requested horizon不进入descriptor/scorer；
- full T policy先完成，最后crop；
- parent、zero、permuted、independent controls均有独立readout mode；
- final zero layer给出exact parent initial function。

### 仍是proxy的部分

local synthetic gate不能证明trained correction有用、teacher能改善fused MSE、或ordered field优于independent。
这些必须由Step7B冻结后的validation selection、formal test四层评估和trained internal artifacts回答。

### Falsifiers

- 任意prefix输出不再等于full forecast crop；
- parent base hash或初始输出不一致；
- zero/permuted controls不再保持参数或映射合同；
- second-step correction hidden gradient为零；
- target、requested H或benchmark bin进入inference path；
- 正式结果被capacity/teacher/independent controls解释。

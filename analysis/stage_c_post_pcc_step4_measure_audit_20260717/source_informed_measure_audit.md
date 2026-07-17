# Post-SIFF Step 4：Projective Measure Source Audit

## 1. 当前节点与结论

| Field | Value |
| --- | --- |
| `current_step` | rollback Step 4 source/narrative audit |
| `trigger` | SIFF long-prefix微正、short-prefix严重负；ETTm2 H1 >100% pathology |
| `search_date` | 2026-07-17 |
| `source_policy` | external primary sources first；Zotero未作为coverage source |
| `candidate_audited` | projective/harmonic horizon-measure-aligned fused loss |
| `narrative_gate` | fail as paper contribution |
| `code-audit correction` | coupling arms already train with exact harmonic-L1 fused loss |
| `protocol_role` | HR is existing control；weighted checkpoint is diagnostic prior |
| `next_step` | `SC-D16-CTD` Step 5/6 checkpoint-trajectory design；implementation/remote/test=false |

[Decision] uniform-horizon harmonic reweighting已经被ElasTST直接覆盖。更关键的是，本项目
`projective_coupling_credit_loss()`与MCCA training path已经对最终fused forecast使用exact harmonic-L1 loss。
因此新增HR training arm既不新颖也不构成新的matched intervention。未决问题只剩：H720 checkpoint selection
是否丢弃了dense-risk更健康的SIFF epoch。

## 2. 为什么问题是真实的

当前StageC primary validation screen使用

$$
L_{\mathrm{dense}}
=\frac1T\sum_{H=1}^{T}\frac1H\sum_{t=1}^{H}e_t
=\sum_{t=1}^{T}
\left(\frac1T\sum_{H=t}^{T}\frac1H\right)e_t.
$$

code audit确认训练协议实际是：

- PCSD/SIFF EQUAL/PCC/MCCA：exact harmonic target measure下的fused **L1** loss，加各自arm/router auxiliary；
- A6 reference：flat full-domain L1；
- 所有arms：用H720 validation **MSE**选择checkpoint；
- primary screen：dense-prefix **MSE** AUC。

ETTm2上SIFF+MCCA相对PCSD+MCCA的H1 gain为-669.49%，H720 gain却为+0.6013%。由于二者共享相同
harmonic-L1 training，该现象不能归因于flat training；剩余解释是Q2 readout optimization、L1/MSE mismatch或
H720 checkpoint selection。

[Fact] 问题存在不等于解决方案新颖。后续必须分开记录：

1. architecture paired comparison已经处于matched harmonic-L1 objective；
2. 是否存在被H720 checkpoint rule丢弃的healthy SIFF epoch；
3. 若不存在，Q2 readout/optimization negative是否可升级为稳定direction evidence。

## 3. External primary-source audit

### 3.1 ElasTST：直接覆盖uniform-horizon harmonic reweighting

NeurIPS 2024
[ElasTST](https://proceedings.neurips.cc/paper_files/paper/2024/file/d7aa002885ccbe68cf6880da583761b2-Paper-Conference.pdf)
在每个training step从$[1,T_{\max}]$随机采样forecast horizon，并推导其期望可由single fixed-horizon weighted
loss近似；论文Equation 4–5明确给出inverse-horizon expectation与harmonic/log reweighting。其研究对象就是
varied-horizon forecasting，不是相邻领域的偶然primitive。

官方实现为
[microsoft/ProbTS `elastst` branch](https://github.com/microsoft/ProbTS/tree/elastst)，本次审计checkout
commit `d49f7e41c2db7ac3208816225885b6e3f61c0fb3`。关键实现：

- `get_weights("random", max_hor)`使用
  $T^{-1}(\log T-\log t)$近似；
- `training_forward()`把该weight逐future step乘到loss tensor后求和；
- validation在启用reweighting时也计算weighted metric；
- checkpoint monitor从普通`val_CRPS`切换到`val_weighted_ND`；
- ETTm1/ETTm2/ETTh1/ETTh2/Weather官方configs均启用`sampling_weight_scheme: random`。

[Strong Evidence] 我们推导的exact discrete weight
$T^{-1}\sum_{H=t}^{T}H^{-1}$与ElasTST的研究问题、数学来源、single-forward training作用和checkpoint
alignment均实质重叠。

### 3.2 Loss Shaping：已覆盖flat average造成step-wise disparity

ICML 2024
[Loss Shaping Constraints](https://icml.cc/virtual/2024/poster/34815)
明确指出最小化forecast window平均loss会造成不同future steps的error distribution显著不均，并用per-step
constraints与primal-dual optimization控制该分布。

它没有专门使用uniform-horizon harmonic measure，但已经覆盖“flat average可能掩盖单step严重错误”和
“training objective应直接shape future-step loss”的problem statement。

### 3.3 QDF：已覆盖non-uniform future-task weighting

ICLR 2026
[Quadratic Direct Forecast](https://openreview.net/pdf/1987c914e2236b7b61b5561103abff145b5a1222.pdf)
把future steps视为heterogeneous forecasting tasks，以non-uniform diagonal weights控制step importance，并以
off-diagonal quadratic terms处理label autocorrelation。

因此“不同future steps不应等权”或“weighted MSE改善multi-step forecasting”均不能作为本项目novelty。

### 3.4 Wider loss-design pressure

NeurIPS 2025
[Time-o1](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0cd62dea69635f4c5b569848267fe5a8-Abstract-Conference.html)
已将training loss design与transformed-label alignment作为forecasting核心贡献。虽然它不直接覆盖uniform
prefix measure，但进一步提高了“仅更换loss”路线的novelty门槛。

## 4. Novelty decision

| Claim | Decision | Reason |
| --- | --- | --- |
| uniform random horizons induce harmonic target weights | prior-covered | ElasTST直接推导 |
| one full forward replaces random-horizon sampling | prior-covered | ElasTST核心training design |
| weighted validation/checkpoint must match training measure | prior-covered protocol | ElasTST official implementation |
| exact discrete harmonic sum beats log approximation | insufficient novelty | numerical exactification |
| per-step nonuniform weights are useful | prior-covered | ElasTST/QDF/Loss Shaping |
| SIFF × exact projective measure interaction | already tested in training | paired main effect fail |

formal result：

`SC2-PHMA = rejected_by_narrative_gate_as_standalone_contribution`

retained role：

`SC-D16-CTD = diagnostic_only_checkpoint_trajectory_audit`

## 5. Source-informed diagnostic boundary

ElasTST官方实现的weighted validation checkpoint说明：启用horizon reweighting时，只改training而仍按full-H
metric选checkpoint是不完整protocol。但本项目不应因此重新训练25个HR arms；training measure已经存在。

下一步只需重新运行exact training trajectory并保存每epoch validation artifacts：

1. training loss/objective、initialization、epochs与optimizer保持不变；
2. early stopping不应在trajectory收集完成前删除epochs，可保留原stop epoch并另设diagnostic max epoch；
3. 每epoch一次full forward计算H1..720 dense MSE/MAE，不读取test；
4. 离线比较best-H720-MSE、best-dense-MSE与best-dense-MAE checkpoint；
5. 不把weighted checkpoint包装成Contribution。

## 6. Proposed SC-D16-CTD diagnostic

### 6.1 Problem

SIFF-v1的short-prefix pathology是否只由H720 checkpoint selection造成，还是贯穿全部training trajectory？

### 6.2 Minimal matched matrix

先只在ETTm2 pathology locus、seed2021运行四条matched trajectories：

1. `PCSD_EQUAL_TRAJECTORY`：same-objective carrier control；
2. `SIFF_EQUAL_TRAJECTORY`：primary pathology arm；
3. `SIFF_CONSTANT_EQUAL_TRAJECTORY`：same-parameter no-order control；
4. `Q1_WIDE_EQUAL_TRAJECTORY`：better-conditioned matched-width control。

该4-run diagnostic只定位failure cause。若dense checkpoint能消除pathology，再冻结five-dataset confirmation；
否则不扩矩阵。MCCA已关闭，不得进入。

### 6.3 Diagnostic gates to freeze in Step 5/6

gate在implementation前仍需正式冻结，至少包括：

- epoch checkpoint hash与dense metric可追溯；
- original best-H720 result reproduction；
- best-dense-MSE与best-dense-MAE epoch是否不同于best-H720；
- SIFF best-dense checkpoint的H1/H1–48 pathology是否消失；
- 同epoch-rule下SIFF vs PCSD/constant/Q1 attribution；
- H337–720 non-catastrophic trade-off。

### 6.4 Decision boundary

- 若SIFF在所有pre-registered epoch rules下仍存在>100% short-prefix degradation，Q2 readout/optimization
  negative获得有效trajectory evidence，回Step2并关闭scale-field direction；
- 若best-dense checkpoint消除pathology但仍不超过PCSD/Q1/constant，则只修checkpoint protocol，SIFF仍关闭；
- 若best-dense checkpoint同时恢复SIFF architecture effect与controls，再授权five-dataset confirmation；
- 无论结果如何，MCCA-v1保持关闭。

## 7. Self-critique

[Speculative] 当前最吸引人的解释是SIFF trajectory中可能存在short/long trade-off，而H720 checkpoint选中了
long-favoring epoch。但也可能全部epochs都存在Q2 field instability。没有per-epoch dense artifacts前不能二选一。

此外，uniform $H=1,\ldots,720$本身是一种application measure选择。论文最终若只报告传统
96/192/336/720 horizons，应使用对应离散measure，而不是因为当前dense diagnostic方便就默认uniform horizon
具有普适业务意义。

# SC2-PCC-v1-TI Step7A 代码说明

## Modules

- `baselines/timealign_official/layers/PCC.py`：无参数objective、credit transport、schedule与diagnostics；
- `baselines/timealign_official/models/TimeAlign.py`：同一次PCSD forward的可选training details与scope-wise RevIN denorm；
- `baselines/timealign_official/train_repo.py`：CLI、optimizer-progress schedule、loss接入与epoch logging；
- `scripts/check_stage_c_sc2_pcc_step7a.py`：35-case local gate。

## Forward Tensor Flow

`TimeAlign.Model.forward`默认路径不变：

```text
x [B,720,C]
 -> normalization_x(norm)
 -> memory [B,C,P,D]
 -> hidden [B,C,R=P*D]
 -> PCSDCouplingFieldReadout
 -> output_normalized [B,720,C]
 -> normalization_x(denorm)
 -> output [B,720,C]
```

训练显式请求`return_pcsd_training_details=True`时，readout在相同hidden上返回：

```text
arms_normalized [B,C,5,720]
policy          [B,C,720,5]
fused_normalized[B,720,C]
```

由于`Normalize(affine=False)`的history statistics为`mean/stdev [B,1,C]`，代码将其reshape为`[B,C,1,1]`并对
arms逐scope执行同一linear denorm，得到`arms_raw [B,C,5,720]`。这保证arm L1和原始target处于相同scale。

## Objective Tensor Flow

训练adapter把arms permute为`[B,C,T,S]`，target由`[B,T,C]`变为`[B,C,T]`：

```text
arm_error [B,C,T,S]
 -> cumsum(T) / [1..T]
 -> prefix_risk [B,C,T,S]
 -> detach + standardize(S) + softmax(-risk/tau)
 -> q_prefix [B,C,T,S]
 -> reverse_cumsum(q/H) / reverse_cumsum(1/H)
 -> c_transport [B,C,T,S]
```

`prefix_measure [T]`由`reverse_cumsum(1/H)/T`生成并且sum为1。所有loss先在scope（若存在）与target维聚合，再对
batch/channel取mean。

## Control Dispatch

`projective_coupling_credit_loss`只接受Step6冻结的九个mode。`skill_kind`在`none/equal/pointwise/transport`中选择，
`route_kind`在`none/pointwise/transport`中选择。inactive term严格为scalar zero；active route再乘continuous
`route_weight`。代码不接受dataset-specific $	au$、floor或loss coefficients CLI覆盖。

## Training Integration

当`--pcc-objective-mode != off`：

1. adapter强制`readout_mode=pcsd-coupling-field`、`policy_mode=direct`、unified $T=720$与`pred_loss_mode=full`；
2. 每个batch只调用model一次；
3. schedule progress使用`epoch * updates_per_epoch + batch_idx // gradient_accumulation_steps`；
4. `pred_loss`替换为PCC/control total，`train_prediction_full_l1`仍记录普通full-720 L1便于审计；
5. twenty PCC diagnostics按epoch batch mean写入`training_log.csv`；
6. `effective_config.json`固定记录全部objective constants与`inference_graph_changed=false`。

## Code-Theory Consistency

- Intended theory：all-prefix risk应守恒地归因到natural target coordinates，而不把requested horizon输入模型；
- Code realization：cumulative risk与reverse harmonic transport为$O(BCTS)$，float64 loop equivalence最大差
  `2.22e-16`，identity gap `0`；
- Inference boundary：默认三元组、parameter count、full output与prefix crop均保持不变；
- Proxy boundary：local one-batch gradient只证明实现可训练，不证明真实arm specialization、policy predictability或性能；
- Falsification：remote Phase A若被equal/pointwise controls解释、arms不恢复或发生numeric pathology，分别回Step4或Step5，
  不能把local implementation pass升级为paper claim。

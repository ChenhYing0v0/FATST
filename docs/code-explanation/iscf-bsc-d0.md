# ISCF-BSC-D0 诊断代码说明

## 输入与角色

`diagnose_stage_c_iscf_bsc_d0.py`只读取Step7B identity-control checkpoint evaluator已经保存的前256条sequential
validation rows：`probe_arms [N,S,T]`、`probe_direct_policy [N,T,S]`、`probe_fused/probe_targets [N,T]`。它不加载
checkpoint、不修改参数、不创建train/test loader，因此是frozen function-level diagnostic，不是method evaluation。

## Tensor flow

1. 先复核$b(t)=\sum_s w_s(t)a_s(t)$与saved fused forecast的max-absolute gap；
2. 对每个arm构造deviation `a_s - b [N,T]`；
3. scope control按`group_indices [G_s,s]`gather deviation，并用local DCT `C_s [s,r_s]`计算
   $P_s(a_s-b)=C_sC_s^\top(a_s-b)$；
4. affine arm为$\tilde a_s=b+P_s(a_s-b)$，shape仍为`[N,T]`；
5. 保存原policy重新组合$\tilde y(t)=\sum_s w_s(t)\tilde a_s(t)$；
6. 对H96/192/336/720计算MSE/MAE，并与parent barycenter比较。

`global` control对五个deviations使用同一个global DCT rank-$K$ projector；`random` control复现production
`partition_seed=15101`与`15101 + 1009 * scale_index + scale`的exact `torch.randperm`规则。

## Statistics

`gain_percent=100(1-transformed/parent)`；`composition_change_normalized_rms`是新composition与parent forecast的RMS差除以
target RMS；`affine_arm_pairwise_normalized_rms`是十对affine arms的RMS距离均值除以target RMS。CSV列均直接来自上述
probe tensors，不混入official-test数据。

## Code-theory boundary

该诊断测试的是“保留policy barycenter、只投影arm-specific deviation”是否在co-adapted identity representation上有即时
validation signal。positive只能授权BSC end-to-end Step4–6 design；negative只能否定这个frozen readout，不能拒绝BSC或ISCF，
因为identity arms和policy并未与affine composition共同训练。

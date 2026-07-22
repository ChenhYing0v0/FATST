# ISCF-FRSC-D1 诊断代码说明

## 诊断角色

`diagnose_stage_c_iscf_frsc_d1.py`只读取SPS identity-control已保存的validation probes，不加载或修改checkpoint。它用于判断
hard SPS的负向结果是否主要来自rank deletion，以及full-rank scope conditioning是否存在低成本lead；不建立method effectiveness。

## Tensor flow

输入为`probe_arms [N,S,T]`、`probe_direct_policy [N,T,S]`、`probe_fused/probe_targets [N,T]`，固定
`S=5,T=720,N=256`。对每个arm先复用SPS projector得到`P_s a_s [N,T]`，再计算

$$
Q_s(\alpha)a_s=P_sa_s+(1-\alpha)(I-P_s)a_s
=a_s+\alpha(P_sa_s-a_s).
$$

`alpha`冻结为`0.05/0.10/0.20/0.35`；对应out-of-scope eigenvalue为`0.95/0.90/0.80/0.65`，所以每个算子均
full-rank且invertible。原policy重新组合conditioned arms，输出shape仍为`[N,T]`。requested horizon不进入计算，只在评估时取
H96/192/336/720 prefix。

## Controls and selection

`scope-canonical`、`global-canonical`与`scope-random`共享arms、policy、rank和alpha grid。只按canonical macro validation MSE
选择一个全局alpha，再在同一alpha比较controls；不允许dataset/horizon-specific alpha。输出`diagnostic_cells.csv`、
`diagnostic_summary.csv`、`run_audit.csv`和`decision.json`。

## Code-theory boundary

FRSC-D1在frozen identity-coadapted representation上测试即时function perturbation。positive仅授权end-to-end Step4–6 design；
negative只能关闭exact frozen conditioning screen。即使算子full-rank，固定上游representation仍可能不适应该conditioning，故不能据此
否定ISCF或未来joint-training版本。

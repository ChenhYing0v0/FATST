# SC-D18-SPC Step 3 Protocol Freeze 与 Prelaunch

## 1. 当前要回答什么

`SC-D18-SPC-v1`只回答一个problem-existence问题：

> 相对同架构、同初始化类、同dataset profile的强`A6_MEASURE` unified control，分别只为
> H96/H192/H336优化的A6 specialist是否在own horizon上具有稳定accuracy headroom？

它不实现soft-projective model，也不把三个specialists视为论文贡献。只有problem gate通过，才允许回到Step 4
设计controlled soft projectivity；失败则回滚Step 2，重新审计fixed-past multi-horizon主线。

## 2. 公平性与tensor contract

所有arms都使用A6 learned-basis operator：

$$
\text{memory}\in\mathbb R^{B\times C\times P\times D}
\rightarrow
\text{coeff}\in\mathbb R^{B\times C\times256}
\rightarrow
\hat y\in\mathbb R^{B\times C\times720}.
$$

固定项包括Encoder、decoder、rank 256、T=720 parameter table、dataset-aware natural profile、seed 2021、
AdamW、learning rate、batch size、epoch budget与from-scratch initialization。变化仅为：

- `A6_MEASURE`：measure-aligned loss；validation四horizon均值选checkpoint；
- `A6_FULL`：full-H720 L1；validation四horizon均值选checkpoint；
- `A6_SPECH`：只对prefix H计算L1；只以validation H MSE选checkpoint。

specialist仍有完整T=720 head。short horizon之外的rows保留参数但不接受该arm的performance claim。

## 3. Local contract evidence

prelaunch gate为`11/11`：

- 25个artifact units：15个new specialist training runs，10个reused unified controls；
- 15个gradient checks全部具有正prefix gradient；
- `max_tail_gradient_abs=0`，证明short-specific loss没有更新own horizon之外的temporal rows；
- `max_prefix_gap=0`，full-domain output crop与native prefix完全一致；
- 每个dataset的五个arms要求相同Encoder/operator initialization hash、total parameters与active parameters；
- one-batch ETTm1 CLI smoke通过；
- 普通A6 evaluator现保存`probe_fused [8,720]`与`probe_targets [8,720]`，protocol/invariant均pass；
- analyzer synthetic positive case与remote runner syntax通过。

## 4. Formal matrix与统计量

五datasets为ETTh1、ETTh2、ETTm1、ETTm2、Weather。primary cells为：

$$
5\ \text{datasets}\times 3\ \text{own horizons}=15.
$$

每个cell的primary gain定义为：

$$
100\left(1-
\frac{\operatorname{MSE}(\mathrm{A6\_SPEC}_H,H)}
{\operatorname{MSE}(\mathrm{A6\_MEASURE},H)}
\right).
$$

same ordered test rows还保存full-domain predictions，并计算：

$$
\mathrm{NRMSE}_{\text{pred}}=
\frac{\sqrt{\mathbb E(\hat y_{\mathrm{SPEC},1:H}-
\hat y_{\mathrm{MEASURE},1:H})^2}}
{\sqrt{\mathbb E\hat y_{\mathrm{MEASURE},1:H}^2}}.
$$

该量只验证specialization是否真的改变shared prefix，不替代MSE/MAE effectiveness。

## 5. Frozen gates

必须同时满足：

1. specialist相对A6_MEASURE的15-cell macro MSE gain至少`0.5%`；
2. 至少`2/3` horizons为正；
3. 至少`4/5` datasets为正；
4. 至少`10/15` cells为正；
5. 任一horizon macro不得低于`-0.5%`；
6. prediction NRMSE必须非零；
7. protocol、initialization、parameter、finite与checkpoint non-mutation全部通过，且无绝对值超过100%的cell
   degradation。

只超过`A6_FULL`而不超过`A6_MEASURE`时，结论为`measure_training_explains`，不进入soft architecture。

## 6. Authorization boundary

- `diagnostic_only`: true；
- method implementation: false；
- new remote training: 15 runs authorized；
- formal official-test audit: authorized as `test_informed primary-problem-existence-diagnostic`；
- per-dataset/per-horizon tuning: forbidden；
- next: commit/push、remote pull、GPU preflight、dry-run、Weather one-batch resource smoke，全部通过后启动matrix。

# SC-D24-CTB-v1.1：Validation Result、Failure Attribution与Rollback

## 1. Decision

| Field | Result |
| --- | --- |
| `candidate_version` | `SC-D24-CTB-v1.1` |
| `machine_decision` | `coarse_linear_deformation_not_supported` |
| `research_decision` | `close_exact_coarse_deformation_probe_return_step2_4_consolidation` |
| `protocol_valid` | true |
| `problem_gate_supported` | false |
| `official_test_access` | 0 |
| `paper_method_status` | none；training/test/router/second loss remain false |

[Strong Evidence] 在两个strong trajectory carriers上，ordered raw-history coarse correction都没有跨
chronological validation transfer，并系统性弱于marginal、sorted-history与target-shuffled controls。故
“固定trajectory synthesis留下稳定、past-identifiable的48-step linear deformation surface”不成立。

该结果不方向级拒绝所有nonlinear history-conditioned synthesis，因为frozen checkpoint、coarse block、
linear feature map仍是有限diagnostic。但在没有新problem evidence时，也不能据此设计dynamic basis、router、
post-hoc adapter或第二loss。

## 2. Protocol integrity

- source checkpoints：D23 seed2021 `A6_MEASURE`与`DENSE_DUAL_MATCHED`；
- datasets：Weather、ETTm1、ETTh1、ETTh2、ETTm2；
- 10/10 frozen-checkpoint validation runs；
- 每个dataset按forecast origin使用first-third fit、middle-third purge、last-third evaluate；
- 全channels随origin进入同一partition；
- 840 metric rows、720 comparison cells；
- all finite、10/10 checkpoint hashes未改变；
- official test rows/cells：0；
- remote training：0。

v1首先暴露unnormalized ridge penalty设计错误，标记`design_fault_suspected`且不参与problem decision。
v1.1使用

$$
X^\top X+n\lambda I,\qquad \lambda\in\{0.01,0.1,1\},
$$

与normalized mean-squared objective一致；primary $\lambda=0.1$。data、features、splits、controls和gates均未
根据v1结果作dataset-specific修改。

## 3. Primary comparison

gain定义为

$$
g=100\left(1-\frac{\operatorname{MSE}_{ordered}}
{\operatorname{MSE}_{control}}\right).
$$

正值表示ordered raw history更好。

| Carrier | Comparison | Macro gain | Cells | Datasets | Horizons |
| --- | --- | ---: | ---: | ---: | ---: |
| A6 | ordered vs marginal | -8.5950% | 3/20 | 1/5 | 0/4 |
| A6 | ordered vs sorted | -9.4741% | 6/20 | 2/5 | 0/4 |
| A6 | ordered vs target-shuffled | -14.1002% | 4/20 | 1/5 | 0/4 |
| DENSE | ordered vs marginal | -8.6168% | 3/20 | 0/5 | 0/4 |
| DENSE | ordered vs sorted | -8.8197% | 6/20 | 2/5 | 0/4 |
| DENSE | ordered vs target-shuffled | -13.4974% | 4/20 | 1/5 | 0/4 |

相对更弱controls也失败：

- ordered vs global：A6 `-17.9501%`，DENSE `-17.6128%`；
- ordered vs channel：`-12.8787%/-12.4661%`；
- ordered vs recent：`-4.0152%/-4.1797%`。

没有一个primary comparison达到11/20 cells、3/5 datasets或3/4 horizons。

## 4. Absolute transfer health

所有corrections相对原frozen forecast的macro gain：

| Feature family | A6, $\lambda=0.1$ | DENSE, $\lambda=0.1$ |
| --- | ---: | ---: |
| global | -11.794% | -12.088% |
| channel | -18.652% | -18.909% |
| marginal | -21.005% | -20.851% |
| recent | -27.088% | -27.115% |
| sorted history | -21.729% | -22.147% |
| ordered history | -32.256% | -32.375% |
| target shuffled | -16.682% | -17.249% |

即使$\lambda=1$强收缩，ordered history仍为`-14.930%/-15.169%`。这排除了“只差一个合理ridge scale”的
解释。global correction本身为负，说明first-third residual mean不能迁移到last-third；conditional features又
进一步扩大误差。

## 5. Dataset pattern

primary ordered-vs-marginal：

- A6仅Weather macro `+1.551%`，其余四datasets为负；
- DENSE五个dataset macro均不为正；
- ETTh1 ordered-vs-sorted与target-shuffled局部为正，但ordered-vs-marginal分别约`-13.38%/-13.19%`；
- ETTh2是最强反例，ordered-vs-marginal约`-26.06%/-24.69%`。

因此不能选择有利dataset或control来promotion。Weather与ETTh1局部信号只保留为exploratory observation。

## 6. Failure attribution

1. `hypothesis_false`：对exact hypothesis成立。stationary coarse linear deformation没有跨chronological
   validation transfer；
2. `intervention_point_wrong`：对broader direction仍可能。48-step constant map可能过粗；
3. `readout_or_head_design_wrong`：v1成立，已由v1.1 normalized ridge修复；v1.1 negative不能再归因于该错误；
4. `optimization_or_numeric_pathology`：v1.1不存在；all finite且$\lambda$ sensitivity一致；
5. `capacity_control_explains`：ordered features没有超过同capacity target-shuffled，反而明显更差。

Direction boundary：

- 关闭`SC-D24-CTB-v1.1` exact diagnostic；
- broader nonlinear history-conditioned output geometry标记`unresolved_but_unsupported`，不是
  `direction_rejected`；
- 不做bin width、feature、lambda、seed、nonlinear head或representation rescue；
- 不把Weather/ETTh1局部结果用于dataset-specific candidate；
- 当前没有active method。

## 7. Latest primary-source boundary

检索日期：2026-07-20。

- [BasisFormer](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e150e6d0a1e5214740c39c6e4503ba7a-Abstract-Conference.html)
  已覆盖adaptive basis learning、history-to-basis coefficients与future basis consolidation；
- [PULSE, ICML 2026](https://openreview.net/forum?id=JJIqZzujgE)已覆盖phase evolution、phase router与
  nonstationary trajectory generation；
- [PhaseFormer, ICLR 2026](https://openreview.net/forum?id=Lk9SqMQzhX)已覆盖phase-wise prediction与
  cross-phase routing；
- [LatentTSF, ICML 2026](https://openreview.net/forum?id=s49fw3BVU0)把问题转向latent-state prediction与
  auxiliary representation objective；
- [Taming Recent-Data Bias, ICML 2026](https://openreview.net/forum?id=gDA03Yn3fi)直接研究recent/global
  context robustness；
- [Current Benchmarking Hinders Real Progress, ICML 2026 position paper](https://openreview.net/forum?id=gtwbLmO7Wb)
  强调architecture比较必须隔离design dimensions与implementation confounds。

因此dynamic basis、phase routing、latent objective、recent/global context本身都不能作为D24 negative后的低成本
新命名候选。当前更可信的论文资产是Bayes boundary、完整capacity controls与一系列可复核negative/positive
diagnostics，而不是未经支持的新module。

## 8. Rollback

Decision：

`close_exact_coarse_deformation_probe_return_step2_4_consolidation`。

下一步做paper-story/evidence consolidation audit：

1. 判断`Bayes task boundary -> horizon frontier negative -> target-access diagnostic positive ->
   capacity-control explanation`是否形成可发表的完整问题链；
2. 重新界定A6/MEASURE在该链中的solution/control角色，而不是单独claim basis或loss；
3. 列出现代varied-horizon/native baselines缺口与最小补齐矩阵；
4. 在该narrative gate完成前，不再启动D25 architecture、remote training或official test。

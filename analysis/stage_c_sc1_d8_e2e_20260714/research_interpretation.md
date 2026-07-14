# SC1-D8-E2E End-to-End Co-Adaptation Screen

## 1. What We Tested

本轮回答两个彼此独立的问题：

1. canonical RGNB geometry相对PERM/RANDOM descriptors是否在end-to-end训练后仍有效；
2. 当前shared-latent PAF是否能作为完整forecast operator追平或超过same-run A6。

矩阵为ETTh1、ETTh2、ETTm1、ETTm2、Weather × A6/GEO/PERM/RANDOM compact+matched七arms ×
seed2021，共35 runs。所有arms从头joint-train Encoder与Decoder；profile、optimizer、full-H720 L1、best-val
selection完全固定。最终只在validation上计算H1..720 dense metrics，test未使用。

## 2. Artifact And Protocol Audit

[Fact] 35/35 runs均具有720-row dense metrics、training log、effective config、model diagnostics、patch
diagnostics和trained checkpoint invariants；35/35 status=`ok`。

[Fact] 所有run均为`from_scratch`、0 frozen parameter tensors、`final_evaluation_split=val`。prefix max gap、
finite、patch artifact与profile hash audits均通过。

[Protocol Repair] RANDOM-c256 Weather的trained patch-block absolute gap为`1.1444e-5`，略超初始化时
`1e-5`阈值，但prefix gap仅`1.885e-6`且该rewrite代数上是同一个linear map。trained patch audit在不改变
模型、训练或performance gate的前提下修正为`2e-5`；重审通过。一次worker recovery race产生了重复
RANDOM-m694 ETTh1启动，最终目录重新验证为20 training rows、720 metric rows、checkpoint invariant pass，
本地复算与远端gate完全一致。

## 3. Metric Definitions

- `dense MSE AUC`：H1..720 cumulative-prefix MSE的算术平均；越低越好；
- `improvement`：$100(1-\mathrm{candidate}/\mathrm{reference})$，正数表示candidate更好；
- `matched median`：同width的PERM与RANDOM dense MSE AUC中位数；
- `patch entropy`：validation上各patch对shared latent的mean-squared contribution share之normalized entropy；
- `atom-patch diversity`：RGNB support groups的normalized analytic Jacobian patch profiles之pairwise mean
  absolute distance。

## 4. Primary Results

| Dataset | GEO-c256 vs A6 | GEO-c256 vs matched | GEO-m694 vs A6 | m694 vs c256 | c256 entropy Δ vs A6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | -35.62% | +5.68% | -34.56% | +0.78% | -0.002 |
| ETTh2 | -21.56% | +13.59% | -20.84% | +0.59% | -0.071 |
| ETTm1 | -46.21% | +18.29% | -44.77% | +0.98% | -0.049 |
| ETTm2 | -14.26% | +18.69% | -14.24% | +0.02% | -0.075 |
| Weather | -22.87% | +15.40% | -22.25% | +0.51% | -0.111 |
| **Macro** | **-28.10%** | **+14.33%** | **-27.33%** | **+0.58%** | **-0.062** |

[Strong Evidence] geometry attribution通过：GEO-c256相对matched median为`+14.33%`、5/5 datasets为正；
GEO-m694相对matched median为`+14.71%`、5/5为正。该效果跨width稳定，复现并强化D7 conditional signal。

[Strong Evidence] 完整operator effectiveness失败：GEO-c256相对A6 macro `-28.10%`、MAE macro
`-20.54%`，5/5 datasets均负；worst dataset ETTm1为`-46.21%`。m694相对c256只改善`+0.58%`，
相对A6仍为`-27.33%`，因此parameter/trunk width不足不能解释主要gap。

## 5. Horizon Behavior

| Segment | c256 vs A6 | c256 vs matched | m694 vs A6 | m694 vs matched |
| --- | ---: | ---: | ---: | ---: |
| H1-48 | -45.52% | +46.65% | -43.91% | +47.19% |
| H49-96 | -30.27% | +36.39% | -28.57% | +37.19% |
| H97-192 | -32.92% | +24.28% | -31.38% | +25.00% |
| H193-336 | -31.52% | +14.33% | -30.55% | +14.78% |
| H337-720 | -25.66% | +7.43% | -25.16% | +7.65% |

[Strong Evidence] geometry benefit随horizon增长单调衰减，在H1-48最强，和D6的short-prefix local-support
证据一致；但PAF相对free A6的绝对gap也在short horizon最大。因此当前方法把geometry变成了有效inductive
bias，却以过强的history-to-atom restriction损失了free operator expressivity/optimisability。

## 6. Optimization And Patch Diagnostics

- A6五dataset运行`7–17` epochs，best epoch为`2–12`；
- GEO-c256与GEO-m694均5/5跑满20 epochs，best epoch多在18–20；
- 但最后5 epochs validation仅改善`0.02%–0.49%`，远小于对A6的`14%–46%` gap；
- GEO-c256 patch entropy均值`0.878`，A6为`0.940`；4/5 datasets明显下降，ETTh1仅下降`0.002`；
- GEO-c256 atom-patch diversity均值`0.0089`，PERM/RANDOM约`0.0012–0.0013`；geometry确实形成更有差异的
  support-group patch profiles，而非完全忽略descriptors。

[Decision] `epoch-cap dominated`使本轮不能拒绝更广的PLGO方向；但曲线已近平台，简单延长同一cosine schedule
缺乏弥补数量级gap的证据，不启动无边界longer-epoch sweep。

[Decision] patch concentration支持`shared_history_interface_suspected`，但ETTh1在entropy几乎不变时仍
`-35.62%`，所以collapse不是充分解释。更核心的风险是
$\alpha_j=\psi(d_j)^T A h$的shared-latent/separable atom-history map：它把A6的free temporal table换成
descriptor-MLP所张成的受限map；即使m694参数近预算，约束与优化路径仍未解除。

## 7. Failure Attribution

1. `hypothesis_false`: **否**。D6 support interaction仍成立，GEO相对matched controls在两种width均5/5通过；
2. `intervention_point_wrong`: **suspected**。部分dataset patch concentration明显，但不是唯一原因；
3. `readout_or_head_design_wrong`: **strongly supported**。exact shared-latent PAF相对A6大幅失败，width扩展无救；
4. `optimization_or_numeric_pathology`: **present but bounded**。无divergence/NaN/>100% degradation；存在
   epoch-cap saturation与一次float32 audit阈值修复；
5. `capacity_control_explains`: **not supported as the main cause**。m694近预算仍失败，且geometry effect跨width稳定。

因此，本轮可以否定的是`SC1-D8 exact shared-latent PAF`作为paper-core完整方法；不能否定RGNB geometry、
projective synthesis或更广的PLGO方向。direction-level标签应为`design_fault_suspected`，而不是
`hypothesis_false`。

## 8. Decision And Rollback

`decision = exact_paf_failed_geometry_retained_rollback_step4`

- 不进入三seed confirmation；
- 不启动SC2-MIPR或joint factorial；
- 回到11-step loop的Step 4，先做source-informed intervention/readout redesign audit；
- 新审计必须同时解释：如何保留已证实的RGNB geometry benefit、如何解除shared separable bottleneck、是否需要
  直接使用`memory [B,C,P,D]`而非只经共享latent、以及如何避开B14已否定的无依据atom retrieval；
- 在新候选重新通过Step4-6 narrative/theory gate前，不实现、不远程训练。

## 9. Artifacts

- raw lightweight artifacts：`artifacts/stage_c_sc1_d8_e2e_remote_20260714/`；
- machine gate：`gate.json`；
- run-level metrics/diagnostics：`run_summary.csv`；
- A6 comparisons：`a6_comparisons.csv`；
- segment comparisons：`horizon_segment_comparisons.csv`。

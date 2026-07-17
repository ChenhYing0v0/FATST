# SC2-PCC-v1-TI Step7B Seed2021 Step9/10 Result

## Current Decision

| Field | Value |
| --- | --- |
| `current_step` | Step9 artifact evaluation完成；Step10 validation-screen decision完成；回Step4 |
| `candidate` | `SC2-PCC-v1-TI` |
| `formal_gate` | `method_pass=false` |
| `formal_decision` | `generic_or_pointwise_control_explains_return_step4` |
| `research_decision` | `exact_pcc_v1_ti_fail_retain_arm_recovery_signal_return_step4` |
| `held` | official test、seed2022/2023、NO_FLOOR/NO_STOPGRAD Phase B |

[Fact] 完整PCC在validation上相对A6取得macro `+0.9627%`、3/5 wins，相对plain PCSD DIRECT取得
`+2.4927%`、5/5 wins，并把25/25个same-run arms相对independent fixed training的退化全部改善。

[Fact] 它仍未通过pre-registered method gate：相对closest `POINTWISE_PRIOR_COMPOSED`仅`+0.1050%`，低于
`0.2%` specificity门槛；五datasets的minimum pairwise arm-output NRMSE只保留plain DIRECT的
`20.57%-41.13%`，全部低于50%门槛。

[Decision] 不能把这次结果写成PCC method成功。它证明“direct arm supervision能够修复credit starvation并改善
carrier”，但没有证明“nested-risk harmonic transport”是主要增益来源，也没有实现论文要求的scope specialization。
因此精确v1停止，不跑test/confirmation/Phase B；保留问题与正向training evidence，回Step4重构architecture-training
contract。

## What Was Tested

Phase A为9 objectives × 5 datasets × seed2021，共45个from-scratch E2E runs。所有runs使用相同dataset-aware
natural profile、PCSD-CF DIRECT inference architecture、full $T=720$ output、best-validation-H720 checkpoint，并在
validation上计算dense $H=1,\ldots,720$ full-crop metrics。A6、plain DIRECT、DENSE_MATCHED与five fixed-scope
runs复用冻结seed2021 references，不重训。

`dense_mse_auc`定义为

$$
\mathrm{AUC}_{\mathrm{MSE}}=\frac1{720}\sum_{H=1}^{720}
\mathrm{MSE}(\hat Y_{1:H},Y_{1:H}).
$$

相对收益为$100(1-\mathrm{candidate}/\mathrm{reference})$，先逐dataset计算，再对five datasets等权平均。

## Artifact And Protocol Audit

- remote、本地复算均得到45/45 PCC runs与15/15 references有效；
- paired Encoder/PCSD initialization hashes逐dataset一致；
- 每个run包含dense 720 horizons、training log、PCSD diagnostics、trained invariants与best-val shared-gradient audit；
- 所有trained invariants与gradient audits通过，未发现Traceback、OOM、NaN或gradient surgery；
- `test_used=false`，本阶段没有读取official test split；
- remote automatic summary与本地从raw artifacts重算的`gate.json`一致。

## Formal Gate Result

| Gate | Result | Evidence |
| --- | --- | --- |
| PCC over A6 | pass | `+0.9627%`，3/5 |
| PCC over plain DIRECT | pass | `+2.4927%`，5/5 |
| PCC over pointwise PCC-v0 | pass | `+0.5179%`，4/5 |
| PCC over pointwise prior composed | **fail** | `+0.1050%`，3/5；要求≥`0.2%` |
| arm degradation recovery | pass | 25/25 improved；median relative reduction `98.01%` |
| pairwise NRMSE retention | **fail** | minimum retention `20.57%`；要求≥`50%` |
| policy non-collapse | pass | entropy min `0.9313`；usage max `0.3588` |

### Dataset-level PCC comparison

| Dataset | PCC AUC | PCC vs A6 | PCC vs plain | PCC vs prior composed |
| --- | ---: | ---: | ---: | ---: |
| ETTh1 | 1.177916 | -0.6413% | +0.1378% | +0.2384% |
| ETTh2 | 0.421791 | +1.5440% | +4.1154% | +0.2621% |
| ETTm1 | 0.646374 | +3.0282% | +5.2462% | -0.0384% |
| ETTm2 | 0.195287 | -0.3027% | +1.5716% | -0.0073% |
| Weather | 0.491216 | +1.1855% | +1.3924% | +0.0700% |

## Objective Attribution

| Objective | Macro gain vs A6 | Wins vs A6 | Macro gain vs plain |
| --- | ---: | ---: | ---: |
| `MEASURE_ONLY` | +0.2881% | 2/5 | +1.8332% |
| `EQUAL_SKILL` | +0.8558% | 3/5 | +2.3881% |
| `POINTWISE_CAPABILITY_SKILL_ONLY` | +0.4959% | 3/5 | +2.0351% |
| `POINTWISE_PRIOR_COMPOSED` | +0.8580% | 3/5 | +2.3894% |
| `POINTWISE_PCC_V0` | +0.4471% | 3/5 | +1.9883% |
| `TRANSPORT_SKILL_ONLY` | +0.7345% | 3/5 | +2.2690% |
| `PCC_TRANSPORT_FULL` | **+0.9627%** | **3/5** | **+2.4927%** |

`EQUAL_SKILL`已经取得完整PCC相对A6增益的`88.90%`。该比例不是严格可加的因果分解，但与PCC仅比
`EQUAL_SKILL`高`0.1091%`、仅比prior composition高`0.1050%`共同说明：主要收益来自generic direct arm
supervision，而不是已被独立识别的harmonic transport作用。

## Mechanism Diagnosis

### 1. Arm starvation was repaired

plain DIRECT的五arms相对independent fixed-scope training严重退化；PCC对25/25 pairs全部改善，median relative
reduction为`98.01%`。多数组合已回到fixed run附近，部分甚至略优。这排除了“auxiliary credit没有进入arms”或
“shared field完全不可训练”的解释。

### 2. Recovery happened mainly by homogenization

| Dataset | PCC minimum pairwise NRMSE | Retention vs plain | Same-run oracle headroom |
| --- | ---: | ---: | ---: |
| ETTh1 | 0.04253 | 29.35% | 10.93% |
| ETTh2 | 0.03265 | 41.13% | 15.44% |
| ETTm1 | 0.02535 | 23.63% | 17.87% |
| ETTm2 | 0.02158 | 20.57% | 6.53% |
| Weather | 0.04617 | 40.36% | 18.35% |

PCC arms虽然都变得准确，但它们的forecast outputs明显靠拢。`EQUAL_SKILL`与prior-composed controls也出现相同
现象；相反，pointwise capability-weighted controls通常保留更多separation但性能较弱。[Strong Evidence] current
objective处在“coverage/skill”与“scope specialization”之间的Pareto trade-off，而不是同时解决两者。

### 3. Router is balanced, but capability alignment remains weak

policy entropy为`0.9313-0.9846`，usage max不超过`0.3588`，因此没有one-hot collapse。但训练末期credit-policy
argmax accuracy仅`0.3083-0.4141`；shared-field scope-loss gradient cosine mean为`0.2456-0.6730`，多数方向高度
同向。[Inference] router保持soft/balanced不等于学到target-specific coupling；shared-field与same-label arm losses
共同把不同scope推向相似函数。

### 4. Horizon profile does not isolate transport

PCC相对A6的macro gain从$H=1..48$的`+7.35%`逐步降到$H=513..720$的`+0.40%`，说明measure-aware direct
supervision确实偏向dense-prefix objective。可是PCC相对prior-composed在八个bins仅约`+0.02%`到`+0.21%`，曲线
整体贴近零线；未形成强、独有的nested-risk transport signature。完整曲线见`horizon_gain_curves.svg/png`。

## Failure Attribution

1. `hypothesis_false`: **不完全成立**。arm starvation是真问题，direct supervision能够修复并改善validation性能；
2. `intervention_point_wrong`: **部分成立**。对每个arm施加same-label skill loss能送达梯度，但会把scope functions推向
   相同conditional predictor；
3. `readout_or_head_design_wrong`: **仍可能**。PCSD所有arms共享`mode_weight/mode_bias/synthesis/bias`，scope只经
   pooled coordinates改变计算；current field没有可辨识的scope-conditioned history operator；
4. `optimization_or_numeric_pathology`: **排除**。45/45 finite、paired、protocol pass；
5. `capacity_control_explains`: **对主要增益成立**。generic equal-skill/prior controls解释绝大部分gain；但不能解释
   PCC比plain 5/5改善与25/25 arm recovery这一正向training signal。

结论是`design-level failure`，不是方向级`hypothesis_false`：exact harmonic transport不足以成为Contribution 2，且
current PCSD field与same-label skill objective之间存在结构性identifiability tension。

## Step10 Decision And Test Boundary

- `SC2-PCC-v1-TI`改为`validation_screen_failed_exact_design`；不升paper claim；
- `EQUAL_SKILL/MEASURE_ONLY`保留为强training controls，不改名为创新；
- 不执行NO_FLOOR/NO_STOPGRAD Phase B：它们按protocol只在Phase A method pass后授权，而且单纯调floor不能解决
  shared-field identifiability；
- 不执行seed2022/2023或official test。test仍是paper-core candidate的primary effectiveness gate，但一个已在
  validation mechanism/specificity gate失败的exact design不因读取test而恢复叙事有效性；
- 回11-step Step4，同时审计architecture中的scope-identifiable degrees与training中的non-homogenizing credit；
  下一节点为Step5 theory feasibility，而不是remote experiment。

## Artifact Definitions

- `objective_scoreboard.csv`：all objectives的five-dataset macro AUC、相对A6/plain收益与wins；
- `horizon_gain_by_reference.csv`：逐dataset、逐$H$的PCC/reference MSE/MAE relative gain；
- `horizon_bin_gain.csv`：冻结八bins内先平均MSE/MAE再算relative gain，并附five-dataset macro；
- `mechanism_control_summary.csv`：all objectives的oracle、separation retention、policy与gradient diagnostics；
- `pcc_training_diagnostics.csv`：full PCC末epoch credit/policy alignment与early-stop状态；
- `deep_dive_gate.json`：formal gate不变，追加post-hoc failure attribution；
- `horizon_gain_curves.svg/png`：PCC相对A6/plain/equal/prior/pointwise-v0的dense horizon curves；
- `raw/`：gitignored、checkpoint/predictions排除后的remote evidence。

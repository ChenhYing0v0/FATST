# SC1-D3 Crossed Basis-Group Diagnostic Protocol

## Status

| Field | Value |
| --- | --- |
| `candidate` | `SC1-D3` |
| `role` | `diagnostic_only` |
| `current_step` | Step 2/3 |
| `problem` | D2的balanced-interval basis优势是独立main effect，还是basis-group interaction？ |
| `carrier` | frozen `A6-LBF-natural-baseline` Encoder checkpoints |
| `suite` | ETTh1, ETTh2, ETTm1, ETTm2, Weather × checkpoint seeds 2021/2022/2023 |
| `new_fits` | 45：5 datasets × 3 checkpoints × 3 paired structure seeds |
| `test_used` | `false` |
| `forecast_model_updated` | `false` |
| `method_training_authorized` | `false` |
| `pass_authorization` | 仅返回Step 4提出新的paper-core idea |

## 1. What We Plan To Test

D2得到两个同时成立的事实：

1. `true basis`相对`random basis`的formal5 macro MSE gain为`+3.0635%`，5/5 datasets通过；
2. `true group`相对`random group`仅`+0.0947%`，2/5 datasets通过。

但是D2只有三个cell：

| Cell | Basis | Group | D2 arm |
| --- | --- | --- | --- |
| `TT` | true | true | `true_scale_grouped` |
| `TR` | true | random | `random_group_s*` |
| `RT` | random | true | `random_basis_s*` |
| `RR` | random | random | **missing** |

因此D2的`RT vs TT`只是在true-group条件下的conditional basis effect，不能识别marginal main effect。
D3只补`RR=random basis × random group`，不重跑已存在的165个D2 fits，不调profile、group sizes、
hidden width、optimizer或seed。

标准factorial literature把main effect与interaction作为不同estimands；D3只借用这一代数分解，
不把probe arms解释为随机化causal treatments。设计依据参考
[Zhao & Ding, 2021](https://arxiv.org/abs/2101.02400)与
[Li, Ding & Rubin, 2018](https://arxiv.org/abs/1812.10911)。检索日期：2026-07-14；
来源为外部primary preprints，非Zotero覆盖判断。

## 2. Tensor And Artifact Construction

D3完全复用D2 frozen-memory tensor contract：

$$
M\in\mathbb R^{B\times C\times P\times D},\qquad
h=\operatorname{vec}(M)\in\mathbb R^{BC\times(PD)},
$$

$$
u=\frac{y-\mu_x}{\sigma_x}\in\mathbb R^{BC\times720}.
$$

对每个`structure_seed`，构造random orthogonal basis
$Q_r\in\mathbb R^{720\times720}$，并把720个coefficient indices随机分到与D2相同大小的11组：

$$
[1,1,2,4,8,16,32,64,128,256,208].
$$

每组使用独立`PD -> GELU(32) -> n_l` block，输出$widehat\alpha$，再通过
$\widehat u=\widehat\alpha Q_r$还原。`basis_seed=group_seed`只用于与D2的同编号controls形成paired
block；三个structure seeds是control replicates，不是三个额外checkpoint observations。

### Frozen split and optimization

- train前16 batches；按sample ID做80/20 fit/inner-holdout；
- official validation前8 batches仅作一次final evaluation；test不加载；
- feature normalization只用fit rows；
- `AdamW(lr=1e-3, weight_decay=1e-4)`；`batch_size=1024`；
- `max_epochs=120`、`patience=15`；
- `data_seed=20260713`、`probe_seed=20260714`、structure seeds=`3101/3102/3103`；
- A6 forecast model全部冻结。

## 3. Estimands

对正的evaluation-space error使用log scale，使2×2 contrasts具有可加性。记
$L_{TT}=\log E_{TT}$，其余同理。true basis相对random basis的两个conditional effects为：

$$
\Delta_{B\mid G_T}=L_{RT}-L_{TT}=\log\frac{E_{RT}}{E_{TT}},
$$

$$
\Delta_{B\mid G_R}=L_{RR}-L_{TR}=\log\frac{E_{RR}}{E_{TR}}.
$$

positive表示true basis error更低。basis main effect与interaction定义为：

$$
\Delta_B=\frac{1}{2}(\Delta_{B\mid G_T}+\Delta_{B\mid G_R}),
$$

$$
I_{BG}=\Delta_{B\mid G_T}-\Delta_{B\mid G_R}.
$$

报告时将log effect换成可读的relative reduction：

$$
r(\Delta)=1-\exp(-\Delta).
$$

同样计算group main effect与MAE effects，但paper-relevant primary estimand是MSE basis main effect。

## 4. Aggregation Without Pseudo-Replication

每个dataset × checkpoint seed中，先平均三个paired structure-seed的**log effects**，再换算relative
reduction。这样形成`5 × 3 = 15`个primary paired units。TT在三个structure blocks中复用，但不会因此被
计作三个独立checkpoint observations。

dataset summary再对其三个checkpoint units求均值、标准差与positive count；macro gate对15个primary
units等权。该设计比把45 blocks直接做显著性计数更保守。

## 5. Preregistered Hard Gate

配置冻结于`configs/stage_c_sc1_d3_crossed_basis_group.json`。formal pass必须同时满足：

1. 45/45 factorial blocks、15/15 primary units、5/5 datasets完整；D2/D3 contract hash相同；
2. test/freeze/official-validation/orthogonality invariants全部通过；
3. basis main MSE reduction至少`0.5%`，且至少3/5 datasets有2/3 checkpoint effects为正；
4. true-group conditional与random-group conditional分别满足同一`0.5% + 3/5` gate；
5. basis main MAE reduction不得低于`-0.25%`；
6. 至少3/5 datasets满足$|I_{BG}|\le|\Delta_B|$，即interaction不主导main effect。

Decision mapping：

| Observation | Decision |
| --- | --- |
| incomplete/invariant fail | `diagnostic_invalid_for_direction_rejection` |
| main与true-group pass，但random-group fail | `basis_advantage_group_dependent_reformulate_step2` |
| main pass但interaction guard fail | `basis_signal_interaction_dominated_refine_step2` |
| 完整gate pass | `basis_main_effect_supported_return_step4` |
| 其余stable fail | `basis_main_effect_not_supported_reformulate_step2` |

## 6. What A Pass Does Not Prove

- 不证明balanced interval basis是novel contribution；wavelet/tree/basis prior art仍需Step 4重审；
- 不证明该basis能改善end-to-end decoder；probe只测试frozen-memory readout family；
- 不证明11个depth groups有效；D2已经否定该exact grouping claim；
- diagonal seed pairing没有覆盖3×3全部random-basis/random-group组合，结论只针对预注册paired control
  distribution；
- 不授权实现decoder、MIPR或MoE。

## 7. Failure Attribution

- non-finite loss、orthogonality gap `>1e-5`、artifact不完整或split污染：
  `optimization_or_numeric_pathology` / invalid，不可否定方向；
- random-group conditional失败：basis signal依赖当前group context，属于`hypothesis_not_identified_as_main_effect`；
- interaction主导：D2差值不能被叙述为独立geometry effect；
- stable main-effect fail：只关闭“balanced interval basis在当前frozen-memory grouped nonlinear probe中具有
  independent main effect”，不关闭全部future-aware operator方向。

## 8. 11-Step Record

| Field | Record |
| --- | --- |
| `current_step` | Step 2/3 diagnostic |
| `problem` | basis geometry是否为独立main effect |
| `existence_evidence` | D2 true-vs-random-basis +3.0635%，但缺RR cell |
| `idea` | none paper-core；补齐paired crossed control |
| `theory_check` | 2×2 log-error factorial decomposition；structure seeds先聚合 |
| `design` | 45 missing-cell fits + 15-unit hard gate |
| `narrative_gate` | not applicable to diagnostic |
| `effectiveness_gate` | preregistered；pending artifacts |
| `artifacts` | remote `stage_c_sc1_d3_crossed`；local `analysis/stage_c_sc1_d3_crossed_20260714/` |
| `decision` | pending；任何pass只返回Step 4 |

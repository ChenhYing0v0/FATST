# StageC D14-P Auxiliary Conditional Patch-Memory Probe

> [Superseded 2026-07-15] 用户指出ordered patch memory只属于Encoder–Decoder interface，不足以支撑
> multi-horizon paper mainline。该protocol未执行，现降为`auxiliary_not_scheduled`。active problem gate已改为
> `stage-c-d14-output-coupling-granularity.md`。以下内容仅保留为未来interface ablation设计，不再决定paper slots。

## Status

| Field | Value |
| --- | --- |
| `stage` | `StageC-UVHF` |
| `current_step` | deferred auxiliary interface probe |
| `role` | `auxiliary_diagnostic_only` |
| `active_candidate` | none；historical `SC1-CADMO` rejected by narrative scope |
| `method_training` | false |
| `remote_training` | false；先local artifact/probe implementation |
| `test_access` | false |
| `effectiveness_claim` | prohibited；frozen representation conditional diagnostic |
| `rollback` | not applicable；不再是paper-mainline gate |

## What We Plan To Test

给定A6 natural baseline的完整patch memory与compressed coefficient state：

$$
M[B,C,P,D]\rightarrow g[B,C,256]\rightarrow\hat Y[B,C,T],
$$

D14检验：

> 在已经知道$g$与A6 global prediction后，$M$是否仍包含可在chronological validation泛化的、与future
> target相关的增量预测信息？

D14不训练CADMO或CPGA，也不比较新decoder与A6的paper-core performance。

## Why It Matters

A6的`flatten`是bijective reshape；真正可能的压缩发生在`PD -> 256`。如果$g$对future已经近似充分，则
patch-level direct path只会增加capacity与optimization burden，CADMO主线应在Step 2关闭。

如果$M$在给定$g$后仍有跨dataset conditional gain，则才有理由进入Step 4-6，设计同时保留global coherence
与target-specific patch access的dual-memory operator。

## Representation And Target Construction

### Frozen checkpoints

- datasets：ETTh1、ETTh2、ETTm1、ETTm2、Weather；
- seeds：2021、2022、2023；
- profiles：`configs/stage_c_five_dataset_natural_profiles.json`；
- checkpoint：各run的best-validation A6 checkpoint；
- model parameters：全部冻结；
- train split：只用于fit probes与fit-only normalization；
- validation split：唯一final decision surface；
- test：禁止读取。

### Exported tensors

每个sample必须导出或流式提供：

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `patch_memory` | `[B,C,P,D]` | A6 Encoder输出、flatten前的ordered patch memory |
| `flat_memory` | `[B,C,PD]` | `patch_memory`的bijective reshape |
| `global_coeff` | `[B,C,256]` | A6 global compressed state $g$ |
| `a6_pred` | `[B,C,T]` | A6 global prediction |
| `label` | `[B,C,T]` | future target |
| `residual` | `[B,C,T]` | `label - a6_pred` |
| `target_coord` | `[T,*]` | fixed, horizon-agnostic future coordinate descriptor |

必须记录dataset/profile/checkpoint/split hashes、effective dtype、sample counts与shape contract。

## D14-A：Linear Incremental Sufficiency

### Arms

1. `A0_ZERO`：不修正A6；
2. `A1_GLOBAL_RIDGE`：`global_coeff -> residual`；
3. `A2_FLAT_RIDGE`：`flat_memory -> residual`；
4. `A3_PARTIAL_MEMORY_RIDGE`：先partial-out `global_coeff`，再用`flat_memory`预测剩余residual；
5. `A4_RANDOM_FEATURE_MATCHED`：与A3同output/parameter scale的random features；
6. `A5_SAMPLE_PATCH_SHUFFLE`：每sample独立打乱patch order；
7. `A6_TARGET_SHIFT`：memory与future coordinates错位；
8. `A7_D2_FULL_AFFINE_REFERENCE`：复核已有full-affine capacity evidence。

所有regularization只能在train内部chronological sub-split选择；不得读取validation进行arm-specific tuning。

### Statistics

对dataset $d$、seed $s$、horizon/bin $h$：

$$
\operatorname{Gain}_{a,d,s,h}
=
\frac{\operatorname{MSE}(\hat Y_{A6})-
\operatorname{MSE}(\hat Y_{A6}+\hat r_a)}
{\operatorname{MSE}(\hat Y_{A6})}.
$$

同时报告：

- validation MSE/MAE；
- residual $R^2$；
- global-vs-memory incremental gain；
- short/mid/long distance bins；
- seed direction count；
- train-validation generalization gap；
- effective rank与condition number。

## D14-B：Structured Target-Memory Interaction

### Purpose

D14-A若只有full-affine收益，可能只是更多自由参数。D14-B检验ordered patches与future targets之间是否存在
超越generic nonlinearity/capacity的结构化interaction。

### Probe contract

probe只读取冻结`patch_memory/global_coeff`，规模小、共享future coordinates、无future-query self-attention。
每个$q_\tau$独立读取history memory，requested $H$只裁剪$q_{1:H}$。

### Arms

1. `B0_COEFF_MLP`：global state nonlinear control；
2. `B1_FLAT_MLP_MATCHED`：generic flattened-memory nonlinear control；
3. `B2_ORDERED_PATCH_QUERY`：future-coordinate query读取ordered patches；
4. `B3_SAMPLE_PERMUTED_PATCH_QUERY`：每sample独立permutation；
5. `B4_RANDOM_QUERY`：固定random future descriptors；
6. `B5_TARGET_SHIFT_QUERY`：target-coordinate shift；
7. `B6_NO_MEMORY_PARAM_MATCHED`：相同parameter scale但无memory；
8. `B7_CATS_LIKE_REFERENCE`：source-informed direct-query primitive reference，不作为本地创新。

### Mandatory implementation controls

- same initialization class；
- same optimizer/epochs/early stopping；
- same train/validation tensors；
- no per-dataset arm-specific hyperparameter tuning；
- params差异不参与选型，但完整报告params/FLOPs；
- per-sample permutation必须破坏temporal identity，不能使用全局固定permutation被probe重新学习；
- target shift必须保持marginal distribution而破坏alignment。

## Gate

单dataset pass需：

1. `B2_ORDERED_PATCH_QUERY`相对`B0_COEFF_MLP` validation MSE gain为正；
2. 相对`B1_FLAT_MLP_MATCHED`仍为正；
3. 优于B3/B4/B5/B6 controls；
4. 至少2/3 seeds同方向；
5. train gain不能由明显validation reversal解释。

总体pass需：

1. 至少3/5 datasets pass；
2. five-dataset macro MSE gain至少`0.5%`；
3. 任一dataset不得出现超过`5%`严重退化；
4. D14-A/B全部shape、split、hash、fit-only、finite invariants通过。

## Decision Matrix

| D14-A | D14-B | Decision |
| --- | --- | --- |
| fail | fail | 当前A6-memory conditional-headroom route关闭；rollback Step 2 |
| fail | pass | 只支持structured nonlinear interaction；CADMO问题需在Step 2重述 |
| pass | fail | 存在generic增量capacity，但target-specific patch mechanism不被支持 |
| pass | pass | CADMO只进入formal Step 4-6；method/remote/test仍false |
| invalid | any | 修复diagnostic；不得方向级否决 |

## Failure Attribution Boundary

D14是frozen representation conditional probe，允许判断：

- A6现有$M$相对$g$是否有可访问conditional information；
- ordered patch-target interaction是否超越probe controls。

D14不允许判断：

- CADMO end-to-end effectiveness；
- 新Encoder是否能创造更好的patch memory；
- frozen A6 representation不适配新head时，整个方向无效；
- positive probe可替代matched end-to-end training。

若出现divergence、>100% degradation、validation reversal或ill-conditioning，只能标记
`optimization_or_numeric_pathology`或`diagnostic_invalid_for_direction_rejection`。

## Required Artifacts

1. effective protocol/config JSON；
2. checkpoint/profile/split hashes；
3. tensor shape/statistics manifest；
4. train-only preprocessing/selection record；
5. per-cell and dataset-seed summary CSV；
6. arm parameter/FLOP table；
7. short/mid/long interval table；
8. MSE/MAE/residual-R2/generalization-gap plots；
9. permutation/target-shift invariant report；
10. Chinese research interpretation；
11. failure attribution与11-step decision。

## Source Boundary

外部primary sources定义mandatory controls；Zotero coverage不用于novelty判断：

- CATS：https://proceedings.neurips.cc/paper_files/paper/2024/file/cf66f995883298c4db2f0dcba28fb211-Paper-Conference.pdf
- CATS official code：https://github.com/dongbeank/cats
- BasisFormer：https://papers.nips.cc/paper_files/paper/2023/hash/e150e6d0a1e5214740c39c6e4503ba7a-Abstract-Conference.html
- MQTransformer：https://openreview.net/forum?id=rxF4IN3R2ml
- TimePerceiver：https://arxiv.org/abs/2512.22550
- Memory Guided Transformer：https://www.vldb.org/pvldb/vol18/p239-cheng.pdf
- TimeCapsule：https://doi.org/10.1145/3711896.3737157
- CIB-MTSF：https://www.ijcai.org/proceedings/2025/627

完整复盘与主线设计见
`analysis/stage_c_fixed_past_mainline_reset_20260715/fixed_past_mainline_reconstruction.md`。

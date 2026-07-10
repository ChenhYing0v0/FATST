# Paper Mainline

本文档记录当前论文主线。旧 StageA 候选、诊断和中间 rollback 细节已从主线文档移出，保存在
`docs/archive/phase5-stage-a/` 与 `analysis/`。

## Current Position

| Field | Content |
| --- | --- |
| `paper_target` | 高水平 SCI 期刊时间序列预测论文 |
| `working_title` | Horizon-Agnostic Supervision Scheduling for Unified Multi-Horizon Forecasting |
| `current_stage` | Phase5 StageA clean A6 validated；StageB post-B14 C0 carrier/protocol audit |
| `active_carrier` | `A6-LBF-r256`；hierarchical patch memory is diagnostic-only |
| `active_stage_ledger` | `docs/stage-ledgers/phase5-timealign-interface.md` |
| `current_11_step` | B14-FURD Step 3 blocked；rollback Step 2/3；C0 Encoder control remotely running |
| `paper_core_status` | A6-LBF-r256 pure operator 是当前唯一 accepted paper-core method；StageB 第二贡献仍未成立 |

## Core Claim

[Claim] A6-LBF-r256 将 TimeAlign 的 final prediction head 改写为 prefix-native learned-basis forecast
operator。它用一个 unified 720-step model 覆盖 96/192/336/720 多个 prediction horizons，并在当前
实验集合上整体优于 fixed-horizon per-horizon TimeAlign。

[Boundary] A6-LBF-r256 不是强意义上的 target-set-native multi-horizon architecture。它更准确地说是
prefix-compatible learned-basis trajectory operator：模型生成同一条 720-step future trajectory，再按
requested horizon 返回 prefix。StageB 已依次排查显式 target-set/stage conditioning、basis-conditioned
coefficient field、subspace-tiled basis operator，以及 GRU-based future-unit composition；当前仍未形成第二个
accepted paper-core method。

## Main Contribution Draft

### Contribution 1: Learned-Basis Unified Forecast Operator

A6-LBF-r256 的机制：

- history encoder 输出 `hidden: [B, C, R]`;
- `learned_basis_coeff(hidden)` 生成 per-channel forecast coefficients；
- `learned_temporal_basis[:H]` 根据 requested horizon `H` 选择 prefix-native temporal basis；
- 输出 `prediction: [B, H, C]`；
- 不依赖 dense-row anchor、teacher、EMA、nested residual、target-query path 或 future-recon-branch。

论文叙事边界：

- 这是一个 unified multi-horizon architecture contribution；
- 它直接挑战 fixed-horizon per-horizon 训练的必要性；
- 更精确地说，它是 learned temporal-coordinate / low-rank trajectory operator，而不是
  target-set-conditioned forecaster；
- 它不是 early-stop、best-val、teacher distillation 或手工 horizon routing。

### Contribution 2 Candidate: Prefix-Native Objective

StageB 尚未成为正式贡献。B1/B3 reliability route 已证明 raw future-unit weighting 会被
forecast-distance confounder 污染，不能作为 method implementation。

TimeAlign dependency route 的最新结论是：

- artifact-only dependency audit 显示 A6-LBF 在 same TimeAlign align/recon setting 下相对 official
  unified TimeAlign 有 `11/12` MSE wins，mean MSE `-1.94%`；
- no-align/no-recon dependency ablation 显示纯 head/operator arm `no_align_no_recon` 相对 current
  A6-LBF mean MSE 仅 `+0.07%`，且有 `7/12` MSE wins；
- `align_no_recon` 的 mean MSE 略好 `-0.04%`，但 effect size 太小，不能单独支撑一个新的
  basis-aware alignment 方法。

因此 Contribution 1 的 head/operator 证据已经更强：A6-LBF-r256 不只是 inherited TimeAlign
alignment/reconstruction 的 artifact。当前代码也已将 A6-LBF 收束为 pure learned-basis forecast
operator：official TimeAlign baseline 保留 future reconstruction/alignment，A6-LBF 不再包含该 branch
或对应 auxiliary losses。

因此曾经提出的 Contribution 2 候选问题是：

> A6-LBF-r256 已经把 prediction head 改成 learned-basis coefficient space；训练目标是否也应该从
> generic time-domain point loss / generic auxiliary loss，转成与 prefix-native label autocorrelation
> 和 learned-basis residual 结构一致的 objective？

但 B6-PLO Step 2/3 diagnostic 已返回负证据：

- train-label PCA top32 与 DCT top32 几乎相同：ETTh2 `0.917/0.889`，ETTm1 `0.939/0.930`，
  Weather `0.832/0.831`；
- A6 learned basis top32 弱于 DCT：label coverage 为 ETTh2 `0.675`、ETTm1 `0.690`、Weather
  `0.251`，residual coverage 为 ETTh2 `0.287`、ETTm1 `0.110`、Weather `0.081`。

StageB 当前不得实现 prefix-native objective。该方向容易退化为 generic low-frequency/frequency auxiliary
loss，难以区别 FreDF/TransDF。B6 因此作为严谨的负诊断边界保留。

当前新的 StageB candidate 是 `B7-UPO`: unified prefix optimization。它不再问 label/basis 是否需要
frequency-like auxiliary objective，而是问 A6-LBF 的 unified forecast operator 是否被 nested
multi-prefix objective 公平、稳定地优化。初步 offline diagnostic 显示，当前 `multi-prefix` loss 使 `0-96`
steps 获得 `336-720` tail steps 的 `14.39x` scalar supervision weight；segment-level A6 gains vs fixed
TimeAlign 从 early `-3.57%` 收窄到 tail `-0.16%`。该方向与 StageA 的 unified prediction 叙事更连贯，
但仍只是 `problem_candidate`：Weather 是反例，且还缺少 gradient/task-interference evidence。

根据用户对 StageB 主贡献的约束，B7 当前降级为 small objective candidate。随后提出的 architecture
candidate 是 `B8-FQA`: Future-Query Aligned Basis Operator。它的核心问题是：A6-LBF 已有
prefix-native learned-basis decoder，但 sample-specific coefficient vector 对 future positions 是不变的；
StageB 可以引入 future-position query/placeholder tokens，在进入 basis operator 前生成
target-position-aware coefficient modulation。该方向更适合作为第二个主创新点，因为它改变 representation
interface，而不是只改 loss。该判断已补充外部网络调研证据：TimeAlign、ElasTST、TimePerceiver 的 arXiv
或 official repository 资料分别支撑 future alignment、future placeholders/masks 与 target-query
decoder 的机制可行性；SRP++ 仅作为本地 note 辅助证据。

`B8-OCD` coefficient-space oracle diagnostic 已返回负向控制结果：learned basis 的 segment-specific
correction 相比 global correction 有明显 headroom，但 DCT control 的绝对 residual reduction 更强。Rank
64 下 learned basis 的 segment reduction 为 ETTh2 `79.05%`、ETTm1 `72.77%`、Weather `61.91%`，而
DCT control 为 ETTh2 `87.61%`、ETTm1 `91.85%`、Weather `91.18%`。因此当前证据不足以说明 B8
对应的是 A6 learned-basis coefficient interface 特有的 architecture problem，StageB 不能实现 B8-FQA，
应回到 Step 2/3 重新定义 architecture-level 第二贡献问题。

用户明确排除 residual-style architecture 作为 paper-core route。因此新的 StageB candidate 是
`B9-FSN-SCF`: Stage-Native Coefficient Field。它不做 `y=A6(x)+correction`，而是让 future stage
信息在 basis projection 前进入 primary coefficient/operator path。`B9-SGC` stage-gradient diagnostic
已给出正向问题证据：四个 future stage losses 对同一个 A6 `coeff[b,c]` 的梯度方向相似度很低，mean
pairwise cosine 为 ETTh2 `0.072`、ETTm1 `0.171`、Weather `0.048`，early-tail cosine 为
`0.041/0.112/0.014`。这说明 single coefficient state 同时服务多个 future stages 时存在 native stage
pressure。

Step 4-6 设计门已通过：B9-FSN-SCF 将 A6 的 `coeff: [B,C,K]` 扩展为
`coeff_field: [B,C,S,K]`，其中 `S=4` 对应当前 multi-prefix stages；每个 stage 在 prediction 前生成
自己的 coefficient field，再与同一组 `learned_temporal_basis` 做 projection。该设计通过 zero-gated
multiplicative coefficient modulation 保持 A6 function-preserving fallback。

Step 7 最小实现与本地 smoke 已通过：`stage-native-coefficient-field` 与
`stage-native-coefficient-field-no-stage` 均可训练/评估；B9/no-stage 对 A6 的初始 fallback max abs 为
`0.0`，`H=96` 与 `H=720` prefix consistency max abs 也为 `0.0`。B9 仍未成为 accepted method；下一步
只能做 remote small gate，比较 `a6_clean`、`b9_fsn_scf`、`b9_no_stage`。

Remote small gate 已返回负向机制结果：`b9_fsn_scf` 相对 `a6_clean` 有 `12/12` MSE wins、mean MSE
`-0.13%`，但 `b9_no_stage` 相对 `a6_clean` 同样有 `12/12` wins、mean MSE `-0.13%`；`b9_fsn_scf`
相对 `b9_no_stage` 只有 `2/12` wins，mean MSE `+0.0036%`。因此 B9-FSN-SCF 被 no-stage control
阻断，不能作为 accepted method，也不能将相对 A6 的微弱收益解释为 future-stage-aware routing。

B9 之后，StageB 回滚到更根本的问题：A6 当前仍是 `f(history) -> y_{1:720}` 后 prefix slicing，而不是
`f(history, J) -> y_J`。新的候选是 `B10-TCO`: Target-Set Conditioned Operator。B10 不再把 stage token
塞进已有 coefficient，而是研究 requested target set $J$ 是否应进入 basis-coeff prediction graph。默认立场是
prefix-invariant target-set computation：`H=720` 的后续 target positions 不应改写 `H=96` 的 prefix outputs。

B10-TSI-A 已完成 checkpoint-only basis geometry audit。结果显示 A6 的 basis 不是 stage-blind：top64 atoms
主要跨 stage，但不同 stage 的 coefficient row spaces overlap 很低。因此 StageB 不能叙事为“给 basis 补
stage 信息”。更准确的论文问题是：`learned_temporal_basis` 已经形成 stage-differentiated coefficient
geometry，但 `learned_basis_coeff(hidden)` 仍然只生成 target-set-blind coefficient/state；requested target set
没有进入 `history -> coeff/state` 生成路径。

B10-TSI-B 进一步显示，真实 A6 `coeff` 同时激活多个低同向性 stage row subspaces；rank64 下三数据集
projection share 为 `0.3882/0.4950/0.2764`，projection cosine 为 `0.3759/0.4702/0.1639`，output
entropy 为 `0.7969/0.8958/0.9042`。这支持继续诊断 target-set interface，但仍不能直接实现方法。

B10-TSI-C 返回了负向且病态的 frozen-coeff linear readout 结果：相对 pooled 4-head no-target control，
ETTh2 为 `-185.5316%`，ETTm1 仅 `+0.2812%`，Weather 为 `-26.5683%`。该结果不能否定
target-set-aware 方向；它只说明 `frozen coeff -> Linear_s(coeff)` 这个信息介入位置太晚、readout/head
过线性且存在数值病态风险。

B10-TSI-D 已进一步做 failure attribution：比较 `coeff_late`、`memory_pool`、`memory_plus_coeff`
三种 feature sources，并加入 rank-truncated row-space target、wrong-target control 和 shrinkage target-set
readout。rank64 stabilized target vs pooled control 为 `-12.3695%/-40.7499%/-44.4687%`；rank16
稳定性对照仍为 `-5.1506%/-32.0345%/-36.3672%`。因此当前结论是：
`offline_readout_route_blocked_but_direction_not_rejected`。

StageB 不能继续用 frozen/offline oracle 反复否定 B10。该节点原本只允许转入 native trainable
target-query memory readout 的 Step 4-6 gate，或 rollback；但后续用户进一步排除了显式
stage/target-set conditioning 作为主线。

基于用户对 unified model 叙事的进一步约束，StageB 现在不继续显式 stage/target-set conditioning。
新的候选是 `B11-ESA`: Emergent Subspace Aggregation。它的问题是：A6 的 `learned_temporal_basis`
已经自发形成 future geometry，是否能让架构更自然地利用这种 geometry，而不是输入 hard `stage_id`
或 `horizon_id`。

B11 basis/coeff diagnostic 已返回正向 problem evidence。Hard KMeans basis-row clusters 只在 ETTh2
上明显，不能支持 hard cluster/stage method；但 sliding-window subspace geometry 更稳定：adjacent/far
subspace overlap 为 ETTh2 `0.3900/0.0649`、ETTm1 `0.4021/0.0811`、Weather `0.3810/0.0700`；
distance-overlap Spearman 为 `-0.7016/-0.5472/-0.2786`。真实 `coeff` 的 projection cosine 也随
window distance 降低：adjacent/far 为 `0.5585/0.1504`、`0.5391/0.2379`、`0.4071/0.0484`。

B11 的 Step 4-6 design gate 已通过，但只对 `B11-BCF` 成立。`B11-BCF` 是 continuous
basis-conditioned coefficient field：用 overlapping basis-window descriptors 生成 soft coefficient states，
再通过同一组 `learned_temporal_basis` 做 primary prediction。它不输入 hard `stage_id` 或
`horizon_id`，也不写成 `A6 + residual repair`。

当前 B11 仍不是 accepted contribution。B11-BCF local implementation 曾通过 smoke，但 required small
gate 返回后被 controls 阻断：`b11_bcf` 相对 A6 mean MSE `-0.1019%`、`5/12` wins；`b11_no_basis`
相对 A6 几乎相同，为 `-0.1007%`；`b11_constant_slot` 更好，为 `-0.1281%`。`b11_bcf` 相对
`b11_no_basis` 只有 `2/12` wins，mean MSE 仅 `-0.0012%`；相对 `constant_slot` mean MSE 反而
`+0.0263%`。

因此 B11-BCF 不能写成 basis-conditioned architecture contribution。论文主线当前只能保留 A6-LBF-r256
作为 accepted paper-core method；StageB 必须回到 Step 4 redesign 或 Step 2/3。

B11 之后，StageB 回到 Step 2/3 并打开 `B12-STBO`: Subspace-Tiled Basis Operator。该方向来自一个更
native 的 multi-horizon 问题：A6 当前是 `learned_temporal_basis[720,K]` 的 full-trajectory operator，
短 horizon 只是 prefix slicing；B12 询问是否能把它改成 stage/tile-local shared or banked basis
operator，使短 horizon 只启动必要 tiles。

B12 的叙事比 B11-BCF 更接近 primary operator redesign，因为它不是 residual，也不是
`coeff + delta`。但 Step 2/3 diagnostic 未通过 method-entry gate：A6 basis 的 `bank4` local basis
相对 local DCT 有弱优势，ETTh2/ETTm1/Weather 分别为 `+0.061/+0.054/+0.081`，但仍低于
independent-tile upper bound `0.067/0.068/0.083`；train-label tile structure 虽然很强，却几乎被
local DCT 解释；真实 A6 `coeff` 的 adjacent/far tile-subspace projection pattern 只在 ETTh2 明显。

因此 B12 的 Step 2/3 offline 初始结论是 `diagnostic_not_enough_for_method`：不能仅凭 A6-derived
basis/coeff evidence 把 shared/bank local-basis operator 写成 Contribution 2。该结果不是所有
basis-operator redesign 的方向级否定。

该边界随后被修正：A6-derived offline diagnostic 不能直接否定 native trainable STBO，因为它只观察已经
训练好的 A6 full-basis 解。StageB 因此允许一个严格控制的 trainable small gate，但必须包含 fixed local DCT
和 independent-tile controls。B12-STBO local implementation 已通过 smoke：新增
`subspace-tiled-basis-operator-shared`、`subspace-tiled-basis-operator-bank`、
`subspace-tiled-basis-operator-dct`、`subspace-tiled-basis-operator-independent` 四个 readout modes；
H96 与 H720 prefix consistency max abs 为 `0.0`，synthetic backward 和 ETTh2 one-batch CPU smoke 均通过。

B12 仍不是 accepted paper-core method。只有当 learned `stbo_shared` / `stbo_bank4` 在 remote small gate
中超过 `stbo_dct`，且收益不能仅由 `stbo_independent` capacity 解释时，才可能进入 StageB Contribution 2
评估。

截至 2026-07-08，B12 remote small gate 已返回但未通过。由于远程 `/home` quota 阻塞持久 repo 写入，
本次使用 `/tmp` clone 和 `/tmp` output root；这只影响执行位置，不改变实验矩阵或 paper claim boundary。
A6 在 `9/12` settings 上仍是 best arm；`stbo_shared` vs A6
为 `+1.59%` mean MSE 且 `0/12` wins，`stbo_bank4` vs A6 为 `+1.98%` 且只有 Weather 的三个短中 horizon
极小正向；learned STBO 没有超过 fixed local DCT，`stbo_bank4` 的 tile-bank entropy 约 `0.999`，说明
stage/tile bank specialization 未形成。因此 B12 当前实现不能进入 Contribution 2。

随后打开一个 diagnostic-only rank/capacity check：由于 A6 使用 `basis_rank=256`，而第一轮 STBO 只使用
`stbo_rank=16`，需要判断失败是否主要来自 local rank bottleneck。初始 `L96-R64` 配置无效，因为 `96`
不整除 `720`；修复后的有效矩阵为 `L48-R32`, `L120-R64`, `L144-R128`,
`L360-R256_capacity_probe`。该诊断不改变 B12 当前未通过的 paper claim 状态；只有当高 rank learned STBO
同时接近 A6、超过 same-rank DCT，并形成非均匀 bank specialization 时，B12 才能重新进入方法候选讨论。

rank/capacity diagnostic 返回后，结论是：rank bottleneck 部分成立，但 B12-STBO 当前方法仍不成立。
`L360-R256:stbo_independent` 几乎追平 A6（`+0.014%` mean MSE），说明低 rank 确实限制了第一轮 STBO。
但 paper-relevant 的 learned shared/bank 仍未超过 A6：最佳 `L360-R256:stbo_shared` 为 `+0.33%` mean MSE，
且 `stbo_bank4` 的 bank entropy 仍接近最大值。B12 因此不能进入 Contribution 2。

当前论文主线已落到 post-B12 restart：StageA 的 A6-LBF-r256 继续作为唯一 accepted paper-core method；
StageB 第二贡献保持 open，回到 Step 2/3 architecture search。新候选必须服务 unified prediction 主线，
并避免 residual repair、hard stage/horizon coding、以及容易退化为 auxiliary loss 的路线。重启入口文档为
`docs/stage-ledgers/phase5-stageb-restart-handoff-20260709.md`。

Post-B12 search 打开了 `B13-FUCO`，将用户提出的“future stage/segment generation 而不是 full-horizon
clipping”具体化为 benchmark-independent large future units。Diagnostic A 在 ETTh2/ETTm1/Weather 的
`120/144/180/240` unit sizes 上全部通过（`12/12`），证明 shared-state gradient pressure 不是 canonical
horizon boundary 或 small-unit artifact。

但 current GRU-based prefix-causal composition 没有通过 mechanism gate。Coefficient-memory B1 只支持
`3/6` settings；pre-coefficient hidden-memory B2 支持 `4/6`，但 ETTh2-U180/U240 仍平均退化
`+5.16%/+5.36%`。更关键的是正向 setting 没有 progressive-depth pattern：ETTm1-U240 最后一个 unit
平均退化 `+7.50%`，Weather-U240 最大收益发生在没有 previous-unit information 的 unit 0。正向 aggregate
gain 因而不能支撑 latent compositional context claim。

论文主线当前只关闭 `GRU-based prefix-causal composition`，不关闭 broader future-unit generation。
StageB 回到 Step 2，下一问题收束为：不同 large future regions 是否需要不同的 history retrieval/state，
并能否在没有 recurrent transition 与 full-horizon clipping 的情况下形成 native generator。该问题尚未通过
literature/problem existence gate，因此不得直接实现。

B14 前置 encoder reconstruction 已完成。直接用 PatchTST-derived contextual encoder替换 A6 carrier的两个
arms均失败：`P16-S8 +4.135% (1/12 wins)`，`P48-S24 +4.799% (0/12 wins)`。这关闭 full contextual
replacement，不否定 local patch memory。Step 5/6 repair改为 hierarchical interface：accepted A6 forecast path
保持，normalized history额外展开为 parameter-free valid `P48-S24` local memory `[B,C,29,48]`。

该 repair在 ETTh2、ETTm1、Weather 全部通过 strict equivalence：state keys与 parameter count相同，
multi-prefix outputs和 full-test MSE/MAE max diff均为 `0.0`。因此 A6性能结论无需修改，B14可以在统一 local
patch supports上执行 Step 3 demand-vs-sensitivity diagnostic；trainable retrieval仍未获授权。

Step 3启动前进一步移除了 initial side path的 right replication padding：29 个 patches都对应完整 48-step
history evidence；coverage-corrected aggregation保持 position attribution total mass。本修改不进入 forecast path。

B14 Step 3已完成并关闭当前 retrieval route。A1 current-gradient mismatch为 `0/6`；由于 A1 demand与
sensitivity共享 A6 Jacobian，追加一次 model-independent DCT-8 linear-CKA repair。A2仅 Weather-U180通过，
整体 `1/6`，没有 dataset的 U180/U240同时支持。结果为
`blocked_by_nonrobust_label_patch_evidence`：不得实现 trainable retrieval，回 Step 2/3。

下一问题回到最小 carrier 修改：ETTm1 inherited `patch_num=1` 是否应单独改为 `patch_num>1`。这必须作为
patch-only/capacity-controlled audit，不复活 full contextual encoder，也不把 diagnostic side path写入论文方法。

C0 carrier/protocol audit进一步收紧了该问题。ETTm1 unified A6同时继承 official H720 preset的
`patch_num=1,d_model=256,dropout=0.9,official-last`，而 fixed H96/H192使用
`d_model=128,dropout=0.2`。因此现有 A6 vs fixed结果是 source-faithful practical comparison，但不是完全
configuration-controlled architecture comparison。该边界不撤销 A6 accepted carrier；正式论文的强
architecture claim需要追加 matched configuration、dropout与 dual-checkpoint control。

Frozen checkpoint evidence反对“高 dropout使 Encoder失效”的解释：移除 ETTm1 residual MLP并保留
LayerNorm后，H96/H192/H336/H720 MSE分别恶化 `12.96%/9.32%/7.29%/5.58%`。进一步的 frozen
inclusion-exclusion diagnostic 在全部 8 个 attenuation-horizon settings 检测到 material cross-patch
interaction。因此当前六臂 `C0-ETTm1-CPA` 保持相同 flattened `C*P` semantics，分离 global width、
state/active capacity、dropout和 last/best selector；不在返回结果前增加 mixer。任何正向结果最多修复
carrier/protocol，不能成为 Contribution 2。

## Evidence Snapshot

### A6-LBF-r256 vs fixed-horizon per-horizon TimeAlign

Protocol: official-last；datasets: ETTh2 / ETTm1 / Weather；horizons: 96/192/336/720。

| Dataset | A6-LBF MSE wins | Mean MSE change |
| --- | ---: | ---: |
| ETTh2 | 4/4 | `-10.53%` |
| ETTm1 | 3/4 | `-1.64%` |
| Weather | 2/4 | `-0.22%` |
| Overall | 9/12 | `-4.13%` |

### A6-LBF-r256 vs official unified TimeAlign

| Dataset | A6-LBF MSE wins | Mean MSE change |
| --- | ---: | ---: |
| ETTh2 | 4/4 | `-2.78%` |
| ETTm1 | 3/4 | `-1.20%` |
| Weather | 4/4 | `-1.26%` |
| Overall | 11/12 | `-1.75%` |

### Clean A6 validation after removing future-recon branch

The clean rerun at `analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/` validates the active
implementation: `effective_w_recon=0.0`, `effective_w_align=0.0`, `readout_mode=learned-basis-forecast-operator`,
`basis_rank=256`, and `pred_loss_mode=multi-prefix`.

Relative to the historical A6-LBF-r256 artifact, the clean rerun changes mean MSE by only `+0.20%` overall
(`6/12` MSE wins). Therefore the future reconstruction/alignment branch removal improves contribution boundary
clarity without materially changing the accepted StageA evidence.

### A6-LBF-r256 no-align/no-recon dependency ablation

Protocol: official-last；datasets: ETTh2 / ETTm1 / Weather；horizons: 96/192/336/720。

| Arm | Mean MSE vs current | MSE wins vs current | Decision |
| --- | ---: | ---: | --- |
| `no_align_recon` | `+0.07%` | 7/12 | inherited align not required |
| `align_no_recon` | `-0.04%` | 8/12 | recon not required; tiny align benefit only |
| `no_align_no_recon` | `+0.07%` | 7/12 | pure A6-LBF operator remains competitive |

## Method Boundary

Accepted into current mainline:

- `official` TimeAlign baseline；
- `A6-LBF-r256` pure learned-basis forecast operator；
- official-last protocol；
- multi-prefix evaluation on 96/192/336/720。

Archived or inactive:

- A2/A3 nested decoders；
- A4 reliability diagnostics；
- A5 target-query / continuous fixed-basis designs；
- A6-DER capacity ceiling；
- A6-QBR query-bilinear readout；
- A6S/A6ST/A7DG/A8TAG stability and teacher routes；
- pre-cleanup B0 pressure ablation。

## Active Files

| File | Purpose |
| --- | --- |
| `baselines/timealign_official/models/TimeAlign.py` | clean official + A6-LBF model |
| `baselines/timealign_official/train_repo.py` | clean training and evaluation adapter |
| `scripts/remote/run_phase5_a6_lbf_r256_main.sh` | clean A6-LBF-r256 remote runner |
| `scripts/analyze_phase5_a6_clean_operator_rerun.py` | clean A6 validation analyzer |
| `docs/code-explanation/phase5-clean-timealign-a6-lbf.md` | code explanation |
| `docs/code-explanation/phase5-clean-a6-rerun-analysis.md` | clean A6 validation analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b9-fsn-scf.md` | B9-FSN-SCF implementation explanation |
| `docs/code-explanation/phase5-stage-b-b10-tsi-basis-geometry.md` | B10-TSI-A basis geometry analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b10-tsi-coeff-usage.md` | B10-TSI-B coefficient usage analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b10-tsi-target-set-oracle.md` | B10-TSI-C target-set oracle/control analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b10-tsi-failure-attribution.md` | B10-TSI-D failure attribution analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b11-esa-basis-coeff-diagnostic.md` | B11-ESA basis/coeff diagnostic analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b11-bcf.md` | B11-BCF implementation explanation |
| `docs/code-explanation/phase5-stage-b-b12-stbo-diagnostic.md` | B12-STBO diagnostic analyzer explanation |
| `docs/code-explanation/phase5-stage-b-b12-stbo.md` | B12-STBO model implementation explanation |
| `docs/code-explanation/phase5-stage-b-b12-stbo-rank-diagnostic.md` | B12-STBO rank/capacity diagnostic explanation |
| `docs/code-explanation/phase5-stage-b-b13-future-unit-granularity.md` | B13 large-unit problem diagnostic explanation |
| `docs/code-explanation/phase5-stage-b-b13-future-unit-composition-probe.md` | B13 parameter-matched composition probe explanation |
| `docs/stage-ledgers/phase5-stageb-restart-handoff-20260709.md` | post-B12 restart handoff for new conversations |
| `docs/stage-ledgers/phase5-stageb-b13-restart-handoff-20260710.md` | post-B13 Step 2 restart handoff |
| `docs/stage-ledgers/phase5-timealign-interface.md` | active StageA/StageB ledger |
| `docs/research-roadmap.md` | active roadmap |
| `docs/experiments/phase5-stage-b-prefix-native-label-objective-diagnostic.md` | B6 rejected objective diagnostic protocol |
| `docs/experiments/phase5-stage-b-unified-prefix-optimization-diagnostic.md` | active B7 unified prefix optimization diagnostic protocol |
| `docs/experiments/phase5-stage-b-future-query-aligned-basis-architecture.md` | B8 rejected architecture candidate protocol |
| `docs/experiments/phase5-stage-b-native-future-stage-operator.md` | B9-FSN-SCF blocked protocol |
| `docs/experiments/phase5-stage-b-target-set-conditioned-operator.md` | B10 target-set-conditioned operator protocol |
| `docs/experiments/phase5-stage-b-emergent-subspace-aggregation.md` | B11 emergent subspace aggregation protocol |
| `docs/experiments/phase5-stage-b-subspace-tiled-basis-operator.md` | B12 subspace-tiled basis operator protocol |
| `docs/experiments/phase5-stage-b-future-unit-compositional-operator.md` | B13 future-unit problem, controls and rollback protocol |
| `scripts/analyze_phase5_stage_b_b10_tsi_basis_geometry.py` | B10-TSI-A basis geometry analyzer |
| `scripts/analyze_phase5_stage_b_b10_tsi_coeff_usage.py` | B10-TSI-B coefficient usage analyzer |
| `scripts/analyze_phase5_stage_b_b10_tsi_target_set_oracle.py` | B10-TSI-C target-set oracle/control analyzer |
| `scripts/analyze_phase5_stage_b_b10_tsi_failure_attribution.py` | B10-TSI-D failure attribution analyzer |
| `scripts/analyze_phase5_stage_b_b11_esa_basis_coeff_diagnostic.py` | B11-ESA basis/coeff diagnostic analyzer |
| `scripts/analyze_phase5_stage_b_b12_stbo_diagnostic.py` | B12-STBO diagnostic analyzer |
| `scripts/analyze_phase5_stage_b_b12_stbo_rank_diagnostic.py` | B12-STBO rank/capacity diagnostic analyzer |
| `scripts/check_phase5_stage_b_b12_stbo_local.py` | B12-STBO local checker |
| `scripts/analyze_phase5_stage_b_b13_future_unit_granularity.py` | B13 large-unit granularity analyzer |
| `scripts/analyze_phase5_stage_b_b13_future_unit_composition_probe.py` | B13 coefficient/hidden-memory composition analyzer |
| `scripts/check_phase5_stage_b_b11_bcf_local.py` | B11-BCF local fallback/prefix/backward checker |
| `scripts/remote/run_phase5_stage_b_b11_bcf_small_gate.sh` | B11-BCF remote small gate runner |
| `scripts/sync_phase5_stage_b_b11_bcf_small_gate_results.sh` | B11-BCF remote artifact sync/analyze wrapper |
| `scripts/analyze_phase5_stage_b_b11_bcf_small_gate.py` | B11-BCF small gate analyzer |
| `scripts/remote/run_phase5_stage_b_b9_fsn_scf_small_gate.sh` | B9-FSN-SCF remote small gate runner |
| `scripts/analyze_phase5_stage_b_b9_fsn_scf_small_gate.py` | B9-FSN-SCF small gate analyzer |
| `scripts/sync_phase5_stage_b_b9_fsn_scf_small_gate_results.sh` | B9-FSN-SCF result sync/analyze wrapper |
| `analysis/phase5_stage_b_timealign_dependency_audit_20260706/` | TimeAlign dependency audit |
| `analysis/phase5_stage_b_timealign_dependency_ablation_20260706/` | no-align/no-recon dependency ablation |
| `analysis/phase5_stage_b_prefix_native_objective_diagnostic_20260706/` | B6 negative diagnostic |
| `analysis/phase5_a6_lbf_r256_clean_operator_rerun_20260706/` | clean A6 validation report |
| `analysis/phase5_stage_b_unified_prefix_optimization_20260707/` | B7 problem-candidate diagnostic |
| `analysis/phase5_stage_b_future_query_aligned_architecture_research_20260707/` | B8 architecture direction research |
| `analysis/phase5_stage_b_b8_ocd_coefficient_oracle_20260707/` | B8-OCD negative oracle diagnostic |
| `analysis/phase5_stage_b_b9_stage_gradient_diagnostic_20260707/` | B9-SGC positive problem-candidate diagnostic |
| `analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/` | B9-FSN-SCF launch record and future small-gate analysis |
| `analysis/phase5_stage_b_b10_tsi_basis_geometry_20260708/` | B10-TSI-A basis geometry diagnostic |
| `analysis/phase5_stage_b_b10_tsi_coeff_usage_20260708/` | B10-TSI-B coefficient usage diagnostic |
| `analysis/phase5_stage_b_b10_tsi_target_set_oracle_20260708/` | B10-TSI-C target-set oracle/control diagnostic |
| `analysis/phase5_stage_b_b10_tsi_failure_attribution_20260708/` | B10-TSI-D rank64 failure attribution diagnostic |
| `analysis/phase5_stage_b_b10_tsi_failure_attribution_rank16_20260708/` | B10-TSI-D rank16 stability control |
| `analysis/phase5_stage_b_b11_esa_basis_coeff_diagnostic_20260708/` | B11-ESA basis/coeff diagnostic |
| `artifacts/smoke_phase5_stage_b_b11_bcf_local/b11_bcf_etth2/` | B11-BCF local ETTh2 smoke |
| `analysis/phase5_stage_b_b11_bcf_small_gate_20260708/launch_record.md` | B11-BCF remote launch record |
| `analysis/phase5_stage_b_b11_bcf_small_gate_20260708/b11_bcf_small_gate_report.md` | B11-BCF small gate decision |
| `analysis/phase5_stage_b_b12_stbo_diagnostic_20260708/b12_stbo_report.md` | B12-STBO Step 2/3 diagnostic decision |
| `analysis/phase5_stage_b_b12_stbo_rank_diagnostic_20260708/b12_stbo_rank_diagnostic_report.md` | B12-STBO rank/capacity diagnostic decision |
| `analysis/phase5_stage_b_b12_stbo_rank_diagnostic_20260708/b12_stbo_rank_deep_analysis.md` | B12-STBO rank/capacity deep analysis |
| `analysis/phase5_stage_b_b13_future_unit_granularity_20260710/` | B13-A large-unit problem evidence |
| `analysis/phase5_stage_b_b13_future_unit_composition_20260710/` | B13-B1 coefficient-memory no-transition control |
| `analysis/phase5_stage_b_b13_future_unit_hidden_composition_20260710/` | B13-B2 hidden-memory final repair and rollback decision |

## Next Step

1. Treat StageA clean A6-LBF-r256 as fixed.
2. Do not revive archived StageA code paths.
3. Treat B5 basis-aware alignment as deferred, not the next implementation target.
4. Do not implement B6 objective under current evidence.
5. Defer B7 objective optimization as a small contribution candidate.
6. Do not implement B8-FQA under current evidence.
7. Do not launch B9-FSN-SCF full matrix.
8. Do not continue explicit stage/horizon conditioning as the main StageB route.
9. B11-BCF is blocked by no-basis and constant-slot controls; do not claim it as paper-core.
10. B12-STBO is blocked by rank/capacity diagnostic; do not continue it as paper-core or full matrix.
11. B13 GRU-based prefix-causal composition is blocked by no-transition control; do not continue GRU/head tuning.
12. Use the exact A6-preserving hierarchical `P48-S24` memory as B14's common history interface.
13. B14-FURD failed its Step 3 gate (`A1 0/6`, `A2 1/6`)；do not implement trainable retrieval.
14. Roll back Step 2/3 and test only the minimal ETTm1 `patch_num=1 -> >1` carrier question with capacity controls.

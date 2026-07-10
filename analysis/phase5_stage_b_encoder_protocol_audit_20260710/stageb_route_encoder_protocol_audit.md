# StageB 路线、ETTm1 Encoder 与训练协议审计

## Executive Decision

| Field | Decision |
| --- | --- |
| `current_step` | B14-FURD Step 3 已失败；StageB 回滚 Step 2/3。本报告是 carrier/protocol diagnostic，不是新 method |
| `gate` | 判断 `patch_num=1` 是否值得作为 immediate carrier audit；判断 Encoder/dropout/checkpoint 是否足以动摇当前研究结论 |
| `rollback` | 若 controlled patch audit失败，关闭 ETTm1 `patch_num` performance defect假设，StageB回到 Step 2/3 或暂停第二贡献搜索 |
| `active_carrier` | `A6-LBF-r256` 保留；其 source-faithful empirical evidence有效，但 architecture-only superiority claim需要 controlled protocol补证 |
| `paper_core` | StageA A6仍是唯一 accepted paper core；StageB 第二贡献仍未成立 |
| `next_action` | 只做 ETTm1 minimal carrier/protocol audit；不实现 retrieval、future query、GRU composition或新 routing |

核心判断：

1. [Fact] StageB 一开始要解决的不是 `patch_num`，而是 A6 unified carrier 上的 future-aware
   supervision/reliability allocation问题；后续证据连续否定了 reliability、generic objective、显式
   stage/target conditioning、basis subspace与 GRU composition的具体实现，路线才逐步转向 carrier/retrieval
   前提审计。
2. [Decision] ETTm1 `patch_num=1` 需要立即做一次**有边界的小型因果审计**，但不应成为 StageB 的论文
   主问题。它目前是 protocol/carrier confounder，不是已证实的 architecture defect。
3. [Fact] `P=1` 不是“Encoder不看局部时间信息”或“没有时序建模”；它把完整 720-step history一次投影到
   256-dimensional global token。但它确实没有 local patch axis，不能支持 B14 那种 patch retrieval解释。
4. [Strong Evidence] frozen A6 checkpoint中，移除 ETTm1 residual MLP后四个 horizons 的 MSE恶化
   `5.58%-12.96%`。`dropout=0.9` 没有把 Encoder branch变成空路径。
5. [Decision] 当前更严重的科学问题不是“大 dropout一定错误”，而是 unified A6、fixed baselines与不同
   horizons同时使用不同的 `P/D/d_ff/dropout`，而且主结果只报告 `official-last`。因此现有结果可称为
   source-faithful practical comparison，尚不能称为完全 configuration-controlled architecture comparison。

## 一、StageB 原始问题与路线变化

### 两层原始问题

[Fact] 论文层面的长期目标一直是：

```text
one model for multi-horizon forecasting
  + future-aware architecture/supervision
  + possible conditional computation
```

[Fact] StageA 先解决 blocking interface：能否不用四个 fixed-horizon heads，而用一个 unified operator覆盖
`96/192/336/720`。A6-LBF-r256 给出的答案是：

```text
history -> hidden [B,C,R]
        -> coeff [B,C,256]
        -> learned temporal basis[:H]
        -> prediction [B,H,C]
```

[Fact] A6 接受之后，StageB 最初的具体研究问题是：

> 不同 future units 的难度/可靠性是否存在可学习、非 forecast-distance 的结构差异，从而可以用
> future-aware supervision allocation 提升 unified model 的稳定性与论文机制深度？

因此 StageB 最初是 `future-aware supervision / reliability allocation`，不是 patch encoder研究，也不是
“必须再造一个 prediction head”。

### 路线变化表

| 路线阶段 | Candidates | 当时研究的问题 | 证据后的变化 |
| --- | --- | --- | --- |
| future-unit reliability | B1-B3 | future units是否存在非 distance-confounded reliability，可否动态分配 supervision | raw difficulty几乎被 forecast distance解释；seasonal residual不够 robust，停止 loss weighting |
| carrier dependency/objective | B4-B7 | A6是否依赖 TimeAlign align/recon；basis-native operator是否需要 basis/prefix-native objective | no-align/no-recon基本不伤 A6；basis结构被 DCT解释；prefix optimization只够 small contribution |
| target/stage-aware architecture | B8-B10 | A6 single coeff是否缺少 future-position/target-set-aware state | oracle受 DCT或 pooled controls解释；offline readout出现 pathology；不再继续显式 stage/horizon coding |
| emergent subspace/operator redesign | B11-B12 | 能否用 learned basis geometry或 tiled local basis产生 native future structure | no-basis/constant-slot controls解释 B11；B12 rank恢复 capacity但 shared/bank mechanism不成立 |
| large future-unit generation | B13 | requested horizon是否应决定生成多少 large units，而不是先生成 720再 clipping | large-unit gradient evidence `12/12`通过，但 GRU composition被 no-transition control阻断 |
| future-region-specific history retrieval | B14 | 不同 U180/U240 future regions是否需要不同 local history evidence | contextual replacement失败；exact HPM解除前提后，A1 `0/6`、A2 `1/6`，问题不跨数据集 robust |
| carrier/protocol audit | 当前 | ETTm1 `P=1`、大 dropout和 last selector是否污染 carrier与既有诊断 | 只授权 minimal controlled audit，不授权新 StageB method |

### 为什么会显得“混乱”

[Diagnosis] 路线并不是没有记录的随机试错；B1-B14 大多有 gate与 control。但有三个认知层级经常被混在
一起：

1. **paper problem**：unified model如何 native 地处理 requested future structure；
2. **candidate mechanism**：reliability weighting、stage field、basis bank、GRU、retrieval；
3. **carrier prerequisite**：A6 encoder tokenization、capacity、dropout、checkpoint policy。

当前 `patch_num` 属于第 3 层。它是在 B14 诊断前提中暴露的，不应反过来被写成 StageB 最初想研究的
paper problem。

[Decision] StageB 后续必须保持以下顺序：

```text
carrier/protocol validity
  -> stable cross-dataset problem evidence
  -> Step 4-6 narrative/method gate
  -> implementation
```

不能因为修正 `patch_num` 后某个 ETTm1 metric提高，就把 patching升级成 Contribution 2。

## 二、ETTm1 `patch_num=1` 到底是什么

### Current Tensor Path

legacy A6 当前执行：

```text
x [B,720,C]
  -> Normalize
  -> x.permute(0,2,1).reshape(B,C*720)
  -> unfold(patch_len=720/P, stride=720/P)
  -> patch projection
  -> tokens [B,C*P,D]
  -> token-wise residual MLP x 2
  -> memory [B,C,P,D]
  -> hidden [B,C,P*D]
  -> coeff [B,C,256]
  -> basis[:H] projection
```

当前 official-720 presets：

| Dataset | `P` | patch length | `D` | hidden `P*D` | `d_ff` | dropout | active A6 params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 48 | 15 | 32 | 1536 | 32 | 0.1 | 583,376 |
| ETTm1 | 1 | 720 | 256 | 256 | 256 | 0.9 | 699,600 |
| Weather | 48 | 15 | 128 | 6144 | 128 | 0.5 | 1,826,256 |

[Fact] ETTm1 `P=1` 对每个 channel形成 `[720 -> 256]` 的 global linear projection，再执行两层 residual
MLP和 LayerNorm。它缺少 local tokens，但不是 input被压成一个 scalar，也不是 rank-1 model。

[Fact] ETTm1 active parameter count并不比 ETTh2 小；它比 ETTh2高约 `19.9%`。因此“ETTm1失败因为
`P=1` 容量太小”不是现有证据支持的说法。真正较小的是 downstream history state width：`256` vs
ETTh2 `1536`、Weather `6144`。

### 为什么仍值得审计

1. B14 要讨论 local history evidence，`P=1` 没有可比 local axis；
2. `P=1` 把所有 temporal information先压入单个 256-dimensional state，局部结构必须在这个压缩前后被
   隐式编码；
3. ETTm1 与 ETTh2/Weather的 tokenization、width、dropout一起变化，任何跨数据集差异都可能是 preset
   interaction；
4. A6 已移除 TimeAlign future reconstruction/alignment branch，因此继续无条件继承 upstream
   `patch_num=1` 缺少机制上的必然性。

### 为什么不能直接判为重点 paper problem

1. clean A6 在 ETTm1 相对 fixed TimeAlign仍有 `3/4` wins、mean MSE `-1.64%`；没有性能 collapse；
2. PatchTST-derived full contextual replacement已经失败：P16-S8在 ETTm1为 `+7.40%`，P48-S24为
   `+4.07%`。这证明“更多 patches自然更好”不成立；
3. 增大 `P` 会同时改变 patch length、weight sharing、hidden width、coefficient head参数和 positional
   semantics；未经 controls 的 patch sweep不可解释；
4. 即使只在 ETTm1有效，也更像 dataset-specific carrier tuning，不具备跨数据集 SCI architecture
   contribution边界。

[Decision] 优先级应标为：

```text
high as a short carrier/protocol blocker
low as a StageB paper-core direction
```

## 三、当前 Encoder 是否有理论漏洞

### 结论分级

- [Fact] 没有证据证明 Encoder 在数学上无效或发生信息断路；A6性能与 frozen ablation都反对这一说法。
- [Strong Evidence] 存在多个明确的 inductive-bias限制与 code-theory不一致。
- [Decision] 应称为 `structural limitations / protocol confounders`，不应直接称为足以推翻 A6 的
  fatal flaw。

### Limitation 1：Encoder 内没有 cross-patch mixing

`encoder[layer]` 是对每个 token独立执行的 MLP：

$$
z_p^{(\ell+1)}=\operatorname{Norm}\left(z_p^{(\ell)}+
W_2\operatorname{Dropout}(\operatorname{GELU}(W_1z_p^{(\ell)}))\right).
$$

不存在 $z_p$ 与 $z_q$ 在 Encoder 内交互的项。`P>1` 时，patches只在
`hidden.flatten -> learned_basis_coeff` 的 dense linear layer汇合。

[Implication] 现有架构能学习“每个 local patch的 nonlinear feature + 全局 linear aggregation”，但不能在
coefficient readout之前表达 conditional cross-patch interaction。例如“只有当早期 patch出现模式 A 且近期
patch出现模式 B 时才激活某 future state”的乘性/条件关系没有直接路径。

[Counterpoint] `P=1` 时，MLP接收的是 whole-history global projection，反而可以在压缩后的 256-dimensional
state内做 global nonlinear interaction。因此 `P>1` 不是对 `P=1` 的单调增强，而是 inductive bias交换。

### Limitation 2：ETTm1 有 early global bottleneck

ETTm1 先执行：

$$
\mathbb{R}^{720}\rightarrow\mathbb{R}^{256}\rightarrow\mathbb{R}^{256}.
$$

A6 output又通过 256-dimensional coefficients生成全部 future rows。对单 channel而言，history-to-future
Jacobian的线性局部 rank上界受这一 state dimension约束。

[Boundary] rank 256 对 ETTm1并非显然不足；Phase4 label audit显示 future labels本身高度 low-rank。该项只
说明 local detail必须经过 early compression，不能证明预测损失由这个 bottleneck造成。

### Limitation 3：positional encoding混入 channel offset

代码先把 `[B,C,720]` reshape为 `[B,C*720]`，unfold后得到 `[B,C*P,D]`，然后在整个 `C*P` token
axis上加 sinusoidal position。于是 channel $c$ 的 patch $p$ 使用位置 $cP+p$，而不是所有 channel共享
位置 $p$。

[Code-Theory Inconsistency] 模型表面上是 channel-wise shared encoder，但 position code隐式注入 channel
identity/order。若要声称 channel independence，应先 reshape `[B*C,P,K]`，让每个 channel从 position 0
开始。

[Boundary] 固定 dataset、固定 channel order下，这种 channel-specific bias可能有利于性能；它不是数据
leakage。但它使 `P=1 -> P>1` 的 naive comparison同时改变 channel identity encoding。

### Limitation 4：`P` 必须整除 `L`

当前实现用 `patch_len=L//P`，没有 padding或边界检查，并在 channel-concatenated axis上 unfold。只有
`P | L` 时才能确保每个 channel正好产生 P 个 non-overlap patches且不跨 channel boundary。该约束也解释
了 upstream issue中 `seq_len=512` 与 `P=24/48` 的复现疑问。

### Limitation 5：参数报告包含 unused dense head

clean A6 虽不调用 official `proj_x`，但当前 model instance仍构造该层。unused parameters为：

| Dataset | unused `proj_x` params | total state-dict params | active-forward params |
| --- | ---: | ---: | ---: |
| ETTh2 | 1,106,640 | 1,690,016 | 583,376 |
| ETTm1 | 185,040 | 884,640 | 699,600 |
| Weather | 4,424,400 | 6,250,656 | 1,826,256 |

[Implication] 若表格使用 total parameters，会严重高估 A6实际 computation capacity，尤其 Weather。未使用
参数因 gradient为 `None` 不会被 AdamW更新，但仍污染 model-size/fairness叙述。

### Limitation 6：Encoder 与 requested horizon无关

A6 history state与 coefficient对所有 $H$ 相同；requested horizon只截取 `basis[:H]`。这正是 StageB
长期存在的叙事缺口，但 B8-B14 说明“缺少 target-aware path”并不自动证明任何具体 query/retrieval机制
必要。

[Decision] 这是 open architecture question，不是已经找到的理论漏洞。

## 四、Frozen Encoder Branch Diagnostic

### Setup

- checkpoint：B9 small gate中的 clean A6 seed-2021 `official-last`；
- splits：ETTh2/ETTm1/Weather test；
- batches：前 64 个，batch size 128；
- `full`：原 checkpoint；
- `no_mlp_keep_norm`：每层 residual MLP替换为 zero update，保留 LayerNorm；
- `embed_only`：同时绕过 residual MLP和 LayerNorm；
- 不 retrain，因此只用于 branch-use diagnosis。

### Eval-Mode Branch Size

| Dataset | layer 1 branch/input norm | layer 2 branch/input norm |
| --- | ---: | ---: |
| ETTh2 | 0.2380 | 0.3004 |
| ETTm1 | 0.0860 | 0.1307 |
| Weather | 0.1852 | 0.1875 |

[Strong Evidence] ETTm1 branch update幅度确实比另两个 datasets小，符合高 dropout强正则化后更依赖 global
embedding/identity path的假设；但它不是 0。

### Removing Residual MLP, Keeping LayerNorm

| Dataset | H96 | H192 | H336 | H720 | mean relative MSE increase |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | +1.62% | +7.59% | +16.12% | +47.12% | +18.11% |
| ETTm1 | +12.96% | +9.32% | +7.29% | +5.58% | +8.79% |
| Weather | +18.12% | +14.75% | +11.00% | +8.07% | +12.99% |

[Decision] “ETTm1 dropout 0.9让 Encoder MLP失效”被当前 frozen evidence反驳。更准确的判断是：MLP
branch较小但 material，并且对短 prefixes尤其重要。

[Self-Critique] frozen bypass会让 downstream coefficient head接收未训练分布，因而可能高估 removing branch
的损害。它能证明 branch被使用，不能证明相同架构 retrain后一定需要该 branch，也不能选择 optimal
dropout。

## 五、Large Dropout 与 Last Checkpoint

### Effective Config Is A Compound Preset

official ETTm1 scripts使用：

| fixed horizon | `d_model` | `d_ff` | dropout | `patch_num` |
| --- | ---: | ---: | ---: | ---: |
| 96 | 128 | 256 | 0.2 | 1 |
| 192 | 128 | 256 | 0.2 | 1 |
| 336 | 128 | 256 | 0.8 | 1 |
| 720 | 256 | 256 | 0.9 | 1 |

unified A6 因 `pred_len=720` 选择 720 preset，因此 H96/H192/H336/H720 全部使用
`D=256,dropout=0.9,P=1`。Weather unified同理继承 H720 的 `d_ff=128,dropout=0.5`。

[Fact] 这确实与 PatchTST/iTransformer等常见 official configs中的 `0.1-0.3` dropout有明显区别。但超参数
“不常见”不是错误证据；TimeAlign release与论文结果本身使用了这种 dataset/horizon-specific tuning。

### Dropout 的实际作用点

dropout只位于 residual MLP 的 GELU后：

```text
token + Linear2(Dropout(GELU(Linear1(token))))
```

它不直接 drop input patch、不直接 drop coefficient、不作用于 learned temporal basis；eval mode下 dropout
关闭。`p=0.9` 时 inverted dropout的额外方差因子为：

$$
\frac{p}{1-p}=9,
$$

而 `p=0.2` 为 `0.25`。相同 activation尺度下，noise factor相差 36 倍。`d_ff=256,p=0.9` 每次 mask
期望保留 `25.6` units；但 residual identity path完整保留，所以网络不会像非 residual MLP那样直接丢失
90%表示。

[Hypothesis] ETTm1 `P=1 + dropout=0.9 + residual path` 可能把模型推向“global low-rank near-linear
forecaster + small nonlinear correction”，这与 ETTm1强周期/low-rank label结构相容。该解释需要 controlled
retraining，不能由 frozen norm alone确认。

### Last Checkpoint Is Source-Faithful, Not Neutral

upstream `EarlyStopping` 的比较逻辑被注释，checkpoint每 epoch覆盖；test path也不 reload validation-best。
TimeAlign作者在 official GitHub issue #2明确回复：论文使用 fixed epochs后的 last checkpoint，并认为
validation/test shift可能使 early stopping训练不足。

clean A6 validation drift：

| Dataset | best epoch | best val mean MSE | last val mean MSE | last vs best |
| --- | ---: | ---: | ---: | ---: |
| ETTh2 | 1 | 0.410330 | 0.475124 | +15.79% |
| ETTm1 | 9 | 0.599628 | 0.599933 | +0.05% |
| Weather | 8 | 0.488468 | 0.489485 | +0.21% |

[Interpretation] last selector对 ETTm1/Weather几乎不是主要风险，但对 ETTh2明显 material。此前 official
TimeAlign best-val sensitivity并未改变 unified-vs-fixed winner pattern，说明 selector未必是机制主因；但
clean A6本身尚未完成同一 trajectory的 dual-checkpoint test。

[Decision] 主论文应同时保留两种证据层：

1. `source-faithful`: official presets + official-last，用于复现 TimeAlign；
2. `controlled`: architecture/config matched + dual last/best reporting，用于支持 FATST 自己的 causal claim。

不能用第 1 层替代第 2 层，也不应因为第 2 层待补就删除第 1 层。

## 六、下一轮最小研究设计

完整预注册见：

`docs/experiments/phase5-stage-b-ettm1-carrier-protocol-audit.md`。

small gate只在 ETTm1 seed-2021执行六臂：

1. exact legacy `P1-D256-F256-drop0.9`；
2. channel-independent PE的 matched P1；
3. channel-independent `P5-D52-F256-drop0.9` lower-capacity control；
4. channel-independent `P5-D52-F2048-drop0.9` near-parameter-matched arm；
5. CI P1-drop0.2；
6. CI parameter-matched P5-drop0.2。

每次训练必须同时保存 last/best states。patch defect只有在以下条件同时满足时成立：

- P5 vs clean P1跨 `dropout=0.9/0.2` 同号；
- last/best selector同号；
- 每个组合至少 `3/4` wins；
- mean MSE `<=-0.5%`；
- 不被 PE语义修复或 parameter control解释；
- confirmation seeds 2022/2023维持方向。

之后另做一个独立 unified-vs-fixed fair control：四个 single-prefix arms与 multi-prefix arm使用完全相同的
720-step A6 architecture/config，只改变 supervision target。该实验才直接检验“一个 unified model是否
优于四个 horizon-specific training objectives”。

## 七、最终研究判断

### 对 `patch_num` 的回答

[Decision] 要关注，但只作为立即、短周期、可杀死的 carrier audit。它不是当前 StageB 的 paper-core
问题。若 controlled gate不通过，应立即关闭，不再做 patch count sweep。

### 对 Encoder 理论漏洞的回答

[Decision] 存在真实 structural limitations：无 cross-patch mixing、P1 early global bottleneck、channel-
offset positional semantics、divisibility contract和 unused-head parameter accounting。但没有证据支持
“Encoder无效”或“P1必然错误”。

### 对 large dropout 的回答

[Decision] `0.8/0.9` 异常激进，且在 unified setting中被所有 prefixes共同继承，必须做 control；但它位于
residual branch，eval时关闭，现有 frozen evidence证明 branch仍 material。不能仅凭数值大就判定 protocol
错误。

### 对 last checkpoint 的回答

[Decision] `official-last` 是 TimeAlign source-faithful policy，但不是 FATST architecture claim的充分
protocol。ETTh2 validation drift要求 clean A6 dual-selector sensitivity；ETTm1当前 drift很小，不能把
ETTm1 `patch_num`现象主要归因于 last selector。

### StageB 路线决策

```text
active: C0-ETTm1 carrier/protocol audit (diagnostic-only)
paused: new future-aware method search
closed: B14-FURD retrieval implementation
rollback if C0 fails: StageB Step 2/3 or pause Contribution 2
```

## Primary Sources

- [TimeAlign paper / arXiv](https://arxiv.org/abs/2509.14181)：TimeAlign 的原始问题是 past-future
  distribution alignment；其 predict backbone被描述为 replaceable，不能自动为 FATST 的 inherited
  encoder preset提供理论必然性。
- [TimeAlign official ETTm1 script](https://github.com/TROUBADOUR000/TimeAlign/blob/main/scripts/ETTm1.sh)：
  `patch_num=1` 与 `dropout=0.2/0.2/0.8/0.9` 的 source配置。
- [TimeAlign official issue #2](https://github.com/TROUBADOUR000/TimeAlign/issues/2)：作者确认论文主结果使用
  last epoch，并说明 released configs有部分 dataset-specific simplification。
- [TimeAlign official model](https://github.com/TROUBADOUR000/TimeAlign/blob/main/models/TimeAlign.py)：
  `PatchEmbed -> token-wise MLP -> flatten projection` 的 source tensor path。
- [PatchTST official repository](https://github.com/yuqinie98/PatchTST) 与
  [paper](https://arxiv.org/abs/2211.14730)：patching 的局部语义、channel independence与 cross-token
  Transformer boundary。
- [Dropout original JMLR paper](https://www.jmlr.org/papers/v15/srivastava14a.html)：dropout的 ensemble/
  anti-coadaptation解释；它不支持“dropout rate越高必然越好/越坏”的无条件结论。

## Artifacts

- `encoder_preset_audit.csv`；
- `checkpoint_drift.csv`；
- `encoder_branch_statistics.csv`；
- `encoder_branch_ablation.csv`；
- `scripts/analyze_phase5_stage_b_encoder_protocol_audit.py`；
- `docs/code-explanation/phase5-stage-b-encoder-protocol-audit.md`；
- `docs/experiments/phase5-stage-b-ettm1-carrier-protocol-audit.md`。

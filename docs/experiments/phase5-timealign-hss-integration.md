# Phase5: TimeAlign-HSS Integration Protocol

## 当前定位

[Decision] 暂时不做 TimeAlign 论文级复现。官方源码版本已经足以证明两件事：

1. TimeAlign 是一个强 baseline / carrier；
2. unified setting 存在 dataset-dependent decrease：ETTm2 与 Weather 在 unified-720
   下稳定弱于 fixed-horizon，而 ETTh2 反而在多数 horizon 受益。

因此，Phase5 的研究重点从 reproduction audit 转向：

> 如何把 TimeAlign 整合进或修改成 Horizon Supervision Scheduling 的论文叙事与方法体系。

## 11-Step 状态

| Field | Content |
| --- | --- |
| `current_step` | Step 2/3/4/5/6：重新定义问题、验证问题价值、提出核心 idea、做理论可行性检查、设计第一轮实验 |
| `problem` | TimeAlign 的 future-aware supervision 在统一 multi-horizon setting 中不是稳定有益：部分数据集受益，部分数据集受损。问题不是“unified 一定退化”，而是同一个 future alignment objective 在不同 future states/units 上有不同可靠性 |
| `existence_evidence` | `official-last` 与 `best-val` 的 winner pattern 一致：ETTh2 `3/4` unified wins，ETTm2/Weather `0/4` unified wins；selector 不是主因。官方源码版本相对 repo-local 实现显著更可信，说明 carrier 本身成立 |
| `idea` | 将 TimeAlign 作为 HSS carrier：HSS 不再只调整预测 loss，而是调度 TimeAlign future reconstruction/alignment supervision 的强度、位置和 gradient path |
| `theory_check` | TimeAlign 的 training-only future branch 会把 ground-truth future distribution pressure 传回 history branch。若 future units 可预测且可对齐，这种 pressure 有益；若 future units 噪声大、状态漂移强或与 history representation 冲突，static full-future alignment 会成为 harmful supervision |
| `design` | 先做 head/interface confounder diagnostic，再做 supervision diagnostic 与 minimal scheduling。D0 诊断 fixed `pred_len=720` head 是否导致短 prefix 监督不足；D1 诊断 supervision reliability；M1 做 unit-level reliability weighting；M2 做 alignment gradient-path routing |
| `gate` | ETTm2/Weather unified gap 缩小；ETTh2 unified benefit 不丢失；机制诊断能说明 schedule 区分了 useful/harmful future supervision，而不是简单淡化 TimeAlign loss |
| `artifacts` | 本文档；后续分析输出放入 `analysis/phase5_timealign_hss_*`；代码优先加在 `baselines/timealign_official/` adapter / train wrapper 层，保留官方 forward 作为对照 |
| `decision` | `active_timealign_hss_integration`。若 D1 不能证明 reliability 与 unified decrease 有关，回 Step 2/3；若 M1/M2 只靠 loss weakening 取胜，回 Step 4/5 |

## 叙事锚点

[Core Claim] Unified multi-horizon forecasting 不只是 output head 问题，也是
training supervision allocation 问题。TimeAlign 暴露了这个问题：同一个 future alignment
objective 对 alignable future states 有益，但对 unstable/noisy future units 可能有害。

[HSS Claim] Horizon Supervision Scheduling 的目标不是按 benchmark horizon id 训练模型，而是在
training 中判断 future supervision 何时、何处、以多大强度、通过哪条 gradient path 进入模型。
evaluation 仍然保持 `{96,192,336,720}` multi-horizon。

[Design Constraint] schedule 必须 horizon-agnostic。允许使用的信号包括：

- future reconstruction difficulty；
- local/global alignment consistency；
- local volatility 或 seasonal residual volatility；
- prediction residual structure；
- training dynamics，例如 unit-level loss rank、loss stability、gradient conflict。

不应直接把 `h96/h192/h336/h720` 作为 schedule 的输入。

## D0：Unified Head / Interface Diagnostic

### 目的

[Question] TimeAlign 的 unified decrease 是否主要来自 fixed `pred_len=720` output head 没有
统一 multi-horizon interface 设计？

[Fact] 官方 TimeAlign 使用固定长度 projection head：
`Linear(d_model * patch_num, pred_len)`。当前 unified 设置是训练 `pred_len=720`，然后在评估时
裁剪 `96/192/336/720` prefix。它具备 tensor-level prefix consistency，但没有像 R.3 /
target-set decoder 那样显式设计 requested-horizon interface。

如果 D0 通过，后续 HSS 不能直接从 supervision reliability 开始；必须先承认 unified head /
interface 是一个 co-factor。

### 最小实验

| Arm | Change | Purpose |
| --- | --- | --- |
| `full` | 官方 full-horizon prediction loss | official unified-720 baseline |
| `multi-prefix` | prediction loss 改为 `mean(L_96,L_192,L_336,L_720)` | 判断短 prefix 是否只是缺少直接 prediction supervision |

两臂都不改 official TimeAlign forward，不改 reconstruction / alignment loss，也不引入 horizon id
作为模型输入。`multi-prefix` 只改变 prediction loss 的计算方式，因此是 head/interface confounder
control，不是最终 HSS 方法。

### 判定

[Pass] `multi-prefix` 明显缩小 ETTm2/Weather 的 unified decrease，同时不破坏 ETTh2 的 unified
benefit。此时先进入 unified head/interface 设计，而不是直接做 D1/M1。

[Partial] 只改善一个 degraded dataset，或只改善短 horizon。继续 D1，但在叙事中把 head/interface
作为 co-factor。

[Fail] `multi-prefix` 不改善 ETTm2/Weather 或明显伤害 ETTh2。此时 head/interface confounder
不是主因，进入 D1。

### Artifacts

- runner: `scripts/remote/run_phase5_timealign_hss_d0_head_gate.sh`
- analyzer: `scripts/analyze_phase5_timealign_hss_d0_head_gate.py`
- sync: `scripts/sync_phase5_timealign_hss_d0_results.sh`
- output root: `/home/yingch/exp_outputs/r-2026-fatst/phase5_timealign_hss_d0_head_gate`

## D1：Supervision Reliability Diagnostic

### 目的

[Question] ETTm2/Weather 的 unified decrease 是否与 future supervision reliability 有关？

如果答案是否定的，TimeAlign-HSS 缺少必要机制支点；后续不应直接实现 scheduling。D1 只在
D0 未完全解释 unified decrease 后进入。

### 最小诊断量

对已完成的 official-source runs，尽量复用已有 logs 与 predictions；如需要新增导出，保持精简：

| Signal | Computation | Meaning |
| --- | --- | --- |
| `unit_recon_error` | future branch reconstruction error by future unit/block | future supervision 本身是否容易学习 |
| `unit_align_gap` | history/future aligned representation 或 aligned prediction 的 unit-level distance | future pressure 与 history branch 是否一致 |
| `unit_residual_volatility` | target residual or first-difference volatility by unit/block | future unit 是否不稳定或噪声强 |
| `unit_unified_gap` | unified prefix prediction error minus fixed prediction error by unit/block | unified decrease 在哪些 future units 发生 |

### 判定

[Pass] 至少在 ETTm2 或 Weather 中，`unit_recon_error`、`unit_align_gap`、
`unit_residual_volatility` 与 `unit_unified_gap` 呈现一致方向的相关或分组差异，同时 ETTh2
表现出不同模式。

[Fail] unified decrease 与这些 reliability proxies 没有可解释关系，或者所有 dataset 模式相同。
此时回 Step 2/3：问题可能不是 supervision scheduling，而是 TimeAlign architecture / data
normalization / look-back choice。

## M1：Reliability-Weighted TimeAlign Supervision

### 方法

[Idea] 不改预测 head，不改官方 TimeAlign forward。只在 train wrapper 中给 reconstruction/alignment
loss 增加 unit-level reliability weight：

$$
L = L_{\text{pred}} +
\lambda_{\text{recon}}\sum_u w_u L_{\text{recon},u} +
\lambda_{\text{align}}\sum_u w_u L_{\text{align},u}.
$$

其中 $w_u$ 来自 D1 中最稳定的 reliability proxy，且不使用 benchmark horizon id。

### 对照

第一轮只跑最小矩阵：

| Arm | Purpose |
| --- | --- |
| `official_unified_720` | strong baseline |
| `hss_m1_recon_only` | 判断 reconstruction reliability 是否是主因 |
| `hss_m1_align_only` | 判断 alignment reliability 是否是主因 |
| `hss_m1_recon_align` | 判断二者组合是否形成稳定收益 |

datasets 优先 `ETTm2/Weather/ETTh2`。Weather 最慢，远程调度时优先启动 Weather 与 ETTm2，
避免 fast GPU 空转。

### Gate

[Pass] `ETTm2/Weather` mean relative MSE gap 明显小于 official unified，同时 `ETTh2` 的
unified benefit 不明显下降。

[Weak Pass] 只改善 ETTm2 或 Weather，且诊断能解释 dataset specificity。可进入 M2，但不能写成
通用方法。

[Fail] 改善只来自整体降低 `lambda`，或 h720 损伤明显，回 Step 4/5。

## M2：Alignment Gradient-Path Routing

### 方法

[Idea] 如果 D1/M1 证明 harmful supervision 存在，但 scalar weighting 不够稳定，则进入
gradient-path routing：可靠 future units 可以更新 shared alignment path；低可靠 units 只更新
auxiliary adapter 或 reconstruction branch，不直接污染 shared history representation。

这对应 HSS 的升级表述：

> HSS decides not only how much a future unit supervises, but also where its
> gradient is allowed to update.

### 最小实现原则

- 保留 official TimeAlign forward 与 fixed/unified baseline；
- routing 放在 adapter/train wrapper 层；
- adapter 应 zero-init 或 near-identity init，避免一开始破坏 baseline；
- diagnostic 只保存必要的 unit-level summary，不保存大规模中间 tensor。

### Gate

[Pass] 相比 M1，M2 在 Weather 或 ETTm2 上进一步缩小 unified gap，且 ETTh2 不发生明显退化。

[Fail] routing path 被使用但性能不改善，说明 carrier 不是 gradient destination，而可能是
representation capacity 或 dataset-specific hyperparameter。

## 当前下一步

1. 先运行 D0 head/interface diagnostic；
2. 若 D0 不能解释 unified decrease，再完成 D1 诊断脚本设计，优先复用 existing predictions/logs；
3. 如果 D1 通过，进入 M1 最小实验；
4. 只有当 M1 显示 scalar scheduling 不足但机制方向成立时，才进入 M2；
5. 不做 look-back horizon sweep，除非后续需要论文表格级 TimeAlign reproduction 作为附录或审稿防线。

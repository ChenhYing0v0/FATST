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

### D0 结果

[Decision] `head_interface_confounder_strong`.

`multi-prefix` 在全部 `12/12` 个 dataset-horizon setting 上优于 full unified loss，平均
MSE 相对 full 下降：

| dataset | wins | mean relative MSE vs full |
| --- | --- | --- |
| ETTh2 | 4/4 | -3.36% |
| ETTm2 | 4/4 | -1.57% |
| Weather | 4/4 | -1.17% |
| ALL | 12/12 | -2.03% |

相对 fixed-horizon reference：

| dataset | full unified gap | multi-prefix gap | gap improvement |
| --- | --- | --- | --- |
| ETTh2 | -8.01% | -11.05% | +3.05 pct-pt |
| ETTm2 | +3.72% | +2.06% | +1.66 pct-pt |
| Weather | +1.05% | -0.13% | +1.18 pct-pt |

[Decision] D0 通过，且强度足以改变研究顺序。D1 supervision reliability diagnostic 后移；
下一步应先研究 TimeAlign-compatible unified prediction interface。

## H0：Prefix-Supervised TimeAlign Carrier

### 目的

[Question] 是否可以把 D0 的 `multi-prefix` control 从 diagnostic formalize 成 TimeAlign-HSS 的
第一层 carrier，即让 TimeAlign 在 unified multi-horizon setting 中先获得
evaluation-consistent prefix supervision？

### 第一轮计划

| Arm | Change | Purpose |
| --- | --- | --- |
| `full` | official full-horizon prediction loss | strong baseline |
| `multi-prefix` | `mean(L_96,L_192,L_336,L_720)` | prefix-supervised interface |
| `balanced-step` | `mean(L_1:96,L_97:192,L_193:336,L_337:720)` | 判断收益是否只是非重叠 region reweight |
| `stochastic-prefix` | 每 batch 从 `{96,192,336,720}` 采样 prefix | 判断 prefix supervision 能否作为 schedule |
| `continuous-prefix` | 每 batch 从连续 prefix pool 采样 | 判断能否脱离 benchmark horizon id |

第一轮仍不改 official TimeAlign forward。若 robustness 成立，再进入轻量 prefix-aware /
target-set readout。

### 实现语义

所有 H0 arms 都保持：

```text
pred: [B, 720, C]
true: [B, 720, C]
```

只改变 `L_pred`：

- `full`: `L1(pred[:, :720], true[:, :720])`；
- `multi-prefix`: `mean(L1(:96), L1(:192), L1(:336), L1(:720))`；
- `balanced-step`: 对不重叠区间分别算 loss 后平均；
- `stochastic-prefix`: 每个 batch 从 fixed prefix set 随机采样一个 prefix；
- `continuous-prefix`: 每个 batch 从 `32,64,...,720` prefix pool 随机采样一个 prefix。

`balanced-step` 是 mechanism control，不是最终 HSS 候选；`stochastic-prefix` 与
`continuous-prefix` 才是 scheduling 候选。

### Gate

[Pass] `multi-prefix` 或其稳健变体在 ETTm2/Weather 继续缩小 fixed gap，且 ETTh2 不退化；
至少一次 seed/checkpoint sensitivity 不改变方向。

[Paper-story Pass] `stochastic-prefix` 或 `continuous-prefix` 接近或超过 `multi-prefix`。这说明
D0 的收益可以从 benchmark-specific multi-prefix loss 升级为 horizon-agnostic supervision
scheduling。

### H0 结果

[Decision] `prefix_scheduling_pass_with_stochastic_candidate`.

| loss_mode | mean MSE vs full | mean MSE vs multi-prefix | wins vs fixed | mean MSE vs fixed |
| --- | ---: | ---: | ---: | ---: |
| `multi-prefix` | -2.03% | 0.00% | 7/12 | -3.04% |
| `balanced-step` | -1.22% | +0.83% | 6/12 | -2.26% |
| `stochastic-prefix` | -1.90% | +0.13% | 7/12 | -2.90% |
| `continuous-prefix` | -1.67% | +0.37% | 7/12 | -2.69% |

[Strong Evidence] `balanced-step` 明显弱于 prefix modes，说明 D0/H0 的收益不是简单 region
reweight。

[Strong Evidence] `stochastic-prefix` 接近 `multi-prefix`，说明 prefix supervision 可以
schedule 化；它是当前最有 paper-story potential 的 H0 候选。

[Evidence] `continuous-prefix` 有效但略弱，说明脱离 benchmark horizon id 有潜力，但当前
single-prefix sample 与 `32` step pool 需要继续校准。

### H0B 计划

| Arm | Change | Purpose |
| --- | --- | --- |
| `stochastic-prefix_k2` | 每 batch 从 `{96,192,336,720}` 采样 2 个 prefix | 检查 schedule strength 是否能超过 `multi-prefix` 或稳定提升 ETTm2 |
| `continuous-prefix_k2` | 每 batch 从 continuous pool 采样 2 个 prefix | 检查 horizon-agnostic schedule 是否能追上 benchmark-specific schedule |
| `continuous-prefix_pool96` | 从较粗 prefix pool 采样，例如 `96,192,...,720` | 判断 continuous 弱势是否来自过短 prefix 噪声 |

H0B 通过后再做 seed/checkpoint sensitivity；H0B 失败则保留 `multi-prefix` 作为 strong interface
control，但主线应转向 prefix-aware / target-set readout，而不是继续调 random schedule。

### H0B 实验落地

| Field | Content |
| --- | --- |
| `current_step` | Step 6/7/8：设计并运行 schedule robustness gate |
| `problem` | H0 证明 `stochastic-prefix` 接近 `multi-prefix`，但还不知道 sample count 与 continuous pool 是否能进一步提升或 horizon-agnostic 化 |
| `idea` | 调整 prefix sample count 和 continuous pool granularity，不改 TimeAlign forward |
| `theory_check` | 若 `k=2` 改善 ETTm2 或整体 mean，说明单 prefix sample 的 supervision signal 不足；若 `pool96` 改善 continuous-prefix，说明过短 prefix 采样引入噪声 |
| `design` | `3 datasets x 3 arms`：`stochastic_prefix_k2`、`continuous_prefix_k2`、`continuous_prefix_pool96` |
| `gate` | 至少一个 schedule arm 接近或超过 `multi-prefix`；优先看 ETTm2 residual gap、Weather no-harm、ETTh2 gain preservation |
| `artifacts` | `scripts/remote/run_phase5_timealign_hss_h0b_schedule_gate.sh`、`scripts/analyze_phase5_timealign_hss_h0b_schedule_gate.py`、`scripts/sync_phase5_timealign_hss_h0b_results.sh` |
| `decision` | H0B 已进入远程实验；返回后再决定 seed/checkpoint sensitivity 或 prefix-aware readout |

[Fail] prefix weighting 的收益不稳定或只来自单次随机性，则回到 D1 reliability diagnostic，
但保留 head/interface 作为 confounder。

## D1：Supervision Reliability Diagnostic

### 目的

[Question] ETTm2/Weather 的 unified decrease 是否与 future supervision reliability 有关？

如果答案是否定的，TimeAlign-HSS 缺少必要机制支点；后续不应直接实现 scheduling。D1 只在
H0 后仍存在 residual unified decrease 时进入。

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

1. D0 已通过，先进入 H0：Prefix-Supervised TimeAlign Carrier；
2. 若 H0 后仍存在 ETTm2/Weather residual gap，再完成 D1 诊断脚本设计，优先复用 existing predictions/logs；
3. 如果 D1 通过，进入 M1 最小实验；
4. 只有当 M1 显示 scalar scheduling 不足但机制方向成立时，才进入 M2；
5. 不做 look-back horizon sweep，除非后续需要论文表格级 TimeAlign reproduction 作为附录或审稿防线。

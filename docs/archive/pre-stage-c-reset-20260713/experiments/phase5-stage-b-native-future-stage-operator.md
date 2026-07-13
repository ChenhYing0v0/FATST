# Phase5 StageB B9 Native Future-Stage Operator

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B9-FSN` |
| `current_step` | Step 9/10：B9-FSN-SCF small gate completed |
| `problem` | A6-LBF 用一个 sample/channel-specific `coeff[b,c]` 服务所有 future stages；不同 stage 可能对该共享 coefficient 施加不同训练方向 |
| `existence_evidence` | `B9-SGC` stage-gradient diagnostic 显示三数据集 stage gradients 对共享 coefficient 的 cosine 很低 |
| `idea` | 将 future stage 作为 primary prediction path 的条件变量，生成 stage-native representation/operator，而不是在 A6 output 后做 residual correction |
| `theory_check` | stage-gradient conflict 下，共享 coefficient 接收多个近正交目标；stage-native coefficient field 可把冲突路由到不同 future-stage state，同时保留 A6 shared basis |
| `design` | `B9-FSN-SCF`: Stage-Native Coefficient Field，在 basis projection 前生成 stage-specific coefficient field，并以 function-preserving gate 初始化为 clean A6 |
| `narrative_gate` | `passed_for_small_gate`; 可进入最小实现与 smoke/gate，不得直接启动 full main matrix |
| `effectiveness_gate` | `blocked_by_no_stage_control` |
| `artifacts` | `analysis/phase5_stage_b_b9_stage_gradient_diagnostic_20260707/`; `analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/` |
| `decision` | `B9-FSN-SCF rejected as current method`; rollback to Step 4 redesign or Step 2/3 if no stronger native-stage mechanism is found |

## 用户约束

[Fact] StageB 不应走 residual-style architecture。Residual adapter、post-hoc correction、error repair 都不适合作为 paper-core method，因为论文叙事弱且贡献边界不干净。

[Decision] B9 必须是 native future-stage-aware architecture：future-stage 信息要进入 primary prediction path，而不是修正已经生成的 A6 prediction。

## 外部网络调研范围

本轮 Step 2/3 调研没有只看 Zotero 或本地 notes，外部网络核验了以下 primary / near-primary sources：

- TimePerceiver arXiv HTML：<https://arxiv.org/html/2512.22550v1>
- TimePerceiver official repository：<https://github.com/efficient-learning-lab/TimePerceiver>
- ElasTST arXiv：<https://arxiv.org/abs/2411.01842>
- MQ-RNN ar5iv full text：<https://ar5iv.labs.arxiv.org/html/1711.11053>
- Temporal Fusion Transformer arXiv：<https://arxiv.org/abs/1912.09363>

本地 note 辅助来源：

- `Papers/timeperceiver-generalized-forecasting.md`
- `Papers/elastst-varied-horizon.md`
- `Papers/srp-step-specific-representation.md`

Step 4-6 补充核验：

- Zotero semantic search 可返回候选 item，但本机 full item fetch 为 `Connection refused`，因此本轮不把
  Zotero API 输出作为完整元数据证据；
- 外部网络重新检索 TimePerceiver、ElasTST、MQ-RNN、TFT 与 SRP++，其中 SRP++ OpenReview 页面仍受
  browser verification 限制，只作为本地 note 辅助证据，不作为外部已完整核验证据；
- 本轮方法设计只抽取 target/stage conditioning、future placeholder、horizon-specific context 与
  step-specific representation 这些 mechanism-level evidence，不复制任何完整外部架构。

## 文献证据

### TimePerceiver

[Fact] TimePerceiver 明确批评 forecasting 研究偏重 encoder，忽略 decoding 和 training 的整合；其 decoder 使用 target timestamp queries 从 input representations 中检索相关信息。

[Implication] target position/stage 不应只是 output index，而可以作为 decoder/operator 的原生条件变量。

### ElasTST

[Fact] ElasTST 使用 future placeholders 与 structured masks，目标是 varied-horizon inference 下 prefix/horizon invariant。

[Implication] future positions 可以在不泄漏 future values 的前提下进入模型结构；B9 若实现，必须保留 prefix stability。

### MQ-RNN

[Fact] MQ-RNN 显式构造 horizon-specific contexts，并指出 horizon-specific context 对 seasonality mapping 等问题是必要的；其 local MLP 共享参数但接收 horizon-specific context。

[Implication] 这直接支持 B9 的 native future-stage framing：future stage 应在主预测路径中形成 context，而不是只由最终 output row 隐式表达。

### Temporal Fusion Transformer

[Fact] TFT 面向 multi-horizon forecasting，显式处理 static covariates、known future inputs 和 observed historical inputs 的混合，并通过专门组件选择相关特征。

[Implication] multi-horizon architecture 可以把 future-known/stage-related 信息作为预测路径中的结构输入；B9 在没有外生 future covariates 的 LTSF setting 中，可以把 future stage 本身作为结构条件。

## A6 数据流中的原生问题

当前 clean A6-LBF 的关键路径是：

```text
hidden = encoder(history)                  # [B, C, R]
coeff = learned_basis_coeff(hidden)        # [B, C, 256]
y[t,c] = learned_temporal_basis[t] @ coeff[b,c] + bias[t]
```

其中：

- `coeff[b,c]` 是 sample-specific / channel-specific；
- `learned_temporal_basis[t]` 是 future-position-specific，但全局共享；
- 四个 future stages `[0,96), [96,192), [192,336), [336,720)` 都共用同一个 `coeff[b,c]`。

B9 的问题不是“如何修 residual”，而是：

> 同一个 `coeff[b,c]` 是否被不同 future stages 的 primary prediction losses 拉向不同方向？

若是，则说明 native stage-aware representation/operator 有问题基础。

## B9-SGC 诊断

`B9-SGC` 不拟合 residual，也不设计 correction module。它只分析 primary prediction path 的 stage gradients。

对 clean A6 checkpoint，在 train split 上取 batches，手动执行 forward 到：

```text
coeff = learned_basis_coeff(hidden)      # [B, C, 256]
prediction = learned_temporal_basis @ coeff
```

然后分别计算四个 non-overlap future stage losses：

```text
loss_early = MSE(y[0:96])
loss_mid   = MSE(y[96:192])
loss_late  = MSE(y[192:336])
loss_tail  = MSE(y[336:720])
```

并求每个 loss 对同一个 `coeff` 的梯度：

```text
g_s = d loss_s / d coeff
```

若 `cos(g_i, g_j)` 低或为负，说明不同 future stages 对共享 coefficient 的训练方向不一致。

## B9-SGC 结果

完整报告：

- `analysis/phase5_stage_b_b9_stage_gradient_diagnostic_20260707/b9_stage_gradient_report.md`
- `docs/code-explanation/phase5-stage-b-b9-stage-gradient-diagnostic.md`

Summary：

| dataset | mean pairwise cosine | early-tail cosine | negative pair rate |
| --- | ---: | ---: | ---: |
| ETTh2 | 0.072 | 0.041 | 0.083 |
| ETTm1 | 0.171 | 0.112 | 0.042 |
| Weather | 0.048 | 0.014 | 0.083 |

[Strong Evidence] 三个 dataset 的 stage-gradient cosine 都很低，尤其 early-tail cosine 接近 0。这说明四个 future stages 对同一个 A6 coefficient 的训练方向近似正交，而不是高度一致。

[Inference] 这支持 B9 的 problem existence：A6 的 single coefficient state 虽然性能强，但从训练信号看，它同时服务多个 future stages 时存在 native stage pressure。

[Decision] `B9-SGC` 暂定通过 problem-candidate gate。该结论已进入下文 Step 4-6 narrative/method design，
用于设计 primary-path stage-native operator 和 capacity-preserving initialization。

## B9 方法边界

B9 不允许写成：

```text
y = A6(x) + correction(stage)
```

也不允许写成：

```text
residual = true - pred
stage_module learns residual
```

B9 只能写成 primary prediction path：

```text
stage_state = StageNative(history, stage_token)
stage_coeff = CoeffGenerator(stage_state)
y[t in stage] = StageAwareBasisOperator(stage_coeff, t)
```

其中 stage 信息必须在生成 prediction 前进入 representation/operator。

## Step 4：Core Idea

[Idea] `B9-FSN` 的首个可实现 variant 命名为 `B9-FSN-SCF`，即
Stage-Native Coefficient Field。

核心思想是：保留 A6 的 learned temporal basis 作为 unified operator 的公共坐标系，但不再让一个
`coeff[b,c]` 同时承担所有 future stages。模型在 basis projection 前为每个 future stage 生成
`coeff_s[b,c]`，然后用同一组 temporal basis rows 生成该 stage 的预测。

这不是：

```text
y = A6(x) + residual(stage)
```

而是：

```text
coeff_base = learned_basis_coeff(hidden)
coeff_s = StageCoefficientField(coeff_base, hidden, stage_token_s)
y[t in stage_s, c] = learned_temporal_basis[t] @ coeff_s[b,c] + bias[t]
```

stage 信息进入的是 primary prediction path 中的 coefficient field，不是 output 之后的误差修正。

## Step 5：Theoretical Feasibility

[Fact] B9-SGC 已显示四个 future stages 对同一个 `coeff` 的梯度方向低相似：
ETTh2/ETTm1/Weather mean pairwise cosine 为 `0.072/0.171/0.048`。

[Inference] 若总训练目标是 $\mathcal{L}=\sum_s \mathcal{L}_s$，A6 的共享 coefficient 更新方向为
$\sum_s \nabla_{coeff}\mathcal{L}_s$。当这些梯度近正交时，单一 coefficient 需要在多个 stage objective
之间折中；这会表现为 long-tail 或 middle-stage 的 representation pressure，而不是单纯的 output-head
capacity 问题。

`B9-FSN-SCF` 将 coefficient 从一个向量场扩展为 stage-indexed coefficient field：

```text
coeff:       [B, C, K]
coeff_field: [B, C, S, K]
```

其中 `S=4` 对应当前 multi-prefix supervision 的 canonical stages：

```text
S0 = [0, 96), S1 = [96, 192), S2 = [192, 336), S3 = [336, 720)
```

这样每个 stage loss 的主要梯度作用在自己的 `coeff_s` 上，仍通过共享 encoder、共享 base coefficient
和共享 temporal basis 维持 unified model。理论上它解决的是“stage pressure routing”问题，而不是
“增加一个后处理模块”。

## Step 6：Concrete Method Design

### Tensor Path

以 ETT setting 为例，`pred_len=720`、`basis_rank=K=256`、`S=4`：

```text
x_norm = Normalize(x)                         # [B, 96, C]
tokens = PatchEmbed(x_norm)                   # [B*C, patch_num, d_model]
encoded = TimeAlignEncoder(tokens)
hidden = reshape(encoded)                     # [B, C, R]

coeff_base = learned_basis_coeff(hidden)      # [B, C, 256]
stage_token = StageEmbedding(stage_id)        # [4, D_s]
stage_context = StageContext(hidden, stage_token)
coeff_field = StageCoefficientField(
    coeff_base, stage_context
)                                             # [B, C, 4, 256]

pred[:, 0:96, :]    = B[0:96]    @ coeff_field[:, :, 0, :]
pred[:, 96:192, :]  = B[96:192]  @ coeff_field[:, :, 1, :]
pred[:, 192:336, :] = B[192:336] @ coeff_field[:, :, 2, :]
pred[:, 336:720, :] = B[336:720] @ coeff_field[:, :, 3, :]
```

其中 `B = learned_temporal_basis` 仍是 A6 的 learned basis，不引入 DCT/Fourier hand-crafted basis。

### Recommended Minimal Parameterization

首版不做 per-step query，也不做 full cross-attention decoder。推荐最小参数化：

```text
z = LayerNorm(coeff_base)                              # [B, C, K]
m_s = low_rank_mlp([z, stage_token_s])                 # [B, C, K]
g_s = sigmoid(stage_gate_s)                            # scalar or [K]
coeff_s = coeff_base * (1 + g_s * tanh(m_s))
```

设计理由：

- `coeff_s` 是 coefficient field 的 primary representation，不是 prediction residual；
- 乘性 modulation 比 additive output correction 更贴合“operator conditioning”；
- low-rank MLP 限制参数量，避免只靠容量赢；
- `stage_gate_s` 可初始化为接近 `0`，使初始函数严格近似 A6。

若首版乘性表达不足，第二候选才允许加入 coefficient-space bias：

```text
coeff_s = coeff_base * (1 + g_s * tanh(m_s)) + g_s * b_s
```

但这必须作为 ablation，而不是默认首选，因为 additive coefficient bias 更容易被误读为 residual-style
repair。

### Capacity-Preserving Initialization

[Requirement] 初始模型必须能回到 clean A6。

实现上：

- 复制 clean A6 的 `learned_basis_coeff`、`learned_temporal_basis`、`learned_temporal_bias` 初始化路径；
- `stage_gate_s` 初始化到接近 `0`，例如 gate logit 取负值；
- 当 `stage_gate_s=0` 时，所有 `coeff_s=coeff_base`，prediction 与 A6-LBF 完全一致；
- warm-start 若使用已训练 A6 checkpoint，必须记录为 trained-checkpoint capacity transfer；随机权重复制不能声称 preserved learned capacity。

### Prefix Stability

`B9-FSN-SCF` 必须保持 prefix-native 输出：

- 请求 `H=96` 时只计算并返回 `[0,96)`；
- 请求 `H=192` 时 `[0,96)` 的 stage id 与 basis rows 不变；
- 不引入 future placeholder 之间的信息流，因此不会让更长 horizon 的 placeholder 改写已有 prefix。

首轮 small gate 必须报告 prefix consistency：

```text
max_abs(pred_H96 - pred_H720[:, :96])
MSE(pred_H96, pred_H720[:, :96])
```

若该数值明显非零，说明实现破坏 unified prefix contract。

## Contribution Boundary

与 TimePerceiver 相比：

- TimePerceiver 是 generalized forecasting encoder-decoder，target timestamp query 通过 decoder cross-attention
  检索 input representations；
- B9-FSN-SCF 不做任意 target segment forecasting，也不引入 full Perceiver decoder；
- B9 的创新边界是 A6 learned-basis coefficient field：future stage 原生条件化 coefficient/operator。

与 ElasTST 相比：

- ElasTST 用 future placeholders、structured masks 和 horizon reweighting 做 varied-horizon robustness；
- B9 不把 future values 或 placeholders 拼入 encoder，也不复刻 structured future mask；
- B9 只在 basis projection 前引入 stage-conditioned coefficient field，并用 prefix consistency 约束验证。

与 MQ-RNN / TFT 相比：

- MQ-RNN/TFT 依赖 known future covariates 或 horizon-specific context 来服务业务多步预测；
- B9 在标准 LTSF benchmark 中没有外部 known future covariates，stage token 是结构条件变量；
- B9 的输出仍是 deterministic point forecast operator，与 quantile/probabilistic forecasting 不绑定。

与 SRP++ 相比：

- SRP++ 的核心是 step/segment-specific representation adaptation，常见实现形态是 LoRA/expert adapter；
- B9 不走 frozen foundation model + adapter fine-tuning，也不对 encoder 每层做 step-specific LoRA；
- B9 把 step-specificity 限定在 learned-basis coefficient interface，贡献边界更贴近 StageA 的 unified basis
  operator。

## Narrative Gate

`B9-FSN-SCF` 通过 Step 4-6 narrative gate，理由如下：

1. [Problem clarity] B9-SGC 直接证明 primary coefficient path 中存在 stage-gradient conflict；
2. [Mechanism novelty] stage-native coefficient field 是 A6 learned-basis operator 的内部结构扩展，不是
   loss/objective 小修，也不是 output residual；
3. [Tensor explainability] 方法的关键张量从 `[B,C,K]` 扩展为 `[B,C,S,K]`，梯度路由和 prediction path
   都可解释；
4. [Contribution continuity] StageA 解决 unified learned-basis operator，StageB 解决该 operator 在不同
   future stages 下的 native conditional coefficient field；
5. [Implementation feasibility] 可以在 `TimeAlign.py` 的 `_learned_basis_forecast_operator` 附近局部实现，
   不需要恢复 TimeAlign future-recon/align branch。

[Decision] 允许进入最小实现与 small gate。不得直接启动 full main matrix。

## Effectiveness Gate Plan

### Required Controls

首轮 small gate 至少包含：

1. `A6-LBF-r256-clean`: 当前 clean carrier；
2. `B9-FSN-SCF`: stage-native coefficient field；
3. `B9-no-stage-control`: 参数量接近，但所有 stages 共用同一个 learned token 或 gate，用于排除纯容量收益；
4. 可选 `A6-rank-control`: 提高 `basis_rank` 的容量对照，用于检查是否只是 rank 增大。

禁止把 output residual correction 作为 control，因为它不符合当前 paper-core 路线。

### Minimal Remote Gate

先做 small gate，而不是 full matrix：

- datasets: `ETTh2`, `ETTm1`, `Weather`;
- horizons: `96/192/336/720`;
- seed: `2021`;
- compare: MSE/MAE vs clean A6, vs no-stage-control；
- report: per-stage segment MSE, prefix consistency, parameter count, gate activation, coefficient stage variance。

### Pass Criteria

`B9-FSN-SCF` 才能进入 main matrix，当且仅当：

1. overall mean MSE 相对 clean A6 不劣于 `+0.20%`，且至少一个 dataset 有明确改善；
2. 相对 `B9-no-stage-control` 有稳定优势，避免“只是多参数”解释；
3. prefix consistency 数值接近浮点误差；
4. stage gates 没有全部塌缩到 `0`，且 `coeff_field` 的 stage variance 非零；
5. long/mid stages 至少不系统性恶化，避免只牺牲 tail 换 early-stage。

### Failure And Rollback

- 若 `B9-no-stage-control` 与 B9 等价或更强：回滚到 Step 4，说明 stage token 无效，不能 claim native
  future-stage mechanism；
- 若 prefix consistency 被破坏：回滚到 Step 6 implementation design；
- 若 B9 明显劣于 A6：回滚到 Step 2/3，不继续堆叠 stage modules；
- 若收益只来自 additive coefficient bias：不得改写成 residual-style paper story，需重新设计 primary-path
  mechanism。

## Step 7：Implementation Status

[Decision] 最小 B9-FSN-SCF 已实现并通过本地 smoke。代码入口：

- `baselines/timealign_official/models/TimeAlign.py`
- `baselines/timealign_official/train_repo.py`
- `scripts/remote/run_phase5_stage_b_b9_fsn_scf_small_gate.sh`
- `scripts/analyze_phase5_stage_b_b9_fsn_scf_small_gate.py`
- `scripts/sync_phase5_stage_b_b9_fsn_scf_small_gate_results.sh`
- `docs/code-explanation/phase5-stage-b-b9-fsn-scf.md`

新增 readout modes：

| Mode | Role |
| --- | --- |
| `stage-native-coefficient-field` | B9-FSN-SCF candidate |
| `stage-native-coefficient-field-no-stage` | no-stage capacity control |

[Verification] 本地验证已完成：

```text
max_abs_b9_vs_a6_h720 = 0.0
max_abs_no_stage_vs_a6_h720 = 0.0
max_abs_b9_h96_vs_h720_prefix = 0.0
max_abs_no_stage_h96_vs_h720_prefix = 0.0
```

CPU smoke：

- `stage-native-coefficient-field` on ETTh2: 1 epoch / 1 train batch / 1 eval batch passed；
- `stage-native-coefficient-field-no-stage` on ETTh2: 1 epoch / 1 train batch / 1 eval batch passed；
- `model_diagnostics.json` exported with stage gate and parameter diagnostics。

[Next] 进入 Step 8 remote small gate。Runner 默认按 dataset-major ordering 调度
`Weather ETTm1 ETTh2`，并在 `a6_clean / b9_fsn_scf / b9_no_stage` arms 间分散 GPU，避免 slow dataset
堆在同一张 GPU 上。

## Step 9/10：Small Gate Result

Remote small gate 已完成，完整分析见：

- `analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/b9_fsn_scf_small_gate_report.md`
- `analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/b9_fsn_scf_small_gate_summary.csv`
- `analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/b9_fsn_scf_small_gate_comparison.csv`
- `analysis/phase5_stage_b_b9_fsn_scf_small_gate_20260707/b9_fsn_scf_model_diagnostics.csv`

Summary:

| Comparison | Overall MSE wins | Mean relative MSE |
| --- | ---: | ---: |
| `b9_fsn_scf` vs `a6_clean` | `12/12` | `-0.13%` |
| `b9_no_stage` vs `a6_clean` | `12/12` | `-0.13%` |
| `b9_fsn_scf` vs `b9_no_stage` | `2/12` | `+0.00%` |

[Fact] B9-FSN-SCF 相比 clean A6 有很小正收益，但 no-stage control 也有同等甚至略强收益。

[Fact] B9-FSN-SCF 相比 no-stage control 的 overall mean relative MSE 为 `+0.0036%`，MSE wins 为
`2/12`。ETTm1 与 Weather 上 B9 对 no-stage 是 `0/4` wins；ETTh2 只有 `2/4` wins，且差异在
`0.01%` 量级。

[Decision] `B9-FSN-SCF` 被 `no-stage` control 阻断，不能作为 paper-core method 继续推进，也不能把
相对 A6 的微弱收益解释成 native future-stage mechanism。

[Interpretation] 当前结果说明，Step 2/3 的 stage-gradient conflict 仍可能是真问题，但首版
Stage-Native Coefficient Field 没有让 stage token 产生可分辨收益。观察到的收益更可能来自额外 coefficient-space
modulation capacity、训练扰动或 zero-gated shared modulation，而不是 future-stage-aware routing。

[Rollback] 不启动 full matrix。回滚到 Step 4 重新设计 native-stage mechanism；若不能提出能击败 no-stage
control 的机制约束，则回滚到 Step 2/3，重新寻找 StageB 第二主创新点。

# Phase5 StageB B9 Native Future-Stage Operator

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `B9-FSN` |
| `current_step` | Step 2/3：native future-stage-aware problem diagnostic completed |
| `problem` | A6-LBF 用一个 sample/channel-specific `coeff[b,c]` 服务所有 future stages；不同 stage 可能对该共享 coefficient 施加不同训练方向 |
| `existence_evidence` | `B9-SGC` stage-gradient diagnostic 显示三数据集 stage gradients 对共享 coefficient 的 cosine 很低 |
| `idea` | 将 future stage 作为 primary prediction path 的条件变量，生成 stage-native representation/operator，而不是在 A6 output 后做 residual correction |
| `theory_check` | 该路线直接深化 unified prediction：StageA 统一 forecast operator，B9 让 operator 输入端原生感知 future stage |
| `design` | 尚未进入 Step 4-6；当前只定义问题和诊断 |
| `narrative_gate` | `problem_candidate_passed_not_method_ready` |
| `effectiveness_gate` | 未评估 |
| `artifacts` | `analysis/phase5_stage_b_b9_stage_gradient_diagnostic_20260707/` |
| `decision` | `problem_candidate_passed`; 下一步写 Step 4-6 method/narrative gate，不得直接实现 |

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

[Decision] `B9-SGC` 暂定通过 problem-candidate gate。但这还不是 method-ready；下一步必须进入 Step 4-6 narrative/method design，设计 primary-path stage-native operator，并定义 capacity-preserving initialization。

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

## 下一步

进入 Step 4-6 前需要完成：

1. 设计 `B9-FSN` 的 concrete tensor path；
2. 明确如何保持 A6 capacity，例如 zero-gated stage mixing 或 function-preserving fallback；
3. 明确与 TimePerceiver、ElasTST、MQ-RNN 的区别；
4. 定义 small gate，不得直接启动 remote main matrix。

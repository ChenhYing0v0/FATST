# Phase5 C1 Global-Anchored Multi-Patch Carrier Gate

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `C1-GAMP`，carrier normalization，不是 StageB method candidate |
| `current_step` | carrier Step 9-10 completed；effectiveness gate failed |
| `problem` | accepted A6 使用同一个 Encoder class，但 ETTm1 为 `P=1`，ETTh2/Weather 为 `P=48`；这不利于统一 local-token interface 与论文叙事 |
| `existence_evidence` | C0 证明 P1 性能有效；B14 full local-only contextual replacement 退化；exact HPM 证明统一 local patch interface 可构造 |
| `idea` | 所有数据集统一使用 full-window global anchor 与多个 valid local patch tokens；标准 attention 只负责让 global/local tokens 交互 |
| `theory_check` | global anchor 保留 full-window inductive bias；local tokens 提供显式 patch axis；coefficient head 只读取 updated global token，避免 `P*D` flatten capacity 随 scale 改变 |
| `design` | 两种 local scale、统一 dropout policy、三数据集 seed-2021 small gate；不做 parameter matching |
| `narrative_gate` | `control_only`；只允许统一 carrier/interface，不构成 StageB 创新点 |
| `effectiveness_gate` | 允许小幅性能换取一致性，但不得抹掉 Contribution 1 的 cross-dataset evidence |
| `rollback` | 若 shared scale 和 validation-selected scale 都越过 degradation budget，恢复 accepted A6 + exact HPM interface，停止 Encoder normalization |
| `artifacts` | `analysis/phase5_c1_global_anchored_multipatch_gate_20260710/` |
| `decision` | `c1_carrier_normalization_gate_failed`；关闭 exact C1 design，回滚 StageB Step 2/3 |

## Source-informed boundary

本实现复用 repo 已完成的 PatchTST official source audit：channel-independent patch projection、learned
position、residual attention 与 explicit patch axis。C1 不复用 PatchTST flatten head，也不再测试 local-only
full replacement；B14 已证明后者整体退化 `+4.135%/+4.799%`。

C1 的新增工程约束只有一个：在 local tokens 前加入 full-window global token，并让 coefficient head只读取
updated global token。这是标准 global-token aggregation，不作为论文贡献。

## Tensor contract

```text
x [B,720,C]
  -> Normalize
  -> per-channel full-window projection
       global [B,C,1,D]
  -> valid unfold(K,S) + shared local projection
       local [B,C,P,D]
  -> concat + learned position
       tokens [B*C,1+P,D]
  -> one pre-norm residual-attention/FFN block
       contextual tokens [B*C,1+P,D]
  -> global token [B,C,D]
  -> coefficient [B,C,256]
  -> learned basis[:H] @ coefficient
  -> prediction [B,H,C]
```

`encode_history()` 返回 forecast global state `[B,C,1,D]`；`encode_retrieval_memory()` 返回 contextual
local memory `[B,C,P,D]`。因此 downstream modules 不再依赖 legacy `patch_num`。

## Dropout decomposition

Legacy ETTm1 `dropout=0.9` 位于 token-wise residual MLP 内，不能机械复制到 attention。C1 将 dropout sites
完全拆开：

| Site | CLI/config | Small-gate value | 含义 |
| --- | --- | ---: | --- |
| token/input | `history_token_dropout` | 0.0 | global/local projection 与 position 相加后 |
| attention weights | `history_attn_dropout` | 0.0 | softmax attention probabilities |
| attention residual | `history_attn_residual_dropout` | 0.1 | attention output进入 residual前 |
| FFN hidden | `history_ffn_dropout` | 0.1 | GELU后、第二个 Linear前 |
| FFN residual | `history_ffn_residual_dropout` | 0.1 | FFN output进入 residual前 |

首轮不做 dropout sweep，原因是 scale 与 dropout 同时变化会使 6-run gate无法归因。只有某个 scale 接近
performance gate、同时 training/validation gap支持 over/under-regularization时，才授权一个单因素 dropout
sensitivity follow-up。

## Local-token scale

“统一架构”定义为相同 tensor topology 与算子，不强制不同 sampling frequency 使用相同物理时间尺度。
首轮仍对所有 datasets 完整运行两个标准 step-scale arms，避免根据 test结果选择：

| Arm | `patch_len` | `stride` | valid local `P` | Active params | 角色 |
| --- | ---: | ---: | ---: | ---: | --- |
| `gamp_p16s8` | 16 | 8 | 89 | 990,416 | fine local evidence |
| `gamp_p48s24` | 48 | 24 | 29 | 983,248 | coarse/compact local evidence |

Primary decision要求一个 shared scale在三数据集整体通过。Secondary decision允许每个 dataset只依据
`best_val_mean_mse` 选择 scale，再读取相应 test artifact；这是 dataset-level hyperparameter selection，不能
用 test MSE反向挑 scale。

不在首轮加入 multi-scale bank。若单尺度失败就堆叠多个 scales，会同时改变 token count、capacity 与
optimization，且容易把 carrier cleanup扩张成新方法。

## Small-gate arms

所有 C1 arms固定：

- datasets：ETTh2、ETTm1、Weather；
- seed 2021，10 epochs，batch size 32；
- `D=256,n_heads=8,d_ff=512,layers=1`；
- `basis_rank=256,multi-prefix,w_recon=w_align=0`；
- learning rate `1e-4`；
- 同轨迹保存/evaluate last 与 best-val；
- 不要求与 A6 parameter matching，但记录 active parameters、epoch time 与 GPU memory；
- accepted A6 在同一 runner中重跑并产生 dual-checkpoint reference；official-last同时检查是否 exact reproduce现有 clean artifacts。

矩阵为 `(A6 reference + 2 scales) x 3 datasets x 1 seed = 9 runs`。

## Pre-registered gates

### Gate 0: implementation

1. global memory `[B,C,1,256]` 与 local memory `[B,C,P,256]` shape正确；
2. fine/coarse valid patch count分别为 89/29，无 right-padding token；
3. 五个 dropout sites的实际概率分别记录且与表格一致；
4. global/local projection、attention与FFN均有 finite gradients；
5. 四个 prefixes deterministic consistency通过；
6. last/best strict reload 与 dual artifacts通过。

### Gate 1: shared-scale carrier feasibility

对每个 scale，在 last 与 best-val 分别与 accepted A6 比较：

1. overall mean MSE regression `<=+1.0%`；
2. 任一 dataset mean regression `<=+1.5%`；
3. 任一 horizon regression `<=+3.0%`；
4. 相对 fixed TimeAlign overall mean MSE improvement至少 `-3.0%`；
5. 相对 fixed至少 `8/12` MSE wins，且每 dataset至少 `2/4` wins。

### Gate 2: validation-selected scale

若无 shared scale通过，允许每 dataset按 minimum validation mean MSE从两 scale中选择。组合后的 test
metrics仍必须通过 Gate 1。选择规则和 test reading必须在 analyzer中分离。

### Gate 3: local-token use

只有 Gate 1/2通过后，才执行 frozen local-token masking与必要的 global-only same-backbone control。若 local
tokens被忽略，C1最多作为统一 API cleanup，不能声称 multi-patch representation有效。

## Decisions

- Gate 1通过：追加 seeds 2022/2023，确认 shared-scale carrier；
- 仅 Gate 2通过：允许 dataset-specific patch hyperparameter，但保持统一 topology；追加 seeds；
- 性能接近但 validation gap显示明确 regularization问题：只追加一个 dropout sensitivity；
- Gate 1/2失败：恢复 accepted A6 + exact HPM，关闭 C1，不继续 scale/mixer/width sweep。

## Returned effectiveness result

9/9 runs、last/best checkpoints、training logs、effective configs与model diagnostics均完整；无 OOM、NaN、
traceback或runtime failure。预注册 gate结果如下：

| Scale | Selector | Overall vs A6 | Max horizon | ETTh2 | ETTm1 | Weather | vs fixed | Fixed wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P16-S8 | last | `+5.37%` | `+12.26%` | `+8.23%` | `+1.10%` | `+6.78%` | `+0.25%` | 5/12 |
| P16-S8 | best-val | `+3.75%` | `+6.18%` | `+2.80%` | `+2.97%` | `+5.49%` | `+0.03%` | 4/12 |
| P48-S24 | last | `+5.87%` | `+12.20%` | `+4.53%` | `+2.05%` | `+11.02%` | `+0.96%` | 4/12 |
| P48-S24 | best-val | `+4.73%` | `+9.63%` | `+3.96%` | `+1.93%` | `+8.31%` | `+0.98%` | 4/12 |

Validation在三个datasets都选择P48-S24，因此validation-selected组合与P48-S24相同并失败。Weather的
validation差仅`-0.11%`，但test在last/best都偏向P16-S8，说明dataset-specific scale selection也不稳定。

### Protocol audit

Runner对全部arms使用`learning_rate=1e-4`。ETTh2 source preset实际为`5e-4`，所以本轮ETTh2 A6不是
source-faithful exact reproduction；ETTm1/Weather无此偏差。该问题不改变C1裁决：同一ETTh2 comparison
仍是matched-LR control，且相对既有source-faithful A6 official-last，P16-S8/P48-S24整体分别退化
`+4.63%/+5.17%`、均为`0/12` wins。

### Failure attribution

- `hypothesis_false`：不能据此否定所有统一multi-patch carrier；
- `intervention_point_wrong`：可能。random global token aggregation替换了已验证的dataset-specific A6 hidden contract；
- `readout_or_head_design_wrong`：强支持。ETTh2/Weather readout state由`1536/6144`压至`256`；
- `optimization_or_numeric_pathology`：无数值异常，但ETTh2/ETTm1出现早期validation overfit；
- `capacity_control_explains`：只部分解释Weather；C1在ETTh2/ETTm1参数更多仍退化。

[Decision] 失败是exact C1 design级别，不是broader multi-patch方向的理论否定。但按预注册rollback，继续做
readout、width、scale、mixer或dropout search会把control-only cleanup扩张为architecture research，因此不再修补。
恢复并冻结`A6-LBF-r256 + exact valid HPM [B,C,29,48]`。

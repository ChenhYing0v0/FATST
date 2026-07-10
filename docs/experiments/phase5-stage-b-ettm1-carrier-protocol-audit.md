# Phase5 StageB ETTm1 Encoder Control Protocol

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `C0-ETTm1-CPA`，Encoder/carrier control，不是 StageB method candidate |
| `current_step` | C0 Step 9/10 已完成；patch-defect gate failed，回滚 StageB Step 2/3 |
| `problem` | ETTm1 unified A6 同时继承 `patch_num=1`、`d_model=256`、`dropout=0.9` 与 `official-last`，现有证据不能区分全局状态宽度、patch granularity、regularization 与 checkpoint selector |
| `existence_evidence` | code/config audit；frozen residual-MLP ablation；last-vs-best drift；冻结 checkpoint 的跨 patch inclusion-exclusion diagnostic |
| `idea` | 不提出新机制；保持相同的 flattened `C*P` token semantics，用 global-width control、state/capacity control、dropout control 和同轨迹 dual checkpoint evaluation 做最小因果分解 |
| `theory_check` | `P=1` 是 full-window global token，不是信息空洞；`P>1` 同时改变局部投影、hidden state 与参数量，必须用宽度/参数控制拆解 |
| `design` | ETTm1 seed-2021 六臂 small gate；只有预注册 gate 通过后才追加 seeds 2022/2023 |
| `narrative_gate` | `diagnostic_only`；任何结果只允许让 Encoder 更可控、合理，不能成为 StageB 创新点或 Contribution 2 |
| `effectiveness_gate` | patch effect 必须跨 dropout、跨 last/best selector 同号，并排除 global-width 与 parameter-capacity 解释 |
| `artifacts` | frozen interaction diagnostic、六臂 dual-checkpoint metrics、training/segment attribution 均已完成 |
| `decision` | `patch_num_performance_defect_not_supported`；保留 ETTm1 P1-D256-drop0.9 carrier |
| `rollback` | 关闭 ETTm1 patch-defect route，不追加 seeds 或 mixer；StageB 回到 Step 2/3 或暂停 |

## Scope boundary

本实验不是新的 StageB 方法。ETTm1 `patch_num=1` 是 B14 过程中暴露出的 carrier/control 问题；即使
`patch_num>1` 提升性能，也最多授权修复 inherited Encoder preset。它不能证明 future-unit retrieval、
target-set conditioning、MoE routing 或新的 future-aware mechanism 成立。

用户指出 Unified FM 常把多变量 patches 展平到 `C*P` token axis。这一设计本身不是主要风险。因此本
protocol 不再把 positional encoding 的 `C*P` 作用域列为独立实验因素；所有 arms 保留同一
flattened-token semantics，只改变 `P/D/d_ff/dropout`。

## Tensor path

```text
x [B,720,C]
  -> Normalize
  -> permute + flatten [B,C*720]
  -> non-overlap unfold, patch_len=720/P
  -> tokens [B,C*P,D]
  -> positional encoding on C*P
  -> token-wise residual MLP x 2
  -> reshape memory [B,C,P,D]
  -> flatten hidden [B,C,P*D]
  -> coeff [B,C,256]
  -> learned_basis[:H] @ coeff
  -> prediction [B,H,C]
```

只要 `P` 整除 720，non-overlap patches 会在每个 channel 内精确闭合，不跨 channel boundary。
`P=1,D=256` 的 readout state width 是 256；`P=5,D=52` 的 width 是 260（`+1.56%`）。

## Frozen cross-patch interaction diagnostic

在 clean A6 ETTm1 official-last checkpoint 上，将历史划为 5 个 144-step canonical patches。对 patch
pair `(i,j)` 定义 coefficient-space inclusion-exclusion interaction：

$$
I_{ij}=c(x)-c(x^{(-i)})-c(x^{(-j)})+c(x^{(-i,-j)}).
$$

投影到每个 target prefix 后，用 interaction RMS 除以两个 single-patch main-effect RMS 的均值。结果：

- attenuation `0.25`：四个 horizon 的 pair-median mean 为 `0.0634--0.0646`，每个 horizon 有
  `9/10` 或 `10/10` pairs 达到 `0.05`；
- attenuation `0.50`：四个 horizon 的 pair-median mean 为 `0.1282--0.1294`，全部 `10/10` pairs
  达到 `0.05`。

[Strong Evidence] P1 global Encoder 存在稳定的跨时间区域非加性交互。[Boundary] 该诊断不证明显式
token mixer 会提升性能；它只说明 P5 no-mix 不是对 P1 计算图的功能保持分解。因而 mixer 不进入当前
六臂 gate，必须等返回结果后才可作为独立的、条件触发的 control。

## Six-arm small gate

所有 arms 固定：ETTm1；`seq_len=pred_len=720`；prefixes `96/192/336/720`；`basis_rank=256`；
`multi-prefix` loss；`w_recon=w_align=0`；seed 2021；10 epochs；batch size 32；相同 optimizer 和
learning rate；同一次训练同时评估 last 与 best-validation states。

| Arm | `P` | patch length | `D` | `d_ff` | dropout | Active-forward params | 作用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `p1_d256_f256_d09` | 1 | 720 | 256 | 256 | 0.9 | 699,600 | accepted A6 exact control |
| `p1_d384_f96_d09` | 1 | 720 | 384 | 96 | 0.9 | 710,416 | near-parameter-matched wider global-state control |
| `p5_d52_f256_d09` | 5 | 144 | 52 | 256 | 0.9 | 313,468 | near-state-matched lower-capacity patch control |
| `p5_d52_f2048_d09` | 5 | 144 | 52 | 2048 | 0.9 | 689,788 | near-state/parameter-matched patch arm |
| `p1_d256_f256_d02` | 1 | 720 | 256 | 256 | 0.2 | 699,600 | regularization control |
| `p5_d52_f2048_d02` | 5 | 144 | 52 | 2048 | 0.2 | 689,788 | patch-by-dropout interaction control |

`unused proj_x` 不计入 active-forward parameters。它由 legacy model 构造但不进入 clean A6 forward，
因此只报告，不用于 capacity matching。

## Required artifacts

每个 arm 必须产出：

```text
checkpoint.pt
checkpoint_last.pt
checkpoint_best_val.pt
metrics_by_target_horizon.csv
metrics_last_by_target_horizon.csv
metrics_best_val_by_target_horizon.csv
model_diagnostics.json
effective_config.json
```

last 与 best-val 必须来自同一 optimization trajectory。主 `checkpoint.pt` 仍由 `checkpoint_policy`
决定，以保持现有 runner compatibility；本 gate 的比较读取显式 dual artifacts。

## Pre-registered gates

### Gate 0: implementation semantics

1. 所有 arms 保持相同 flattened `C*P` positional semantics；
2. `P` 整除 720，patch 不跨 channel boundary；
3. effective config 精确记录 `P/D/d_ff/dropout`；
4. active/unused parameter count 与预注册值一致；
5. 四个 requested prefixes 通过 deterministic prefix consistency；
6. last/best checkpoints 均可 strict reload，并来自同一训练轨迹。

### Gate 1: global-state bottleneck

比较 `p1_d384_f96_d09` 与 `p1_d256_f256_d09`。它只诊断 P1 global state width 是否限制性能；即使
通过，也只授权合理调整 Encoder capacity，不支持 patch 或新机制结论。

### Gate 2: `patch_num` performance defect

比较 `p5_d52_f2048` 与同 dropout 的 `p1_d256_f256`。只有每个 dropout-selector 组合均满足，才标记
ETTm1 `P=1` 为 carrier defect：

1. 四 horizon mean MSE delta `<= -0.5%`；
2. 至少 `3/4` horizon MSE wins；
3. 任一 horizon regression 不超过 `+1.0%`；
4. effect 在 dropout `0.9/0.2` 与 last/best-val 下同号；
5. `p5_d52_f256_d09` 无 numeric/optimization collapse；
6. 收益不能由 wider P1 control 或 active parameter 差异单独解释。

small gate 通过后才追加 seeds 2022/2023。三 seed confirmation 要求 mean effect 的 95% bootstrap
interval 不跨 0，且至少 `2/3` seeds 保持相同 dataset-level 方向。

### Gate 3: protocol confound

若 patch 收益只存在于 `dropout=0.9` 或只存在于某个 selector，decision 必须是
`patch_effect_confounded_by_regularization_or_selector`，不得修改 active Encoder topology。

### Conditional mixer boundary

只有 P5 no-mix 在 state/parameter controls 下出现退化，且退化与冻结 interaction 证据一致时，才允许
设计独立 mixer control。该 control 的目的仍是恢复可控的 interaction capacity，不是 StageB innovation。

## Decisions

- Gate 2 + multi-seed confirmation 通过：修复 ETTm1 carrier preset，重跑 A6 controlled evidence；不产生
  StageB Contribution 2。
- 仅 Gate 1 通过：调整 global-state capacity control，关闭 patch-defect 强结论。
- 仅 Gate 3 模式：保留 architecture，报告 selector/regularization sensitivity。
- 全部 patch arms 失败：保留 `P=1`，定义为 global-token inductive bias；StageB 回 Step 2/3 或暂停。

## Returned results and final decision

[Fact] Gate 0 完整通过：6/6 arms 的 effective config、active parameter count 与 dual metrics 完整；accepted
P1-D256-F256-drop0.9 official-last control 与先前 clean A6 ETTm1 metrics 逐值一致，MSE/MAE max abs diff
为 `0.0`。

[Strong Evidence] Gate 1 未通过。更宽的 P1-D384-F96 在四 horizons 为 `0/4` wins，mean MSE 相对 accepted
P1 在 last/best-val 下分别为 `+1.34%/+1.44%`。因此没有 global-state-width bottleneck evidence。

[Strong Evidence] Gate 2 反向失败。parameter-matched P5-D52-F2048 在 dropout 0.9 下为
`+4.22%/+4.17%`，dropout 0.2 下为 `+1.92%/+2.50%`；四个 dropout-selector 组合全部 `0/4` wins。
H720 的 8 个 disjoint segments 也全部 `0/8` wins。该结果不授权 seeds 2022/2023。

[Strong Evidence] Gate 3 不是主解释：降低 P1 dropout 到 0.2 使 mean MSE 变化 `+0.79%/+0.34%`，没有
改善 accepted P1；best-val 相对 last 在 accepted P1 上仅改善 mean test MSE `0.15%`，也不改变任何结构
排序。高 dropout 和 official-last 在本 ETTm1 gate 中都不是 patch 结论的 confounder。

[Failure Attribution] 当前结果不支持“ETTm1 `P=1` 是 performance defect”这一窄假设。P5 no-mix 仍可能
属于 `intervention_point_wrong/readout_or_head_design_wrong`，因为 frozen P1 已证明存在跨时间区域 interaction；
但这只说明不能用当前失败否定所有 patchwise encoders。所有 runs 稳定，无 numeric pathology；把 P5
`d_ff` 从 256 提升到 2048 仅带来 mean `-0.30%/-0.38%` 且长 horizons 反向，capacity matching 不足以
恢复 P1。

[Decision] 保留 ETTm1 `P=1,D=256,d_ff=256,dropout=0.9,official-last` 作为 accepted A6 carrier。关闭
patch-defect route，不追加 seeds，不启动 mixer control。后者即使成功，也只回答“P5 如何恢复 P1 interaction
capacity”，已不再回答 inherited P1 是否有缺陷。

## Separate unified-vs-fixed control

Contribution 1 的 fair-task control 与本实验分开：固定同一个 720-step A6 architecture、同一
`P/D/d_ff/dropout/checkpoint policy`，分别训练 single-prefix loss arms，再与 multi-prefix unified arm
比较。本 Encoder control 不回答 unified training 是否有效。

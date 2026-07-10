# Phase5 StageB ETTm1 Carrier Protocol Audit

## 阶段记录

| 字段 | 内容 |
| --- | --- |
| `candidate_id` | `C0-ETTm1-CPA`，carrier/control audit，不是 StageB method candidate |
| `current_step` | B14-FURD Step 3 失败后回滚 Step 2/3；先审计 carrier 与 protocol confounder |
| `problem` | ETTm1 unified A6 同时继承 `patch_num=1`、`d_model=256`、`dropout=0.9` 与 `official-last`；现有证据不能区分 single-token inductive bias、regularization、capacity 和 checkpoint selector |
| `existence_evidence` | code/config audit；ETTm1 `P=1` 无 local patch axis；frozen branch ablation 证明 residual MLP 仍有效；A6 last-vs-best validation drift在 ETTm1 很小、ETTh2 很大 |
| `idea` | 不提出新机制；用 channel-independent patch semantics、state/capacity controls、dropout control 和同一次训练的 dual checkpoint evaluation 做最小因果分解 |
| `theory_check` | `P=1` 是 full-window global token，不是信息空洞；`P>1` 改变的是局部共享投影与 token aggregation，因此必须和参数/hidden state一起控制 |
| `design` | ETTm1 seed-2021 六臂 small gate；只在预注册 gate 通过后追加 seeds 2022/2023 |
| `narrative_gate` | `diagnostic_only`；任何正向结果只允许修复 carrier/protocol，不能成为 Contribution 2 |
| `effectiveness_gate` | patch effect必须跨 dropout、跨 last/best selector同号，且不能由 channel-position语义或参数量解释 |
| `artifacts` | 当前只有 source/config/frozen-checkpoint audit；尚未实现或启动训练矩阵 |
| `decision` | `preregistered_not_launched` |
| `rollback` | 若 patch effect不稳，关闭 ETTm1 `patch_num` performance defect假设；StageB回到 Step 2/3 或暂停第二贡献搜索 |

## 为什么这不是新的 StageB 方法

当前 paper-level StageB 问题仍是：A6 作为 full-trajectory prefix operator 后，是否存在稳定、跨数据集的
future-aware architecture problem。B14-FURD 已在 Step 3 被 `A1 0/6`、`A2 1/6` 阻断。

ETTm1 `patch_num=1` 是 B14 过程中暴露出的 carrier/protocol 问题，不是新的 future-aware mechanism。即使
`patch_num>1` 提升 ETTm1，也最多说明 accepted carrier 的 inherited preset 需要修复；它不能证明
future-unit retrieval、target-set conditioning 或 MoE routing 成立。

## 已确认的 Tensor 事实

legacy A6 history path为：

```text
x [B,720,C]
  -> Normalize
  -> permute + flatten [B,C*720]
  -> non-overlap unfold, patch_len=720/P
  -> tokens [B,C*P,D]
  -> token-wise residual MLP x 2
  -> reshape memory [B,C,P,D]
  -> flatten hidden [B,C,P*D]
  -> coeff [B,C,256]
  -> learned_basis[:H] @ coeff
  -> prediction [B,H,C]
```

ETTm1 official-720 preset为 `P=1,D=256,d_ff=256,dropout=0.9`。因此每个 channel 的整个 720-step
history先投影到一个 256-dimensional global token。该路径没有 local patch axis，但不是“没有使用 history”。

legacy `PositionalEmbedding` 在 `[B,C*P,D]` 上执行，position index同时包含 channel offset与 patch
offset。干净的 channel-independent control必须先 reshape为 `[B*C,P,K]`，再对每个 channel 从 patch
position 0 重新编码；否则 `P=1 -> P>1` 会同时改变 channel identity leakage。

## Small Gate Arms

所有 arms固定：

- dataset：ETTm1；
- `seq_len=pred_len=720`；
- target prefixes：`96/192/336/720`；
- `basis_rank=256`；
- `pred_loss_mode=multi-prefix`；
- `w_recon=w_align=0`；
- optimizer、learning rate、epochs、batch size与 clean A6一致；
- seed-2021；同一次训练同时保存 last 与 best-validation states，并分别评估；
- 不增加 attention、retrieval、future query、residual forecast 或 auxiliary loss。

| Arm | Patch semantics | `P` | patch length | `D` | `d_ff` | dropout | Active-forward params（约） | 作用 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `legacy_p1_d256_f256_d09` | legacy channel-offset PE | 1 | 720 | 256 | 256 | 0.9 | 699.6K | accepted A6 exact control |
| `ci_p1_d256_f256_d09` | channel-independent PE | 1 | 720 | 256 | 256 | 0.9 | 699.6K | isolate channel-position semantics |
| `ci_p5_d52_f256_d09` | channel-independent PE | 5 | 144 | 52 | 256 | 0.9 | 313.5K | near-state-matched lower-capacity patch control |
| `ci_p5_d52_f2048_d09` | channel-independent PE | 5 | 144 | 52 | 2048 | 0.9 | 689.8K | near-state/parameter-matched patch arm |
| `ci_p1_d256_f256_d02` | channel-independent PE | 1 | 720 | 256 | 256 | 0.2 | 699.6K | dropout control |
| `ci_p5_d52_f2048_d02` | channel-independent PE | 5 | 144 | 52 | 2048 | 0.2 | 689.8K | patch-by-dropout interaction control |

`P=5,D=52` 给出 `P*D=260`，与 legacy hidden width 256 相差 `+1.56%`；`d_ff=2048` 后 active
forward parameter count与 legacy相差约 `-1.4%`。`d_ff=256` arm用于防止 parameter matching本身掩盖
或制造 patch收益。

`unused proj_x` 不计入 active-forward parameters。当前 A6 实例仍构造但不调用 official dense
`proj_x`；它在 ETTh2/ETTm1/Weather 分别约有 `1.107M/0.185M/4.424M` parameters，不能用于论证 A6
实际 forecast capacity。

## Required Artifact Change

训练 adapter在本 diagnostic中必须同时保存并评估：

```text
checkpoint_last.pt
checkpoint_best_val.pt
metrics_last_by_target_horizon.csv
metrics_best_val_by_target_horizon.csv
```

这不是运行两次相同训练。两个 checkpoint必须来自同一 optimization trajectory，避免把 seed/CUDA
nondeterminism误当作 selector effect。`patience` 只记录兼容字段，不得声称发生 early stopping。

## Pre-Registered Gates

### Gate 0：实现语义

每个 arm必须通过：

1. patch不跨 channel boundary；
2. channel-independent PE在每个 channel从 position 0重新开始；
3. `P=1` 时 legacy与 CI arm的差异只能来自 channel-offset PE；
4. 记录 active/unused parameter count、state keys和 tensor shapes；
5. 四个 requested prefixes均通过 deterministic prefix consistency；
6. last/best checkpoints均可 strict reload。

### Gate 1：`patch_num` performance defect

比较 `ci_p5_d52_f2048` 与同 dropout的 `ci_p1_d256_f256`。只有同时满足以下条件，才把 ETTm1
`P=1` 标记为 carrier defect：

1. dropout `0.9` 与 `0.2` 下 mean MSE delta均 `<=-0.5%`；
2. last 与 best-val selector下 effect方向一致；
3. 每个 dropout-selector组合至少 `3/4` horizon MSE wins；
4. 任一 horizon不出现 `>+1.0%` regression；
5. `ci_p5_d52_f256_d09` 不发生 numeric/optimization collapse；
6. `ci_p5` 的收益不能只来自 `ci_p1` 相对 legacy的 PE语义修复。

small gate通过后，才给相关 arms追加 seeds 2022/2023。三 seed confirmation要求 mean effect的 95%
bootstrap interval不跨 0，且至少 `2/3` seeds保持相同 dataset-level方向。

### Gate 2：protocol-only confound

若 patch收益只存在于 `dropout=0.9` 或只存在于 `official-last`，decision必须是：

```text
patch_effect_confounded_by_regularization_or_selector
```

此时不得修改 active Encoder；应优先修正实验报告边界与统一/固定对照协议。

### Gate 3：channel-position semantic defect

若 `ci_p1` 已稳定优于 `legacy_p1`，但 `ci_p5` 不优于 `ci_p1`，则问题是 legacy positional/channel
semantics，而不是 `patch_num=1`。该结果只授权 channel-independent implementation repair。

## Separate Unified-vs-Fixed Control

carrier audit之后，Contribution 1 还需要一个独立的 fair-task control：固定同一个 720-step A6 architecture、
同一 `P/D/d_ff/dropout` 与同一 checkpoint policy，分别训练四个 single-prefix loss arms，再与 multi-prefix
unified arm比较。这样 fixed controls仍保留 720-step parameterization，只改变 supervision target，避免把
official per-horizon preset的 width/dropout/patch差异误归因于 unified learning。

该 control与本 patch audit分开报告；不能用一个实验同时回答 carrier tokenization与 unified training
是否有效两个问题。

## Decision Rules

- Gate 1 + confirmation通过：修复 ETTm1 carrier preset，重跑 A6 controlled evidence；仍不产生 StageB
  Contribution 2。
- Gate 3通过：修复 channel-independent PE semantics，`patch_num` route关闭。
- 仅 Gate 2模式：保留 architecture，新增 dual-selector/regularization sensitivity报告。
- 所有 patch arms失败：保留 ETTm1 `P=1`；将其定义为 global-token inductive bias，而不是漏洞；StageB回
  Step 2/3 或暂停。

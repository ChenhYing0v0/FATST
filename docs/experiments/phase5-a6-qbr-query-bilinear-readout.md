# Phase5-A6-QBR Query-Bilinear Readout

本文档是 A6-QBR 的 Step 6 design/code-theory gate。它接在 A8TAG 失败与
`phase5-post-a8tag-candidate-backtracking.md` 之后，目标是先判断 A6-QBR 是否值得实现，
再决定是否进入 local smoke 与 remote gate。

## 11-Step State

| Field | Content |
| --- | --- |
| `current_step` | Step 6：method design 与 code-theory gate |
| `problem` | A5-Q 有 target-query semantics 但 capacity collapse；A6-LBF 恢复 dense-level capacity 但 target position 只作为 learned row table index，缺少 query-side机制解释 |
| `existence_evidence` | A6-LBF-r256 已接近 A6-DER dense-equivalent control，但仍 `0/12` wins vs best controls；A8TAG 证明继续调 self-teacher gate 不是主路径 |
| `idea` | 用 future coordinate/query 生成 bilinear row key，让 requested target position 进入主 readout，同时保留 A6-LBF 的 dense-equivalent coefficient path |
| `theory_check` | A6-QBR 是 A6-LBF 的 coordinate-generated row-key 版本：`feature=P(hidden)`、`row_key=G(q_t)`、`y_t=<feature,row_key_t>+b_t` |
| `design` | 已实现 prefix-native direct head：`query-bilinear-readout`，先做 rank 256/512 两档；不加 teacher、不加 self-distillation |
| `narrative_gate` | conditional pass：若 row-key generator 不退化为 dense table，A6-QBR 可以连接 A5-Q 的 target-query 语义与 A6-LBF 的 capacity path |
| `effectiveness_gate` | pending；remote gate 必须比较 A6-QBR-r256/r512、A6-LBF-r256、A6-DER 与 A7DG best |
| `artifacts` | 本文档；`baselines/timealign_official/models/TimeAlign.py`；`docs/code-explanation/phase5-a6-capacity-native-heads.md`；remote wrapper |
| `decision` | 通过 design/code-theory gate 与本地验证，进入 `ready_for_remote_gate`；remote 前必须 commit/push 并按 GPU policy 启动 |

## Mechanism

### Tensor Contract

沿用 official TimeAlign backbone：

```text
batch_x: [B, L, C]
patch_emb_x + encoder
x_tokens: [B, C, patch_num, d_model]
hidden = flatten(x_tokens): [B, C, R]
```

A6-QBR head：

```text
feature = feature_proj(hidden): [B, C, K]
tau_H = [1 / 720, ..., H / 720]: [H]
query_features = [tau, tau^2, sin/cos bands, optional global-free coordinate features]
row_key = row_key_mlp(query_features): [H, K]
bias_H = row_bias[:H]: [H]
output = einsum("bck,hk->bch", feature, row_key) + bias_H
return output.permute(0, 2, 1): [B, H, C]
```

### Prefix Consistency

`row_key_t` 只依赖 absolute coordinate `t / 720`，不依赖 requested horizon `H`。因此
`decode(96)` 与 `decode(720)[:, :96]` 在 eval/no-dropout 下共享同一 `row_key[:96]`、同一
`feature_proj(hidden)` 与同一 `row_bias[:96]`，应达到 numerical prefix-invariance。

### Capacity Path

若 `K` 足够大，`feature_proj(hidden)` 与 `row_key_t` 的 bilinear product 可以近似 A6-LBF 的
`learned_basis_coeff(hidden)` 和 `learned_temporal_basis[t]`。差异在于：

- A6-LBF 的 row basis 是自由参数表 `learned_temporal_basis[t]`；
- A6-QBR 的 row key 来自 coordinate/query generator `G(q_t)`；
- 因此 QBR 不是简单 dense row table，而是给 future position 一个可解释的 continuous row-key path。

## Design Choices

| Item | Decision | Reason |
| --- | --- | --- |
| row key source | coordinate MLP from absolute step features | 避免直接 learnable row embedding 退化成 dense rows |
| prefix feature | 不输入 requested `H`，只输入 absolute `t / 720` | 保证 architecture-level prefix invariance |
| rank | `K=256` 与 `K=512` | 与 A6-LBF-r256/r512 对齐，避免参数量 confounder |
| bias | learnable `row_bias[:H]` | 对齐 A6-LBF/A6-DER 的 row bias capacity |
| dropout | 不在 row-key generator 内加 dropout | prefix-invariance smoke 应先排除 stochastic path |
| teacher/self-teacher | disabled | A8TAG 后 self-teacher route 暂停，避免机制混合 |

## Narrative Gate

| Gate Item | Assessment |
| --- | --- |
| problem motivation | strong：直接回应 A5-Q semantics 与 A6-LBF capacity 的断裂 |
| mechanism novelty | medium-strong：future coordinate generates bilinear row keys rather than selecting dense rows |
| tensor/gradient path | strong：每个 requested position 的 `row_key_t` 直接接收 supervised loss 梯度 |
| capacity preservation | medium-high：rank 与 A6-LBF 对齐；若 `K=512` 仍不如 A6-LBF，则 row-key generator 是瓶颈 |
| contribution boundary | conditional：必须证明不是 A6-LBF 改名，也不是 generated dense table |

## Code-Theory Consistency Requirements

实现后必须满足：

1. `readout_mode == "query-bilinear-readout"` 属于 direct-prefix mode；
2. `target_prefix=H` forward 直接返回 `[B,H,C]`；
3. eval/no-dropout prefix-invariance smoke：`decode(96)` 与 `decode(720)[:, :96]` max diff 应接近
   numerical zero；
4. 参数量记录必须报告 `feature_proj`、`row_key_mlp`、`row_bias`，并与 A6-LBF-r256/r512 对照；
5. training log 不加入 teacher/self-teacher 字段作为机制证据。

## Minimal Experiment Plan

### Local Verification

- `python -m py_compile` touched Python files；
- CPU smoke：ETTh2，`max_train_batches=1`、`max_eval_batches=1`；
- prefix-invariance helper：same batch 下 `H=96` 与 `H=720` overlap max diff；
- output shape check：`H=96/192/336/720` 均为 `[B,H,C]`。

### Remote Gate

若本地验证通过，启动 ETTh2/ETTm1/Weather × 2 variants：

| Variant | readout mode | rank | Role |
| --- | --- | ---: | --- |
| `a6qbr_r256` | `query-bilinear-readout` | 256 | 与 A6-LBF-r256 同 rank 的 primary check |
| `a6qbr_r512` | `query-bilinear-readout` | 512 | 检查 row-key generator 是否需要更高 rank |

Controls 不重跑，使用已有 artifacts：

- A6-LBF-r256 / A6-DER：`analysis/phase5_timealign_hss_a6_capacity_native_gate_20260703/`
- A7DG best：`analysis/phase5_timealign_hss_a7dg_selective_self_teacher_gate_20260704/`
- A8TAG negative evidence：`analysis/phase5_timealign_hss_a8tag_teacher_advantage_gate_20260705/`

## Pass / Fail Rules

[Pass Candidate] A6-QBR 至少需要满足：

- 相对 A6-LBF-r256 有整体改善，且不能只来自单一 dataset/horizon；
- ETTh2 不劣于 A7DG 的主要 positive signal 太多，或能解释为什么 capacity-side route 不需要 self-teacher；
- ETTm1/Weather 不出现 A6ST/A7DG 式 consistency damage；
- 与 best stage controls 的 gap 必须小于 A7DG best，最好产生新的 wins。

[Fail] 若 A6-QBR-r512 仍弱于 A6-LBF-r256，说明 coordinate-generated row key 是 bottleneck，不能继续加 rank
或加入 teacher；应回 Step 2/3 重审 Stage A architecture 是否已接近上限。

## Decision

[Decision] A6-QBR 通过 Step 6 design/code-theory gate，可进入实现与本地验证。它不是 self-teacher
补丁，也不是 A5-Q attention decoder 的简单复活；其核心是将 target-query semantics 放入
dense-equivalent bilinear readout 的 row-key generation。

## Implementation And Local Verification

[Code] 新增 `readout_mode == "query-bilinear-readout"`：

- `qbr_feature_proj(hidden): [B,C,R] -> [B,C,K]`;
- `qbr_row_key(query_features): [H,9] -> [H,K]`;
- `qbr_row_bias[:H]: [H]`;
- `einsum("bck,hk->bch") -> [B,C,H] -> [B,H,C]`。

[Verification] 已完成：

- `python -m py_compile baselines/timealign_official/models/TimeAlign.py baselines/timealign_official/train_repo.py`;
- `bash -n scripts/remote/run_phase5_timealign_hss_a6qbr_query_bilinear_gate.sh`;
- model-level shape smoke：`H=96/192/336/720` 分别输出 `[2,H,7]`;
- prefix-invariance smoke：`decode(96)` 与 `decode(720)[:, :96]` 的 `prefix_overlap_max_diff=0`;
- CPU data-loader smoke：ETTh2，`max_train_batches=1`、`max_eval_batches=1`，成功写出
  `training_log.csv` 与 `metrics_by_target_horizon.csv`。

[Decision] 本地验证通过，A6-QBR 可进入 remote gate。remote gate 仍保持 ETTh2/ETTm1/Weather ×
`a6qbr_r256/a6qbr_r512`，不加入 teacher/self-teacher。

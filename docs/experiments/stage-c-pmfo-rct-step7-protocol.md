# StageC PMFO-RCT Step 7 Protocol

## Protocol Status

| Field | Content |
| --- | --- |
| `candidate` | `SC1-PMFO-RCT` |
| `current_step` | Step 7A local implementation/invariant gate |
| `carrier` | frozen `A6-LBF-natural-baseline` Encoder contract |
| `objective` | full H720 MSE only；MIPR forbidden in SC1 gate |
| `implementation_authorized` | local module/tests only |
| `remote_training_authorized` | `false` |
| `rollback` | invariant/readout fault -> Step 6；matched control explains -> Step 4 |

## Question

PMFO-RCT的refinement-conservative future tree，是否在不修改A6 Encoder、不读取horizon semantic feature的
条件下，形成exact prefix projectivity、local support以及不能被dense/no-transition capacity control解释的
forecast improvement？

## Frozen Tensor Contract

```text
history memory M: [B,C,P,D], P*D=768
flattened history z: [B,C,768]
coarse node state u0: [B,C,8,d_state]
coarse scaling a0: [B,C,8]
radices: [3,3,2,5]
detail groups: [16,48,72,576]
unit leaves / normalized forecast: [B,C,H]
denormalized output: [B,H,C]
```

`H`只允许进入active-node count、weight slicing与final prefix slicing；不得进入`seed/split/detail` modules、
embedding、normalization statistics、attention或router。

## Step 7A: Local Implementation Gate

### Required variants

1. `pmfo-rct`：shared parent-to-child state transition + orthogonal conservative synthesis；
2. `pmfo-rct-no-transition`：每层直接从history state产生，保留相同synthesis；
3. `pmfo-rct-no-conservation`：保留tree states，但children readout不受orthogonal detail constraint；
4. `dense-mlp-matched`：相同nonlinearity与近似parameter budget，直接输出future rows。

### Required local tests

| Test | Gate |
| --- | --- |
| all natural profiles instantiate | Weather/ETTm1/ETTh2全部通过 |
| output shape | horizons `1/48/96/192/336/720`精确为`[B,H,C]` |
| full-versus-prefix | max absolute gap `<=1e-6` in float32 eval mode |
| refinement recovery | parent/detail reconstruction max gap `<=1e-6` |
| conservation | detail perturbation后的parent projection max gap `<=1e-6` |
| locality | detail perturbation在其support外max change `<=1e-6` |
| horizon path audit | learned modules不接收H；无node-count-dependent normalization |
| parameter/FLOP report | 四个variants与A6均报告；不得隐去capacity差异 |

[Boundary] params差异不参与dataset profile选择；但在本mechanism gate中，capacity control用于failure
attribution。`dense-mlp-matched`与`no-transition`若解释收益，SC1不能通过。

Step 7A通过后，先更新ledger、code explanation、commit/push；再按remote policy检查GPU并单独授权Step 7B。

## Step 7B: Architecture-Only Screening

### Matrix

- datasets：ETTm1、ETTh2；
- seed：2021；
- profiles：已冻结natural profiles；
- arms：A6、dense matched、no-transition、no-conservation、PMFO-RCT；
- loss：full H720 MSE；
- evaluation：all integer horizons `1..720`可由一次full prediction聚合，同时报告
  `48/96/192/336/720`与deployment-risk AUC；
- checkpoint/stopping：沿用natural carrier contract，不重新调参。

选择ETTm1是因为D1 ordered-memory/linear probe较强；选择ETTh2是因为其linear probe为负但frozen nonlinear
head有效，是对decoder interface的stress test。该选择在method results前固定，不使用test performance做选择。

### Screening gate

Step 7B只形成`partial_pass`或rollback，不直接形成paper claim。`partial_pass`要求同时满足：

1. PMFO-RCT相对A6在两dataset合并dense-horizon mean MSE至少改善`1.0%`；
2. 任一dataset的dense-horizon mean MSE不得稳定恶化超过`0.5%`；
3. 相对best matched structural control仍至少改善`0.5%`；
4. prefix/refinement invariants在trained checkpoint继续通过；
5. 无divergence、>100% degradation、validation/test protocol mismatch或明显numeric pathology。

单seed通过只授权3-seed confirmation；不允许写成effectiveness gate pass。

## Failure Attribution

- invariant失败：`readout_or_head_design_wrong`或`implementation_fault`，回Step 6；
- PMFO≈dense matched：`capacity_control_explains`，回Step 4；
- PMFO≈no-transition：recursive refinement mechanism不成立，回Step 4；
- 仅ETTh2失败且出现interface证据：`intervention_point_wrong`待审计，不直接更换Encoder；
- divergence/异常退化：`optimization_or_numeric_pathology`，只能否定本实现；
- 稳定且被controls击败：`failed_as_core_candidate`，不得叠加MIPR/Encoder/MoE掩盖。

## Expected Artifacts

- model/code explanation与unit tests；
- variant parameter/FLOP table；
- effective configs与environment manifest；
- per-horizon MSE/MAE、deployment-risk aggregates与invariant checks；
- blocking-control comparison与failure-attribution report。

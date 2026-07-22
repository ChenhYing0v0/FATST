# ISCF-SCC Step7A Implementation and Remote Gate

## 1. Long-stage record

| Field | Record |
| --- | --- |
| `current_step` | SCC Step7A complete；Step7B remote validation prelaunch authorized |
| `problem` | 实现exact train-only coalition credit且保持ISCF inference architecture不变 |
| `existence_evidence` | corrected D0B全部gate通过 |
| `idea` | 在existing PCC objective infrastructure新增coalition与shuffled modes |
| `theory_check` | credit inputs全部detach；route KL只更新policy；fused L1仍joint更新arms/policy |
| `design` | 20 new runs + 5 frozen parent；EQUAL/FUSED/ARMERR/SCC/SHUFFLED |
| `narrative_gate` | passed in Step5–6 |
| `effectiveness_gate` | pending validation artifacts |
| `artifacts` | SCC loss、dedicated shuffle RNG、gradient logging、checker、config、runner |
| `decision` | `step7a_pass_step7b_remote_validation_authorized` |

## 2. Code realization

`PCC.py`新增两种mode：`scope_coalition_credit`和`scope_coalition_credit_shuffled`。closed-form removal在
`fused [B,T,C]`与`arms/policy [B,C,T,5]`上完成；fused、arms、policy、target在credit path入口detach。all-negative
coordinate回退uniform，route weight沿用冻结的25% ramp-to-`.1`。

SHUFFLED使用`seed + 20260722`的dedicated generator逐coordinate产生scope permutation；不消费global RNG。
`train_repo.py`记录SCC effective contract，并在backward后、step前保存five independent arm mode gradient norms。
model forward、readout parameters与inference graph均未修改。

## 3. Matrix and launch contract

`configs/stage_c_iscf_scc_step7b.json`冻结4 new objectives × 5 datasets × seed2021=20 runs，外加5个historical
EQUAL parent。dataset profiles、initialization class、rank、optimizer、checkpoint selector和evaluation horizons matched。
runner从arm config读取objective mode；historical SPS/FRSC configs缺省仍为`equal_skill`，保持兼容。

remote outputs固定到：

```text
/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_scc_v0_step7b
```

先在Weather的SCC与SHUFFLED执行2-batch/1-epoch resource smoke；通过后再启动20-run validation matrix。正式test入口
硬禁用。Step9 analyzer不在launch critical path，将在artifact返回前按冻结comparison/gate单独实现并验证。

## 4. Verification

通过：

- `py_compile` for `PCC.py`、`train_repo.py`、SCC checker；
- SCC exact credit、uniform fallback、route-gradient boundary；
- shuffled credit reproducibility、marginal preservation、global RNG isolation；
- existing PCC 36/36 regression checks；
- config JSON parse、20-job runner dry-run、shell syntax、diff check。

historical `check_stage_c_siff_equal_attribution_step7a.py`曾因其已完成阶段的remote authorization guard状态变化报告
11/13；其model/gradient/evaluator 11项通过，失败项只检查旧阶段authorization boundary，不是本次SCC code regression，故不作为
SCC gate。该旧checker未被修改。

## 5. Authorization boundary

```text
active_method = SC-ISCF-SCC-v0_step7b_validation_candidate
step7a_implementation = passed
remote_resource_smoke_authorized = true
remote_20_run_validation_authorized = conditional_on_resource_smoke
formal_test_authorized = false
modern_baseline_matrix_authorized = false
```

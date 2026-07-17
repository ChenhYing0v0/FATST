# StageC Post-PCC SIFF/MCCA Protocol

## Current Position

| Field | Value |
| --- | --- |
| `architecture_candidate` | `SC1-SIFF` |
| `training_candidate` | `SC2-MCCA` |
| `current_step` | Step8 / Step7B seed2021 validation-only 55-run remote running |
| `problem_gate` | PCC arm recovery positive；same-label homogenization confirmed |
| `theory_gate` | 10/10 pass |
| `narrative_gate` | conditional pass |
| `implementation/remote/test` | Step7A complete / commit `7a9e5c7` on GPU0/1/2 running / false |

## Candidate Contracts

`SC1-SIFF`把coupling scale作为decoder内部连续坐标，生成scope-conditioned history modes；requested horizon不进入
decoder computation，模型仍先生成完整$T=720$ forecast，再做prefix crop。

`SC2-MCCA`在projective target measure row marginal与scope skill-budget column marginal下分配competitive arm
credit。assignment只用于one-stage training，inference graph不变；balanced OT/Sinkhorn本身不作为novelty。

## Step5 Evidence

- SIFF Q1 containment gap `3.5527e-15`；prefix gap `0`；
- current constant-coordinate scope gap `0`，SIFF witness gap `1.0`；
- MCCA row/column gap `6.25e-17/1.11e-16`；
- crossed specialization gain `0.6667`，dominant-arm case minimum scope mass `0.2`；
- skill/router gradients finite；test=false。

## Step6 Frozen Design

SIFF-v1固定$Q=2,D=4,K=256$，scale basis为`[constant, centered unit-RMS log-scale]`：

```text
hidden [B,C,R]
 -> component modes [B,C,Q,D,K]
 -> scale basis [S,Q]
 -> scale-indexed modes [B,C,S,D,K]
 -> existing scope pooling/shared synthesis
 -> arms [B,C,S,T]
```

MCCA-v1以batch-channel-target rows和projective measure构造assignment。其column marginal严格等于current PCC在同一
training progress下给予各scope的总skill mass；区别仅是PCC将floor逐target均匀分配，MCCA通过log-domain
I-projection竞争性分配。assignment stop-gradient，inference graph不变。

22/22 design cases通过：float64/float32 marginal gap分别`3.86e-10/1.04e-7`，same-mass PCC column gap
`5.55e-17/2.98e-8`，MCCA reference-KL advantage `0.107352`；integer-rank matched controls最大parameter gap
`0.3893%`。

Step7B必须使用`PCSD/SIFF × EQUAL/PCC/MCCA`的$2\times3$ factorial，并加入SIFF-constant、scale-permuted、
Q1-wide、independent-scope、dense matched、pointwise MCCA与uniform balanced OT controls。参数量只作mechanism
attribution，不作dataset profile或candidate选择。

若SIFF被constant/generic width/independent experts解释，或MCCA不超过same-mass PCC与generic balanced OT，则按
冻结规则回Step4；不得堆叠新机制。

## Active Artifacts

- `analysis/stage_c_sc2_pcc_step7b_seed2021_20260717/step9_10_result_and_failure_attribution.md`；
- `analysis/stage_c_post_pcc_step4_redesign_20260717/source_informed_redesign_audit.md`；
- `analysis/stage_c_post_pcc_step5_theory_20260717/step5_theory_feasibility.md`；
- `analysis/stage_c_post_pcc_step6_design_20260717/step6_source_method_control_design.md`；
- `analysis/stage_c_post_pcc_step7a_local_20260717/step7a_implementation_gate_report.md`；
- `analysis/stage_c_post_pcc_step7b_prelaunch_20260717/prelaunch_report.md`；
- `analysis/stage_c_post_pcc_step7b_prelaunch_20260717/remote_launch_record.md`；
- `configs/stage_c_post_pcc_step6.json`；
- `configs/stage_c_post_pcc_step7b.json`。

## Step7A / Step7B Prelaunch Result

Step7A共36/36 cases通过：SIFF Q1/A6 containment gap均为`0`，constant collapse gap `3.55e-15`，
five-profile matched-rank最大gap `0.383856%`；MCCA float64/float32 marginal gap分别为`1.11e-16`与
`4.47e-8`，same-mass PCC gap最大`2.78e-17`，arm/policy gradient均非零。

Step7B prelaunch 8/8 categories通过，冻结55个new runs。runner按dataset-major slow-first在3个GPU worker间调度，
先执行Weather；每个run保存dense validation metrics、trained invariants与mechanism diagnostics。remote启动前必须完成
resource smoke。`2026-07-17T14:59:22+08:00`已从commit `7a9e5c7`在GPU0/1/2启动；启动后不轮询值守，
等待用户通知完成。test与confirmation seeds仍未授权。

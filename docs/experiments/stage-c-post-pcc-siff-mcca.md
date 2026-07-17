# StageC Post-PCC SIFF/MCCA Protocol

## Current Position

| Field | Value |
| --- | --- |
| `architecture_candidate` | `SC1-SIFF` |
| `training_candidate` | `SC2-MCCA` |
| `current_step` | Step5 theory feasibility pass；Step6 source/control design next |
| `problem_gate` | PCC arm recovery positive；same-label homogenization confirmed |
| `theory_gate` | 10/10 pass |
| `narrative_gate` | pending Step6 |
| `implementation/remote/test` | false / false / false |

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

## Step6 Gate

Step6必须冻结：

1. production SIFF tensor flow、$Q/rank$与parameter budget；
2. MCCA batch-level $\rho_s$定义、Sinkhorn numeric policy与stop-gradient boundary；
3. `PCSD/SIFF × EQUAL_SKILL/MCCA`的$2\times2$ factorial；
4. Q1-wider、dense-matched、independent-scope、generic balanced OT、pointwise capability与current PCC controls；
5. complete-chain novelty boundary，明确AME-TS/MoHETS/Expert Loss Integration/BASE/SSR/orthogonality methods覆盖内容；
6. code-theory falsification gates。

Step6未通过前禁止Step7A implementation。若SIFF被generic width吸收、MCCA只等价于generic load balance，或两者无法
形成可归因factorial，则回Step4。

## Active Artifacts

- `analysis/stage_c_sc2_pcc_step7b_seed2021_20260717/step9_10_result_and_failure_attribution.md`；
- `analysis/stage_c_post_pcc_step4_redesign_20260717/source_informed_redesign_audit.md`；
- `analysis/stage_c_post_pcc_step5_theory_20260717/step5_theory_feasibility.md`；
- `configs/stage_c_post_pcc_step5.json`。

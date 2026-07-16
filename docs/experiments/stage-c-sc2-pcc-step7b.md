# StageC SC2-PCC-v1-TI Step 7B Protocol

## Current Position

| Field | Value |
| --- | --- |
| `candidate` | `SC2-PCC-v1-TI` |
| `current_step` | Step7B prelaunch pass；11-step Step8 remote seed2021 authorized |
| `prelaunch_gate` | 8/8 categories；45 CLI contracts |
| `matrix` | 9 objective modes × 5 datasets × seed2021 = 45 runs |
| `evaluation` | validation dense H1..720 full-crop |
| `checkpoint` | best validation H720 MSE |
| `test` | false |
| `confirmation/Phase B` | false / false |

## Frozen Matrix

所有新runs使用同一个PCSD-CF DIRECT architecture与from-scratch seed2021 initialization，仅改变
`--pcc-objective-mode`：

1. `measure_only`；
2. `equal_skill`；
3. `pointwise_route_only`；
4. `pointwise_capability_skill_only`；
5. `pointwise_prior_composed`；
6. `pointwise_pcc_v0`；
7. `transport_skill_only`；
8. `transport_route_only`；
9. `pcc_transport_full`。

Datasets按`Weather -> ETTm1 -> ETTm2 -> ETTh1 -> ETTh2`排列，每个dataset内部遍历nine modes。三个fixed GPU
workers用stride领取任务，使慢dataset优先铺满GPU，避免arm-major queue造成某个fast worker长期空闲。

## Locked References

以下seed2021 checkpoints来自已完成的PCSD-CF matrix，只读、不重训：

- `A6`；
- plain `PCSD DIRECT`；
- `DENSE_MATCHED`；
- five independently trained fixed-scope arms。

remote reference root：`/home/yingch/exp_outputs/r-2026-fatst/stage_c_pcsd_cf_step7b`。

## Evaluation And Diagnostics

每个run必须产生：

- `metrics_by_target_horizon.csv`：validation H1..720 MSE/MAE；
- `training_log.csv`：20个PCC optimization diagnostics；
- `pcsd_validation_diagnostics.npz`：arms、policy、oracle、pairwise diversity；
- `trained_invariants.json`：prefix、protocol、finite与test=false；
- `pcc_shared_gradient_diagnostics.json`：best-val checkpoint在first sequential train row上的five scope shared-field
  gradient norms与10个pairwise cosines。

Shared-gradient diagnostic只作failure attribution，不进入optimizer，不执行gradient surgery。

## Hard Gates

full `pcc_transport_full`必须同时满足：

- vs A6：至少3/5 wins且macro gain ≥ 0.3%；
- vs plain DIRECT：至少3/5 wins且macro gain ≥ 0.5%；
- vs `pointwise_pcc_v0`：至少3/5 wins且macro gain ≥ 0.2%；
- vs `pointwise_prior_composed`：至少3/5 wins且macro gain ≥ 0.2%；
- 25 arm pairs中至少15个degradation改善，median relative reduction ≥ 30%；
- 每个dataset的minimum pairwise NRMSE至少保留plain DIRECT的50%；
- policy normalized entropy min ≥ 0.3，usage max ≤ 0.9。

45/45 complete前禁止partial method judgment。Phase-A pass也只进入conditional Phase-B review，不自动授权test或seeds。

## Commands

Local prelaunch：

```bash
conda run -n r2026-fsa python scripts/check_stage_c_sc2_pcc_step7b.py
CONDA_BIN=/opt/anaconda3/bin/conda CONDA_ENV=r2026-fsa DRY_RUN=1 \
  bash scripts/remote/run_stage_c_sc2_pcc_step7b.sh
```

Remote resource smoke：

```bash
GPU_IDS="0" RESOURCE_SMOKE=1 bash scripts/remote/run_stage_c_sc2_pcc_step7b.sh
```

Remote full matrix：

```bash
GPU_IDS="0 1 2" SEED=2021 bash scripts/remote/run_stage_c_sc2_pcc_step7b.sh
```

## Prelaunch Decision

`step7b_prelaunch_pass_remote_seed2021_authorized`。本结果只证明protocol/tooling可以启动。remote run返回前，
PCC effectiveness、Contribution 2 claim与joint paper story仍为unknown。

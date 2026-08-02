# ISCF-BSCA-MAIN-v1 H4J Launch Record

## Launch summary

| Field | Value |
| --- | --- |
| `date` | 2026-08-02 |
| `remote_project` | `/home/yingch/projects/FATST` |
| `commit` | `a7d23780311e75f05d04dd3843dd7b29ceef6b08` |
| `config` | `configs/iscf_bsca_main_v1_hpo_joint_h4j.json` |
| `config_hash` | `0a82eff46821776d365b2cf61c832b213f3ab56c91504652c7414a3b09468591` |
| `search_space_hash` | `3d6ba6fd10dfe5053a387e50f5191502f35973ac75d2f68bcef254f2c2326b14` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h4j_r1` |
| `pipeline_pid` | `4032407` |
| `resource_smoke` | 40/40 pass after fail-closed protocol repair |
| `training_start` | `2026-08-02T12:10:47+08:00` |
| `training_matrix` | 40 jobs；seed2021；validation checkpoint selection only |
| `official_test_at_launch` | 0/40 |
| `decision` | `H4J_40_job_training_active_test_zero` |

## 1. Remote preflight

Launch前remote HEAD与pushed commit一致。远端worktree保留三个与本轮无关的historical analysis CSV modifications；未reset、未覆盖。GPU 0/1/2均为RTX 3090，launch前memory used=`18 MiB`、utilization=`0%`；`/home`剩余约871 GB。Remote H4J checker通过40 jobs、dataset counts、28 weak-dataset jobs、joint selector、40/56 gate、evidence hashes与`seq_len % patch_num == 0` invariant。

## 2. Fail-closed smoke and repair boundary

首次output root `.../h4j`的resource smoke中，三个`seq_len=512` profiles因原patch count不能整除lookback而在trainer参数检查阶段退出。Pipeline按`set -euo pipefail`没有进入training，test=0。该root只保留failed preflight evidence，不进入training/test manifest。

Repair commit `a7d2378`把三项profile改为`patch_num=16`并新增static divisibility invariant。新的`.../h4j_r1` root重新执行全部40 jobs，而不是只补三个失败项；40/40 two-batch train/validation smoke于`2026-08-02T12:10:46+08:00`完成，无OOM、NaN或Inf。Observed smoke memory peak约4.1 GiB，保留了约20 GiB safety margin。

## 3. Full training launch

Smoke通过后同一fail-closed pipeline自动切换到full train/validation：

```text
GPU0: ECL__h4j_exact_budget60
GPU1: ECL__h4j_exact_lr3e4
GPU2: ECL__h4j_intermediate_budget45
```

首批三项在epoch 1运行，约100秒内达到1400--1500 iterations；observed GPU memory约2.0--2.1 GiB、utilization约69--77%。当前complete=`0/40`、test=`0/40`。Global queue继续按ECL -> Solar -> Weather -> ETT/Exchange填充空闲GPU。

[Estimate] 综合H3A历史吞吐、H4J job composition和首批ECL速度，conservative wall-clock ETA为24--40小时；early stopping可缩短。每次进度报告必须给出complete jobs、active dataset/trial、active epoch与更新后的ETA。

## 4. Next gate

Full training必须达到40/40 artifact completeness、finite H96/H192/H336/H720 validation MSE/MAE与checkpoint hashes。Validation不进行profile ranking。完成后立即：

1. 运行H4J analyzer做artifact/numeric audit；
2. 冻结40-row checkpoint manifest；
3. 核对remote commit、GPU与test target directory为零；
4. 对40个checkpoints执行complete official-test audit；
5. 将H4J与53个existing trials合并，按frozen joint selector产生每dataset唯一profile；
6. 报告MSE、MAE及combined leading-cell gates，不隐去negative trials。

Decision=`H4J_40_job_training_active_test_zero`。

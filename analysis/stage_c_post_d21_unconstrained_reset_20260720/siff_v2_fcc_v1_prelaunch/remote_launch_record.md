# SIFF-v2 FCC-v1 Remote Launch Record

## 1. Launch identity

| Field | Record |
| --- | --- |
| `launch_time` | `2026-07-21T12:54:37+08:00` |
| `remote_host` | `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `commit` | `87bea35678475d652a9de0df2e8e969ff9bd2c70` |
| `candidate_version` | `SC1-SIFF-v2-FCC-v1` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_v2_fcc_v1` |
| `historical_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_equal_attribution_v2` |
| `config_sha256` | `bd936240b3e77c028a51373ef910264900d178f6cda8617a8ed71e2623ea680a` |
| `profile_sha256` | `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a` |
| `supervisor_pid` | `1908153`；driver PID `1908154` |
| `training_matrix` | 30 new runs；45 effective runs after seed2021 reuse |
| `formal_test` | not launched；blocked until 30/30 training |

## 2. Remote repository boundary

remote执行`git pull --ff-only`并从`4cc96f2` fast-forward到`87bea35`。pull前已有三项与FCC无关的generated CSV
数值微差修改：

- `analysis/stage_c_d14a1_dual_carrier_grouped_mlp_20260715/local_gate/parameter_and_theory_cases.csv`；
- `analysis/stage_c_pcsd_cf_step7b_prelaunch_20260716/arm_matrix_checks.csv`；
- `analysis/stage_c_sc2_pcc_step7b_prelaunch_20260717/initialization_pairing.csv`。

这些remote changes已保留，未清理、未提交；与本次FCC新文件不重叠。

## 3. GPU preflight and resource smoke

launch前GPU0/1/2均为RTX 3090，memory used均18 MiB、free均24107 MiB、utilization均0%，且无compute
process。resource smoke并行使用GPU0/1：

| Smoke | Contract | Result |
| --- | --- | --- |
| Weather `siff_equal` seed2022 | `siff-coupling-field` + `equal_skill` | finite；train loss `1.7185802`；no OOM |
| ETTm2 `siff_independent_equal` seed2023 | independent field + `equal_skill` | finite；train loss `1.0052406`；no OOM |

smoke artifacts写入`OUTPUT_ROOT/_resource_smoke`，不计入30-run matrix。

## 4. Launch command and initial progress

training以冻结runner后台启动：

```bash
OUTPUT_ROOT=/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_v2_fcc_v1 \
GPU_IDS="0 1 2" \
bash scripts/remote/run_stage_c_siff_v2_fcc_v1.sh
```

首次状态为`training=0/30 test=0/30`，三个workers分别进入launch order的jobs 1–3：

- GPU0：Weather `siff_equal` seed2022，epoch 1，约300/1108 iterations；
- GPU1：Weather `a6_full` seed2022，epoch 3，约1100/1108 iterations；
- GPU2：Weather `siff_independent_equal` seed2022，epoch 1，约700/1108 iterations。

active memory约GPU0/1/2=`2003/434/1474 MiB`，保留充足安全余量。根据首次三臂速度与后续每GPU十个job，
粗略完成时间为launch后约`3–8 h`，会受early stopping与dataset速度影响；该ETA只作运行管理，不改变matrix。

## 5. Execution boundary

1. training期间不改config、rank、selector、launch order或gates；
2. `A6_MEASURE`不进入matrix、metrics或machine decision；
3. 只有runner确认30/30 training artifacts完整后，才允许独立调用`FORMAL_TEST_ONLY=1`；
4. formal test必须一次执行完整30-run new matrix，并在test前后核对checkpoint SHA256；
5. final analyzer必须联合seed2021/2022/2023的45 runs与180 cells。

当前decision：

```text
step8_training_active_formal_test_not_started
```

## 6. Completion update

- 30/30 training于`2026-07-21T14:06:38+08:00`完成；
- formal test于`2026-07-21T14:09:02+08:00`启动；
- 30/30 new test与45-run/180-cell analyzer于`2026-07-21T14:12:05+08:00`完成；
- checkpoint nonmutation、matrix completeness、unique hashes与initialization pairing全部通过；
- final decision=`performance_pass_attribution_blocked_stop_fcc_promotion`。

结果报告：
`analysis/stage_c_post_d21_unconstrained_reset_20260720/siff_v2_fcc_v1/step9_10_result_and_portfolio_decision.md`。

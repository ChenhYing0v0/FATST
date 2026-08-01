# Solar H3B Remote Launch

## Launch record

| Field | Value |
| --- | --- |
| `candidate_version` | `ISCF-BSCA-MAIN-v1-solar-h3b-test-informed-20260801` |
| `date` | 2026-08-01 |
| `commit` | `4ac5650402624a4ccf0c165790b0976ea561220f` |
| `config_hash` | `f7b8cd2f10d32fa034fdbdb84586481a45beea4465e53ee06e161014e01cb241` |
| `search_space_hash` | `d86da4644ab657e486b3925ff80183af9dfc2d2dfd9a1243dd8a37105c324ab8` |
| `remote_host` | `529_Lab-3090` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/solar_h3b` |
| `orchestrator_pid` | `2827267` |
| `start_time` | `2026-08-01T20:33:48+08:00` |
| `GPUs` | 0, 1, 2；RTX 3090 |
| `GPU_preflight` | each 18 MiB used，0% utilization |
| `resource_smoke` | 4/4 complete；no OOM/NaN/Inf |
| `full_matrix` | Solar 4 profiles；seed2021；45 epochs/patience10 |
| `test_during_training` | 0 |

## Active jobs

At `2026-08-01T20:34:13+08:00`, 0/4 jobs were complete and three were active:

- GPU0：`Solar__h3b_lr3e4_rank64`；
- GPU1：`Solar__h3b_lr3e4_dropout4`；
- GPU2：`Solar__h3b_lr2e4`。

Observed memory was approximately3157/3960/3960 MiB with 86--89% utilization。The fourth job `Solar__h3b_lr4e4` will be claimed by the first free GPU。No failure signature appeared。

## Completion gate

Training must reach4/4 complete checkpoints with finite four-H validation metrics and frozen SHA256。Validation only selects each trial checkpoint；no profile ranking is performed before direct complete test。After H3B test, the final Solar profile is selected jointly over all retained H1/H2/H3A/H3B candidates by four-H official-test mean MSE。

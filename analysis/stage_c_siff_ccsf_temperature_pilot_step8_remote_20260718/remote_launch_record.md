# CCSF shared-temperature pilot Step8远程启动记录

## Launch decision

[Fact] `SC1-SIFF-v2-CCSF-TEMP-PILOT-v1`已于2026-07-18 15:31:11 +08:00在
`529_Lab-3090`启动。该运行只访问validation split，用于从`{0.05,0.1,0.25}`选择一个跨5 datasets共享的
temperature；formal Phase A、official test与confirmation均未启动。

## Provenance

| Field | Value |
| --- | --- |
| `remote_host` | `star3090.iai.zju.edu.cn` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `commit` | `06d0ffcf555260b6aac8e067a24b2ca7d42629f3` |
| `driver_pid` | `654232` |
| `conda_env` | `moe` |
| `gpu_ids` | `0 1 2` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_ccsf_temperature_pilot_v1` |
| `config_sha256` | `730e8021e1c324071057b87863028d41a443c09093c78d4181ebd3d1f2672be5` |
| `profile_sha256` | `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a` |
| `matrix` | 15 runs / 60 validation cells |
| `evaluation_split` | validation only |
| `formal_test_access` | false |

远端fast-forward前存在3个历史analysis CSV的未提交修改；本次pull没有覆盖或修改它们，它们仍原样保留。

## Prelaunch evidence

1. 启动前GPU0/1/2均used=15 MiB、free=24110 MiB、utilization=0%；
2. 无活跃`train_repo.py`或同名runner；
3. remote HEAD从`c4c4730` fast-forward到`06d0ffc`；
4. dry-run输出15/15 jobs，config/profile hashes匹配，并显示
   `validation_only=true formal_test_authorized=false`；
5. Weather/tau0.1单batch resource smoke于15:31:00通过，artifact位于external output root下的
   `_resource_smoke/weather_tau01_seed2021`。

## Start snapshot

三个worker于15:31:11同时启动首批慢数据集Weather：

- job 1/15：tau0.05，GPU0；
- job 2/15：tau0.1，GPU1；
- job 3/15：tau0.25，GPU2。

启动后GPU0/1/2显存约3841/3842/3842 MiB，利用率93%/91%/92%，三个training processes均存活。

## Current boundary

`current_step=Step8 validation-temperature-pilot running`。运行期间不得pull新代码、修改temperature grid、选择规则、
dataset profiles或追加arms。用户明确不需要值守；结果完成后先核验15/15 runs、60/60 validation cells与analyzer
artifacts，再固定唯一shared temperature并返回formal-candidate prelaunch audit。pilot结果不得直接判定CCSF机制有效。

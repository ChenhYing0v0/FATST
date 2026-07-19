# D19 IF Control Step 8 Remote Launch Record

## Frozen identity

- `candidate_version`: `SC-D19-IFC-control-v1.1`
- `role`: `control_only`
- `commit`: `da011c8b40f815f69aea94a7784326b1f7b73f0c`
- `config`: `configs/stage_c_d19_if_control_step7b.json`
- `config_sha256`: `60c30fa672884c1acf01ac81e9042f1fccc45dcc21981749af4b8544ac354fe8`
- `profile_sha256`: `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a`
- `matrix`: 15 new training runs + 5 reused A6 runs；80 official-test cells
- `checkpoint_selection`: validation mean MSE over H96/H192/H336/H720
- `formal_evaluation`: official test H96/H192/H336/H720 MSE/MAE

## Preflight and resource smoke

2026-07-19 remote preflight显示GPU0/1/2均为RTX 3090，启动前各约15 MiB used，
无compute process。远端已有3个历史generated CSV修改，与D19文件不重叠，均保留。

Step7B prelaunch在remote conda env `moe`再次通过31/31。随后执行两个real-batch smoke：

1. Weather `if_measure`：2 train batches，epoch-1 train loss `0.5402943`，
   validation loss `0.8306245`；
2. ETTm2 `direct_nonlinear_matched_measure`：2 train batches，epoch-1 train loss
   `0.3914795`，validation loss `0.2717429`。

两者均完成、finite且无OOM。第一次smoke还暴露remote无`rg`；runner已改为`rg/grep`
fallback并以commit `da011c8`重新smoke通过，未改变模型、矩阵、gates或训练协议。

## Launch

- `launch_time`: `2026-07-19T18:52:43+08:00`
- `remote_host`: `529_Lab-3090`
- `remote_repo`: `/home/yingch/projects/FATST`
- `output_root`: `/home/yingch/exp_outputs/r-2026-fatst/stage_c_d19_if_control_v1_1`
- `driver_log`: `driver_seed2021.log`
- `driver_pid`: `2700799`
- `gpu_ids`: `0 1 2`

启动后的单次确认显示driver存活，三个worker分别进入：

1. GPU0：job 1/15，Weather `if_measure`；
2. GPU1：job 2/15，Weather `if_noskip_measure`；
3. GPU2：job 3/15，ETTm1 `if_measure`。

## Authorization boundary

只授权seed2021完整matrix与一次冻结的official-test audit。seeds2022/2023 confirmation、
candidate redesign、per-dataset/horizon tuning与paper-method promotion均未授权。

按用户要求，正式启动后不持续值守。待用户告知远程完成后，再同步完整artifacts并执行Step9四层分析。

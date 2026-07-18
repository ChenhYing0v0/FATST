# CCSF temperature pilot retry1 Step8远程重启记录

## Decision

[Fact] runtime repair通过local 3/3、Step7B recheck 15/15与真实Weather三batch smoke后，同一冻结validation pilot
已于2026-07-18 15:54:21 +08:00重启。retry1不改变temperature grid、datasets、seed、profiles、selection或
no-test合同。

## Provenance

| Field | Value |
| --- | --- |
| `commit` | `7045c8005a0b27716601f88fcdc4f2cc598f1261` |
| `driver_pid` | `683945` |
| `worker_pids` | `683970,683974,683981` |
| `gpu_ids` | `0,1,2` |
| `config_sha256` | `46cf93dc6cf537e42081471b50480da356c060419bc282d880ae2b8809c80f43` |
| `profile_sha256` | `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_ccsf_temperature_pilot_v1_retry1` |
| `matrix` | 15 runs / 60 validation cells |
| `formal_test_access` | false |

## Repair smoke

Weather/tau0.1 resource smoke现执行3个train batches。结果：train loss=1.785318，validation mean MSE=1.221736；
checkpoint与四horizon metrics均存在，training log没有`nan/inf`。这验证repair跨过了首次失败发生的parameter-update
边界，但仍不等于15-run完成或CCSF effectiveness。

## Start snapshot

首批三个Weather runs分别为tau0.05/0.1/0.25，运行于GPU0/1/2。启动快照显存约3841/3842/3842 MiB，GPU
utilization=98%/93%/92%，三个train processes均存活。

`current_step=Step8 repaired validation pilot running`。用户已明确无需值守；不再轮询。完成后必须核验15/15、
60/60、finite metrics、no-test provenance与shared selection artifacts。formal Phase A/test/confirmation仍未授权。

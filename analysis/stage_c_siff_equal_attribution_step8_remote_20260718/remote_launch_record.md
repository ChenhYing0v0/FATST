# SC1-SIFF-v2-EQ-ATTR Step 8 Remote Launch

## Launch identity

| Field | Content |
| --- | --- |
| `candidate_version` | `SC1-SIFF-v2-EQ-ATTR-v1` |
| `launch_time` | 2026-07-18 11:12:03 +08:00 |
| `host` | `star3090.iai.zju.edu.cn` / `529_Lab-3090` |
| `remote_repo` | `/home/yingch/projects/FATST` |
| `source_commit` | `c4c4730be09f4c6471653018a39b6a9cba365bee` |
| `config_hash` | `f0600be7c5b79445a1055463e0a41486a52f16daa20068c610d56457bd03836a` |
| `profile_hash` | `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a` |
| `seed` | 2021 |
| `matrix` | 50 runs / 200 official-test cells |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_equal_attribution_v2` |
| `confirmation` | false |

## Prelaunch evidence

1. local Step7B gate：9/9；
2. remote `git pull --ff-only`：`b3f42ea -> c4c4730`；
3. remote dry-run：50/50 jobs，config/profile hash一致；
4. Weather + `SIFF_EQUAL` 1-batch resource smoke：2026-07-18 11:11:41完成；
5. launch前GPU 0/1/2均约15 MiB used、24110 MiB free、0% utilization；
6. 三份历史dirty analysis CSV保留且未被覆盖。

## Process layout

| Role | PID | GPU / job |
| --- | ---: | --- |
| SSH wrapper | 319686 | waits for detached runner |
| runner | 319688 | 50-job scheduler |
| worker 0 | 319707 | GPU0 |
| worker 1 | 319711 | GPU1 |
| worker 2 | 319716 | GPU2 |

首批dataset-major jobs：

1. job 1/50：`Weather / A6_FULL / GPU0`；
2. job 2/50：`Weather / A6_MEASURE / GPU1`；
3. job 3/50：`Weather / PCSD_MEASURE / GPU2`。

## First progress confirmation

2026-07-18 11:12:48：

- `A6_FULL / Weather`：epoch 3 active；
- `A6_MEASURE / Weather`：epoch 3 active；
- `PCSD_MEASURE / Weather`：epoch 1 active；
- GPU memory used：约433 / 434 / 1984 MiB；
- GPU utilization：约18% / 17% / 82%；
- completed runs：0/50；
- active position：jobs 1–3/50。

基于首批速度，粗略ETA为2–4小时；SIFF与Q1-wide arms计算更重，因此该估计仅作早期参考。

## Boundary

- 当前只运行seed2021；
- 每个run先由validation四horizon均值选择checkpoint，再读取official test；
- test不选择epoch、不修改checkpoint；
- runner完成后自动执行四层analyzer；
- seeds2022/2023不会自动启动；
- 不进行高频轮询，结果完成后再进入Step9。

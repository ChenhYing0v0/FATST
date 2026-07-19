# SC-D18-SPC Step 8 Remote Launch Record

## Launch

- date/time：`2026-07-19T14:31:07+08:00`；
- remote repo：`/home/yingch/projects/FATST`；
- commit：`c843178`；
- output root：`/home/yingch/exp_outputs/r-2026-fatst/stage_c_d18_soft_projectivity_cost_v1`；
- environment：`moe`；
- GPUs：0、1、2；
- preflight memory：三卡均`15 MiB used / 24110 MiB free`；
- driver script PID：`2372498`；
- matrix：25 artifact units，其中15个new specialist training、10个reused control audits。

## Prelaunch evidence

1. local prelaunch `11/11`；
2. remote prelaunch `11/11`；
3. local/remote 25-job dry-run通过；
4. Weather `A6_SPEC96` one-batch resource smoke通过；
5. remote control lineage直接从
   `/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_equal_attribution_v2`
   核对；
6. config hash：
   `8314e44bdeef948a4dd3c248c856c3433fa43c04cf4730730d115b4b598b1764`；
7. profile hash：
   `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a`。

## First-task confirmation

- Weather `A6_MEASURE`与`A6_FULL` control probe audits已完成；
- Weather `A6_SPEC96`、`A6_SPEC192`、`A6_SPEC336`已分别在GPU2/0/1启动；
- 启动后显存约433–436 MiB，三卡utilization约17%–18%，无capacity风险；
- checkpoint selection分别为H96/H192/H336 validation MSE；
- official test只在每个best-validation checkpoint冻结后执行；
- 不修改历史control checkpoint。

## Boundary

本次是`test_informed problem-existence diagnostic`，不是method training。运行中不pull、不改config/gates、不按
dataset或horizon筛选结果。无需高频值守；25/25 artifacts完成后必须先做protocol、initialization、parameter、
prediction NRMSE和15-cell own-H gate审计，再决定是否返回Step 4。

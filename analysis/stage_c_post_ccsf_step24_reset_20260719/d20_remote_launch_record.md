# D20 CST Step 8 Remote Launch Record

## Launch identity

| Field | Value |
| --- | --- |
| candidate | `SC-D20-CST-v1 diagnostic_only` |
| remote host | `529_Lab-3090` / `star3090.iai.zju.edu.cn` |
| remote repo | `/home/yingch/projects/FATST` |
| commit | `9573cd7b5d651675d1ea77fc4cc2ed365017ba12` |
| launch time | `2026-07-20T11:55:45+08:00` |
| driver PID | `13904` |
| GPUs | `0 1 2` |
| output root | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_d20_cst_v1` |
| config hash | `4c67b038bd85bca511cc3f4d69084135b391c3947be79573089796977b06101f` |
| profile hash | `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a` |

## Preflight and repository boundary

首次SSH连接短暂出现`Network is unreachable`与`Connection refused`，随后恢复；未在失败连接期间执行远端操作。
恢复后GPU 0/1/2均为RTX 3090、约24,107 MiB free、0% utilization，且没有compute process。

远端repo在pull前为commit`da011c8`，存在三份历史analysis CSV的未提交改动。它们与D20文件不重叠，已保留原样；
`git pull --ff-only origin main`成功将代码推进到`9573cd7`。远端Step7B checker再次为`10/10 pass`。

## Resource smoke

在GPU 0/1并行运行：

- `A6_CST_SPEC / Weather`；
- `A6_CST_RANDOM / ETTm2`。

每项使用2个train batches、2个eval batches、1 epoch且不访问test。两项均产生finite training/validation loss、
checkpoint、effective config、environment、initialization contract与model diagnostics；没有Traceback、OOM、NaN或Inf。

## Formal matrix launch

runner固定运行15个from-scratch jobs与60个official-test cells，checkpoint由validation四horizon mean MSE选择。
启动检查确认driver和三个worker存活，首批任务为：

1. GPU0：job 1/15，`A6_CST_SPEC / Weather`；
2. GPU1：job 2/15，`A6_CST_RANDOM / Weather`；
3. GPU2：job 3/15，`A6_CST_SPEC / ETTm1`。

一次性启动检查时三项均在epoch 1，log分别已推进约900、900与1000 iterations；GPU显存占用约433、434与
400 MiB。未观察到启动期异常。

## Monitoring boundary and next action

按用户要求，正式启动后不持续值守、不轮询、不改变config、checkpoint rule或launch order。用户通知训练结束后，
进入Step 9：先审计15/15 artifact units与60/60 official-test cells，再按paper-facing effectiveness、matched
attribution、internal health和failure attribution四层形成结论。

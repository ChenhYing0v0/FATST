# ISCF-BSCA-MAIN-v1 H4K Targeted Matrix and Prelaunch Gate

## 1. Decision

H4K在用户明确要求下冻结为24个seed2021 dataset-level profiles，目标是修复H4J暴露的ETTm2、Weather和H720缺口。Architecture、objective、scales、partition与inference graph均不改变；该阶段是test-informed hyperparameter optimization，不是architecture search。

Decision=`H4K_24_job_targeted_matrix_frozen_remote_training_authorized_test_pending_manifest_gate`。

## 2. Evidence-to-matrix mapping

| Block | Existing weakness | Frozen jobs | Search rationale |
| --- | --- | ---: | --- |
| ETTm2 | 0/8 leads；H96--H336仍有1.8%--4.6% gaps | 8 | 围绕H4J `rank64` winner搜索rank32/48/80及rank×dropout/lr/patch/capacity interactions |
| Weather | 2/8 leads；H96--H336弱，H720已领先 | 6 | 组合`current` family的short-horizon per-cell winners与`timealign_dropout3` joint winner |
| ETTh1 H720 | MSE/MAE gaps 4.03%/3.46% | 2 | `lr3e4` winner的lr与rank邻域 |
| ETTh2 H720 | MSE/MAE gaps 1.56%/0.41% | 2 | `lr5e4` winner的rank64/96邻域 |
| ETTm1 H720 | MSE gap仅0.03% | 2 | optimizer与rank微邻域，同时保留four-H joint guard |
| ECL H720 | best MSE/MAE gaps 1.77%/1.25% | 2 | exact budget60 anchor的dropout/rank邻域 |
| Solar H336/H720 MSE | gaps 3.49%/1.84%；MAE已4/4领先 | 2 | 在patch1 H720与patch4 joint winner之间插值patch2/3 |

总量=ETTm2 8 + Weather 6 + H720 supplementary 10 = 24 jobs。没有Exchange job，因为本轮目标是冻结published-context 56-cell surface；没有复跑任何H1--H4J trial。

## 3. Protocol

- 每个trial从scratch end-to-end joint training；
- seed=`2021`；
- 训练输出一个长度720的unified forecast；
- validation horizons=`{96,192,336,720}`；
- checkpoint selector=`mean validation MSE over four horizons`；
- training阶段`official_test_mode=false`、`final_evaluation_split=val`；
- 每个dataset最终仍只能选择一个profile共同服务四个horizons；
- 禁止per-H、per-metric、per-seed或per-cell profile selection；
- formal test当前未授权，必须等待24/24 checkpoints、artifact audit和manifest freeze。

## 4. Gates and rollback

Local gate要求config JSON、24-job materialization、dataset counts、trial-ID noncollision、`seq_len % patch_num == 0`、source SHA256、test=0和authorization全部pass。Remote launch前必须commit/push、remote fast-forward、quota/GPU检查并完成24/24 two-batch resource smoke。

Performance目标继续为MSE>=20/28、MAE>=20/28、combined>=40/56；局部目标为ETTm2>=4/8、Weather>=6/8、H720>=6/14。局部目标只作failure attribution，不能替代global gate。

若resource smoke出现OOM、NaN、Inf或tensor contract错误，只修复invalid profile并在fresh root重跑完整smoke。若正式结果仍失败，保留全部negative trials，不得拼接per-horizon profiles或在H4K内部修改architecture。

## 5. Resource and scheduling

预计training+smoke新增storage约1--3 GiB，training约12--24 GPU-hours。LPT queue先交错Weather与ETTm2，使三个GPU优先覆盖主要弱项；随后运行ECL/Solar和较短ETT jobs。Remote output root=`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h4k`。

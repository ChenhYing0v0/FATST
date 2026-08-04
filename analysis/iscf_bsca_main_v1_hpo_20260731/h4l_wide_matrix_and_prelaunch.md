# ISCF-BSCA-MAIN-v1 H4L Wide Matrix and Prelaunch Gate

## 1. Decision

H4L在用户明确授权下冻结为ETTm2与Weather各24个、合计48个seed2021 dataset-level profiles。它针对H4K后仍为ETTm2 0/8、Weather 2/8的缺口，将搜索从winner附近的局部邻域扩大为curated space-filling design。Architecture、objective、scales、partition和inference graph保持不变；本阶段是test-informed HPO，不是architecture search。

Decision=`H4L_48_job_wide_matrix_frozen_remote_training_authorized_formal_test_false`。

## 2. Why the search must widen

H1--H4K共117个formal-test trials中，ETTm2仅覆盖`seq_len={336,512,720}`、`d_ff=128`为主且`weight_decay=0.01`固定；Weather主要覆盖`mode_rank=116`且`weight_decay=0.01`固定。H4K只在两个datasets带来0.1%量级的连续改善，没有新增leading cell；合法dataset-level selector与逐cell diagnostic oracle均停在30/56。因此，继续在原winner附近做细邻域搜索没有证据支持。

H4L不是完整Cartesian grid。每个dataset用24个profiles覆盖：

- context/patch边界：短context、`patch_num=1`到120；
- capacity边界：`d_model=32`到256、`d_ff=64`到512；
- decoder边界：ETTm2 `mode_rank=8`到256，Weather 16到256；
- optimizer/regularization边界：`lr=1e-5`到`5e-4`、`weight_decay=0`到0.05；
- 一个cross-dimension space-filling interaction profile；
- 四个TimeAlign encoder-inspired profiles。

Checker按dataset、11个effective HPO fields和`layer_norm`对117个历史profiles做fingerprint audit；48个H4L profiles内部无重复，且与历史profiles无重复。

## 3. TimeAlign source prior

本轮直接核对本地official scripts，而不是凭印象借用参数：

| Dataset role | TimeAlign encoder tuple used as prior | H4L recombination |
| --- | --- | --- |
| ETTm2 short | `L720, patch12, d128, ff128, dropout0.3, LN1` | 与`rank128, wd0`组合 |
| ETTm2 long | `L720, patch12, d128, ff128, dropout0.9, LN1` | 与`rank64, wd1e-3`组合 |
| Weather short | `L720, patch48, d128, ff256, dropout0.1, LN0` | 与`rank64, wd1e-3`组合 |
| Weather long | `L720, patch48, d128, ff128, dropout0.5, LN0` | 与`rank192, lr5e-5`组合 |

借鉴边界只到encoder parameter coupling。TimeAlign的alignment losses、margin设置和prediction head不进入ISCF-BSCA；ISCF-BSCA decoder、BSCA objective与joint training graph不变。由于TimeAlign官方是horizon-specific配置，而H4L要求每dataset一个profile共享四个horizons，这四项只是source-informed search points，不是协议等价复现。

## 4. Frozen protocol

- ETTm2 24 jobs + Weather 24 jobs = 48；
- seed=`2021`，from-scratch end-to-end joint training；
- 一个长度720的unified forecast，evaluation horizons=`{96,192,336,720}`；
- checkpoint selector=`mean validation MSE over four horizons`；
- 每个trial最多60 epochs，early-stopping patience=12；
- effective batch固定为32，避免batch-size变化混入capacity点；
- training与resource smoke均`official_test_mode=false`、`final_evaluation_split=val`；
- 每dataset最终只能选择一个profile共同服务四个horizons；禁止per-H、per-metric、per-seed和per-cell selection；
- formal test未授权。48/48 checkpoints与artifact manifest冻结后，必须另行请求授权。

## 5. Gates, resources, scheduling and rollback

Local gate要求JSON parse、48-job materialization、24/24 dataset counts、117-profile noncollision、`seq_len % patch_num == 0`、effective batch一致、source SHA256、test=0与authorization全部pass。Remote launch前要求focused commit/push、remote exact-commit、quota/GPU preflight和48/48 two-batch resource smoke。

预计training为24--48 GPU-hours，在3×3090上的wall time约8--18小时；training+smoke新增storage约3--5 GiB，不含未来formal test。LPT queue优先交错两个dataset的high-capacity与TimeAlign-inspired profiles，随后覆盖patch/context/rank/optimizer边界。正式输出根为`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h4l`。

若resource smoke出现OOM、NaN、Inf或tensor contract错误，停止正式训练，只修复invalid profile并在fresh root重跑完整smoke。若48/48训练完整，则冻结checkpoint hash和manifest后请求formal-test授权；不得先看部分test。未来效果目标仍为MSE>=20/28、MAE>=20/28、combined>=40/56，局部目标ETTm2>=4/8、Weather>=6/8。H4M、baseline、3-seed和architecture redesign均未授权。

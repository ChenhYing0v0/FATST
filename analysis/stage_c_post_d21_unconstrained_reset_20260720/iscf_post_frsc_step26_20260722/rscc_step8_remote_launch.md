# ISCF-RSCC-v1 Step8 Remote Launch

## 1. 本轮要验证什么

SCC-v0 已在完整 validation matrix 上失败，failure attribution 为
`intervention_point_wrong`：删除 `equal_skill` reliability supervision 后，arm/coalition
health 明显恶化。RSCC-v1 只检验一个收紧后的命题：保留 EQUAL 的 fused + uniform
individual-arm L1 reliability path，再附加 exact leave-one-scope-out coalition policy KL，
能否把已有 coalition information 转成超过 matched controls 的 fused forecast gain。

本轮是 validation-only development screen。它不访问 official test，不构成 paper-facing
effectiveness pass，也不授权 confirmation seeds、modern baselines 或新的 coefficient/router rescue。

## 2. Resource smoke

- date: `2026-07-22`
- remote repo: `/home/yingch/projects/FATST`
- source commit: `020eea3d4a1813bb398c0aa6f317450bdfd8d2bd`
- GPUs before smoke: GPU0/1/2 均 `18 MiB / 24576 MiB`，utilization `0%`
- smoke arms: `iscf_rscc`、`iscf_rscc_shuffled`
- dataset: Weather；每臂2个train batches、1 epoch；final evaluation disabled
- common initialization hash:
  `8828e47bb0fcacba2798493596efe690444298c4da15bb8d99673e218d89f368`

两臂 effective objective 分别为：

- `equal_scope_coalition_credit`
- `equal_scope_coalition_credit_shuffled`

两臂的 `train_pcc_skill_loss` 都为 `0.7632371485`，说明 shuffled control 没有改变
EQUAL reliability path。weighted route loss 分别为 `0.0266938601` 和 `0.0266925879`，
均为非零；五个 scope gradient norms 分别约为
`[0.2399, 0.2364, 0.2003, 0.1734, 0.1017]`，全部 finite/nonzero。日志未出现
Traceback、OOM、NaN 或 Inf。

Decision=`rscc_resource_smoke_pass_full_validation_launch_authorized`。

## 3. Frozen validation matrix

完整矩阵由3个 new arms × 5 datasets × seed2021组成，共15个 from-scratch runs：

| Role | Arm | Objective |
| --- | --- | --- |
| candidate | `iscf_rscc` | EQUAL reliability + exact coalition KL |
| order control | `iscf_rscc_shuffled` | 同上，但batch-local credit target按dedicated RNG shuffle |
| standalone-error control | `iscf_equal_armerr` | existing `pointwise_prior_composed` |

Datasets 为 ETTh1、ETTh2、ETTm1、ETTm2、Weather。所有 arms 使用相同 natural profile、
dataset-matched rank、seed、initialization class、optimizer、four-horizon validation checkpoint
selector及 H96/H192/H336/H720 MSE/MAE scorecard。历史 `ISCF-EQUAL` parent checkpoints只在
Step9 作为 matched reference 复用，不在本轮重训。

冻结 gates 不变：

1. RSCC vs EQUAL macro MSE 至少 `+0.3%`、MAE严格正、至少3/5 datasets和3/4 horizons为正；
2. RSCC vs EQUAL-ARMERR 与 RSCC-SHUFFLED macro MSE 均至少 `+0.1%`；
3. artifacts、initialization、checkpoint selection与evaluation split contract完整；
4. 五个 scope gradients finite/nonzero，policy alignment改善，coalition oracle headroom保持正值；
5. 任一 primary/control gate 失败都关闭 exact SCC/RSCC route，不做seed、lambda、epsilon、
   fallback或router-width rescue。

## 4. Formal launch state

- launch time: `2026-07-22T14:12:34+08:00`
- runner shell PID: `3836251`
- GPUs: `0 1 2`
- config hash: `fba748ff0a6abe087f58677c8aa6e277e66c65a23f74f82c5e3e70837de52fc7`
- profile hash: `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a`
- remote output:
  `/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_rscc_v1_step7b`
- supervisor log:
  `/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_rscc_v1_step7b/supervisor_seed2021.log`
- initial jobs: Weather RSCC on GPU0、Weather RSCC-SHUFFLED on GPU1、Weather
  EQUAL-ARMERR on GPU2
- observed initial model memory: about `1.49--1.51 GiB` per GPU
- progress at `14:18:15+08:00`: `0/15` complete；三个Weather jobs均到epoch 6，loss均finite
- official test access: `false`

Decision=`rscc_step8_validation_training_active_formal_test_disabled`。

## 5. Validation/test角色与下一步

Validation只用于本轮 candidate continuation、checkpoint selection和机制/数值诊断。不得查看或挑选
partial favorable cells；必须等待15/15 runs及60/60 standard-horizon rows完整后一次性执行 Step9。
若完整 validation gate通过，也只能形成 `validation_continuation_supported`，formal test仍需新的用户授权和
预注册完整 test matrix。若 gate失败，直接关闭 exact coalition-credit route并回到Step2/4。

按首批Weather约6分钟到epoch 6的速度估计，完整matrix约还需45--90分钟；early stopping与dataset
吞吐会造成波动。按用户先前指示，不持续高频值守。当前无需修改训练配置。

## 6. Artifacts

- `rscc_step8_remote_records/training_launch_record_seed2021.txt`
- `rscc_step8_remote_records/training_jobs_seed2021.tsv`
- `rscc_step8_remote_records/resource_smoke/`
- `configs/stage_c_iscf_rscc_step7b.json`
- `scripts/remote/run_stage_c_iscf_rscc_step7b.sh`

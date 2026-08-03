# ISCF-BSCA-MAIN-v1 H4K Remote Launch Record

## 1. Frozen identity

H4K以commit `dc52471d585906f1f2251fdb682557f8e5686931`在`529_Lab-3090`启动。Machine contract=`configs/iscf_bsca_main_v1_hpo_targeted_h4k.json`，config SHA256=`0cd6f30a935ec2fef781f81cde0be8fdb78df6bac61c3ebc1cb7ba7db74382c0`，search-space SHA256=`b5594235ab031cf9de12dbb5b285555bee7a4a9299a907a2689663f0debb3f65`。正式矩阵保持24个seed2021 dataset-level profiles，未修改architecture、objective、scales、partition或inference graph。

## 2. Remote preflight and resource smoke

Remote project=`/home/yingch/projects/FATST`已fast-forward至上述commit。启动前GPU0/1/2均为18 MiB、0% utilization；用户quota约190 GiB used / 200 GiB soft / 220 GiB hard，既有HPO root约31 GiB，预计本轮新增1--3 GiB。

24/24 resource-smoke jobs于2026-08-03 16:41:59--16:43:21完成，生成24个metrics artifacts与24个temporary checkpoints；未检出OOM、NaN、Inf、Traceback或tensor-contract failure。Smoke root占用约569 MiB，结束后三张GPU均回到18 MiB idle。Resource smoke始终`test_jobs=0`。

## 3. Formal train/validation launch

正式train/validation于2026-08-03 16:44:09 +08:00 detached启动：

- runner PID=`2412083`；
- GPU IDs=`0,1,2`；
- output root=`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h4k`；
- pipeline log=`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h4k/pipeline.log`；
- matrix=`24 jobs`，`test_jobs=0`；
- first batch=`Weather__h4k_current_lr5e5_patch24`、`ETTm2__h4k_rank32`、`Weather__h4k_current_lr5e5_patch24_dropout0`。

2026-08-03 16:45:26首轮检查时，三个jobs均进入训练，Weather两项和ETTm2一项均已完成epoch 1并进入epoch 2；GPU0/1/2分别约1613/522/1580 MiB，utilization约59%/14%/63%。当时正式结果为0/24 complete，未检出OOM、Traceback或RuntimeError，formal test仍为0/24。

## 4. Active gate

Decision=`H4K_24_job_train_validation_active_test_zero`。当前只等待24/24 train/validation完成；完成后必须审计numeric health、effective configs、checkpoint hashes和artifact completeness并冻结manifest。Formal test仍未授权，不得由runner自动启动。只有manifest gate通过并取得分级授权后，才可对24个checkpoints执行complete four-horizon official test；automatic H4L=false。

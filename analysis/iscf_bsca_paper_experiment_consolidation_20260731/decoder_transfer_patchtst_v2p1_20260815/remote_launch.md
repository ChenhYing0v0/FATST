# PatchTST Decoder-Transfer v2.1 Remote Launch

日期：2026-08-15  
Decision：`v2p1_matched_training_active_guarded_formal_pipeline_queued`

## Frozen execution

- commit：`9cf0e8e87f4ba1060db3c7fe9f026a7baf6d37df`；
- server：`529_Lab-3090`；conda env：`moe`；
- output root：`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_decoder_transfer_patchtst_v2p1_formal_20260815`；
- parent HPO root：`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_decoder_transfer_patchtst_hpo_v2_20260815`；
- protocol hash：`88b71e797fd9ee00fb4533a5fa942eb2e2a512eea4457ce827d201dfeea7f0ee`；
- selected-profile artifact hash：`9044d0534b34761a9353ee7d0854ec0fa2ee42b16f85a9b517f3571fc7ead9f4`；
- profile hash：`3a0b8864f1389ebb1346bebe0954b3fa7466b1ae8b054de597f263b8089c95ec`。

## Resource gate

Launch前GPU0/1/2均为RTX 3090、memory used=18 MiB、utilization=0%。三个代表性matched-ISCF resource smokes全部通过。`/home` filesystem为1.8T，总可用890G；parent HPO artifacts为1.7G，v2.1 launch前含smoke artifacts为108M。

## Scheduling

- start：`2026-08-15T14:43:04+08:00`；
- training driver parent PID：`663673`；
- GPU0：Weather，完成后ETTh1；
- GPU1：ETTm1，完成后ETTh2；
- GPU2：ETTm2；
- initial status：0/5 complete，Weather/ETTm1/ETTm2均已进入epoch 1且loss finite；GPU memory约950--1657 MiB。

## Guarded continuation

Guarded pipeline PID=`665646`。该进程只执行以下严格串行链：

1. 等待matched training driver退出；
2. 运行`check_iscf_bsca_decoder_transfer_patchtst_v2p1_artifacts.py`；
3. 只有checker确认10/10 unique hashes、5/5 matched initialization pairs、protocol/selection hash和test-artifact absence后，才设置`FORMAL_TEST_ONLY=1`；
4. 对5个selected BSCA与5个matched ISCF checkpoints执行一次formal test；
5. 复用v1 DLinear三arms与PatchTST Original的80 cells，生成完整120-cell result bundle。

任一步失败均由`set -euo pipefail`终止后续步骤；尤其manifest失败时不会创建test loader或test artifact。

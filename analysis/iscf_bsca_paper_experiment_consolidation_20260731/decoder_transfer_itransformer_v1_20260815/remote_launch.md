# iTransformer-style Decoder-Transfer v1 Remote Launch

Decision：`itransformer_transfer_v1_three_gpu_train_validation_active_test_zero`

## Exact state

- git commit：`62418769b3afb205a1ef81ba7adf55a6f071ac46`；
- remote project：`/home/yingch/projects/FATST`；
- output root：`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_decoder_transfer_itransformer_v1_20260815`；
- profile SHA256：`5e8dc7f04765fae36d25f696cf66be00cc58ee59f7559871119751abb4ae8b51`；
- protocol SHA256：`21c777aa96f5ea4e5d23658f7dd2d84b3249fef138ffbf28def26c8ee0e3a78b`；
- start：`2026-08-15T18:36:14+08:00`；
- nohup wrapper PID：`956718`；driver PID：`956722`；
- GPUs：0, 1, 2，均为RTX 3090。

## Resource gate

Launch前GPU0/1/2均为18 MiB occupied、0% utilization。User quota为194G used / 200G soft / 220G hard；`r-2026-fatst` experiment root为84G。Remote local gate重复通过28/28。

3-job Weather smoke覆盖Original、+ISCF、+ISCF-BSCA，全部完成2 train batches和2 validation batches，loss finite、无OOM。Smoke输出约60 MiB，确认后已删除；只删除本轮可重建的`_resource_smoke`目录。

## Full queue

15-run train/validation queue使用GPU0/1/2并行。每个GPU固定一个decoder arm并按`Weather -> ETTm1 -> ETTm2 -> ETTh1 -> ETTh2`推进，从而保持三臂dataset workload对齐。首批三条Weather run均已进入epoch 1；observed memory约为Original 0.5 GiB、+ISCF 1.2 GiB、+ISCF-BSCA 1.2 GiB，GPU utilization约32--34%。

Formal test、table mutation、extra HPO和extra seeds均为false。下一 gate 是15/15 training artifacts、15 unique checkpoint hashes与5/5 matched encoder-initialization triplets；immutable manifest前test必须保持0。

Post-run status audit确认本轮train/validation path不会生成`trained_invariants.json`；此前只对Original放宽该文件的completion predicate，因而在15/15实际完成后仍误报为5/15。所有15个worker job均正常写出checkpoint与validation scorecard，没有run丢失。Runner现统一要求checkpoint、effective config、environment、initialization contract、four-H metrics、model diagnostics和training log。该修复只改变status/resume判定，不改变model、optimizer、seed、selector、checkpoint或已有artifact。

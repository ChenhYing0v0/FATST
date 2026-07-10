# B14-FURD Step 3 Remote Launch Record

## Environment

- host：`529_Lab-3090`；
- remote repo：`/home/yingch/projects/FATST`；
- conda env：`moe`；
- datasets：Weather / ETTm1 / ETTh2；
- checkpoint root：`/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b13_probe_inputs/a6_clean`；
- GPUs before launch：GPU 0/1/2各约 `15 MiB` used、`24110 MiB` free、0% utilization。

## A1 Current-Gradient Diagnostic

- commit：`d8f3f065eb6fc2acd950c165bd575297d2ea2dc8`；
- output：`/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b14_future_unit_retrieval_demand_d8f3f06`；
- GPU mapping：Weather->0，ETTm1->1，ETTh2->2；
- config：train split、8 batches、batch size 16、U180/U240、4 Hutchinson draws、seed 2021；
- start/end：2026-07-10 15:54:56 / 15:55:04 +08:00；
- status：3/3 process success。

## A2 Model-Independent Repair

- commit：`bc976311167c67c2cfbb4a4d3113d6c2ef5ad8f4`；
- output：`/home/yingch/exp_outputs/r-2026-fatst/phase5_stage_b_b14_label_patch_dependence_bc97631`；
- GPU mapping：Weather->0，ETTm1->1，ETTh2->2；
- config：A1 config + rank-8 DCT descriptors + 4 shuffled-target CKA draws；
- start/end：2026-07-10 16:04:48 / 16:04:56 +08:00；
- status：3/3 process success。

两个 runs均为 frozen-checkpoint diagnostics，不包含 training或 parameter update。

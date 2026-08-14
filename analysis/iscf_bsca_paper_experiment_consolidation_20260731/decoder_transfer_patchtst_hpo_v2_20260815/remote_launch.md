# PatchTST Decoder-Transfer HPO v2 Remote Launch

- launch time：2026-08-15 01:39:23 +08:00；
- remote：`529_Lab-3090`；
- repo commit：`7480ffc4fbcea5a1047a15e42420522c0fdd7125`；
- output root：`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_decoder_transfer_patchtst_hpo_v2_20260815`；
- runner PID：`3782929`；
- GPUs：0、1、2；launch前均为18 MiB used、0% utilization，无其他compute process；
- quota：191G used / 200G soft / 220G hard；预计本轮新增3--5 GB；
- remote prelaunch：27/27 checks pass；
- resource smoke：3/3 pass，覆盖default-rank、low-rank与high-rank optimizer groups；
- formal test：0；table mutation：false。

正式queue为50 jobs，dataset-major顺序先运行10个Weather profiles，再运行ETTm1、ETTm2、ETTh1、ETTh2。启动后首批：

1. GPU0：Weather / `p01_lr0p25`；
2. GPU1：Weather / `p02_lr0p50`；
3. GPU2：Weather / `p03_lr2p00`。

三项均已进入epoch 1，约完成500/2240 training iterations，loss finite；GPU memory约1.66 GB、utilization约51--54%。按v1各dataset epoch/time与early-stopping分布估计，总wall time约8--11小时，预计2026-08-15 10:00--13:00完成。完成前不作partial profile selection；50/50 artifacts后运行training-only artifact/hash audit，再执行validation selector。

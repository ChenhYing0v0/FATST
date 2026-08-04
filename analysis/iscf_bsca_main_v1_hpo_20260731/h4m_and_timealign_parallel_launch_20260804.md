# H4M与TimeAlign并行remote launch

日期：2026-08-04

## Freeze与preflight

- exact experiment commit=`24eaac1176efd7b7139a692ae3830be8d4d198fc`；
- remote project=`/home/yingch/projects/FATST`；
- remote output root分别为：
  - `/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h4m`；
  - `/home/yingch/exp_outputs/r-2026-fatst/timealign_official_reproduction/ettm2_weather_20260804`；
- launch前GPU0/1/2均为18 MiB used、约24.1 GiB free、0% utilization；
- quota=`165G / 200G soft / 220G hard`；
- TimeAlign两个dataset与六个executed source files的SHA256 preflight通过；
- remote worktree原有三份unrelated dirty CSV保持不变。

## Resource smoke

H4M 24/24 two-batch smoke通过，test=`0/24`，无OOM/NaN/Inf/Traceback。

TimeAlign首次调度命令因background shell丢失working directory而在runner启动前fail-close；第二次在argument parsing阶段发现`official` readout不允许Stage C `legacy-*` overrides；第三次发现unused grouped-MLP default 144不能整除H336。修复只删除冗余legacy overrides并设置不进入official computation path的`grouped_mlp_scale=48`。最终8/8 one-epoch/two-batch smoke通过，`final_evaluation_split=none`、test=`0/8`。这些故障均发生在正式训练或test前，failure attribution=`source_protocol_adapter_preflight_fault_repaired`。

## Full launch

启动时间=`2026-08-04T21:34:39+08:00`。

- H4M：GPU0--1 dynamic queue，PID=`393804`，24 jobs，train/validation only，test=`0/24`；初始jobs为Weather `L1152/p36`与`L960/p30`，检查时均进入epoch1 iteration200。
- TimeAlign：GPU2 serial workload queue，PID=`393805`，8 jobs；每个fixed-H system训练10 epochs后一次official test；初始job为Weather H720，检查时进入epoch1 iteration400。
- observed memory：GPU0/1/2约`1.66/1.63/1.72 GiB`，utilization约`65/62/80%`，安全余量充足。

H4M预计18--30 GPU-hours、约9--16 wall-hours。TimeAlign先完成Weather长horizon，再穿插ETTm2；其8/8 artifact gate通过后才能把本轮结果标为artifact-complete reproduction。

Decision=`H4M_24_train_validation_and_TimeAlign_8_native_reproduction_active_in_parallel`。

# StageC Step 7B Remote Launch Record

- launch time: `2026-07-13T19:13:48+08:00`；
- remote host: `529_Lab-3090`；
- remote checkout: `/home/yingch/projects/FATST`；
- git commit: `8abe9b7d2a4f9b55c68b51169d6594e01ebcd047`；
- conda environment: `moe`；PyTorch `2.9.0+cu128`；
- output root: `/home/yingch/exp_outputs/r-2026-fatst/stage_c_step7b_pmfo_rct`；
- launch PID: `168103`；
- command: `GPU_IDS="0 1 2" bash scripts/remote/run_stage_c_step7b_pmfo_rct.sh`；
- matrix: 3 datasets × 5 arms × seed2021 = 15 runs；
- contract hash: `254d85d47a9e5b7c212f8a8b88decf17a0328a1ea1df324c9cc65be4c672a50c`。

## GPU Preflight

| GPU | Model | Used MiB | Total MiB | Free MiB | Utilization |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | NVIDIA GeForce RTX 3090 | 15 | 24576 | 24110 | 0% |
| 1 | NVIDIA GeForce RTX 3090 | 15 | 24576 | 24110 | 0% |
| 2 | NVIDIA GeForce RTX 3090 | 15 | 24576 | 24110 | 0% |

preflight时未发现其他`train_repo.py`或StageC runner。首批固定worker任务：

- GPU0：Weather / `pmfo_rct`；
- GPU1：Weather / `pmfo_no_conservation`；
- GPU2：Weather / `pmfo_no_transition`。

## Protocol Smoke

launch前在GPU2执行ETTh2 PMFO-RCT的1 epoch/1 train-batch/1 eval-batch smoke。真实dataset、CUDA、
best-val checkpoint、test full-crop metrics与checkpoint reload均通过。trained invariant：prefix
`2.98e-8`、refinement `3.58e-7`、conservation `2.98e-7`、locality outside support `0`。

[Boundary] smoke不进入performance比较；正式gate只读取完整15-run output root。

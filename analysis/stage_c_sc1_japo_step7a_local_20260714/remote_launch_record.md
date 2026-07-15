# SC1-JAPO Step 8 Remote Launch Record

| Field | Value |
| --- | --- |
| launch time | `2026-07-15T10:48:04+08:00` |
| remote host | `529_Lab-3090` |
| remote repo | `/home/yingch/projects/FATST` |
| commit | `90e4164aaaf9fc31b12f7b4b37459106a5df580d` |
| conda env | `moe` |
| GPUs | `0,1,2`，均为RTX 3090 |
| preflight memory | GPU0/1/2均`15 MiB used / 24110 MiB free`，无compute process |
| matrix | seed2021；5 datasets × 7 arms=`35 runs` |
| output root | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_japo_e2e` |
| supervisor PID | `3115861` |
| evaluation | validation dense H1..720；test=false |

启动命令由`bash scripts/remote/run_stage_c_sc1_japo_e2e.sh`执行，effective environment为：

```text
GPU_IDS="0 1 2"
SEED=2021
OUTPUT_ROOT=/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_japo_e2e
```

远端在launch前完成：

1. clean `main` fast-forward到commit `90e4164`；
2. 35-job dry-run、七个synthetic checkpoint audits与analyzer fixture通过；
3. GPU memory与active process preflight通过。

首批jobs为：

- GPU0：job 1/35，A6 / Weather；
- GPU1：job 2/35，A6 / ETTm1；
- GPU2：job 3/35，A6 / ETTm2。

启动后约23秒，三jobs均已进入训练：Weather epoch1 iter1100，ETTm1 epoch1完成，ETTm2 epoch2 iter300；
GPU memory约`400–456 MiB`，无OOM或protocol error。按首批速度粗略估计完整matrix约需1–1.5小时，实际受
early stopping与validation evaluation影响。

[Boundary] 该记录只证明实验已安全启动，不构成effectiveness result。完成后必须同步全部artifacts并由
`analyze_stage_c_sc1_japo_e2e.py`独立重算gate。

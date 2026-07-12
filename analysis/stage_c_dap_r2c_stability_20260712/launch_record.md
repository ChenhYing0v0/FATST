# SC0-DAP-R2C Remote Launch Record

- Date: 2026-07-12 21:41-21:47 Asia/Shanghai
- Host: `529_Lab-3090`
- Repo: `/home/yingch/projects/FATST`
- Commit: `23dba396b7be1a8d9101895f67aebee2290f8f33`
- Conda environment: `moe`
- GPUs: RTX 3090 indices 0/1/2
- Preflight: each GPU `15 MiB / 24576 MiB`, utilization `0%`
- Profile hash: `6872f5ae5b55724dfc760da85a12c21e84dc7bd54ab4cdbd12a6205885c46921`
- Command: `bash scripts/remote/run_stage_c_dap_r2c_stability.sh`
- Output: `/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2c_stability`
- Matrix: selected profiles × seeds `{2022,2023}` = 6 new runs
- Result: 6/6 training runs completed; no OOM or training error

The original static round-robin launcher could briefly place a newly released job on a GPU that still had another job,
while another GPU was idle. Peak observed memory remained below 1 GiB, so this was not a safety failure. The runner was
subsequently changed to one sequential worker per GPU for future reruns.

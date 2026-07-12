# StageC SC0-DAP-R2 Phase A Launch Record

- `candidate`: `SC0-DAP-R2A`
- `role`: validation-only natural patch profile calibration
- `launch_time`: `2026-07-12T21:17+08:00`
- `commit`: `d0c8014`
- `profile_hash`: `2bfb1f88d38a7dfca691302d166b543db74928813653873ffc1a05e93f285f19`
- `remote_host`: `529_Lab-3090`
- `conda_env`: `moe`
- `GPUs`: `0,1,2`；preflight均15 MiB used、无compute process
- `output_root`: `/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2a_patch_screen`
- `launcher_pid`: `2738760`
- `matrix`: 3 datasets × 3 patch profiles × seed2021 = 9 runs
- `selection`: eight-horizon validation normalized regret；params/test不参与

Profiles固定`D=64,d_ff=128`，比较P12/P24/P48；observed active-forward params为
419,216/613,904/1,006,160，只报告不做gate。

```bash
OUTPUT_ROOT=/home/yingch/exp_outputs/r-2026-fatst/stage_c_dap_r2a_patch_screen \
GPU_IDS="0 1 2" \
bash scripts/remote/run_stage_c_dap_r2a_patch_screen.sh
```

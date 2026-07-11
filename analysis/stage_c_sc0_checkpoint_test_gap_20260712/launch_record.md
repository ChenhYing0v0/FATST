# StageC SC0 Best-vs-Last Checkpoint Test Diagnostic Launch

- `role`: post-freeze diagnostic；不用于hyperparameter selection
- `date`: `2026-07-12`
- `remote_host`: `529_Lab-3090`
- `commit`: `3459959e111c67cfb3e9593c2689a11f8b84e80b`
- `conda_env`: `moe`
- `GPUs`: Weather=0，ETTm1=1，ETTh2=2
- `preflight`: 三卡均15 MiB used、无compute process
- `checkpoint_source`: `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc0_carrier_calibration`
- `output_root`: `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc0_checkpoint_test_gap`
- `matrix`: 3 datasets × 3 arms × 2 checkpoints × 8 horizons；汇总为72个best-vs-last comparisons
- `start`: `2026-07-12T00:59:08+08:00`
- `finish`: `2026-07-12T01:02:19+08:00`
- `retraining`: false

```bash
bash scripts/remote/run_stage_c_sc0_checkpoint_test_gap.sh
```

三个dataset并行评估，随后调用统一analyzer。所有test读取均发生在原SC0/SC0-R1 calibration完成之后，
且active dataset mapping仍只由三seedvalidation artifacts决定。

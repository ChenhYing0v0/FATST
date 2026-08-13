# H5B ETTh1 Remote Launch

## 1. Exact state

- `launch_date`: 2026-08-13
- `server`: `529_Lab-3090`
- `remote_repo`: `/home/yingch/projects/FATST`
- `exact_commit`: `776e6bc2ecaf4d199d1e03cffbc9f5cd92bac519`
- `config_hash`: `5ed823ae7ac54e5183ffa06ff02502ff06a3280253dc69e38f1320e945485c8c`
- `search_space_hash`: `e00444bae1a01ab4651b3577e2d96e9fd5148c2c14b1c07ae3de41ce512c4557`
- `output_root`: `/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h5b`

## 2. Storage cleanup

Launch前用户quota为`206G/200G soft/220G hard`。H5A formal-test dense diagnostics共有48个NPZ、约21.4 GiB；保留三个selected profiles的NPZ，删除45个nonselected NPZ。Checkpoint、metrics、effective config、manifest、本地同步结果与selected NPZ均保留，删除对象可由checkpoint重建。清理后quota=`186G`，H5A root由24G降至3.9G。

## 3. GPU and resource smoke

启动前GPU 0/1/2均为RTX 3090，显存占用`18 MiB`、free=`24107 MiB`、utilization=`0%`。第一次PTY smoke暴露runner使用Bash特殊变量`LINES`，导致首项被terminal row count污染；在任何formal training开始前停止，改为`JOB_LINES`，本地dry-run与shell syntax通过后以exact fix commit重启。

修复后36/36 resource smoke通过：36 checkpoints、36 metrics、36 effective configs、36 unique checkpoint hashes，failure tokens=`0`，test jobs=`0`。最长context `L960/L840/L768`与`d48/ff48`均未OOM。

## 4. Full train/validation launch

- `start`: `2026-08-13T10:04:01+08:00`
- `PID`: `700269`
- `GPU_IDS`: `0 1 2`
- `jobs`: 36 ETTh1 profiles
- `test_jobs`: 0
- `command`: `MODE=train GPU_IDS="0 1 2" bash scripts/remote/run_iscf_bsca_main_v1_hpo_etth1_h5b.sh`
- `queue`: shared dynamic queue；LPT order先执行最长context和高成本profiles。
- initial active jobs：GPU1=`seq960_p32`、GPU0=`seq840_p28`、GPU2=`seq768_p32`。
- first progress check：三个jobs均进入epoch 1并至少完成200 iterations；GPU memory约`770–777 MiB`，utilization约24%，安全裕量充分。

Formal test仍由36/36 training-artifact manifest gate阻断；禁止根据partial validation结果提前test、选择profile或修改Main I/Main II。Decision=`H5B_three_GPU_train_validation_active_test_zero`。

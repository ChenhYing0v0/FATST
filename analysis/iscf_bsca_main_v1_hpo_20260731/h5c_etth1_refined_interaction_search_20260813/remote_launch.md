# H5C ETTh1 Remote Launch

## 1. Exact state

- `launch_date`: 2026-08-13
- `server`: `529_Lab-3090`
- `remote_repo`: `/home/yingch/projects/FATST`
- `exact_commit`: `fe9ac10b49779b46e5d1e1aaba2566af796cb8e4`
- `config_hash`: `55375b735cb87534625272d32fd07044e7ad9b122b65d57b1e4848c7c9ff9449`
- `search_space_hash`: `500e3139e91b7770eca469fa762a8c6f9fb1ee167754e0485e7150f2811ff646`
- `output_root`: `/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h5c`

Remote `git pull --ff-only`保留了三份与本轮无关的历史dirty analysis CSV；H5C source、config和runner均来自上述exact commit，未修改这些remote-local files。

## 2. Resource preflight and smoke

启动前GPU0/1/2均为RTX 3090，memory=`18 MiB`、free=`24107 MiB`、utilization=`0%`。User quota=`187G/200G soft/220G hard`；H5B完整root为1.3 GiB，H5C formal training projection仍低于soft limit，未删除任何H5B formal evidence。

Full 54-profile two-batch resource smoke于`11:40:12--11:42:32 +08:00`完成：

- 54/54 checkpoints；
- 54/54 validation metrics和effective configs；
- 54 unique checkpoint SHA256；
- failure-token logs=`0`；
- test artifacts=`0`；
- post-smoke quota=`188G`。

所有L570--736 context/patch、LR/dropout interactions、weight decay和rank profiles均未出现OOM、NaN/Inf、Traceback或RuntimeError。

## 3. Full train/validation launch

- `start`: `2026-08-13T11:43:04+08:00`
- `PID`: `843554`
- `GPU_IDS`: `0 1 2`
- `jobs`: 54 ETTh1 profiles
- `test_jobs`: 0
- `command`: `MODE=train GPU_IDS="0 1 2" bash scripts/remote/run_iscf_bsca_main_v1_hpo_etth1_h5c.sh`
- `queue`: shared dynamic LPT queue；longer context first，随后winner-centered interactions。
- initial jobs：GPU0=`ctx736_p23`、GPU1=`ctx726_p22`、GPU2=`ctx713_p23`。
- first health check：三个jobs均完成epoch2，validation loss finite；GPU memory=`766--784 MiB`，utilization约`24--25%`。

根据H5B实测与H5C trial数量，预计约`1.5--3.5 wall-hours`。54/54 training artifacts与immutable manifest完成前，禁止official test、partial profile selection或paper-table mutation。Decision=`H5C_three_GPU_train_validation_active_test_zero`。

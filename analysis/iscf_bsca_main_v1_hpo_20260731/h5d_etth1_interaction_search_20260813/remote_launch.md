# H5D ETTh1 Remote Launch

## 1. Frozen identity

| Item | Value |
| --- | --- |
| remote host | `529_Lab-3090` |
| repo | `/home/yingch/projects/FATST` |
| exact training commit | `21df4c80cd484350ea4ae777d6453bae94d8512c` |
| config | `configs/iscf_bsca_main_v1_hpo_etth1_h5d.json` |
| config SHA256 | `1c0cf53197501652f5bafe5282a136e22913199c210a2c1210d4b6dfd3bdfaec` |
| search-space SHA256 | `8dd043c3a0a40000bfc9dc917475ba4914d768cb77fa4644f9b18630d58b2f10` |
| output root | `/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h5d` |
| GPUs | 0, 1, 2 |
| formal test during training | 0/48 |

Remote fast-forward保留了三个与本轮无关的historical analysis CSV修改；未stash、覆盖或提交。

## 2. Storage cleanup and preflight

Launch前精确删除H4M/H4N/H5A/H5B/H5C五个completed `_resource_smoke`
目录，释放约3.2 GiB。它们均可由冻结config重建；正式checkpoints、metrics、manifests、
test artifacts和logs全部保留。Quota由189G降至186G（soft 200G，hard 220G）。

GPU0/1/2在preflight均为18 MiB、0% utilization，未发现compute process。H5D full-training
checkpoints为0，remote checker确认48 new profiles、115 historical profiles、0 duplicate
fingerprints及formal-test authorization=false。

## 3. Resource smoke

- interval：2026-08-13 14:40:05--14:42:09 +08:00；
- 48/48 checkpoints、metrics、effective configs、diagnostics与logs；
- 48 unique checkpoint hashes；
- OOM/NaN/Inf/Traceback/RuntimeError hits=0；
- evaluation split均为validation，test jobs=0；
- smoke storage约498 MiB；post-smoke quota仍为186G。

Batch64/high-LR/high-rank边界均通过resource gate。

## 4. Full train/validation launch

Full queue于2026-08-13 14:43:23 +08:00 detached启动，PID=`1092478`：

```bash
MODE=train GPU_IDS="0 1 2" \
  bash scripts/remote/run_iscf_bsca_main_v1_hpo_etth1_h5d.sh
```

首批三个batch16 jobs分别进入GPU0/1/2；14:43:55时均完成epoch1，其中GPU2已进入
epoch2 iteration100。Epoch1 validation losses为`1.139454/1.141646/1.143178`，均finite。
Observed memory为561--562 MiB/GPU，utilization约0--17%。Status=`0/48 complete, test=0/48`。

按H5C历史吞吐与batch16前置队列估计，完成窗口约为16:15--18:45。训练期间不读取partial
validation作profile selection、不pull新代码、不启动formal test。48/48 artifacts完成后，先运行
training-artifact/provenance/numeric-health audit并冻结immutable checkpoint manifest，再请求新的
formal-test授权。

Decision=`H5D_three_GPU_train_validation_active_test_zero`。

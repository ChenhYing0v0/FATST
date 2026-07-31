# ISCF-BSCA-MAIN-v1 H2 Prelaunch and Launch

## Status

| Field | Value |
| --- | --- |
| `date` | `2026-07-31` |
| `candidate` | `ISCF-BSCA-MAIN-v1` |
| `phase` | `H2` |
| `launch_commit` | `b94d47b08d0a7ba1a374347efc817f5a038ee2cd` |
| `config_sha256` | `9300721b22ce2be013f44165132310d5c7ce654bf449d67f4f79a5f877afafaf` |
| `search_space_sha256` | `de3845442c4632ad7160c0c27f557bec05ac299d0179090ea47c21f599a4adc4` |
| `matrix` | `8 datasets × 3 additional profiles = 24 jobs` |
| `total_H1_plus_H2` | `40 trials` |
| `seed` | `2021` |
| `official_test_jobs` | `0` |
| `status` | `H2_train_validation_running` |

## Version and repository audit

Local focused commit `b94d47b`已push至`origin/main`。Remote在
`/home/yingch/projects/FATST`使用`git pull --ff-only`从`7361d9e`更新到
`b94d47b`。Remote原有三处unrelated dirty CSV与incoming paths无冲突并完整保留。

## Prelaunch gates

1. H1 artifact audit：16/16 complete，test=0；
2. local H1 regression checker：pass；
3. local/remote H2 contract checker：pass；
4. H2 dry-run：24 jobs，test=0；
5. new-dataset resource canary：9/9 pass；
6. full-matrix two-batch resource smoke：24/24 pass；
7. failure scan：无Traceback、OOM、NaN或Inf；
8. storage：`/home`约889 GiB free。

Launch前GPUs 0/1/2均为RTX 3090，各18 MiB used、24107 MiB free、0%
utilization，无compute process。

## Launch

Launch time=`2026-07-31T19:02:12+08:00`。

```text
MODE=train bash scripts/remote/run_iscf_bsca_main_v1_hpo_h2.sh
```

Orchestrator PID=`905874`，output root=
`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h2`。

首发jobs按实测workload-aware ordering使用三张GPU：

- GPU0：`ECL__h2_lookback336`；
- GPU1：`ECL__h2_dropout3`；
- GPU2：`ECL__h2_intermediate_capacity`。

首次snapshot分别使用约2083、2108、2036 MiB，GPU utilization约74%、77%、
78%。三个jobs均进入training，loss尚未形成完整epoch score。根据H1实测耗时和
H2的30-epoch上限，完整矩阵预计约5--7小时；early stopping可能缩短该时间。

## Next gate

H2必须24/24生成完整training、four-H validation、checkpoint hash和numeric-health
artifacts。H2完成前不执行official test。完成后先运行artifact analyzer；只有完整
矩阵通过，才对H1+H2全部40个trials执行four-H official-test scorecard和
dataset-level profile ranking。

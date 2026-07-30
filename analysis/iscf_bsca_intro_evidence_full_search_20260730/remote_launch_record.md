# Introduction Evidence Full Search Remote Launch

## Launch state

| Field | Content |
| --- | --- |
| `date` | `2026-07-30` |
| `host` | `529_Lab-3090` |
| `commit` | `0808c80a6eea44ac56be956f7900b6a3033fa140` |
| `environment` | `moe` |
| `gpu_ids` | `0,1,2` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/intro_evidence_visualization_pilot_v1` |
| `driver_log` | `_logs/full_search_driver_retry1.log` |
| `driver_pid` | `2888928` |
| `split` | validation |
| `test_accessed` | false |

Retry preflight：

- remote五dataset construction通过：
  ETTh1/ETTh2 train=`7825`、val=`2161`；
  ETTm1/ETTm2 train=`33745`、val=`10801`；
  Weather train=`36072`、val=`4551`；
- full local/remote dry-run通过；
- GPU 0/1/2均为RTX 3090，launch前各18 MiB used，无compute process；
- remote三份历史CSV修改保持不动；
- first-attempt已完成artifacts restart-safe复用：
  neutral=`13/25`，DLinear=`6/20`，因此retry实际remaining=`26`。

## Launch command

```bash
setsid -f env \
  GPU_IDS="0 1 2" \
  CONDA_BIN=/home/anaconda3/bin/conda \
  CONDA_ENV=moe \
  bash scripts/remote/run_intro_evidence_full_search.sh \
  > /home/yingch/exp_outputs/r-2026-fatst/intro_evidence_visualization_pilot_v1/_logs/full_search_driver_retry1.log \
  2>&1 < /dev/null
```

## Initial health check

Retry启动后：

```text
GPU 0: neutral ETTm2 s1, 567 MiB
GPU 1: neutral ETTh1 s1, 568 MiB
GPU 2: DLinear ETTm2 H720, 388 MiB
```

未出现argument、dataset、CUDA或OOM错误。ETTh2-s1在first attempt中约20秒完成，
而旧逐block实现的s1曾是约80分钟级长尾；该单次observed runtime支持vectorization
已经移除主要operator-launch bottleneck，但不作为正式benchmark。

Current decision=`full_search_retry1_running_wait_for_user_completion_notice`。
不高频polling；完成后同步45/45 artifacts、运行五dataset ranking与final-size
visual QA。

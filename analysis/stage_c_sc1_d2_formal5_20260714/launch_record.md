# SC1-D2 Formal5 Launch Record

| Field | Value |
| --- | --- |
| `server` | `529_Lab-3090` |
| `repo_commit` | `24c910484156cfeeed4a29d5f7e8b0f8db2b10bf` |
| `contract_hash` | `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a` |
| `environment` | conda `moe`；Python 3.12.13；torch 2.9.0+cu128；CUDA 12.8 |
| `gpu` | RTX 3090 ×3；launch preflight各15 MiB、0% utilization |
| `start` | 2026-07-14 00:15:39 +08:00 |
| `finish` | 2026-07-14 00:18:28 +08:00 |
| `output` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc1_d2_formal5` |

正式命令：

```bash
GPU_IDS="0 1 2" bash scripts/remote/run_stage_c_sc1_d2_formal5.sh
```

调度：GPU0 `Weather -> ETTh1`；GPU1 `ETTm1 -> ETTh2`；GPU2 `ETTm2`。正式launch前已完成synthetic
768/1536/3072-width invariants与ETTh1 actual-checkpoint 33-fit shortened smoke；smoke metrics不进入formal gate。

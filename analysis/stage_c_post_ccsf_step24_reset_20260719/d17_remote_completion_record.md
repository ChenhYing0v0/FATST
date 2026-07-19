# SC-D17-PFC-v1 Remote Completion Record

| Field | Value |
| --- | --- |
| `commit` | `746721be77c3cfd20ffa9d6c925240c1470491b7` |
| `server` | `529_Lab-3090` |
| `environment` | `moe` |
| `gpu_ids` | `0,1,2` |
| `observed_memory_before_launch` | GPU0/1/2 each 15 MiB used, 24110 MiB free |
| `start_time` | `2026-07-19T12:42:49+08:00` |
| `validation_complete_time` | `2026-07-19T12:43:51+08:00` |
| `analysis_complete_time` | `2026-07-19T12:44:27+08:00` |
| `matrix` | 2 carriers × 5 datasets = 10 validation inference jobs |
| `completion` | 10/10 |
| `checkpoint_mutation` | false；runner前后SHA-256一致 |
| `new_test_access` | false；analysis复用既有authorized test probes |
| `remote_root` | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_d17_projective_future_context_v1` |

所有10个`trained_invariants.json`均满足`pass=true`、`protocol_pass=true`与
`readout_contract_pass=true`。本次没有训练新模型，也没有修改原run directories中的checkpoint。

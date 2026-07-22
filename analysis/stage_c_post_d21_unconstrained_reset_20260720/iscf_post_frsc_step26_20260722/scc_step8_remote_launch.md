# ISCF-SCC Step8 Remote Validation Launch

## 1. Decision record

| Field | Record |
| --- | --- |
| `current_step` | Step8 20-run validation matrix running |
| `candidate` | `SC-ISCF-SCC-v0-step7b` |
| `source_commit` | `91e466a58435d5653a8b0d7d65bb567d6dc241ca` |
| `config_sha256` | `bf34ce789f18b0a7db6300f8ee51d36a634b4f307fcfe90c6ca614e6507f7837` |
| `profile_sha256` | `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a` |
| `matrix` | 4 new objectives × 5 datasets × seed2021=20 validation runs |
| `decision` | `remote_validation_running_no_formal_test` |

## 2. Resource smoke

在GPU0/1上先执行Weather SCC与SHUFFLED的2-batch/1-epoch smoke。两者finite并正常退出；mean positive
contributors=`2.697`、all-nonpositive fraction=`0`。五个scope gradient norms均非零，范围约`.0527–.1110`。
SHUFFLED与SCC arm gradients相同符合设计：route credit完整detach且KL只改变policy gradient。

## 3. Full launch

launch前GPU0/1/2均为NVIDIA RTX 3090、`18 MiB / 24576 MiB`、0% utilization。remote project已fast-forward到
source commit。执行：

```text
GPU_IDS="0 1 2" bash scripts/remote/run_stage_c_iscf_scc_step7b.sh
```

detached supervisor PID=`3726827`；output root：

```text
/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_scc_v0_step7b
```

首次status确认`0/20` complete，GPU0/1/2分别运行Weather SCC、SHUFFLED、FUSED，均已进入epoch 1且超过500
iterations。runner按dataset-major order继续其余jobs；formal test access=false。预计完整matrix约1–2小时，受early stopping
影响可能提前；按用户约定不持续短轮询。

## 4. Authorization boundary

```text
remote_validation = running
formal_test_authorized = false
step9_decision = pending_full_20_run_artifacts
partial_result_selection = forbidden
```

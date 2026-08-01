# ECL/Solar H3A Remote Launch

## Launch record

| Field | Value |
| --- | --- |
| `candidate_version` | `ISCF-BSCA-MAIN-v1-ecl-solar-h3a-test-informed-20260801` |
| `date` | 2026-08-01 |
| `commit` | `72e1f8f60a20ec8c1a60712601217a92f7048f47` |
| `config_hash` | `1a1bd8ed70619934c1442694b0eda209a3ff1a98ed68b75620f43291b13521ad` |
| `search_space_hash` | `a786d7a86a1e4fd47d2087a51742e81b776ea610c8d02a7e6b0d5e9571f121f4` |
| `remote_host` | `529_Lab-3090` |
| `remote_project` | `/home/yingch/projects/FATST` |
| `output_root` | `/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/ecl_solar_h3a` |
| `orchestrator_pid` | `1996557` |
| `start_time` | `2026-08-01T09:28:45+08:00` |
| `GPUs` | 0, 1, 2；RTX 3090 |
| `GPU_preflight` | each 18 MiB used，0% utilization，no compute process |
| `resource_smoke` | 9/9 complete；two train/eval batches；no OOM/NaN/Inf |
| `full_matrix` | ECL 1 + Solar 8 = 9 jobs；seed2021；45 epochs/patience10 |
| `test_during_training` | 0 |

## Command

```bash
MODE=train GPU_IDS="0 1 2" \
  bash scripts/remote/run_iscf_bsca_main_v1_hpo_ecl_solar_h3a.sh
```

The command runs under `nohup`; remote orchestrator output is `ecl_solar_h3a/orchestrator.log`. Per-trial logs are under `_logs/train_<trial_id>.log`.

## Initial progress

At `2026-08-01T09:29:43+08:00`, 3/9 jobs were active:

- GPU0：`ECL__h3a_budget45`，epoch1，at least 800 training iterations；
- GPU1：`Solar__h3a_budget45`，epoch1，at least 200 iterations；
- GPU2：`Solar__h3a_lr3e4`，epoch1，at least 200 iterations。

Observed GPU memory/utilization was approximately `2107 MiB/74%`, `3960 MiB/87%`, and `3960 MiB/98%`. No failure signature appeared. Based on initial throughput and three waves of Solar jobs, conservative wall-clock ETA is 12--18 hours; early stopping can shorten it.

## Completion and direct-test gate

Training must reach 9/9 complete artifacts with finite four-H validation metrics and frozen checkpoint hashes. Validation is used only for each trial's checkpoint selection and completeness audit; no validation-based profile ranking will delay test. After the 9-row manifest is frozen, all H3A checkpoints receive the same complete four-H official-test audit. No per-horizon or partial selection is allowed.

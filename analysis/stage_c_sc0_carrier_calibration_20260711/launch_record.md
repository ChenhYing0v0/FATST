# StageC SC0 Standardized Carrier Calibration Launch Record

## Decision Context

| Field | Content |
| --- | --- |
| `candidate` | `SC0-MCP` |
| `role` | validation-only standardized carrier control；not a paper-core method |
| `current_step` | StageC Step 3 calibration |
| `gate` | choose one global profile by full-720 validation MSE macro regret；test split forbidden |
| `rollback` | if no common arm passes or selector ranking is unstable, return StageC Step 2/3；do not restore dataset-specific presets |

## Code And Protocol

| Field | Content |
| --- | --- |
| local/remote commit | `31730cded931f9d1dceb940424d13676daaec7cc` |
| protocol profile | `stage_c_sc0_calibration_v1` |
| profile SHA256 | `79a037f751c0c24eea98ff0b516cb0dfeaef950871b3bbc515904754f54fd900` |
| config | `configs/stage_c_mechanism_control.json` |
| runner | `scripts/remote/run_stage_c_sc0_carrier_calibration.sh` |
| environment | remote conda `moe` |
| remote project | `/home/yingch/projects/FATST` |
| output root | `/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc0_carrier_calibration` |

## GPU Preflight

Preflight time：`2026-07-11T14:05+08:00`。

| GPU | Model | Used MiB | Free MiB | Utilization | Assigned Work |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | NVIDIA GeForce RTX 3090 | 15 | 24110 | 0% | queue slot 0 |
| 1 | NVIDIA GeForce RTX 3090 | 15 | 24110 | 0% | queue slot 1 |
| 2 | NVIDIA GeForce RTX 3090 | 15 | 24110 | 0% | queue slot 2 |

No compute process was active. `/home` had approximately `964 GB` free.

## Launch

- start time：`2026-07-11T14:06:11+08:00`；
- launcher wrapper PID：`431994`；
- SC0 runner child PID：`431996`；
- GPU assignment：`GPU_IDS="0 1 2"`；
- scheduling：dataset-major `Weather -> ETTm1 -> ETTh2`，每个 dataset依次包含三个arms，并用
  `wait -n` 填充空闲GPU；
- launcher log：`<output_root>/_launcher.log`。

Equivalent command：

```bash
GPU_IDS="0 1 2" \
OUTPUT_ROOT=/home/yingch/exp_outputs/r-2026-fatst/stage_c_sc0_carrier_calibration \
DATASET_ROOT=/home/yingch/dataset \
CONDA_ENV=moe \
CONDA_BIN=/home/anaconda3/bin/conda \
bash scripts/remote/run_stage_c_sc0_carrier_calibration.sh
```

## Matrix

| Dimension | Values |
| --- | --- |
| datasets | `Weather, ETTm1, ETTh2` |
| arms | `sc0_p12_d128, sc0_p24_d64, sc0_p48_d32` |
| seed | 2021 |
| runs | 9 |
| train objective | full-720 L1 only |
| checkpoint selection | best full-720 validation MSE；last state sensitivity |
| exported diagnostics | validation horizons `48,96,144,192,288,336,512,720` |
| test evaluation | disabled by CLI contract |

## Initial Progress

At `2026-07-11T14:07:04+08:00`：

- active position：Weather arms `1-3/9`；
- `P12/P24/P48` latest epochs：`3/20`, `2/20`, `2/20`；
- active GPU memory：approximately `487/682/1252 MiB`；
- completed runs：`0/9`；
- no runtime/numeric pathology observed；
- estimated completion：approximately `2026-07-11T14:20-14:25+08:00`，以returned logs为准。

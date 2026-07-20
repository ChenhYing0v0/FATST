# SC-D23-FCMI Step8 Remote Launch Record

## 1. Scope

- candidate: `SC-D23-FCMI-v1`
- role: seed2021 paper-facing effectiveness and matched-attribution audit
- matrix: 8 arms × 5 datasets = 40 training runs
- scorecard: 160 validation cells + 160 official-test cells
- confirmation seeds: false
- paper-method promotion: false

## 2. Authorization and immutable identity

- user authorization: 2026-07-20，用户指令“按计划继续推进工作”
- commit: `4ff439c682f03f48a9c8a5f565e01c6ba70f6e4e`
- config SHA256:
  `488dd248b115a556674275e7b86a10b542e584846f9b8bff68880b60cf5d08ca`
- profile SHA256:
  `80912741f9da5560234c400a36e2ec48461cef70bf96701b19fcb90ea278990a`
- remote repository: `/home/yingch/projects/FATST`
- output root:
  `/home/yingch/exp_outputs/r-2026-fatst/stage_c_d23_fcmi_v1`
- environment: `moe`，PyTorch `2.9.0+cu128`，CUDA runtime `12.8`

远端仓库在pull前存在三处与D23无关的historical analysis CSV修改；它们未被清理或覆盖。
`git pull --ff-only`成功把remote HEAD更新到上述commit。

## 3. GPU preflight

`2026-07-20T17:55+08:00`检查GPU0/1/2：

| GPU | Model | Total MiB | Used MiB | Free MiB | Utilization |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | RTX 3090 | 24576 | 18 | 24107 | 0% |
| 1 | RTX 3090 | 24576 | 18 | 24107 | 0% |
| 2 | RTX 3090 | 24576 | 18 | 24107 | 0% |

没有active compute process。

## 4. Resource smoke

两项真实smoke均使用2 train batches、2 validation batches、1 epoch：

| GPU | Arm / Dataset | Train loss | Validation loss | Result |
| ---: | --- | ---: | ---: | --- |
| 0 | FCMI / Weather | 0.5390141 | 0.7558621 | finite/pass |
| 1 | DENSE_DUAL_MATCHED / ETTm2 | 0.5136287 | 0.2242425 | finite/pass |

两项均生成checkpoint、effective config、environment、initialization contract、model diagnostics与
training log；未发现NaN、Inf、traceback或OOM。smoke完成时间为
`2026-07-20T17:56:42+08:00`。

## 5. Formal launch

- start: `2026-07-20T17:57:10+08:00`
- runner PID: `480809`（wrapper PID `480808`）
- worker/GPU: `480828/0`、`480832/1`、`480838/2`
- first jobs:
  - GPU0: Weather / FCMI
  - GPU1: Weather / DENSE_DUAL_MATCHED
  - GPU2: Weather / A6_MEASURE

启动后短检查显示三个training processes均存活。FCMI与DENSE完成epoch 1时validation loss分别为
`0.5688369`与`0.4973443`；A6进入epoch 4。这些数值只证明runtime健康，不用于checkpoint之外的选择，
也不构成effectiveness证据。

## 6. Running boundary

matrix运行期间不得remote pull、修改config/profile/arms/gates、选择dataset/horizon结果或启动
confirmation seeds。40/40 artifacts完整返回后，必须由冻结analyzer一次性生成四层Step9/10 decision。

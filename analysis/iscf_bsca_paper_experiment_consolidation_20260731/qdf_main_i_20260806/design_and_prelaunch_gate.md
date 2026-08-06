# QDF Main I 与 Solar 复现：设计与 prelaunch gate（2026-08-06）

## 1. 当前结论与边界

- `current_step=Step 6 protocol freeze -> Step 8 remote execution authorized`。
- 用户已明确授权拉取 QDF 官方仓库、补写 Solar script、执行 Solar 四个标准 horizon 的训练与正式 test，并将结果加入 Main I。
- ISCF-BSCA HPO 保持 terminal freeze；本任务不改 ISCF architecture、objective、selected profiles 或任何既有 scorecard。
- QDF 在 Main I 中属于 `modern horizon-specific baseline`。它可比较 paper-facing effectiveness，但不构成 ISCF-BSCA 的 matched mechanism attribution。

## 2. Source 与 published-result 审计

论文为 *Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models*，Main I 使用的 QDF backbone 为 TQNet。官方源码固定为：

- repository：`https://github.com/Master-PLC/QDF`
- commit：`eb0693a962928e229417fd80b401c37b0dac6a67`
- license：MIT
- vendored source：`baselines/qdf_official/`
- published transcription：`qdf_table6_published.csv`

QDF Table 6 报告 ETTm1、ETTm2、ETTh1、ETTh2、ECL、Weather 的 `L=96`、`H={96,192,336,720}` 三次运行均值，但没有 Solar。因此六个数据集直接保留 published values；Solar 必须独立标记为 local source-informed reproduction，禁止将其写成论文原值。

## 3. Solar contract

官方代码已有 `Dataset_Solar`，读取无时间戳的 `solar_AL.txt`，137 channels，并使用 70/10/20 顺序切分和 train-only scaler。官方 release 没有 Solar shell script，因此以最接近的高维 electricity preset `scripts/ECL.sh` 为来源：

- 保留每个 horizon 的 `learning_rate / inner_lr / meta_lr / warmup_steps / num_tasks / meta_inner_steps / batch_size`；
- 保留 `seed=2023`、30 epochs、patience 5、TQNet、QDF loss、RevIN、loss weights 与 gradient clipping；
- 只做必要的 Solar 语义替换：`enc_in=dec_in=c_out=137`，`data=Solar`，daily `cycle=144`（10-minute sampling）。
- 远程 file-descriptor safety 固定 `num_workers=0`；该设置只改变 host-side loading，并不改变模型、split、batch order、loss 或 checkpoint selector。

需要披露的 source discrepancy：论文 Appendix C 表述 early stopping patience 3 且不 drop last，而 release 的 ECL script/runner 实际为 patience 5，且 train loader `drop_last=True`。本复现以可执行 released code 为准，不对这一差异进行静默修正。

## 4. Frozen matrix 与调度

| GPU | 顺序 | 理由 |
|---|---|---|
| 0 | H720 | 最长 horizon 独占一个 GPU |
| 1 | H336 -> H96 | 先执行较慢 H336，再填充 H96 |
| 2 | H192 | 与最长任务并行 |

完整矩阵为 `Solar × {96,192,336,720} × seed2023`，每个 cell 是一个 fixed-H system。checkpoint 由 upstream validation early stopping 选择；每个 run 仅在训练结束后访问一次 test。MSE/MAE 均完整报告，不做 per-cell rescue、test-based checkpoint/seed selection 或选择性删除。

## 5. Gates、资源与 rollback

1. `prelaunch_pass`：source/data hashes、JSON、shell syntax、Python compile、四行 dry-run 和 published 24-row transcription 全部通过。
2. `resource_smoke_pass`：四个 horizon 各执行至多两个 train/validation batches，禁止 test，须无 OOM、NaN、Traceback。
3. `formal_pass`：4/4 checkpoint、4/4 learned-QDF-loss artifact、4/4 effective configs 和 4/4 test metrics 完整；记录 hashes、environment 与 logs。
4. `rollback`：任一 formal cell 失败则停止 QDF Solar block；六个 published QDF datasets 仍可保留，但 Solar 保持 missing，禁止插值或用其他模型结果代替。

远程结果根目录固定为 `/home/yingch/exp_outputs/r-2026-fatst/qdf_solar_reproduction_20260806`。prelaunch 时 remote quota 约 175/200 GiB soft、220 GiB hard，GPU 0--2 均约 18 MiB/0%；不保存 prediction arrays，以控制存储占用。

## 6. Prelaunch decision

`PASS`。用户对该精确 Solar block 的 local patch、remote training 与 formal test 已完整授权；允许在 commit/push、remote pull、resource smoke 通过后直接执行 formal matrix。结果只在 4/4 artifact audit 完成后进入 Main I。

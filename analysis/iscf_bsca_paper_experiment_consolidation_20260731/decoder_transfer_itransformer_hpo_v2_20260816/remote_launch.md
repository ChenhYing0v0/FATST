# iTransformer-style Decoder-HPO v2 Remote Launch

Decision：`itransformer_decoder_hpo_v2_three_gpu_training_active_test_zero`。

## 1. Launch identity

- launch time：`2026-08-16T02:04:40+08:00`；
- local/remote commit：`219d52708e22e651725de4b7c027c93c16566677`；
- source profile SHA256：`5e8dc7f04765fae36d25f696cf66be00cc58ee59f7559871119751abb4ae8b51`；
- frozen search/config SHA256：`291b23be097fd991a325f5b83c8464bfa2e74264f2e69496c288cb8d92ea9c4d`；
- remote driver PID：`1517179`；
- GPUs：RTX 3090 `0,1,2`；
- output root：`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_decoder_transfer_itransformer_hpo_v2_20260816`；
- matrix：14 profiles × 5 datasets × seed2021=`70` training runs；
- launch-time formal test jobs：`0`。

## 2. Resource gate

启动前GPU0/1/2均为18 MiB、0% utilization。用户quota为194 GiB used、200 GiB soft、220 GiB hard，project experiment root为85 GiB。预计本轮新增2--4 GiB，未超过hard limit，但运行期间不得保留重复prediction tensors或临时smoke checkpoint。

Remote prelaunch在`moe`环境通过22/22 checks，确认70-run matrix唯一且formal-test jobs为0。三个Weather representative profiles完成resource smoke：

- `p00_budget30`：default decoder geometry；
- `p09_coord8_capmatch_lr0p50`：8-dimensional future coordinate与inverse-rank capacity match；
- `p13_scopes_long_lr0p50`：alternate long-scope geometry。

三者均以`evaluation_split=none`退出，未访问test labels；无OOM、Traceback或non-finite错误。60 MiB smoke目录在正式启动前已删除。

## 3. Active queue evidence

正式命令使用绝对conda入口与repo-external output root：

```bash
nohup env GPU_IDS="0 1 2" \
  bash scripts/remote/run_iscf_bsca_decoder_transfer_itransformer_hpo_v2.sh \
  > /home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_decoder_transfer_itransformer_hpo_v2_20260816/driver.log \
  2>&1 < /dev/null &
```

启动后25秒，driver仍存活，status为`training=0/70 test=0`。workload-aware队列首先占满三个GPU运行Weather：

| GPU | Job | Profile | Observed state |
| --- | --- | --- | --- |
| 0 | 1/70 | `p00_budget30` | epoch 1, iteration 300, finite loss |
| 1 | 2/70 | `p01_lr0p25` | epoch 1, iteration 300, finite loss |
| 2 | 3/70 | `p02_lr0p50` | epoch 1, iteration 300, finite loss |

三卡显存均约1.23 GiB，utilization约36--37%，保留充分安全余量。

## 4. Frozen completion gate

训练完成不自动触发formal test。下一步必须先运行artifact checker并满足：

1. 70/70 checkpoint、training log、four-H validation metrics与effective config完整；
2. 70个checkpoint SHA256唯一；
3. manifest写入后保持immutable；
4. training阶段test artifacts=`0`；
5. numeric health与frozen protocol fields全部通过。

只有上述gate闭合后，才执行已授权的完整`70 checkpoints × 4 horizons = 280` test-tuned scorecard。每个dataset仅按four-H mean test MSE选择一个shared profile；禁止per-H、per-seed、per-metric或per-cell选参，且所有负向trials必须保留。Canonical paper table mutation与extra seeds仍未授权。

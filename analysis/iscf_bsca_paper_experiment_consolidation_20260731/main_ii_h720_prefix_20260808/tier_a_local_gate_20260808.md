# Main II Tier A local gate（2026-08-08）

## Decision

`pass_for_remote_resource_smoke`。该结论只授权进入已获批的 Tier B no-test smoke，不表示 remote formal training、formal test 或 Main II result table 已完成。

## Verified surface

- exact upstream commits：iTransformer `c2426e6...`、PatchTST `204c21e...`、DLinear `0c11366...`；
- key source hashes与19个released script hashes：全部匹配 execution config；
- runtime workspaces：3/3 prepare pass；
- official H720 command extraction：21/21 pass；
- training-time test access：0；validation early stopping保持；
- PatchTST/DLinear Solar：loader registration与ECL-profile-derived command均闭合，角色=`source_informed_not_official`；
- matrix ledger：56 system-dataset rows、49 reusable checkpoint objects、21 new H720 jobs、70 formal checkpoint evaluations、280 raw prefix rows、224 aggregate cells；
- synthetic canonical tensor test：4/4 horizon rows，MSE/MAE与exact prefix identity通过；
- Python compile、JSON parse、remote shell `bash -n`：通过。

## Frozen local hashes

| Artifact | SHA256 |
| --- | --- |
| execution config | `2680358fceb51f2e51638b17a924d19334e81665fab731b206d52a8546aef0c1` |
| source/training adapter | `9ac3a04b7dd54e8ebd65430d6fa81fdc8fefa5c31b507b65b30cd5c52cf39183` |
| prefix evaluator | `a64da68cd55e5ce218c896c5b224c7ed1263e2140ee5079b99779ee53125631e` |
| prelaunch checker | `10b9844f3c89cdebc55e1332df8c6b4c41e09ec27c3580a591ea8f50b4e65fe3` |
| remote launcher | `b0d00301af57bd391a5c34312417abeed6a723fc5a857522863033423d1f0580` |

上述 hashes 是本地 gate 时点值；commit 后 remote launch record 必须再次记录实际 checkout commit及文件 hashes。任何差异都回到 Tier A source/patch audit，不启动 affected jobs。

## Remote entry gate

Remote side须使用 clean exact-commit worktree，不覆盖主仓库现有的三处无关 CSV 修改。启动前重新检查 `quota -s`、`nvidia-smi`、dataset hashes与40 GiB新增storage budget。21个resource smokes全部产生checkpoint且test rows=0后，才能进入formal training。

## Remote smoke compatibility repair

首轮remote smoke为20/21 pass；DLinear-Solar在构建train dataset时因native `data_factory`额外传入`train_only`而触发constructor `TypeError`，test loader尚未构建。失败归因为`artifact/protocol_adapter_compatibility_defect`，不是model behavior。v2 runtime workspace仅给DLinear Solar constructor增加unused `**kwargs`，不改变split、scaler、tensor或optimization；失败目录保留并versioned recovery只重跑该缺失unit。

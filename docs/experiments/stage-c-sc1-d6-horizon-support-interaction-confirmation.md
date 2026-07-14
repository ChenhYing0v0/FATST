# SC1-D6 Horizon-Support Interaction Confirmation Protocol

## Status

| Field | Value |
| --- | --- |
| `candidate` | `SC1-D6` |
| `role` | `diagnostic_only_confirmation` |
| `current_step` | Step 2/3 complete；pass，return Step 4 |
| `question` | b144 local DCT的short-positive/long-negative crossed interaction能否在未使用validation window复现？ |
| `suite` | 5 datasets × 3 checkpoints × 3 grouping seeds × 5 bases = 225 fits |
| `evaluation_window` | official validation batches 8-15；D5使用0-7 |
| `test_used` | false |
| `method_training_authorized` | false；Step 4 conditional narrative audit complete，Step 5 next |

## Design

训练数据、fit/inner-holdout、A6 memory、random coefficient groups、optimizer与D5相同，只移动final official
validation evaluation window。bases为balanced interval、global DCT-II及block DCT-II b48/b96/b144。b48/b96
用于检查support-size ordering；primary candidate固定为b144，不通过新window重新选择。

short horizons为`48,96,144`，long horizons为`336,512,720`。effect仍为
$\log(E_{control}/E_{b144})$，positive表示b144更好。

## Gate

必须先通过225/225、15 metadata、validation offset=8、no-test、no-model-update、finite与orthogonality checks。

interaction gate：

- b144 vs global DCT：short MSE reduction至少`+0.5%`，long最多`-0.5%`；
- b144 vs balanced：short MSE reduction至少`+0.5%`；
- short vs DCT MAE不低于0，long vs DCT MAE不高于0；
- 至少9/15 primary units同时short-positive、long-negative；
- short-positive vs DCT、long-negative vs DCT、short-positive vs balanced均至少3/5 datasets达到2/3 checkpoints。

pass decision=`horizon_support_scale_interaction_supported_return_step4`；否则为
`horizon_support_scale_interaction_not_confirmed_step2`。任何pass只授权Step 4，不授权method training。

## Failure Boundary

D6确认的是support-scale × horizon problem，不确认任何specific operator。若发生numeric pathology或window样本
不足，只能标记diagnostic invalid。若interaction稳定失败，不能用D5同split exploratory result推进method。

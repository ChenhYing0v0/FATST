# SC-D15-T1 PCSD-CF-v1 Test Audit Prelaunch Gate

## Result

`21/21` local cases通过，decision=`prelaunch_pass_remote_test_audit_only`。该gate只授权读取既有60个
best-validation checkpoints并执行一次完整official test audit；不授权checkpoint retraining/mutation、partial matrix
decision或PCC Step6。

## What Was Checked

1. candidate固定为`SC1-PCSD-CF-v1`，5 datasets × 12 arms × seed2021恰好60 runs；
2. dense H1..720、historical best-validation checkpoint、一次正式test access与用户授权字段完整；
3. retraining、checkpoint mutation、partial reporting及per-dataset/horizon tuning全部为false；
4. remote runner不调用`train_repo.py`，调用test evaluator并在每个run前后执行checkpoint SHA-256；
5. cumulative step SSE/SAE得到的720个prefix MSE/MAE与直接prefix reduction一致，最大误差分别为
   `6.66e-16/9.99e-16`。

## Boundary

本gate没有访问test data，也没有证明任何test performance。完整60/60返回前，不得用partial artifacts改变
PCSD/PCC研究方向。若checkpoint hash、split、finite或protocol invariant任一失败，audit整体无效，只允许artifact
repair。

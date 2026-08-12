# H5A Formal-Test Launch

## Gate

2026-08-13 preflight通过：training=`48/48`、test target files=`0`、temporary files=`0`、
48 unique checkpoint hashes、manifest SHA256=
`ee5940c8f66aceab5710f17a4bc8ce2efb9ae3c44fa9cec1459fcd9589fe6643`，remote exact
commit=`cb496f31d0d7cf23b7cfc870e78b3557c9e71004`，GPU0--2均为18 MiB/0%。

## Launch

Once-only complete formal-test queue于2026-08-13 00:34:37启动：

- PID=`4169962`；GPUs=`0 1 2`；`ALLOW_RESUME=0`；
- matrix=`48 checkpoints × four standard horizons = 192 rows`；
- 每个checkpoint还输出dense H1--720 diagnostics，但dataset-level selector只使用冻结的
  H96/192/336/720 MSE/MAE；
- 首批为ECL `budget90/dropout6/dropout7`，三项均已进入GPU evaluation；
- test artifacts先写入per-trial temporary directory，只有完整性、provenance、candidate
  identity和checkpoint hash immutability通过后才atomic publish；任一failure触发ABORT。

Current status=`formal_test_active`。完成48/48前不得执行profile selection、partial table
replacement或H5B extension。

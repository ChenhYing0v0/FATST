# D1-v1 Invalidity Audit

## What Failed

- [Fact] ETTh2 coarse/mid probe为`full_hidden R2=-39.7831`、`patch_shuffled R2=-50.8411`；旧gate仅检查
  差值，因此把两个无效负R2之间的差异误判为Encoder evidence。
- [Strong Evidence] Weather与ETTh2中，label和frozen-A6 residual的nested capture几乎完全相同。结合
  history-std normalized error量级，说明低history-variance窗口主导了source energy，D1-v1没有可靠测量
  evaluation-relevant residual structure。

## What Remains Untested

- evaluation-space future deviation与A6 residual是否具有可利用的nested structure；
- 当前frozen decoder是否实际利用patch order/content；
- evaluation-space deployment measure与projected risk的gradient separation是否跨dataset成立。

## Attribution And Rollback

状态：`diagnostic_invalid_for_direction_rejection`，属于`optimization_or_numeric_pathology`中的measurement/gate
fault，而非`hypothesis_false`。回滚到Step 2-3 diagnostic design；保留candidate、checkpoint、dataset、seed与
batch budget，修订source space和Encoder gate后运行D1-v2。v1 raw artifacts保持不变以供审计。

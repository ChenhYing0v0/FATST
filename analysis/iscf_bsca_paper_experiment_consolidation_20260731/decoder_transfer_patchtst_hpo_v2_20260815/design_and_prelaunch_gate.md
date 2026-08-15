# PatchTST Decoder-Transfer HPO v2：设计与启动门

Decision：`PatchTST decoder HPO v2 frozen; remote training authorized; formal test blocked`。

## 1. 为什么重开、但不改写 v1

既有 `ISCF-BSCA-DECODER-TRANSFER-v1` 是完整且有效的负向结果：PatchTST-style 的 `+ISCF-BSCA` 相对 Original Decoder 的 macro MSE/MAE 为 `-0.733%/-0.062%`，只赢 `2/5` dataset MSE means。该结果继续保留，不能事后解释为无效实验，也不能从原 transfer table 中删除。

本轮建立新的 test-informed candidate `ISCF-BSCA-DECODER-TRANSFER-PATCHTST-HPO-v2`。它回答更窄的问题：固定 PatchTST-style encoder 和 BSCA objective 时，decoder 自身的 capacity/optimization scale 是否不匹配。

## 2. 证据诊断

v1 的四-horizon validation selector 对 `+ISCF-BSCA` 与 Original Decoder 的比较为：

| Dataset | Original val mean MSE | ISCF-BSCA val mean MSE | BSCA方向 |
| --- | ---: | ---: | --- |
| ETTh1 | 1.097532 | 1.095635 | 更低 |
| ETTh2 | 0.381310 | 0.377967 | 更低 |
| ETTm1 | 0.627780 | 0.626720 | 更低 |
| ETTm2 | 0.192916 | 0.185339 | 更低 |
| Weather | 0.524641 | 0.522395 | 更低 |

[Fact] BSCA 在 validation 上是 `5/5` 正向，却在 official test 上只有 `2/5` 正向。这意味着当前主要风险是 split-specific overfitting，而不是 scope path 没有梯度或 BSCA 对 ISCF 完全无作用。因此搜索优先改变 `pcsd_readout` 的 learning-rate scale 与 decoder-only weight decay，并仅用两个 rank 边界点检查 capacity；不调整 PatchTST encoder。

## 3. 冻结矩阵

- datasets：Weather、ETTm1、ETTm2、ETTh1、ETTh2；
- seed：2021；
- horizons：96、192、336、720；
- arm：只搜索 `patchtst_iscf_bsca`；
- 训练：from-scratch、end-to-end joint training、H720 full loss；
- checkpoint selector：four-H validation mean MSE；
- 新 trials：10 profiles × 5 datasets = 50 runs；
- reference：复用 v1 的5个 BSCA checkpoints，仅进入selector，不重复训练；
- official test jobs：0。

10个profiles采用 OFAT 加两个预注册交互点：decoder LR multiplier `{0.25,0.5,2,4}`、decoder-only weight decay `{1e-5,1e-4,1e-3}`、`0.5×LR + 1e-4 WD`，以及在该组合上的 `0.5×/1.5× mode rank`。完整机器可读定义见 `configs/iscf_bsca_decoder_transfer_patchtst_hpo_v2.json`。

## 4. 公平性与选择边界

encoder 的 patch geometry、width、depth、dropout、base learning rate、batch size、patience及初始化均沿用v1 source-informed profile。新增 optimizer parameter group 只作用于 `pcsd_readout.*`；encoder仍使用原base LR和weight decay。每个dataset只选择一个profile，该profile共同服务四个horizons，禁止per-H/per-cell选择。

v2若通过validation gate，只能进入“候选profile冻结”状态。之后需用相同selected decoder hyperparameters为 `+ISCF` 重新做5个matched joint-training runs，再另行请求一次新的formal test授权。v1 Original Decoder checkpoints可以作为冻结anchor，但v2应表述为best-config transfer performance；严格共享optimization的matched attribution仍由v1提供。

## 5. Gate 与 rollback

进入formal-test申请前必须同时满足：

1. 50/50 training artifacts完整且有50个unique checkpoint hashes；
2. 所有validation metrics finite；
3. 按dataset选择后，相对v1 BSCA reference的macro validation MSE改善严格大于0.25%；
4. 至少3/5 datasets的validation MSE改善严格大于0.1%。

若未通过，不访问test，返回Step 4--6实现新的iTransformer-style carrier。该fallback必须继续采用from-scratch end-to-end joint training，不允许用frozen replacement或cross-swap补救方向结论。

### 5.1 启动后artifact-contract修复

2026-08-15首次进度审计发现runner状态函数错误要求`trained_invariants.json`，但HPO runner没有调用生成该文件的重型validation diagnostic evaluator，导致已完成checkpoint被误报为未完成。该问题不影响训练、checkpoint或validation metrics。Artifact contract现改为检查每个run已有的`checkpoint.pt`、`training_log.csv`、four-row `metrics_by_target_horizon.csv`、`effective_config.json`、`initialization_contract.json`、`model_diagnostics.json`与`environment.json`，并要求全部MSE/MAE finite及50个unique hashes。该repair不改变search space、selector、gate或任何已运行trial；formal test仍为0。

## 6. 资源与调度

依据v1日志，Weather、ETTm1、ETTm2单run约40--55分钟，ETTh1/ETTh2明显更短。总预算约22--28 GPU-hours、3张RTX 3090下约8--11 wall-hours，新增存储约3--5 GB。任务按dataset-major顺序先铺Weather和ETTm，再执行ETTh，以减少慢任务尾部。启动前必须检查3张GPU；正式训练不与Efficiency profiling并发。

远程当前quota为191G/200G soft limit、220G hard limit，项目output root约82G。预计新增量仍低于soft limit，但启动后必须只保留本轮必要artifacts，不复制历史checkpoints。

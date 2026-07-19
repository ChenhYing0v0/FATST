# StageC D18 soft-projectivity cost diagnostic 代码说明

## 1. 功能边界

D18没有新增模型结构。它复用`learned-basis-forecast-operator`，只为problem diagnostic增加：

1. horizon-specific prefix loss与对应validation selector的受控training path；
2. A6 prediction probe导出；
3. 25-unit remote matrix、protocol audit与problem gate analyzer。

因此D18仍是`diagnostic_only`，不是Contribution 1 implementation。

## 2. Training flow

`baselines/timealign_official/train_repo.py`只对
`protocol_profile=stage_c_d18_soft_projectivity_cost_v1`允许A6使用`pred_loss_mode=multi-prefix`。

对`A6_SPECH`：

1. `batch_x [B,720,C]`进入A6 Encoder；
2. decoder完整参数域保持T=720；
3. forward以`target_prefix=H`返回`outputs [B,H,C]`；
4. L1只在`target_y[:,:H]`计算；
5. validation只评估H，并据此保存best checkpoint。

local gradient gate直接检查`learned_temporal_basis [720,256]`和
`learned_temporal_bias [720]`：前H rows必须有梯度，H之后必须严格为0。

## 3. Evaluator extension

`scripts/evaluate_stage_c_pcsd_cf_checkpoint.py`做了两个向后兼容扩展：

- 普通A6 readout也保存`probe_fused [N,720]`与`probe_targets [N,720]`；
- protocol config可提供多个`training_contracts`，从而审计统一controls与三个specialists的不同loss/selector，
  而不是把合法协议硬编码为H720/full-loss。

旧config未提供`training_contracts`时仍沿用原单一training contract。D18允许的test role仅扩展为
`primary-problem-existence-diagnostic`；checkpoint hash、non-mutation、full-crop、from-scratch和zero-frozen
checks保持不变。

## 4. Remote runner

`scripts/remote/run_stage_c_d18_soft_projectivity_cost.sh`生成25个artifact units：

- 15个`train`：3 specialists × 5 datasets；
- 10个`reuse`：既有A6_MEASURE/A6_FULL checkpoints重新做D18 test probe export。

reuse只写新的audit artifacts与`source_manifest.json`，不复制或修改历史checkpoint。runner在evaluation前后计算
SHA-256，任何变化立即失败。任务顺序以Weather、ETTm1优先，并分配到GPU0/1/2。

prelaunch checker在3090上优先审计`control_source.remote_root`中的原始controls；本地不存在该路径时才读取
`local_audit_root`同步副本。两条路径执行相同initialization、parameter与invariant checks，不构成fallback放宽。

## 5. Analyzer

`scripts/analyze_stage_c_d18_soft_projectivity_cost.py`读取：

- dense test metrics；
- `probe_fused`；
- effective config、initialization contract与model diagnostics；
- checkpoint SHA-256和test invariants。

它输出own-H cell表、prediction divergence表、25-run audit、gate summary与中文结果报告。paper-facing MSE/MAE
决定problem evidence；prediction NRMSE只证明specialization确实改变了prediction。

## 6. Code-theory consistency

理论要求specialization只放松“所有horizons共享同一预测函数”的optimization约束，而不能通过改capacity、
backbone或输入requested-H制造优势。当前代码保持同一A6 function class，仅改变loss support和validation selector，
符合该诊断目标。

仍然只是proxy的部分：separate specialist models估计的是soft-projective method可能利用的upper headroom，
并不证明单模型能学习该frontier。若specialists不稳定超过A6_MEASURE，problem gate失败；若通过，也必须回Step 4
重新设计单模型、连续、低自由度且可归因的deformation operator。

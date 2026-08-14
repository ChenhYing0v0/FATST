# ISCF-BSCA-v1 code explanation

## 2026-08-15 PatchTST decoder-HPO optimizer groups

`train_repo.py::build_optimizer`为带`pcsd_readout`的HPO trial增加可选的readout参数组。默认`readout_learning_rate_multiplier=1.0`且`readout_weight_decay=None`时仍走历史single-group AdamW，不改变既有run；本轮PatchTST HPO显式把`pcsd_readout.*`与其余encoder参数分组。encoder组保持source profile的base learning rate与weight decay，readout组使用`base_lr × readout_learning_rate_multiplier`及独立`readout_weight_decay`。

`utils/tools.py::adjust_learning_rate`读取每组的`lr_scale`，因此cosine schedule后仍保留readout相对倍率。`training_log.csv`同时记录`lr`与`readout_lr`，`effective_config.json`记录两个新增CLI参数。该patch只改变optimization path，不改变forward tensor、scope集合、policy、BSCA objective或checkpoint selector。

代码理论一致性边界：该设置检验PatchTST representation下decoder optimization scale mismatch，不是新decoder mechanism。所有trial仍从scratch联合训练encoder和decoder；若validation gate失败，应转向新的backbone/carrier设计，而不能用frozen replacement或cross-swap作方向级补救。

## 2026-08-14 Decoder-Transfer carrier extension

为匹配Section 5.7的end-to-end transfer contract，`TimeAlign.Model`新增两个仅供冻结transfer protocol使用的入口。`dlinear-decomposition`把归一化历史`x [B,L,C]`经kernel-25 moving average拆为seasonal/trend，并输出memory `[B,C,2,L]`；`contextual-patch-transformer`沿用既有PatchTST-derived memory `[B,C,P,D]`，但现在允许接入transfer readouts。

`direct-unified-original`在DLinear memory上使用两个共享channel的`L -> 720`线性投影并求和，在PatchTST memory上flatten为`[B,C,P·D]`后使用一个`P·D -> 720`投影。ISCF与ISCF-BSCA两列把同一memory flatten为`hidden [B,C,R]`后输入`siff-independent-scope-control`。因此backbone–dataset block内encoder tensor、初始化class和训练路径一致，差别只位于decoder/objective；所有参数均jointly trainable。

代码理论一致性边界：该实现检验decoder对两种source-informed representation family的可迁移性，不声称与official DLinear/PatchTST逐tensor等价。Original输出与ISCF输出都由一个H720轨迹按prefix返回，因而CHPC由实现构造保证；若两类end-to-end block不能同时通过formal gate，应收窄或撤回transfer claim，而不能用frozen replacement结果补救。

## Forward 与 shape

模型 forward 与 ISCF-v0 完全相同。五个 independent scope arms 产生 `arm_forecasts:[B,C,T,5]`，direct policy 产生 `policy:[B,C,T,5]`，融合结果为 `fused_forecast:[B,T,C]`。BSCA 不增加 module、parameter、requested-H input 或 inference operation。

## Training objective

`PCC.py::projective_coupling_credit_loss` 的 `equal_uniform_scope_anchor` mode 保留 EQUAL 的 uniform arm-skill loss，并构造 `uniform_credit=full_like(policy, 1/5)`。`_weighted_route_kl` 计算按 dense-prefix measure 加权且以 `log(5)` 标准化的 `KL(uniform || policy)`。route weight 在前 25% optimizer progress 从 0 ramp 到 0.1。

Route-only backward 直接更新 policy logits，不直接更新 `arm_forecasts`；joint objective 中，policy 改变 fused loss 对各 scope arm 和 shared encoder 的梯度权重。这是“balanced co-adaptation”的实际实现路径。

## Code-theory consistency

- Intended theory：训练期避免 direct policy 过早集中造成 scope-gradient starvation。
- Code realization：target-free uniform route credit、固定 ramp、同一 ISCF inference graph。
- Proxy boundary：uniform policy 只是 broad gradient access 的 proxy，不保证 arms 自动学到不同 features。
- Falsification：完整 official-test 不优于 EQUAL，或 gain 出现但 policy/internal health 不发生相应改变，均不能支持 BSCA mechanism claim。

## Three-seed confirmation orchestration

Confirmation不修改model或objective。`run_stage_c_iscf_bsca_v1_confirmation.sh`复用single-seed runner，通过`SEED`与`PROTOCOL_PROFILE`注入seeds2022/2023；seed2022固定GPU0/1，seed2023固定GPU2，从而并行两个Weather而不在同一GPU叠加job。Wrapper在formal-test mode前逐项检查10个training、validation diagnostic与invariant artifacts。

`analyze_stage_c_iscf_bsca_confirmation.py`将seed2021 candidate/control与新增两个seeds合并为60-cell test surface，并分别计算seed、dataset、horizon方向、paired initialization、checkpoint hash、policy entropy、arm diversity和oracle headroom。它只读取冻结artifacts，不参与checkpoint或超参数选择。

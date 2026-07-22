# ISCF-BSCA-v1 code explanation

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

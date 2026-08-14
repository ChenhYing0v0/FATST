# ISCF-BSCA Core-Ablation 实现说明

## 1. 目的与边界

本实现不修改 Full `ISCF-BSCA-v1` 的 forward graph，而是在既有 SIFF/PCSD primitives 上开放冻结的 formal control 组合。唯一 source-level change 位于 training-argument validation：允许 `measure_only` 与 `equal`/`fixed` policy、以及 `equal_uniform_scope_anchor` 与 `equal` policy 的合法组合。

## 2. Forward tensor contract

所有变体首先由相同的 TimeAlign encoder 将输入 `batch_x [B,720,C]` 编码为 history memory，并经 SIFF readout 产生：

- `arm_forecasts [B,C,720,5]`：五个 sharing scopes `{1,48,144,360,720}` 的预测；
- `policy [B,C,720,5]`：每个 target coordinate 的 scope 权重；
- `outputs [B,720,C]`：按 `policy` 融合后的 prefix-consistent trajectory。

变体只改变下列路径：

- Full：independent scope projections + learned direct policy；
- `w/o BSCA`：forward 不变，但 loss 只使用 fused Uniform-Prefix Forecasting Loss；
- `w/o Target-Adaptive Allocation`：`policy` 固定为每个 scope 0.2，保留 equal scope-wise supervision；
- `Shared Scope Projection`：五个 scope 共享一个 Q=1 projection，并用 dataset-specific rank 扩宽以匹配 Full active parameters；
- `Fixed Scope (s=144)`：`policy` 为 scope index 2 的 one-hot，`outputs` 仅取 `s=144` arm，并只用 fused prefix loss。

## 3. Loss semantics

`projective_coupling_credit_loss` 已有两种所需 objective：

- `measure_only`：`total_loss = fused_loss`，`skill_loss = 0`，`route_loss = 0`；
- `equal_uniform_scope_anchor`：`total_loss = fused_loss + equal_scope_skill + scheduled_uniform_route_KL`。

在 equal policy 下，policy 本身等于 uniform target，因此 uniform route KL 恒为零；模型仍通过 fused loss 与 equal scope-wise skill end-to-end 更新。fixed policy 使用 `measure_only`，未被选择的 scope arms 不从 formal loss 接收梯度，这与单一固定 scope control 的理论角色一致。

## 4. Source validation change

`train_repo.py` 原先将所有 coupling objectives 限制为 learned policies。这会错误阻止冻结的 equal/fixed matched controls。新 validation 按 objective 建立允许集合：

- learned policy modes：所有既有 coupling objectives 保持原合同；
- `measure_only`：额外允许 `equal` 与 `fixed`；
- `equal_uniform_scope_anchor`：额外允许 `equal`；
- 其他未冻结组合继续报错。

因此，该 patch 只开放本 ablation 所需的已有 forward/loss path，不引入新 architecture、loss 或 H-conditioning。

## 5. Artifact chain

runner 将 training 与 formal test 分开。每个新 run 先生成 validation artifacts；checker 核对 effective config、initialization、finite invariants 与 20 个 unique checkpoint hashes，并写入 immutable manifest。formal-test runner 会在访问 test split 前重新计算 hashes。analyzer 最终合并 20 个 Full cells 与 80 个新 control cells，并输出逐 cell、逐 dataset、overall、control gates 和 checkpoint manifest。

## 6. Code-theory consistency

理论要求是用 matched end-to-end training 分离四个设计因素。代码通过相同 seed、encoder profile、optimizer、budget、validation selector 和 official-test evaluator实现这一要求；shared-projection control 另用 active-parameter gap 检查约束 capacity confound。

仍然只能将观察到的差异解释为冻结的 exact controls 下的 matched attribution，而不能证明每个 primitive 在所有架构中的普遍必要性。若 equal/fixed path 出现数值异常、未选 scope 意外收到梯度，或 shared projection 超出 parameter-match tolerance，则该 control 无效，必须修复实现后重跑，不能用其否定 BSCA 方向。

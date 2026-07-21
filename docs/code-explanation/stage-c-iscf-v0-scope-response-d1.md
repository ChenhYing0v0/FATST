# ISCF-v0 Scope-Response D1 代码说明

## 1. 功能边界

`scripts/diagnose_stage_c_iscf_v0_scope_response.py`只读取frozen ISCF-v0 checkpoint和validation histories。
它不修改checkpoint、不读取target values、不训练参数，也不产生paper-facing effectiveness metric。目的仅是把上一轮被
target difficulty混淆的residual correlation替换为label-free local operator response evidence。

## 2. Tensor flow

1. `batch_x [B,720,C]`经原checkpoint的`encode_history`得到`memory [B,C,P,D]`；
2. flatten为`hidden [B,C,R]`，顺序取32个channel rows形成`h [N=32,R]`；
3. 固定16个Rademacher directions `u [M=16,R]`，按每row hidden RMS设置
   $\delta=10^{-3}\operatorname{RMS}(h)u$；
4. 对`h+delta`和`h-delta`调用原`pcsd_readout.arm_forecasts`，得到
   `arms_plus/arms_minus [M*N,1,S=5,T=720]`；
5. central difference形成`response_bank [M,N,S,T]`；
6. 每个scope的response沿`[M,N,T]`展平、中心化并unit-RMS normalize，再计算common/private energy、pairwise
   cosine/distance和cross-seed topology。

## 3. Controls

- `direction_null`：每个scope独立置换16个direction identities，保留各自response marginal但破坏同步stimulus；
- `random_init_control`：复制完全相同的readout architecture并调用`reset_parameters()`，在相同trained hidden rows和
  directions上重复计算。它隔离scope pooling/synthesis结构本身可能造成的response相似性；
- validation-only：不用test inputs或labels，避免继续用formal-test surface选择问题定义。

## 4. Code-theory consistency

理论问题是“independent scopes是否学得了共享hidden perturbation下的stable response relation”。代码直接测量
$J_s(h)u$的central finite-difference proxy，因此比parameter similarity更接近local forecasting function，也不含target
residual。它仍然只是在frozen、co-adapted hidden representation上的conditional diagnostic；即使通过，也不能证明一个
relation-aware module会在end-to-end训练中提升性能。

会证伪当前问题的证据包括：同步response不超过direction null、learned checkpoint不超过matched random-init readout、
或pairwise topology跨seed不稳定。数值不有限、epsilon敏感或response退化则属于diagnostic design fault，不能拒绝方向。

CLI允许覆盖`hidden_rows`、`hidden_row_offset`、`directions`、`relative_epsilon`、`null_repetitions`和
`random_controls`。`hidden_row_offset`支持confirmatory protocol使用disjoint sequential validation rows；
`summary.json`保存`effective_probe`，避免robustness或confirmation结果被误写成primary gate。

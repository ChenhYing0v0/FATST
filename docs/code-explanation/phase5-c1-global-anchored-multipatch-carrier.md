# C1 Global-Anchored Multi-Patch Carrier 代码说明

## Scope

C1 只统一 history carrier 与 local-token interface，不是 StageB method。新增
`encoder_mode=global-anchored-patch-transformer`，official baseline、accepted legacy A6 与 exact HPM 路径
保持不变。

## Forward flow

输入 `x [B,720,C]` 先经 `normalization_x`。`GlobalAnchoredPatchEncoder` 接收 `[B,C,720]`：

1. `global_projection: Linear(720,D)` 产生 `[B,C,1,D]`；
2. valid `unfold(K,S)` 产生 `[B,C,P,K]`，不使用 right padding；
3. `local_projection: Linear(K,D)` 产生 `[B,C,P,D]`；
4. concat为 `[B*C,1+P,D]`，加 learned position；
5. 一个 pre-norm attention/FFN residual block混合 global/local tokens；
6. output LayerNorm后，token 0作为 forecast state `[B,C,D]`；
7. model把它表示为 memory `[B,C,1,D]`，flatten后经 `learned_basis_coeff: D->256`；
8. `learned_temporal_basis[:H]` 生成 `[B,H,C]` prediction。

`encode_retrieval_memory()` 返回 contextual local tokens `[B,C,P,D]`，供后续模块统一消费。Forecast head
不再 flatten所有 local tokens，因此 parameter count基本不随 `P` 增长。

## Dropout sites

新 Encoder没有复用 legacy `configs.dropout` 的单值语义。五个 sites独立：

- `history_token_dropout`：position后的全部 tokens；
- `history_attn_dropout`：softmax attention weights；
- `history_attn_residual_dropout`：attention output进入 residual前；
- `history_ffn_dropout`：GELU后、第二个 Linear前；
- `history_ffn_residual_dropout`：FFN output进入 residual前。

Small gate使用 `0.0/0.0/0.1/0.1/0.1`。`ResidualPatchAttention.output` 的内部 projection dropout固定为
0，避免 attention output被内部和外部重复 dropout。Legacy ETTm1 `0.9` 只保留在 legacy A6 reference，
不能解释为 C1 attention dropout。

## Scale contracts

- `P16-S8`：valid patch count `(720-16)//8+1=89`；
- `P48-S24`：valid patch count `(720-48)//24+1=29`。

两 scale使用相同 topology与权重类型。Analyzer先判断 shared scale，再允许根据 minimum validation MSE
选择 dataset-specific scale；test metrics不参与选择。

## Runner and analyzer

`scripts/remote/run_phase5_c1_global_anchored_multipatch_gate.sh` 运行：

```text
3 datasets x (A6 dual reference + P16-S8 + P48-S24) x seed2021 = 9 runs
```

`scripts/analyze_phase5_c1_global_anchored_multipatch_gate.py` 输出：

- `c1_comparisons.csv`：每个 dataset/horizon/selector相对 A6 与 fixed TimeAlign的 MSE/MAE；
- `c1_gate_summary.csv`：shared scale与 validation-selected scale的预注册 gate；
- `c1_validation_scale_selection.csv`：只由 training log minimum validation MSE决定的 scale；
- `c1_model_diagnostics.csv`：active parameters、local patch count与实际 dropout概率。
- `c1_protocol_audit.csv`：effective learning rate、source preset learning rate及是否一致；
- `c1_global_anchored_multipatch_deep_analysis.md`：training dynamics、scale stability、source-A6 comparison、
  parameter/state-width attribution与failure attribution；
- `c1_training_dynamics.csv`：best epoch、last/best validation gap、last train loss与best-vs-last test差异；
- `c1_capacity_state_width.csv`：active parameter变化与forecast-head实际读取的`patch_num*d_model` state width；
- `c1_scale_stability.csv`：validation选择和last/best test偏好的scale是否一致；
- `c1_vs_source_a6_summary.csv`：候选相对既有source-faithful A6 official-last的独立comparison。

`c1_capacity_state_width.csv`中的state width来自`model_diagnostics.json`的实际`patch_num*d_model`，不读取
`effective_config.official_args.patch_num`，因为legacy A6 model可能在初始化时把preset patch count改写为
实际Encoder输出patch count。

## Code-theory consistency

理论目标是同时保留 full-window global compression 与显式 local patch axis。代码中 global token确实直接
来自完整 720-step projection，local tokens确实通过 attention更新 global state，coefficient head只读取更新后
global token。

仍未证明的是 local tokens是否被训练后实际使用。只有 performance gate通过后，frozen local masking与
global-only same-backbone control才可验证这一点。若 C1性能通过但 local branch可被无损屏蔽，它只能作为 API
cleanup，不能支持 multi-patch representation claim。

## Returned code-theory evaluation

本轮performance gate失败。ETTh2/Weather的legacy A6 head分别读取`1536/6144`维flattened state，而C1只读取
`256`维global state；因此“避免state随P变化”的设计同时形成了readout bottleneck。ETTm1两者state width都为
`256`但C1仍退化，说明该bottleneck不是唯一原因。

Deep analyzer还检测到runner把ETTh2 learning rate从source preset `5e-4`覆盖成`1e-4`。主gate中的
same-run ETTh2 comparison仍是matched control，但不是source-faithful reproduction；所以最终裁决同时读取既有
source-faithful A6 artifact。两个C1 scales相对该reference均为`0/12` wins，失败结论不依赖该protocol mismatch。

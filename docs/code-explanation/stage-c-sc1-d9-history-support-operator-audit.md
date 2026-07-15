# SC1-D9 History-Support Operator Audit Code Explanation

## Scope

本次只新增`diagnostic_only` analyzer与remote wrapper，不修改model forward、training objective或checkpoint。
完整预注册协议见
`analysis/stage_c_sc1_d9_history_support_operator_audit_20260715/d9_diagnostic_design.md`。

## Artifact Flow

1. analyzer按frozen five-dataset contract定位15个natural A6 checkpoint；
2. 从`effective_config.json`核对`patch_num/d_model/d_ff/pred_len/basis_rank/readout_mode`；
3. 从`checkpoint.pt`读取`learned_temporal_basis [720,256]`与
   `learned_basis_coeff.weight [256,PD]`；
4. 精确合成`operator [720,PD]`并reshape为`[720,P,D]`；
5. future axis乘RGNB transpose，history patch axis乘DCT或random orthogonal rows；
6. 聚合atom energy、scale correlation与matched controls，写出CSV、gate JSON与中文decision report。

## Statistic Sources

- `unit_metrics.csv`：每个dataset/checkpoint的exact operator统计；
- `group_profiles.csv`：七个future support groups的history normalized-frequency centroid；
- `control_distributions.csv`：1024次atom-label permutation与64次random-history-basis的`scale_rho`；
- `dataset_metrics.csv`：先对三seed聚合，再计算dataset-level empirical controls；
- `gate.json`：严格执行config中预注册的六项primary gates；
- `research_interpretation.md`：面向研究决策的简表与授权边界。

## Code-Theory Consistency

理论对象是A6实际memory-to-future linear map$W=BC$。代码直接相乘checkpoint tensors，因此不会把learned
basis内部rotation误写成机制。RGNB与DCT均为orthogonal coordinate transforms；`parseval_relative_gap`验证
能量守恒。atom-label permutation只破坏future support metadata，random orthogonal control只破坏ordered
history-scale coordinate，两者分别对应预注册的两类替代解释。

仍属于proxy的部分：memory patch-axis frequency不等于raw-input local sensitivity。只有D9-A通过后，D9-B才会
以sample-dependent JVP/Jacobian确认Encoder与normalization后的真实input sensitivity。D9-A结果不能单独支持
新decoder或论文method claim。

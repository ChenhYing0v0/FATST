# CCSF Step9深度审计代码说明

## 1. Formal analyzer artifacts

`scripts/analyze_stage_c_siff_ccsf_tau25_phase_a.py`读取50个run的official-test scorecard，执行Step6冻结的10项hard
comparisons、一个interaction term和internal health gate。输出`comparison_cells.csv`、
`comparison_summary.csv`、`mechanism_health.csv`及`summary.json`。

## 2. Post-E2E diagnostic

`scripts/analyze_stage_c_siff_ccsf_post_e2e.py`读取full CCSF的五个diagnostic NPZ：

- `probe_arms [row,S,T]`与`probe_targets [row,T]`构造region arm MSE；
- `probe_base_logits [row,T,S]`提供v1 policy information；
- `probe_contrast_descriptor [row,T,S,6]`提供production target-free contrast；
- `probe_base_policy/probe_policy [row,T,S]`比较learned correction前后。

每dataset按row前后各一半做two-fold cross-fit。feature只从arms/logits/coordinates构造，fit fold的future labels只用于
训练probe，test fold只用于离线评分。输出expected arm MSE、actual mixture MSE、best-arm accuracy、final/base
policy对比及correction-skill alignment。该脚本是test-derived diagnostic，不能进入training path。

## 3. Validation/test deep audit

`scripts/analyze_stage_c_siff_ccsf_step9_deep.py`读取50个run的轻量metadata，重新构造validation与test的同一comparison
matrix，输出：

- `validation_test_comparison_summary.csv`：split-specific macro/cell/dataset/horizon wins；
- `validation_test_direction_audit.csv`：validation/test方向是否一致；
- `arm_macro_score.csv`：每个arm的20-cell raw MSE/MAE平均；
- `checkpoint_training_audit.csv`：best epoch、early stopping及CCSF teacher/policy training diagnostics。

## 4. Statistic boundary

`expected_arm_mse`是$\sum_s p_s e_s$，只评价allocation；`mixture_mse`先融合forecast再计算误差，包含不同arms残差间
cross terms。二者不能互换。`best_arm_accuracy`只比较argmax，也不能替代soft allocation或fused MSE。

## 5. D2 granularity diagnostic

`scripts/analyze_stage_c_siff_ccsf_d2_granularity.py`把完整720-domain按width
`{1,48,144,360,720}`切成固定连续regions。每个region从`probe_arms`和`probe_targets`计算
`arm_losses [row,region,S]`，从`probe_base_logits`与`probe_contrast_descriptor`构造target-free features。
two-fold classifier只在train rows读取winner labels，在held-out rows同时评价expected arm MSE与actual mixture MSE。

## 6. D3 mixture-risk decomposition

`scripts/analyze_stage_c_siff_ccsf_d3_mixture_risk.py`读取：

- `probe_arms [N,S,T]`；
- `probe_targets [N,T]`；
- `probe_base_policy/probe_policy [N,T,S]`。

对width大于1的region，先构造residual Gram matrix
$Q=E^\top E/|R| [M,S,S]$，再枚举五个arms的31个non-empty active sets，求解simplex-constrained quadratic
minimum。候选weights由加微小ridge后的linear solve获得，但loss在原始$Q$上计算；width1使用residual是否跨0的
解析解。输出CSV中：

- `best_single_arm_oracle_mse`：region内最小single-arm MSE；
- `simplex_mixture_oracle_mse`：non-negative sum-to-one oracle mixture MSE；
- `cross_term_share_of_uniform_to_simplex_gap`：best-arm到simplex的增量占uniform到simplex总空间的比例；
- `zero_simplex_loss_fraction`：仅width1可能因正负residual精确互消而为正。

这些量均使用target labels，只解释objective geometry。

## 7. D4 readout-sharpness diagnostic

`scripts/analyze_stage_c_siff_ccsf_d4_readout_sharpness.py`重用D2的two-fold classifier。对每份held-out probability
$p$计算$p^\alpha/\sum p^\alpha$，$\alpha\in\{0.5,1,2,4,8\}$，并同时计算hard argmax。所有exponents均为
simultaneous diagnostic arms，输出`gain_over_exponent_1_pct`与同exponent下true-vs-shuffled contrast gain。
该脚本不能选择production temperature。

## 8. Code-theory consistency

Intended diagnostic theory是：若production contrast仍能在held-out rows预测region competence，但learned correction
与skill反向，则failure应归因于policy/readout/objective，而不是contrast information absence。代码实现了这一
conditional test。D2消除了benchmark-bin依赖，D3排除了residual cross terms为主要矛盾，D4排除了简单
temperature/sharpness修复。三者仍全部使用test-derived probe rows，因此最终只支持关闭exact contrast-policy route
并回滚Step2/4，不支持替代方法promotion。

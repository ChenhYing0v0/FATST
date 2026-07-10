# C0 Frozen Cross-Patch Interaction Diagnostic 代码说明

## 目的与边界

`scripts/analyze_phase5_stage_b_c0_cross_patch_interaction.py` 检查 frozen A6 ETTm1 Encoder 是否包含跨
时间 patch 的非加性交互。脚本不训练模型、不改 checkpoint，也不提出 StageB 新机制。

## Forward data path

1. 从 clean A6 run 的 `effective_config.json` 重建 `TimeAlign.Model`，strict load `checkpoint.pt`；
2. test input `batch_x [B,720,C]` 经 `normalization_x` 得到 normalized history；
3. `attenuate_patches` 在原始时间轴上对 5 个 144-step regions 做可逆的 scale intervention；
4. `_encode_normalized_history` 输出 `memory [B,C,1,256]`，flatten 后经
   `learned_basis_coeff` 得到 `coeff [B,C,256]`；
5. `full/single/pair` coefficients 构成 inclusion-exclusion interaction `I [B,C,256]`；
6. 对每个 horizon，`learned_temporal_basis[:H] [H,256]` 将 delta 投影为 `[B,C,H]`，再计算 RMS ratio。

## Output columns

`cross_patch_interaction_pairs.csv`：

- `first_patch/second_patch`：0-based canonical patch indices；
- `interaction_to_main_mean/median/q25/q75`：sample-level interaction/main-effect RMS ratio 的统计；
- `main_effect_rms_mean`：两个 single-patch projected main-effect RMS 的均值；
- `examples`：参与统计的 test examples 数。

`cross_patch_interaction_summary.csv`：

- `pair_median_mean/min/max`：10 个 pair-level median ratios 的跨 pair 统计；
- `pairs_ge_0_05`：median ratio 至少 0.05 的 pairs 数；
- `material_interaction`：mean 至少 0.05 且至少 75% pairs 达标。

## Code-theory consistency

理论目标是判断 frozen nonlinear Encoder 对两个时间区域的联合响应是否可由各自响应相加。代码用标准
inclusion-exclusion finite intervention 实现这一点，并在实际 forecast basis 上度量影响。

仍是 proxy 的部分：attenuation 不是自然数据生成干预；ratio 不表示 interaction 对 test MSE 的因果贡献。
若不同 attenuation 下 ratio 不稳定、只集中于少数 pairs，或更换 batches 后消失，则会 falsify 当前
“稳定 material interaction”的解释。

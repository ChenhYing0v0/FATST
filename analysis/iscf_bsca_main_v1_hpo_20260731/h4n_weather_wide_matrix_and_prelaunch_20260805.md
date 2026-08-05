# H4N Weather-only Wide HPO Matrix and Prelaunch Gate

日期：2026-08-05

当前步骤：Step 6 Weather HPO redesign 完成，进入 Step 7 local protocol gate；通过
focused commit、remote preflight 和 resource smoke 后进入 Step 8 remote training。

用户授权：重点优化 Weather，扩大超参数调整范围，尽量达到该数据集的最优性能。
本轮将该指令解释为对新的、非重复 H4N Weather-only HPO contract 的 local patch、
remote smoke、remote training，以及 40/40 checkpoint manifest 完成后的一次完整
official-test audit 授权；不扩展到其他数据集、seed2022/2023、H4O 或架构/目标函数重设计。

## 1. 已有证据与剩余缺口

H1--H4M 共 189 个 dataset-level trials，其中 Weather 56 个。所有历史 profile 均由
validation four-H mean MSE 选择 checkpoint，并在完整 official-test scorecard 中保留。
当前两个互补 frontier 为：

| Role | Trial | Four-H mean MSE | Four-H mean MAE | Leads vs full displayed baseline table |
| --- | --- | ---: | ---: | ---: |
| MSE frontier / current selector | `Weather__h4m_seq640_p20` | 0.214752 | 0.247324 | 4/8 |
| MAE and joint frontier | `Weather__h4m_seq512_p16_lr2e5` | 0.215595 | 0.246084 | 2/8 |

相对本地 official-native TimeAlign Weather reproduction，当前 selector 的 mean MSE
领先约 0.49%，mean MAE 落后约 1.06%；历史 MAE frontier 的 mean MAE 缺口已缩小到
约 0.56%。逐 horizon 看，MSE 已在 H192/H336/H720 领先，MAE 仅 H720 领先；主要
缺口是 H96--H336 MAE，尤其 H96。

此前影响排序与 H4M 返回结果一致：

1. `seq_len × patch geometry` 是最大影响轴；constant patch length 32 下
   `L512` 更利于 MAE，`L640` 更利于 MSE，`L768` 显著退化，说明响应非单调；
2. `learning_rate` 在 `L512/p16` 上显著影响 MAE，`2e-5` 优于已试的 `1e-5/5e-5`；
3. `patch_num` 和 `mode_rank` 有次级影响，但此前没有在两个 frontier 上完成 matched
   coverage；
4. width/capacity 的历史单因素影响较小，但未在 `L512/p16/lr2e-5` frontier 上测试；
5. `weight_decay`、`dropout` 和 `layer_norm` 的 matched span 很小，本轮不继续消耗预算。

## 2. Frozen H4N matrix

H4N 只包含 `Weather × seed2021` 共 40 个 from-scratch joint-training profiles，且
与 189 个历史 effective fingerprints 零重复：

| Block | Runs | Frozen coverage | Purpose |
| --- | ---: | --- | --- |
| context × LR interpolation | 16 | `L={448,480,544,576,608,640,672,704}`，patch length=32，`lr={2e-5,4e-5}`为主并在L640补`3e-5` | 填补 L512--L640 两个 frontier 之间及其两侧的大空白 |
| LR wide boundary | 8 | `L512/p16`，`lr={5e-6,1.5e-5,2.5e-5,3e-5,4e-5,7.5e-5,1e-4,2e-4}` | 同时覆盖低 LR 插值和此前未联合测试的外边界 |
| patch geometry | 8 | `L512,p={4,8,32,64},lr2e-5`；`L640,p={8,10,16,32},lr4e-5` | 在 MAE/MSE frontier 上隔离 patch count / patch length |
| mode rank | 5 | `L512/p16/lr2e-5,r={80,96,128,160,192}` | 填补历史 `64 -> 116 -> 256` 周围的未覆盖区间 |
| encoder capacity | 3 | `32/64/r64`、`96/192/r128`、`128/256/r128` | 只在最强 MAE context 上测试 width interaction |

Architecture invariants、scales、partition、policy/objective mode 与 inference graph 均
不变。本轮是 frozen architecture family 内的 HPO，不支持新的 mechanism claim。

## 3. Selection and test boundary

- 每个 trial 仍由 validation `{96,192,336,720}` mean MSE 选择 epoch/checkpoint；
- training 阶段 `official_test=0`，不允许用 test 选择 epoch；
- 40/40 checkpoint、hash、metrics、effective config、logs 与 provenance 完整后，才允许
  一次完整 formal test；
- 一个 Weather profile 共同服务四个 horizons；禁止 per-H、per-metric、per-seed 或
  per-cell 选择；
- H4N primary dataset score 是相对 frozen full-table baseline targets 的 four-H
  MSE/MAE equal-weight relative mean；
- primary score 差异不超过 0.1% 才视为 near tie，并依次使用 lead count、MSE/MAE
  balance、validation score、parameter count 与 lexical ID 打破；
- 所有 40 个 negative trials 与 160 个 MSE/MAE cells 必须保留。

这将用户要求的“每个数据集 MSE/MAE 平均最优”置于 lead-cell tie-break 之前，同时
保留足够多领先 cells 的 secondary objective。

## 4. Success/failure gates

Full pass 同时要求：

1. Weather primary joint score 相对历史 joint frontier 至少改善 0.3%；
2. four-H mean MSE 不高于 0.214752；
3. four-H mean MAE 不高于本地 TimeAlign reproduction mean 0.244725；
4. 相对当前 12-baseline displayed comparison surface 至少 6/8 MSE+MAE cells 领先。

若 primary score 有实质改善但未同时达到 2--4，记为
`weather_performance_partial_pass`；若 40 个 profiles 均未改善 primary score，则记为
`expanded_hpo_performance_shortfall`，回滚到 Main I/II completion，而不是自动启动
H4O。任何 architecture/objective 修改都必须创建新的 test-informed candidate，并回到
Step 4--6。

## 5. Budget, storage and schedule

所有 Weather jobs 统一扩展到 120 epochs / patience 24。原因是历史低 LR frontier 的
best epoch 达到 49--65；统一预算避免只给看似有利的 profile 额外训练。

- estimated training：45--75 GPU-hours；
- three-3090 wall time：约 15--26 hours；
- training storage：4--6 GiB；future formal test：1--2 GiB；
- 2026-08-05 preflight：GPU0/1/2 均为 18 MiB used、0% utilization；remote quota
  169G/200G soft/220G hard；H4L/H4M 分别约 1.7G/2.4G；无需为本轮删除 formal artifacts。

40 jobs 使用 dynamic three-GPU queue，先运行完整 resource smoke。Smoke 只验证
execution/memory，不访问 test，也不构成性能证据。

## 6. Local gate

Machine contract：`configs/iscf_bsca_main_v1_hpo_weather_h4n.json`。

Local checker 必须证明：JSON/Python/shell parse 通过；40/40 trial IDs 和 fingerprints
内部唯一；与 189 个历史 profiles 零重复；五个 blocks 数量为 16/8/8/5/3；所有
`seq_len % patch_num == 0`；effective batch=32；training dry-run 明确输出
`jobs=40 test_jobs=0 remote_authorized=true`。

Decision=`H4N_Weather_40_profile_wide_high_impact_matrix_frozen_remote_training_and_post_manifest_complete_test_authorized`。

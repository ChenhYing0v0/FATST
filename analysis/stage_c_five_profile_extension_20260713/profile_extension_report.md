# StageC ETTh1/ETTm2 Natural Profile Extension Report

## Decision Summary

| Field | Value |
| --- | --- |
| `role` | `mechanism_control_profile_extension` |
| `split` | validation only |
| `remote_runs` | 14/14 complete |
| `selection_uses_test` | `false` |
| `selection_uses_parameter_count` | `false` |
| `stability_gate` | `pass` |
| `decision` | `five_dataset_profile_contract_frozen` |

## Selected Profiles

| Dataset | Profile | patch_num | d_model | d_ff | Mean dense MSE CV | Max dense MSE CV | 3-seed H720 val MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | `r2b_p24_d64_ff128_medium` | 24 | 64 | 128 | 0.8087% | 1.1889% | 1.414194 |
| ETTm2 | `r2b_p48_d64_ff128_medium` | 48 | 64 | 128 | 1.1807% | 2.7958% | 0.279620 |

两套profile均通过预注册的mean CV不高于3%、max CV不高于5% gate。ETTh1在patch和width两阶段均为
8/8 horizons的最低regret。ETTm2的P24或narrow/wide在H720可略优，但它们在短中horizons退化，故
dense-horizon macro选择P48/D64/FF128；这与unified varied-horizon control目标一致。

## Protocol Audit

Phase A固定D64/FF128并比较P12/P24/P48；Phase B固定selected patch并比较narrow/medium/wide；Phase C只为
selected profile补seeds2022/2023。所有14个run的effective config、profile hash、validation split、finite
training log和8-horizon completeness均通过。本地同步后独立重算得到与远端相同的A/B/C decisions。

active parameter count只报告：ETTh1为613,904，ETTm2为1,006,160。二者没有进入selection/tie-break或
stability gate。由此产生的frozen memory width分别为1536与3072；D2 formal5必须适配实际width，而不能为
保持旧768维而反向修改natural profiles。

## Failure Attribution

首次remote launch在argument parsing阶段退出，因为`train_repo.py`的dataset registry尚未暴露ETTh1；
`data_provider`和dataset文件本身均已支持。修复采用vendored upstream `scripts/ETTh1.sh`的routing defaults，
1-batch ETTh1 remote smoke完成train/val/checkpoint/evaluation后才重启矩阵。因此该事件归类为
`protocol_adapter_missing_repaired`，没有进入GPU training，不能视为architecture、optimization或hypothesis
failure。

## Decision And Boundary

[Decision] 新contract为`configs/stage_c_five_dataset_natural_profiles.json`。后续五dataset的SC1-D2与任何
paper-core mechanism/matched controls必须复用这些profiles，不得为D2或新candidate重新选择。

[Boundary] 本实验只冻结control，不提供Contribution 1的正向机制证据。下一步仍是SC1-D2 formal5：若
true-scale grouping不能同时超过dense、random-group与random-basis gates，回到Step 2重定义问题。

## Artifacts

- `phase_a/phase_a_patch_selection.csv`：patch selection regrets；
- `phase_b/phase_b_width_selection.csv`：width selection regrets；
- `phase_c/phase_c_stability.csv`：逐dataset、逐horizon 3-seed CV；
- `raw/`：gitignored effective configs、training logs与validation metrics；
- remote roots：`/home/yingch/exp_outputs/r-2026-fatst/stage_c_five_profile_extension_{a,b,c}`。

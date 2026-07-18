# SC1-SIFF-v2-EQ-ATTR Step 7B Prelaunch

## 1. Gate summary

| Field | Content |
| --- | --- |
| `candidate_version` | `SC1-SIFF-v2-EQ-ATTR-v1` |
| `current_step` | Step 7B prelaunch passed |
| `user_authorization` | 2026-07-18用户要求继续推进并进行远程实验 |
| `phase_a` | 10 arms × 5 datasets × seed2021 = 50 runs / 200 test cells |
| `checkpoint_rule` | validation mean MSE over H96/H192/H336/H720 |
| `test_role` | primary mechanism-effectiveness and paper benchmark |
| `test_informed` | true |
| `confirmation` | seeds2022/2023 held |
| `local_gate` | 9/9 categories pass |
| `decision` | remote dry-run、resource smoke与Phase-A launch authorized |

## 2. Frozen authorization

该candidate version的正式访问边界为：

- `remote_training_authorized=true`；
- `formal_test_access_authorized=true`；
- `formal_test_access_count_for_version=1`；
- checkpoint允许按冻结four-horizon validation selector从头训练；
- test不得选择epoch、修改checkpoint或触发dataset/horizon/cell tuning；
- 只有seed2021 Phase A获授权；
- `confirmation_authorized=false`。

所有50个arms共享profile、seed、training schedule、checkpoint selector与official test scorecard。七项hard
comparisons必须逐项报告，不能筛选有利cell。

## 3. Machine prelaunch gate

`check_stage_c_siff_equal_attribution_step7b.py`验证：

1. candidate identity与`test_informed`；
2. Step7A 13/13 committed evidence；
3. five-dataset profile hash；
4. 50-job/10-arm/seed2021 manifest；
5. 200-cell H96/H192/H336/H720 MSE/MAE matrix；
6. formal user/test/checkpoint authorization；
7. checkpoint evaluator会接受且只接受该冻结config；
8. confirmation仍关闭；
9. remote runner executable与bash syntax。

结果=`9/9 pass`。

## 4. Read-only remote preflight

2026-07-18 launch前只读检查：

- host：`star3090.iai.zju.edu.cn`；
- repo：`/home/yingch/projects/FATST`；
- conda：`moe`；
- torch：`2.9.0+cu128`，CUDA available；
- dataset root：`/home/yingch/dataset` exists；
- GPU 0/1/2：RTX 3090，均约15 MiB used、24110 MiB free、0% utilization；
- 未发现`train_repo.py`或本candidate的运行进程。

远程worktree存在三份历史analysis CSV修改：

- `analysis/stage_c_d14a1_dual_carrier_grouped_mlp_20260715/local_gate/parameter_and_theory_cases.csv`；
- `analysis/stage_c_pcsd_cf_step7b_prelaunch_20260716/arm_matrix_checks.csv`；
- `analysis/stage_c_sc2_pcc_step7b_prelaunch_20260717/initialization_pairing.csv`。

它们与本次提交路径不重叠。远程更新只允许`git pull --ff-only`，不得清理、stash或覆盖这些用户/历史改动。

## 5. Launch order

1. 提交并推送本prelaunch freeze；
2. remote `git pull --ff-only`；
3. 再次读取`nvidia-smi`；
4. external output root下执行50-job dry-run；
5. 在Weather + SIFF_EQUAL上执行1-batch resource smoke；
6. smoke artifacts通过后，在GPU 0/1/2后台启动seed2021 Phase A；
7. 仅确认driver与首批jobs启动，不进行高频值守。

output root固定为
`/home/yingch/exp_outputs/r-2026-fatst/stage_c_siff_equal_attribution_v2`。

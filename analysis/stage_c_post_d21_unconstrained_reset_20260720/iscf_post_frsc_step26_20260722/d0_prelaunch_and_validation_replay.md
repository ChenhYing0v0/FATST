# ISCF-SCC D0 Prelaunch and Validation Replay

## 1. Decision record

| Field | Record |
| --- | --- |
| `current_step` | D0 artifact audit complete；validation-only frozen replay prelaunch passed |
| `problem` | historical ISCF NPZ是否包含closed-form coalition credit所需的exact direct policy |
| `existence_evidence` | 15个source checkpoints与test NPZ完整；historical function audit只有bin-level policy usage |
| `idea` | 不反演欠定policy；用same frozen checkpoints在validation split replay，保存exact `probe_direct_policy` |
| `theory_check` | frozen replay不改变checkpoint，不训练，不读取new test；只能建立diagnostic problem evidence |
| `design` | 5 datasets × 3 seeds=15 validation replays；256 probe rows；three-GPU forward-only evaluation |
| `narrative_gate` | unchanged=`conditional_pass_to_d0_only` |
| `effectiveness_gate` | not applicable；no method candidate trained |
| `artifacts` | D0 config、analyzer、remote runner、code explanation、local smokes |
| `decision` | `d0_validation_replay_prelaunch_pass_remote_forward_authorized` |

## 2. Historical artifact gap

[Fact] remote source roots完整包含15个`pcsd_test_audit_diagnostics.npz`，总大小约`634.54 MiB`：

- seed2021 root：5 files，`221,488,174` bytes；
- seed2022/2023 root：10 files，`443,877,211` bytes。

逐key检查确认历史NPZ包含`probe_arms [256,5,720]`、`probe_fused [256,720]`、
`probe_targets [256,720]`和`policy_row_bin_usage [N,8,5]`，但没有
`probe_direct_policy [256,720,5]`。

[Decision] 单个coordinate只有一个fused equation和五个unknown policy weights；仅凭arms/fused无法唯一恢复policy。
任何least-squares或bin-average substitution都会改变counterfactual，故禁止用反演结果推进SCC。

## 3. Frozen validation replay

同一15个source checkpoints只做validation forward replay。输出写入：

```text
/home/yingch/exp_outputs/r-2026-fatst/stage_c_iscf_scc_d0_validation_replay
```

每个run必须产生：

- `pcsd_validation_diagnostics.npz`；
- `trained_invariants.json`；
- source `checkpoint.pt` replay前后SHA256一致；
- `evaluation_split=val`、`uses_test_split=false`；
- exact direct policy、arms、fused、targets均finite。

runner不包含train command或optimizer；配置硬冻结`new_training_authorized=false`与
`formal_test_access_authorized=false`。若传入test split没有入口，不能访问official test loader。

## 4. D0 statistics and controls

analyzer使用parent-matched L1 risk，计算：

1. fusion reconstruction max gap；
2. renormalized leave-one-scope-out $\Delta_s$；
3. positive contributor count；
4. policy-credit和standalone-credit Spearman；
5. standalone-best/coalition-best match；
6. target-visible coalition reweighting headroom；
7. eight-bin × five-scope credit topology与three-seed stability；
8. uniform、standalone-error和32个deterministic non-identity scope shuffles。

详细定义与gate以
`step2_6_innovation_portfolio_and_scc_gate.md`第7节和
`configs/stage_c_iscf_scc_d0.json`为准。

## 5. Preflight and authorization

- user authorization：`继续按计划推进工作`；只解释为推进已冻结D0，不扩展到method training/test；
- remote project：`/home/yingch/projects/FATST`；
- source commit before code update：`2fadafc565529c603e7cfc0f506818699a4732a0`；
- GPU preflight：GPU0/1/2均`18 MiB / 24576 MiB`、`0%` utilization；
- local verification：analyzer `py_compile`、synthetic smoke、checkpoint evaluator synthetic smoke、JSON parse、
  `bash -n`和`git diff --check`均通过。

remote replay必须在本次code commit/push后由remote fast-forward启动。remote analysis只同步CSV/JSON/log summary，
不提交大体积NPZ。

## 6. Authorization boundary

```text
active_method = none
validation_replay_authorized = true
new_training_authorized = false
formal_test_access_authorized = false
method_implementation_authorized = false
next_action = commit_push_remote_fast_forward_and_run_15_validation_replays
```

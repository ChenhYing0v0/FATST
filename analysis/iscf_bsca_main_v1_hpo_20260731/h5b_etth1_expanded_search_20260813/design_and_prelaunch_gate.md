# H5B ETTh1 Expanded HPO Design and Prelaunch Gate

## 1. Current step and authorization

- `current_step`: Step 6 frozen design -> Step 8 remote resource gate
- `candidate_version`: `ISCF-BSCA-MAIN-v1-etth1-h5b-test-informed-20260813`
- `user_authorization`: 2026-08-13，继续ETTh1 HPO、扩大参数范围，并充分使用三张远程GPU。
- `architecture_search`: false；encoder、ISCF/BSCA decoder、objective、scope set与H720-prefix inference graph保持不变。
- `formal_test_boundary`: training阶段test=0；只有36/36 checkpoint及artifact manifest完整后，才可执行一次完整`36 × 4 H = 144` row formal test，禁止partial selection。

## 2. Prior-result audit

H5A与历史池共审计25个ETTh1 profiles。当前profile为`ETTh1__h5a_lr3p5e4`，four-H mean MSE/MAE=`0.392803/0.419707`，Main II为`2/8` best cells。主要经验如下。

| Factor | Observed evidence | H5B consequence |
| --- | --- | --- |
| learning rate | `3.5e-4`取得最低joint mean MSE；`4e-4`的MAE略低但MSE cells减少；`2.5e-4`更弱，历史`5e-4`明显退化 | 重点搜索`3.2e-4`到`4.2e-4`的细网格 |
| context × patch | `L336`可改善MAE但mean MSE恶化约1.9%；`L512`整体偏弱；`L720,p24`仍最稳 | 扩到`L=576–960`并联合不同patch，搜索未覆盖的长context区域 |
| dropout / weight decay | dropout `0–0.15`和`wd=1e-3`均处在近似平台 | 在最佳LR上搜索更细regularization interactions |
| mode rank | `rank64`明显弱于当前`109`，更高rank未充分覆盖 | 搜索`80/96/128/160`，同时保留109 anchor |
| capacity / normalization | `d16/ff16`、`d32/ff64`、historical大容量与`layer_norm=0`均负向 | 只保留一个`d48/ff48`温和probe，LayerNorm固定开启 |
| training budget | H5A budget90未改变早期best epoch，但更宽context可能需要更长优化 | 全部新trial统一`max_epochs=120, patience=24`，仍由validation selector决定checkpoint |

Self-critique：ETTh1当前与external best的差距多为三位小数边界附近，新搜索可能增加rounded best ties，但不保证full-precision SOTA。为避免表格导向过拟合，selector仍固定为dataset-level shared profile，并对mean MSE与MAE同时设置0.3%退化guard。

## 3. Frozen matrix

- 36个from-scratch、seed2021、ETTh1-only trials；不是Cartesian product。
- 9个LR/budget profiles，5个dropout profiles，5个weight-decay profiles，12个context/patch profiles，4个rank profiles，1个moderate-capacity profile。
- `seq_len`范围从历史`336/512/720`扩展为`576/640/672/720/768/840/960`；全部满足`seq_len % patch_num == 0`。
- effective batch固定32；`layer_norm=1`；完整negative trials全部保留。
- machine-readable contract：`configs/iscf_bsca_main_v1_hpo_etth1_h5b.json`。

## 4. Selection, success and rollback

1. 每trial用validation mean MSE over `{96,192,336,720}`选择唯一checkpoint。
2. 36/36 manifest后，用official test完整评估所有profiles；一个profile同时服务四个horizons。
3. Primary rank为Main II ETTh1 best metric cells；Main I best cells、top-2、mean MSE/MAE依次作为tie-break。
4. Eligibility要求mean MSE和mean MAE均不超过H5A当前profile的`1.003×`。
5. Minimum success=`3/8` Main II best cells；stretch=`4/8`。若无eligible improvement，保留`h5a_lr3p5e4`并关闭H5B，不修改论文表。
6. OOM或numeric pathology只阻断对应runtime profile；architecture/objective变更必须创建新candidate并回到Step 4–6。

## 5. Resource and scheduling gate

- 预计`24–54 GPU-hours`，三GPU动态队列约`8–18 wall-hours`。
- provisional LPT order先放`L960/L840/L768`、moderate capacity与high-rank任务；三个workers从共享队列取任务，避免静态分组导致空闲。
- remote output=`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h5b`，storage budget=`12 GiB`。
- 启动前必须依次通过：远程磁盘审计与安全清理、`nvidia-smi`、exact commit pull、36/36 resource smoke、test=0检查，然后才可后台启动full train/validation queue。

## 6. Local verification

- JSON parse：pass
- touched Python `py_compile`：pass
- frozen-contract checker：pass，`jobs=36`、`seq_len_range=[576,960]`、`test_jobs=0`、three-GPU workers=3
- Decision：`H5B_frozen_authorized_remote_resource_gate_next`

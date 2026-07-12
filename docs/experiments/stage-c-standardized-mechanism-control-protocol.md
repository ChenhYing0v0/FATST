# StageC Standardized Mechanism-Control Protocol

## 1. Purpose

本 protocol 为 StageC 的所有 mechanism diagnostics、ablations 与 small gates 建立统一、冻结、可审计的
research carrier。它解决的是 attribution validity，而不是复现 TimeAlign 的最佳结果。

当前 `train_repo.py` 的 unified mode读取各 dataset 的 TimeAlign H720 preset：

| Dataset | `patch_num` | `d_model` | `d_ff` | LR | dropout | layer norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETTh2 | 48 | 32 | 32 | `5e-4` | 0.1 | 1 |
| ETTm1 | 1 | 256 | 256 | `1e-4` | 0.9 | 1 |
| Weather | 48 | 128 | 128 | `1e-4` | 0.5 | 0 |

[Fact] 这些配置适合 source-faithful reproduction，但同时改变 patch topology、hidden width、active
capacity、regularization、optimization 与 normalization。若后续每个 mechanism 都继承这些 presets，失败
可能只是破坏了某个 dataset 的 finetuned operating point，正向结果也可能只在特调容量下成立。

[Decision] StageC 建立独立 standardized mechanism-control protocol。TimeAlign presets 继续保留用于
source baseline，不再决定 StageC mechanism runner。

## 2. Protocol Classes

| Protocol | Purpose | Hyperparameter Rule | Allowed Claim |
| --- | --- | --- | --- |
| `source` | 复现 TimeAlign 与历史 A6 | 完整保留 upstream per-dataset/per-horizon presets | source-faithful performance only |
| `mechanism_control` | 因果归因、消融、候选 small gate | dataset-aware frozen mapping；同dataset的method/control共用profile；禁止 `patch_num=1` | mechanism evidence under controlled carrier |
| `native_external` | 复现外部 baseline | 在其官方 repository 使用原生 protocol | external baseline reproduction only |

不同 protocol 的结果必须分表报告，不能混成同一 matched comparison。

## 3. Standardization Principle

“统一”定义为：

- 同一dataset内保持相同computation topology与patch policy；
- dataset可从预注册小grid选择结构profile，但所有profiles保持相同state width与近似active parameter budget；
- 同一 optimizer、LR schedule、effective batch、epoch budget、loss 与 checkpoint selector；
- 同一 global selection rule；
- dataset 只改变无法避免的数据 schema。

“统一”不等于把 upstream 某个 dataset 的 preset复制给所有 dataset，也不等于看到 test 结果后选择每个
dataset 的最佳配置。

## 4. Allowed Dataset-Specific Fields

StageC mechanism-control runner允许下列dataset-specific fields：

- file path、dataset loader、sampling frequency；
- `enc_in/dec_in/c_out` 与 channel identity；
- normalization statistics；
- physical micro-batch size，但必须用 gradient accumulation保持相同 effective batch；
- worker/GPU 等不改变 optimization semantics 的 runtime setting。
- 从预注册三臂grid一次性选择的`patch_num/patch_len/d_model/d_ff`。

禁止 dataset-specific 修改：

- grid之外的`patch_num/d_model/d_ff`，或为不同mechanism重新选择这些字段；
- layers、basis rank；
- dropout；
- learning rate、optimizer、schedule、epoch count；
- prediction loss 与 horizon distribution；
- checkpoint selector；
- mechanism-specific rank/width/bank count。

## 5. SC0 Calibration Design

### 5.1 Frozen common settings

| Field | SC0 Value |
| --- | --- |
| datasets | `ETTh2, ETTm1, Weather` |
| `seq_len/pred_len` | `720/720` |
| encoder | clean TimeAlign-derived token MLP, two layers |
| readout | `learned-basis-forecast-operator` |
| basis rank | 256 |
| reconstruction/alignment | disabled；`w_recon=w_align=0` |
| training objective | neutral full-720 pointwise L1；不使用 `multi-prefix` |
| validation selector | lowest full-720 validation MSE |
| primary checkpoint | `best-val-full720` |
| sensitivity checkpoint | last epoch from the same trajectory |
| optimizer | AdamW |
| learning rate | `1e-4` |
| LR schedule | cosine |
| effective batch | 32 |
| epochs | fixed 20；不 early stop |
| dropout | 0.1 |
| layer norm | enabled |
| initial calibration seed | 2021 |
| confirmation seeds | 2022, 2023 only after global selection |

Full-720 L1/MSE 只用于 neutral carrier calibration，避免在 SC0 阶段预先采用 SC2 的 horizon-measure
hypothesis。dense-horizon metrics仍需导出用于检查，但不得参与 config selection。

### 5.2 Capacity-matched arms

所有 arms 满足 `patch_num * d_model = 1536`。对 two-layer residual MLP选择 `d_ff`，使包含 patch
projection、active encoder、LayerNorm、coefficient head和temporal basis的 active-forward parameters
近似相等：

| Arm | `P` | patch length | `D` | `d_ff` | Approx. active params |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sc0_p12_d128` | 12 | 60 | 128 | 256 | 718,672 |
| `sc0_p24_d64` | 24 | 30 | 64 | 536 | 719,168 |
| `sc0_p48_d32` | 48 | 15 | 32 | 1072 | 718,576 |

参数 spread 约 `0.08%`。当前 `TimeAlign.Model` 实例化审计已确认三臂 active count与表格一致，且
unused `proj_x` 均为 `1,106,640`。实现后 checker仍必须以真实 module graph重算
active/total/unused parameters；若与表格不一致，先修订 protocol，不得直接 launch。

## 6. Global Selection Rule

SC0 不能逐 dataset 选配置。对每个 arm $g$ 与 dataset $d$，用 validation full-720 MSE 定义归一化
regret：

$$
r_{d,g}=\frac{\operatorname{MSE}^{val}_{d,g}}
{\min_{g'}\operatorname{MSE}^{val}_{d,g'}}-1.
$$

global score 为：

$$
S_g=\frac{1}{|D|}\sum_{d\in D} r_{d,g}.
$$

选择 $S_g$ 最小的单一 arm，并满足：

1. 每个 dataset 的 $r_{d,g}\le 3\%$；
2. 不出现 numeric pathology；
3. last/best sensitivity 不改变 global ranking；
4. active parameter spread符合预注册边界；
5. selected arm 追加 seeds 2022/2023 后，至少 `2/3` seeds保持 global winner方向；
6. 若两个 arms 的 $S_g$ 差异小于 `0.5%`，选择 latency/memory更低者。

只读取 training/validation artifacts 进行选择。Test set 在 frozen profile hash写入前不得用于 config
selection或 gate 判定。

若没有 arm 满足每个 dataset `<=3%` regret，decision 为
`common_token_mlp_profile_not_supported`。此时回 StageC Step 2/3 重审 common carrier topology，不允许
为每个 dataset 恢复不同 preset。

## 7. Required Implementation Artifacts

已实现且必须通过 local gate的文件：

```text
configs/stage_c_mechanism_control.json
scripts/check_stage_c_sc0_carrier_local.py
scripts/remote/run_stage_c_sc0_carrier_calibration.sh
scripts/analyze_stage_c_sc0_carrier_calibration.py
scripts/sync_stage_c_sc0_carrier_calibration_results.sh
```

`train_repo.py` 需要提供显式 standardized profile override，至少包括：

- `layer_norm` override；
- separation of training/validation/evaluation horizon sets；
- fixed-epoch dual checkpoint export；
- effective batch与 gradient accumulation记录；
- protocol class/profile hash写入 `effective_config.json`。

不得通过 shell 隐式覆盖而不写入 effective config。

[Local Result] 当前实现已通过三臂 structural checks与 `ETTh2/ETTm1/Weather × 3 arms` one-batch CPU
smoke。验证范围包括 patch boundary、`[B,C,1536]` state、active/unused parameters、full-objective单路径、
validation-only dense metrics、dual checkpoint strict reload与analyzer complete summary。当前 profile SHA256为
`79a037f751c0c24eea98ff0b516cb0dfeaef950871b3bbc515904754f54fd900`。

## 8. Local Gate

Remote launch前必须验证：

1. 三 arms 均无 `patch_num=1`，且 `P` 整除 720；
2. patch 不跨 channel boundary；
3. `hidden: [B,C,P*D]` 对三 arms 均为 `[B,C,1536]`；
4. active parameter count与预注册值一致；unused `proj_x` 单独报告；
5. A6 output shape为 `[B,H,C]`，requested-prefix consistency max abs为 0；
6. full objective只反传一次 720-step loss，不隐含四 benchmark-prefix weights；
7. last/best checkpoints可 strict reload，且来自同一 trajectory；
8. JSON/CSV明确记录 protocol class、profile、seed、environment与 selector metric；
9. ETTh2/ETTm1/Weather one-batch CPU/CUDA smoke通过。

## 9. Freeze Rule For Later StageC Experiments

SC0 通过后生成 immutable profile hash。后续 SC1/SC2/SC3 mechanism experiments必须：

- 引用相同 profile hash；
- 使用相同 source data split和preprocessing；
- 除被检验 mechanism 的参数外，不改变 carrier/optimization；
- 新增参数时提供 exact/no-mechanism capacity control；
- 所有候选共享同一 global mechanism hyperparameter，不逐 dataset调参；
- test result不能触发 profile修改。

若某个 mechanism理论上必须改变 carrier field，必须将其注册为新的 candidate family，并提供 matched base
arm；它不能继续引用 SC0 frozen comparison作为严格因果对照。

## 10. Reporting Boundary

论文与报告应并列给出：

1. source-faithful TimeAlign/A6 performance；
2. frozen StageC mechanism-control results；
3. native external baseline results；
4. dense seen/unseen horizon metrics；
5. consistency、worst-horizon regret、horizon-error AUC、params/FLOPs/latency；
6. last/best sensitivity和multi-seed uncertainty。

[Decision] 从 StageC 起，任何 paper-core mechanism claim必须以 frozen mechanism-control protocol 为主要
归因证据；source-finetuned result只能作为外部 performance context。

## 11. SC0 Seed2021 Outcome And Rollback

[Result, 2026-07-11] 预注册矩阵已完成 `9/9`，无运行错误，selection只使用 validation artifacts。
best-validation winner为 `sc0_p24_d64`，macro regret `0.4051%`、max dataset regret `1.2153%`，通过
per-dataset regret gate；last-checkpoint winner为 `sc0_p48_d32`，因此条款 3 的 selector stability失败。

[Failure Attribution] ETTh2 三臂 best epoch均在 1-2，last validation MSE较best恶化
`31.63%-44.95%`；ETTm1恶化`2.42%-3.38%`。这说明 exact fixed-20 training policy存在共同的
optimization/checkpoint pathology。结果只否定本版 SC0 research instrument，不否定 common token-MLP
topology或StageC论文方向。

[Protocol Audit] 条款 5 与“selected arm only追加 confirmation seeds”无法同时证明 global winner：若其他
arms未在相同 seed运行，就无法重算winner。后续 SC0-R1 必须在launch前明确区分 full-arm winner
confirmation 与 selected-arm absolute stability confirmation。

[Decision] 原 profile不冻结，不启动原定 seeds 2022/2023；回到StageC Step 2/3设计统一的
validation-controlled stopping/checkpoint policy。完整归因见
`analysis/stage_c_sc0_carrier_calibration_20260711/sc0_failure_attribution_and_rollback.md`。

## 12. SC0-R1 Preregistered Repair

### 12.1 Offline stopping-policy gate

在SC0九条完整validation trajectory上模拟统一patience候选：

| Patience | Retains full-trajectory best | Early-stopped runs | Total epochs saved |
| ---: | ---: | ---: | ---: |
| 3 | 7/9 | 9/9 | 136 |
| 5 | 9/9 | 9/9 | 105 |
| 7 | 9/9 | 9/9 | 87 |

[Decision] 选择最小且保留9/9已知best的`patience=5`。该结果只证明不会在已有trajectory上过早截断；
不证明新seed稳定。完整定义和逐run结果见
`analysis/stage_c_sc0_r1_protocol_gate_20260711/`。

### 12.2 Frozen training rule

- `max_epochs=20`；
- `enable_early_stopping=true`；
- `patience=5`、`min_delta=0`；
- `checkpoint_policy=best-val`、`restore_best=true`；
- optimizer、cosine schedule、LR、effective batch、loss与SC0一致；
- 所有dataset、arms和seeds使用同一规则；realized stop epoch可以由validation trajectory决定。

这意味着“统一超参数”要求相同decision rule，而不是要求不同dataset无视validation状态、机械训练同样多的
epochs。禁止根据dataset修改patience、max epochs或selector。

### 12.3 Full-arm multi-seed gate

SC0-R1一次运行全部`3 datasets × 3 arms × seeds {2021,2022,2023}=27 runs`。不再使用
selected-arm-only confirmation。只读取full-720 validation MSE：

1. pooled-mean与median-seed selector必须选择同一arm；
2. selected arm必须在至少2/3 seeds中成为global winner；
3. pooled-mean下每dataset regret不超过3%；
4. selected arm任一seed-dataset regret不超过5%；
5. 27 runs配置、参数、数值与dense validation artifacts完整；
6. profile freeze前禁止读取test。

若通过，直接冻结`configs/stage_c_mechanism_control_r1.json`的profile hash；若失败，回Step 2/3重审
carrier topology，不再修改stopping rule追逐本次结果。

## 13. SC0-R1 Result And Frozen Profile

[Result] 27/27 validation-only runs完成，0 errors。pooled-mean与median-seed均选择`sc0_p24_d64`；seed
winners为`p24/p12/p24`，达到2/3。selected arm pooled max dataset regret为1.277%，max
seed-dataset regret为1.440%，全部通过预注册阈值；test metrics未参与selection。

[Decision] StageC mechanism-control carrier冻结为`P=24, D=64, d_ff=536`。两层hash分工如下：

- calibration evidence hash：`3ebd07d647cdd4b0e8ea36a53eea9451d21f438a79164f74b8f4e8095426f31a`；
- resolved frozen contract：`configs/stage_c_mechanism_control_frozen.json`，SHA256
  `86a30f990370eb760feb27fad96fdc972893f196f7cfad4831e6264d2e582b6f`。

后续mechanism runner必须记录resolved contract hash，并保留calibration evidence hash作为selection
provenance。完整结果与dense-horizon claim boundary见
`analysis/stage_c_sc0_r1_carrier_calibration_20260711/sc0_r1_deep_analysis_and_freeze.md`。

[Boundary] `p24/d64`按full-720 validation选择，并非逐horizon最优。ETTm1 H48相对三臂oracle仍有
11.23% diagnostic regret；因此该profile用于因果归因而非all-horizon tuning claim，dense horizons仍必须
完整报告。

## 14. Active Dataset-Aware Governance Revision

[Correction] SC0中31.63%-44.95%是ETTh2 validation trajectory degradation，不是test degradation。
冻结后用原fixed20 checkpoints做test诊断，H720 last在9/9均差于best，mean test MSE为+6.11%；但dense
horizons中last有29/72个MSE wins。StageC A6 mechanism-control据此保留best-val；official TimeAlign
source reproduction仍遵循作者说明的native last protocol。证据见
`analysis/stage_c_sc0_checkpoint_test_gap_20260712/`与
[TimeAlign issue #2](https://github.com/TROUBADOUR000/TimeAlign/issues/2)。

[Governance Revision] 用户目标是避免精细特调，而非禁止dataset偏好。Section 13的uniform P24 contract
保留为历史更严格control，但active profile改为：

| Dataset | Active profile | P/D/d_ff | Three-seed dataset winner count |
| --- | --- | --- | ---: |
| Weather | `p12/d128` | 12/128/256 | 3/3 |
| ETTm1 | `p48/d32` | 48/32/1072 | 2/3 |
| ETTh2 | `p24/d64` | 24/64/536 | 2/3 |

选择只使用SC0-R1三seed pooled full-720 validation MSE；test没有改变mapping。active contract为
`configs/stage_c_mechanism_control_dataset_aware.json`，SHA256
`a10414acb23961225bb944e5939bac96fbecc3d332ff9dab3af71938f972fd88`。

同dataset内baseline、method、ablation和capacity/no-mechanism control必须使用同一profile。禁止test-driven
切换、per-mechanism重选和扩大continuous grid。新dataset只允许同样三组registered profiles的一次性
validation calibration。

## 15. SC0-DAP-R2 Natural-Profile Calibration

[Correction] Section 14 mapping来自capacity-matched grid，适合回答fixed-budget token allocation，却不应
作为dataset profile选择的唯一搜索空间。active parameter差异不再作为carrier selection约束。Section 14
mapping降级为`capacity_control_only`。

R2使用两阶段自然grid：

1. Phase A固定`D=64,d_ff=128`，比较`P={12,24,48}`；
2. Phase B固定各dataset的selected P，比较`D/d_ff={32/64,64/128,128/256}`；
3. Phase A的medium width可在Phase B复用，总remote budget为21 runs；
4. params、FLOPs和latency只报告，不进入winner排序。

profile selection使用H48/96/144/192/288/336/512/720 validation MSE。先在每个horizon内计算相对三候选
best的normalized regret，再平均八个regrets；tie依次比较max regret、H720 regret和name。test完全禁止。

seed2021用于coarse selection。最终selected profile补跑2022/2023只确认absolute stability，不声称
selected-only confirmation能证明relative winner。完整config与实现见：

Phase C在launch前冻结absolute stability gate。对每个dataset、每个dense horizon，用seeds
`{2021,2022,2023}`的MSE计算sample coefficient of variation：

$$CV_{d,h}=\frac{\operatorname{sample\_std}_s(MSE_{d,h,s})}{\operatorname{mean}_s(MSE_{d,h,s})}.$$

每个dataset必须同时满足mean dense-horizon CV不超过3%、maximum dense-horizon CV不超过5%，且9个
profile-seed实例、72个dense metrics完整、finite、validation-only。active params不参与gate。若失败，
decision为`protocol_audit_required`并回Step 2/3；禁止读取test、重选winner或扩展grid。

- `configs/stage_c_dataset_profile_calibration_r2.json`；
- `scripts/remote/run_stage_c_dap_r2a_patch_screen.sh`；
- `scripts/analyze_stage_c_dap_r2a_patch_screen.py`；
- `scripts/remote/run_stage_c_dap_r2c_stability.sh`；
- `scripts/analyze_stage_c_dap_r2c_stability.py`；
- `docs/code-explanation/stage-c-dap-r2-profile-calibration.md`。

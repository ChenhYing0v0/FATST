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
| `mechanism_control` | 因果归因、消融、候选 small gate | 一个跨 dataset frozen profile；禁止 `patch_num=1` | mechanism evidence under controlled carrier |
| `native_external` | 复现外部 baseline | 在其官方 repository 使用原生 protocol | external baseline reproduction only |

不同 protocol 的结果必须分表报告，不能混成同一 matched comparison。

## 3. Standardization Principle

“统一”定义为：

- 同一 computation topology；
- 同一 patch policy；
- 同一 active state width 与近似 active parameter budget；
- 同一 optimizer、LR schedule、effective batch、epoch budget、loss 与 checkpoint selector；
- 同一 global selection rule；
- dataset 只改变无法避免的数据 schema。

“统一”不等于把 upstream 某个 dataset 的 preset复制给所有 dataset，也不等于看到 test 结果后选择每个
dataset 的最佳配置。

## 4. Allowed Dataset-Specific Fields

StageC mechanism-control runner只允许下列 dataset-specific fields：

- file path、dataset loader、sampling frequency；
- `enc_in/dec_in/c_out` 与 channel identity；
- normalization statistics；
- physical micro-batch size，但必须用 gradient accumulation保持相同 effective batch；
- worker/GPU 等不改变 optimization semantics 的 runtime setting。

禁止 dataset-specific 修改：

- `patch_num`、patch topology；
- `d_model`、`d_ff`、layers、basis rank；
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

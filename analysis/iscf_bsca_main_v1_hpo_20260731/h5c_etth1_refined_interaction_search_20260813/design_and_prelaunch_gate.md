# H5C ETTh1 Refined-Interaction HPO Design and Prelaunch Gate

## 1. Current step and authorization

- `current_step`: Step 6 result-informed search redesign -> Step 8 remote resource gate
- `candidate_version`: `ISCF-BSCA-MAIN-v1-etth1-h5c-test-informed-20260813`
- `user_authorization`: 2026-08-13，ETTh1训练较快，将HPO matrix相对H5B扩大约50%，结合此前结果继续优化。
- `architecture_search`: false；encoder mode、ISCF/BSCA decoder、objective、scope set、training path与H720-prefix inference graph均不变。
- `formal_test_boundary`: training阶段test=0；54/54 validation-selected checkpoints及immutable manifest完整后，才允许一次完整`54 × 4 H = 216` row formal test，禁止partial selection。
- `table_boundary`: H5C结束后也不自动修改Main I/Main II；table mutation需单独授权。

## 2. H5B result audit

H5B审计36个新profiles并选择`h5b_seq640_p20`。相对H5A，four-H mean MSE/MAE改善`0.363%/0.584%`，Main II ETTh1 best cells由2/8提高到4/8。完整结果表明参数影响并不均匀：

| Factor | H5B evidence | H5C allocation |
| --- | --- | --- |
| context × patch | `L640/p20`的mean MSE/MAE=`0.391378/0.417255`，显著优于所有其他H5B profiles；`L576/p18`、`L672/p21`及更长context呈非单调变化 | 18个local geometry points + 4个context/dropout interactions；最大预算 |
| learning rate | 在L720/p24上，lower LR更利于H336 MSE，higher LR更利于H336 MAE；尚未与L640/p20联合测试 | 10个winner-LR trials + 5个LR/dropout joint trials |
| dropout | `dropout=0.05`是H5B aggregate第二名，并取得H336最低MSE | 7个winner-dropout trials，并在四个context点联合`dropout=0.05` |
| weight decay | `0--0.03`形成窄平台，影响明显小于context | 5个bounded winner interactions |
| mode rank | rank80取得H720最低MAE，但aggregate不占优 | 5个bounded winner interactions |
| capacity | `d48/ff48`为负向，历史width变化也未形成frontier | H5C不再投入capacity trials |

H5B selected profile在H96与H720的MSE/MAE均为Main II best；当前主要缺口为H192与H336。H192距离三位小数best约为MSE `0.0022`、MAE `0.0014`；H336的历史单因素frontier已达到MSE best显示值，但MAE仍有约`0.0014` full-precision差距。因此H5C重点测试context、LR和dropout之间尚未覆盖的interaction，而不是继续盲目拉长context或增大capacity。

Self-critique：这些差距接近three-decimal ranking boundary，增加best ties不等于在所有full-precision cells上SOTA。H5C仍以一个dataset-level profile服务全部four H，并以mean MSE/MAE双guard阻止用aggregate退化换取单cell显示优势。

## 3. Frozen 54-trial matrix

H5C相对H5B `36 -> 54`，精确增加50%。全部为ETTh1、seed2021、from-scratch、joint encoder-decoder training，且与此前61个ETTh1 effective training profiles零重复。

| Block | Trials | Purpose |
| --- | ---: | --- |
| Local context/patch refinement | 18 | 在L570--736、patch count 18--23内细化L640/p20邻域；全部满足`seq_len % patch_num = 0` |
| Winner × learning rate | 10 | 在L640/p20上覆盖`2.2e-4--4.2e-4` |
| Winner × dropout | 7 | 在L640/p20上覆盖`0--0.12`，重点加密0.04--0.08 |
| Winner × weight decay | 5 | 检查`0--0.03`平台在新context下是否变化 |
| Winner × mode rank | 5 | 检查rank64--160在新context下的H720/aggregate trade-off |
| Winner LR × dropout | 5 | 联合较高LR与dropout0.05/0.08，重点针对H336 MAE |
| Context × dropout | 4 | 在L608/L620/L660/L682上联合dropout0.05 |

Machine-readable contract=`configs/iscf_bsca_main_v1_hpo_etth1_h5c.json`。训练预算继续固定`120 epochs/patience24`；H5B winner的validation best epoch很早，因此本轮增加matrix breadth而不增加epoch budget。

## 4. Selection, success and rollback

1. 每trial仅以validation mean MSE over `{96,192,336,720}`选择checkpoint。
2. 54/54 checkpoint及hash manifest完整后，对全部profiles执行一次完整official test。
3. Primary rank为Main II best metric cells；Main I best、Main II top-2、mean MSE/MAE、validation score、parameter count与lexical ID依次tie-break。
4. Eligibility要求four-H mean MSE与MAE均不超过H5B selected profile的`1.002×`。
5. Current=`4/8` Main II best cells；minimum success=`5/8`，stretch=`6/8`。
6. 若无eligible profile严格增加best cells，继续保留`h5b_seq640_p20`；不因H5C启动而撤销H5B frozen selection。
7. OOM/numeric failure只允许修复对应runtime profile；architecture/objective变化需要新candidate与新narrative gate。

## 5. Resource and scheduling gate

- H5B 36个train/validation jobs实际约1 wall-hour；H5C context上界更低但trial增加50%，预计`4--9 GPU-hours`、三GPU约`1.5--3.5 wall-hours`。
- 三个GPU使用shared dynamic queue；L736/L726/L713等较长context优先，随后执行context interactions与L640/p20单因素任务。
- Remote output=`/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h5c`；storage budget=`6 GiB`。
- Preflight观测GPU0/1/2均为18 MiB、0% utilization；quota=`187G/200G soft/220G hard`。H5C projection保持soft limit内，当前无需删除H5B formal evidence。
- Launch sequence：local gate -> focused commit/push -> remote `git pull` -> 54/54 resource smoke且test=0 -> full train/validation background queue。

## 6. Local prelaunch gate

- JSON parse：pending final verification
- Python/shell syntax：pending final verification
- Frozen contract checker：已确认54 jobs、7 blocks、61 historical ETTh1 jobs audited、0 effective-profile duplicates、training test=0
- Target CSV/source hashes：pass
- Remote resource state：GPU idle、quota margin pass
- Decision：`H5C_frozen_authorized_remote_resource_smoke_next`

# H4M高影响参数搜索与TimeAlign并行复现prelaunch

日期：2026-08-04

当前步骤：Step 6实验设计已冻结，进入Step 7 local protocol gate；通过后执行Step 8 remote smoke与training。

用户授权：清理远程结果后，同时继续ETTm2/Weather HPO并从官方代码复现TimeAlign。

## 1. 当前证据与问题

H1--H4L共165个ISCF-BSCA-MAIN-v1 trials已完成完整official-test审计。合法dataset-level selector为MSE `15/28`、MAE `16/28`、combined `31/56`；ETTm2只由H4L `weight_decay=1e-3`新增H720 MAE领先，Weather没有切换selector。现有pool的dataset-level、unrestricted profile与逐cell diagnostic oracle上限均为31/56，因此继续搜索必须生成新的强profile，而不能再调整selector。

H4M不改变architecture、objective、scales、inference graph或dataset-level shared-profile规则。它只在既有frozen candidate内做test-informed hyperparameter optimization。

## 2. 哪些参数影响最大

为避免把H4K到H4L的budget变化误当成parameter effect，主排序使用H4L同阶段、单因素或可比profile的joint MSE/MAE跨度；跨阶段结果只作辅助解释。

| Dataset | Parameter / interaction | Observed joint span | Decision |
| --- | --- | ---: | --- |
| ETTm2 | `patch_num` | 3.903% | 与low LR组成主grid |
| ETTm2 | `learning_rate` | 3.319% | 扩展到`1e-5/2e-5/5e-5` |
| ETTm2 | `seq_len` | 2.834% | 增加`L=960`组合 |
| ETTm2 | `mode_rank` | 2.427% | 在最佳patch/LR附近补`48/80` |
| ETTm2 | width/capacity | 0.58--0.92% | 固定`d_model=d_ff=128` |
| ETTm2 | `dropout` / `weight_decay` | 约0.14% / 0.028% | 除已选`wd=1e-3`外不再扩张 |
| Weather | `seq_len × patch_num` | 4.220% | 固定patch length 32扩大context |
| Weather | `learning_rate` | 1.486% | 在`L512/p16`补low LR |
| Weather | `patch_num` | 1.435% | 在low LR/rank附近补边界 |
| Weather | `mode_rank` | 0.820% | 在最佳context/LR补`16/64` |
| Weather | width/capacity | 0.20--0.30% | 固定`d_model=64,d_ff=128` |
| Weather | `dropout` / `weight_decay` | 0.02--0.10% / 0.003% | 冻结；只保留一个LN matched diagnostic |

关键interaction尚未被历史矩阵完整覆盖：ETTm2的`patch4 × lr1e-5`兼具逐H MSE/MAE oracle信号，但此前没有联合训练；Weather的`L512/p16`是global joint最好profile，low LR与rank16/64也分别改善部分长horizon，却未在同一profile中组合。

## 3. H4M frozen matrix

H4M共24个seed2021 jobs：

- ETTm2 12：`patch_num={2,4,6} × lr={1e-5,2e-5,5e-5}`九项；在`p4/lr1e-5`补`rank={48,80}`；再补`L960/p4/lr1e-5/r64`。
- Weather 12：五个constant patch-length-32且未在165-trial pool出现的context pairs：`(L,p)={(384,12),(640,20),(768,24),(960,30),(1152,36)}`；`L512/p16`补`lr={1e-5,2e-5}`；在`lr2e-5`补`rank={16,64}`；在`rank64`补`p={8,32}`；另有一个H4K anchor的`LN=0` matched diagnostic。

已有`Weather L512/p16/lr5e-5/r116`与`L720/p24/lr5e-5/r116`保留在165-trial selector pool中，不重复训练。24个新profiles必须与历史165个effective profiles全部不重复。

ETTm2保持60 epochs/patience12。Weather H4L `lr1e-5`与`patch4`均在epoch47/60才达到最佳validation，因此H4M Weather统一扩展至90 epochs/patience18，避免只给有利profile增加budget。每个trial仍由four-H validation mean MSE选择checkpoint，训练阶段test=0；24/24 checkpoint manifest通过后，才执行已授权的完整four-H formal test。

## 4. TimeAlign official reproduction

当前使用的TimeAlign published comparison来自论文Table 6；2026-06-26本地曾复跑ETTm2/Weather且四H均值与论文非常接近，但remote raw directories、checkpoints、effective configs和logs已经删除，因此只能作sanity reference，不能作artifact-complete正式复现。

本轮冻结`2 datasets × 4 horizons × seed2021 = 8`个independent fixed-H systems，使用官方配置：

- ETTm2：`L720,p12,d_model=d_ff128,w_align1,LN1`；H96/192 dropout0.3，H336/720 dropout0.9。
- Weather：`L720,p48,d_model128,w_align0.1,LN0`；H96/192/336 `d_ff256,dropout0.1`，H720 `d_ff128,dropout0.5`。
- 共同契约：label48、batch32、10 epochs、AdamW wd0.01、cosine、official-last、无early stopping、训练后一次official test、MSE/MAE与predictions全部保留。

不直接执行官方`run.py`，因为原训练循环每个epoch读取test loss。执行标签固定为`official-source model/config + FATST test-hygiene/artifact adapter`：`train_repo.py`只在epoch内使用train/validation，训练结束后一次加载test。上游2026-08-04审计commit为`ab2dff5bde250f82e29d8755f87a494921857d71`。官方仓库未提供顶层LICENSE且API为`license=null`，因此标记`license_unresolved`，本轮仅作research-only local reproduction，不主张redistribution clearance。

## 5. 清理、资源与调度

清理已在实验前完成：删除7个resource-smoke目录和157个未被当前selector使用的diagnostic NPZ，保留8个selected NPZ、全部165个metrics/invariants、checkpoints、manifests与logs。精确删除约36.51 GiB，remote quota由201G降至165G，未触碰三个既有remote dirty CSV。

调度冻结为：

- GPU0--1：H4M 24-job dynamic queue，预计18--30 GPU-hours、2--3 GiB training artifacts；
- GPU2：TimeAlign 8-job workload-ordered queue，Weather长horizon优先；
- 两条线分别先完成resource smoke。smoke只证明execution/memory健康；H4M训练smoke不得访问test，TimeAlign smoke使用`final_evaluation_split=none`。

## 6. Gates与rollback

1. Local gate：JSON parse、Python compile、shell syntax、24/24 profile nonduplicate、source/evidence hash与8/8 official preset一致性全部通过。
2. Remote preflight：exact commit、dataset SHA256、source SHA256、quota、GPU process/memory均通过。
3. H4M smoke：24/24 artifacts，无OOM/NaN/Inf/Traceback；失败则只修复runner/runtime并重做smoke，不改变search space。
4. TimeAlign smoke：8/8 checkpoint/config/log artifacts，无failure pattern；失败按source/runtime/data wiring归因。
5. H4M training gate：24/24 validation-selected checkpoints、unique hashes和provenance完整后才进入formal test。
6. TimeAlign result gate：8/8 official-last checkpoints和8/8 test rows完整，逐cell对照论文与历史复跑；partial matrix不得报告为复现完成。
7. H4M effectiveness gate继续为MSE `>=20/28`、MAE `>=20/28`、combined `>=40/56`。所有negative trials/cells保留；不得per-H/per-metric/per-cell选profile。

若H4M仍失败，回滚Step 6并停止无依据的局部HPO扩张，优先完成完整baseline矩阵与matched Main II；若TimeAlign偏差明显，先审计runtime deviation、dataset identity与adapter computation path，不用选择性删除弱cell。

Machine-readable contracts：

- `configs/iscf_bsca_main_v1_hpo_targeted_h4m.json`
- `configs/timealign_official_ettm2_weather_reproduction.json`

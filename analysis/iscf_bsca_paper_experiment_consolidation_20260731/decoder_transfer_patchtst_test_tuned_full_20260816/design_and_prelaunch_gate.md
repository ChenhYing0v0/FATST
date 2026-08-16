# PatchTST Decoder-HPO 全量 Test-Tuned Audit：设计与启动门

日期：2026-08-16  
Decision：`40 unique checkpoints frozen; 35 missing formal tests authorized`。

## 1. 目的与证据边界

用户明确授权对parent PatchTST decoder-HPO中所有尚未测试的unique checkpoints执行official test，并按dataset选择一套共同服务四个horizons的参数。本轮建立新candidate `ISCF-BSCA-DECODER-TRANSFER-PATCHTST-HPO-v2-TT`，属于`test_informed/test_tuned` audit，不把test描述为untouched holdout。

本轮只回答：在已冻结的10-profile decoder search space内，按每个dataset的four-H mean official-test MSE选择一个profile后，PatchTST-style +ISCF-BSCA能否超过其Original Decoder。禁止per-horizon、per-seed、per-metric或per-cell反选。

## 2. 冻结矩阵与复用

- parent training：10 profiles × 5 datasets × seed2021 = 50 runs；
- observed checkpoint objects：50；unique hashes：40；
- duplicate mapping：`p02=p08`与`p05=p06`分别在五个datasets内bitwise collapse；
- unique formal matrix：40 checkpoints × 4 horizons = 160 cells；
- 已测且可复用：v2.1 selected 5 checkpoints / 20 cells；
- 本轮新增访问：35 unique checkpoints / 140 cells；
- profile-expanded matrix：50 profile-dataset objects / 200 cells；
- 加入v1 BSCA reference后，candidate pool为55 profile-dataset objects / 220 cells；
- Original Decoder anchor：复用v1的20 cells，不重新访问test。

Duplicate profiles共享同一checkpoint与formal metrics，但在完整profile scorecard中保留两个trial IDs；它们不被伪装成独立checkpoint。

## 3. 选择与gate

每个dataset在10个HPO profiles与v1 reference之间，以四个standard horizons的mean official-test MSE最小为唯一primary selector；MAE只完整报告，不参与选择。一个dataset只允许一个profile，四个horizons共同使用。

相对Original Decoder的performance gate保持：macro MSE gain > 0、macro MAE gain > 0、dataset-mean MSE wins >= 3/5。全部20 selected cells与所有负向trials必须保留。

即使gate通过，该结果也只能支持“在显式披露test-tuned dataset-level decoder HPO后，PatchTST carrier可以获得正向performance compatibility”。由于新winner未必具有逐profile matched +ISCF control，不能单独建立BSCA objective attribution，也不能恢复architecture-agnostic portability claim；后续若要进入论文transfer table，需先冻结并补训selected profiles对应的matched +ISCF controls。

## 4. 资源、回滚与授权

GPU0/1/2启动前必须空闲检查。新增35次evaluation预计1--3 GPU-hours；不训练，不改checkpoint。远程quota已接近soft limit，本轮只写metrics、invariants、diagnostics和logs，不复制checkpoint。

- 若gate失败：保持PatchTST negative result，不追加search或选择性汇报；
- 若gate通过但matched attribution缺失：状态为`performance_partial_pass`，canonical Decoder-Transfer table保持不变，等待单独matched-training授权；
- 若artifact/hash/invariant失败：在test或汇总阶段停止，只修复失败artifact，不改搜索/选择规则。

Authorization：用户于2026-08-16明确授权全部未测unique checkpoints的formal test与逐dataset test-tuned selection；remote training=false，formal test=true，canonical table mutation=false。

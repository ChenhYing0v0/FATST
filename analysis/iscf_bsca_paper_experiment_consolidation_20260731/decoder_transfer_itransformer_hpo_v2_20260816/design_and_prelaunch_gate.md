# iTransformer-style Decoder-HPO v2 设计与启动门

Decision：`itransformer_decoder_hpo_v2_70_profile_dataset_runs_frozen_test_tuned_remote_training_authorized`。

## 1. 问题与既有证据

上一版 `ISCF-BSCA-DECODER-TRANSFER-ITRANSFORMER-v1` 的 formal test 已完整结束。+ISCF-BSCA 相对 Original Decoder 的 macro MSE/MAE gain 为 `-2.690%/-2.513%`，dataset-mean MSE wins=`1/5`。+ISCF 自身也落后 Original Decoder `1.300%/1.337%`，因此当前最直接的问题是 replacement readout 与 iTransformer variate-token representation 的兼容性，而不是单独的 BSCA loss failure。

训练记录还显示 ETTm1 的 +ISCF 与 +ISCF-BSCA 都在 epoch 10 的预算上限取得 best validation epoch；v1 的10 epochs/patience3可能不足。ETTh1则呈现更明显的decoder capacity/geometry gap。因此本轮同时搜索训练预算、decoder optimization scale 与有限的结构轴，而不修改encoder。

## 2. Candidate 与 narrative gate

新候选为 `ISCF-BSCA-DECODER-TRANSFER-ITRANSFORMER-HPO-v2`。它是观察过v1 official-test结果后的 `test_informed/test_tuned decoder rescue`，不是新的untouched-holdout验证，也不是architecture-agnostic portability证据。

本轮只回答：固定 iTransformer-style encoder 与 ISCF-BSCA objective 时，是否存在一个dataset-level decoder profile，使一个模型共同服务四个horizons并超过v1 Original Decoder。即使通过performance gate，BSCA attribution仍需之后用selected profile补齐matched +ISCF controls；不能只凭best-tuned BSCA结果建立机制归因。

## 3. 冻结 search space

- datasets：Weather、ETTm1、ETTm2、ETTh1、ETTh2；
- seed：2021；
- horizons：`{96,192,336,720}`；
- arm：只训练 `itransformer_iscf_bsca_hpo_v2`；
- encoder：完全复用v1 source-informed iTransformer profiles；
- training：from-scratch end-to-end joint training，H720 full loss；
- checkpoint selector：four-H mean validation MSE；
- budget：30 epochs、patience6；
- profiles：14；new runs=`14 × 5 = 70`。

14个profiles覆盖以下轴：

| Axis | Frozen values |
| --- | --- |
| Budget control | v1 geometry + 30 epochs/patience6 |
| Readout LR multiplier | 0.25、0.5、1、2、4 |
| Mode rank | 0.5×、1×、1.5×、2× dataset-matched rank |
| Future coordinate dimension | 2、4、8；2/8维点以inverse-rank近似capacity match |
| Allocation MLP | 16×32、32×64、64×128 |
| Scope geometry | `[1,48,144,360,720]`、`[1,24,72,240,720]`、`[1,72,180,360,720]` |
| Readout weight decay | 0；不重复PatchTST HPO中已出现bitwise collapse的微小WD网格 |

完整profile定义只以 `configs/iscf_bsca_decoder_transfer_itransformer_hpo_v2.json` 为准。所有profile对四个horizons共同生效，禁止per-H/per-cell架构选择。

## 4. Test-tuned selection

用户于2026-08-16明确允许按official-test结果反选参数。70个training artifacts与70个unique checkpoint hashes闭合后，允许一次完整 `70 checkpoints × 4 horizons = 280` test-tuned formal cells。v1 BSCA reference和v1 Original anchor各20 cells复用既有formal evidence，不重复访问。

每个dataset从14个新profiles与v1 reference中，仅按四个standard horizons的mean test MSE选择一个shared profile；MAE作为完整报告指标和tie-breaker。禁止按单个horizon、seed、metric或table cell选参。所有70个trial的MSE/MAE，包括负向trial，都必须进入scorecard。

Selected BSCA performance gate相对v1 Original Decoder要求：

1. macro MSE gain > 0；
2. macro MAE gain > 0；
3. dataset-mean MSE wins >=3/5；
4. checkpoint nonmutation、numeric health与完整trial reporting全部通过。

## 5. Source patch 与公平性

本轮只给既有SIFF decoder开放三个原本被v1固定的构造参数：`pcsd_scales`、`pcsd_coordinate_dim`、`pcsd_policy_{history,hidden}_dim`。默认值与所有旧protocol行为不变；宽松验证仅对新的 iTransformer HPO protocol profile 生效。iTransformer encoder、normalization、input length、width/depth/dropout、base LR、batch、seed和BSCA objective不变。

Variable scope geometry仍使用相同SIFF inference path：`hidden [B,C,D] -> component modes [B,C,S,D_q,K] -> scope forecasts [B,C,S,720] -> target-adaptive fusion [B,C,720]`。没有加入native decoder skip、residual adapter、warm start或frozen replacement。

## 6. 资源、调度与rollback

- 预计20--32 GPU-hours；3张RTX 3090约7--12 wall-hours；
- 预计新增2--4 GB；
- dataset-major LPT顺序：Weather -> ETTm1 -> ETTm2 -> ETTh1 -> ETTh2；
- 正式启动前检查GPU与quota，resource smoke覆盖default、high-coordinate与alternate-scope profiles；
- training阶段test=0；必须先生成immutable 70-row manifest，之后才执行test-tuned ranking。

Rollback：resource/numeric失败只修复exact protocol；若没有profile改善v1 BSCA，则关闭iTransformer decoder rescue；若改善v1 BSCA但仍不超过Original，则透明报告best-tuned negative result；若超过Original，再补matched ISCF attribution，不直接宣称BSCA portability。

## 7. Local prelaunch

`scripts/check_iscf_bsca_decoder_transfer_itransformer_hpo_v2_prelaunch.py` 已在本地 `r2026-fsa` 环境完成22/22 checks：

- 14 effective profiles与70-run matrix唯一；
- 14种decoder实例的rank、coordinate、policy、scope与optimizer group均匹配config；
- 全部forward finite，H96与H720 prediction满足exact prefix；
- remote runner `bash -n`与70-job dry-run通过；
- 本地test jobs=0，paper table mutation=false。

下一步为focused commit/push、remote GPU/quota gate、3-profile resource smoke和70-run training launch。

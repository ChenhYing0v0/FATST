# PatchTST Decoder-Transfer v2.1 Formal Result Audit

Decision：`decoder_transfer_v2p1_complete_portability_gate_not_passed`

## HPO与修订边界

- parent v2完成50/50训练，validation gate通过：macro MSE改善0.813%，5/5 datasets改善超过0.1%；
- parent v2仅有40/50 unique hashes，两组极小decoder weight decay配置在各dataset上收敛为bitwise-identical checkpoints；因此parent artifact gate保持FAIL，未被事后放宽；
- v2.1只冻结5个互异的validation-selected BSCA checkpoints，并from scratch补训5个matched ISCF checkpoints；每个dataset只选一个profile且四个horizon共用；
- 新formal access覆盖10 checkpoints/40 cells；DLinear完整block与PatchTST Original Decoder的80 cells复用v1 evidence，不再次访问test。

## Formal integrity

- 10/10 rows、10 unique checkpoint hashes、5/5 matched initialization pairs；
- 40/40 new official-test cells与120/120 combined cells完整；
- checkpoint由four-H validation mean MSE选择，test不选择epoch、seed、horizon或table cell；
- candidate为test-informed validation-HPO rescue，不声称untouched holdout。

## Pre-registered gates

| Backbone | BSCA vs Original MSE gain | MAE gain | Dataset/Cell MSE wins | ISCF vs Original MSE gain | BSCA vs ISCF MSE gain/cell wins | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DLinear-style | +15.702% | +8.184% | 4/5, 15/20 | +10.606% | +5.701%, 8/20 | PASS |
| PatchTST-style | -0.436% | -0.568% | 2/5, 9/20 | -1.360% | +0.912%, 16/20 | FAIL |

## HPO effect relative to v1

- PatchTST +ISCF macro MSE由0.316895降至0.314448（改善0.772%），MAE改善0.342%；
- PatchTST +ISCF-BSCA macro MSE由0.312504降至0.311581（改善0.295%），但MAE恶化0.506%；
- BSCA相对Original的MSE deficit从-0.733%缩小到-0.436%，但MAE deficit从-0.062%扩大到-0.568%；
- 新BSCA只在ETTm1和ETTm2的dataset-mean MSE上超过Original；Weather、ETTh1与ETTh2仍落后。

## Four-layer decision

1. `paper_facing_effectiveness`：完整120-cell表面决定最终performance viability；validation HPO本身不构成正式有效性证据。
2. `matched_mechanism_attribution`：PatchTST +ISCF和+ISCF-BSCA共享encoder、rank、optimizer scale、seed与initialization class，仅objective不同；Original Decoder仍是同backbone native-readout control。
3. `internal_mechanism_health`：diagnostics只用于解释，不替代相对Original的formal gate。
4. `failure_attribution`：完整结果无numeric/artifact pathology。BSCA相对matched ISCF改善0.912% MSE并赢16/20 MSE cells，说明BSCA objective在该replacement head内部仍有作用；但两者都未超过native Original Decoder。因此exact two-backbone portability claim记为`hypothesis_false_for_cross_backbone_portability_after_decoder_HPO`，设计层更具体地指向`readout_or_head_design_wrong_for_PatchTST_representation_compatibility`，不能据此否定BSCA objective本身。

## Claim boundary

至少一个backbone未通过gate，因此不得恢复cross-backbone decoder portability总体正向结论；v1负结果的主要结论保持有效。

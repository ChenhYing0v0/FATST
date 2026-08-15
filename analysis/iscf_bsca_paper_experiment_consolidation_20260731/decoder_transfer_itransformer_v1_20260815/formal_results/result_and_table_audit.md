# iTransformer-style Decoder-Transfer v1 Formal Result 与 Step 9--10 审计

Decision：`itransformer_transfer_v1_complete_portability_gate_failed_no_fourth_carrier_or_hpo`

## 1. 完整性与来源

本轮一次性 formal audit 于 2026-08-15 20:15:40 +08:00 完成：

- 15/15 immutable checkpoint objects 与 60/60 standard-horizon cells 完整；
- 15/15 `test_audit_invariants.pass=true`，dense-prefix metric rows 为 10,800/10,800；
- test access 前后均为 15 个唯一 checkpoint SHA256；
- 全部 MSE/MAE 为有限值，15 份 evaluator logs 中没有 error token；
- checkpoint retraining=false，test access date=`2026-08-15`；
- formal artifact manifest SHA256=`aad13a13e2c1f486798caf0a67829396ed5ac334ffb1fb47e8b4277470d4e39a`。

Training manifest 仍为 SHA256=`062588a140ecd4fae385aa9d194c039355bef3c7d9f49f685d796779626eecc9`。Formal evaluation 未修改任何 checkpoint。

## 2. Formal results

下表每个 dataset 的数值均为 `{96,192,336,720}` 四个 horizon 的均值。

| Decoder | ETTm1 | ETTm2 | ETTh1 | ETTh2 | Weather | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original Decoder MSE/MAE | .387/.388 | .287/.324 | .425/.424 | .338/.378 | .252/.272 | .338/.357 |
| +ISCF MSE/MAE | .390/.393 | .286/.324 | .445/.443 | .337/.378 | .252/.273 | .342/.362 |
| +ISCF-BSCA MSE/MAE | .393/.396 | .293/.331 | .458/.453 | .337/.378 | .253/.273 | .347/.366 |

+ISCF-BSCA 相对 Original Decoder：

- macro MSE/MAE gain=`-2.690%/-2.513%`;
- dataset-mean MSE wins=`1/5`;
- MSE/MAE cell wins=`4/20` and `3/20`;
- 唯一取得正向 MSE dataset mean 的数据集是 ETTh2。

+ISCF-BSCA 相对 matched +ISCF：

- macro MSE/MAE gain=`-1.372%/-1.160%`;
- dataset-mean MSE wins=`1/5`;
- MSE/MAE cell wins=`3/20` and `3/20`.

+ISCF 自身也落后 Original Decoder：MSE 为 `1.300%`，MAE 为 `1.337%`。因此，这不是 BSCA 单独造成的失败：exact ISCF replacement readout 与该 iTransformer-style representation 的兼容性较弱，而 BSCA objective 在此设置中进一步降低了 aggregate result。

## 3. Four-layer decision

1. `paper_facing_effectiveness`：完整但未通过。预注册 gate 要求 MSE、MAE gain 均为正，且至少赢得 3/5 dataset MSE means；实测三项均未达到。
2. `matched_mechanism_attribution`：负向。Same-profile、matched-initialization、end-to-end 的 Original/+ISCF/+ISCF-BSCA 对比显示两个 replacement arms 均未超过 Original，且 BSCA 也落后 matched ISCF。
3. `internal_mechanism_health`：仅通过 implementation-health。Exact-prefix、finite diagnostics、source profile、encoder initialization 与 checkpoint contracts 均通过，但执行健康不能补救负向 effectiveness 或 attribution。
4. `failure_attribution`：claim-level=`hypothesis_false_for_cross_backbone_portability_after_two_failed_transformer_carriers`；exact-design level=`readout_or_head_design_wrong_for_itransformer_representation_compatibility`，并伴随 `bsca_objective_not_effective_on_exact_itransformer_replacement_head`。没有 numeric 或 artifact pathology 可以解释该差距。

## 4. 论文与 rollback 边界

DLinear-style 仍是正向 transfer block；validation-tuned PatchTST-style 与新的 iTransformer-style carrier 均未超过各自 native Original Decoder。因此，现有证据不支持 general cross-backbone decoder-portability claim。该结果也不等于否定 native iTransformer 或 BSCA 在所有架构中的有效性：本地 carrier 是 source-informed implementation，而非 exact official iTransformer reproduction，结论仅适用于本次冻结的 matched implementation。

按照冻结的 rollback，不自动开启第四个 carrier、额外 decoder HPO、extra seed 或选择性 dataset rescue。Canonical Decoder-Transfer table mutation 未获授权，因此当前 DLinear/PatchTST v2.1 表保持不变。iTransformer result block 与 review table 在此冻结，等待作者决定是否用于透明披露或 appendix。

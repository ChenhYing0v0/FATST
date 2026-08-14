# Decoder-Transfer v1：design and prelaunch gate

## 状态

- `candidate_version=ISCF-BSCA-DECODER-TRANSFER-v1`
- `authorization=2026-08-14 author explicit request`
- `matrix=2 backbones × 3 decoder arms × 5 datasets × seed2021 = 30 checkpoints / 120 test cells`
- `checkpoint_selector=mean validation MSE over H={96,192,336,720}`
- `training=end-to-end joint from scratch`
- `test_role=test-informed formal mechanism-effectiveness benchmark`

## 公平比较

两类backbone分别固定为DLinear-style decomposition memory与PatchTST-style channel-independent contextual patch memory。每个backbone–dataset block内，Original、+ISCF、+ISCF-BSCA共享同一lookback、encoder architecture、optimizer profile、budget、seed、split与checkpoint selector；只改变decoder与BSCA objective。DLinear/PatchTST official repositories只提供source prior，本地arms不是native external baseline reproduction。

Original使用一个unified H720 direct decoder；+ISCF使用同一ISCF graph但仅保留Uniform-Prefix Forecasting Loss；+ISCF-BSCA再加入Scope-Wise Forecasting Loss与Allocation-Balance Regularizer。三列都从随机初始化端到端训练。frozen replacement、warm-start与cross-swap被排除，不得用于方向级结论。

## Profile来源与边界

`configs/iscf_bsca_decoder_transfer_profiles.json`在任何新transfer结果产生前冻结。Encoder architecture与learning-rate prior来自已审计的official DLinear/PatchTST H720 scripts；batch size仅作同一arm-block内一致的3090安全上限。该profile是`source_informed matched profile`，不是对native baseline的复现，也不是把ISCF-BSCA-v1 ablation carrier机械迁移到新backbone。

## Gates与rollback

formal test必须等待30/30 training artifacts、30 unique checkpoint hashes、matched encoder initialization与checkpoint non-mutation全部通过。两类backbone都需在macro MSE/MAE上使+ISCF-BSCA优于Original且至少赢3/5 dataset MSE，才允许“decoder transfers across backbones”。若只通过一个block，结论收窄到该backbone；若+ISCF有效而BSCA无增益，只支持ISCF decoder transfer；numeric/resource pathology只回滚stable shared profile，不否定方向。

## 最小诚实验证

本地prelaunch已验证13项：profile hash、30-row唯一matrix、joint-training governance、六个arm的shape/finite/exact-prefix contract、两类backbone各自的matched encoder initialization、runner syntax与30-job dry-run。结果位于`local_prelaunch/prelaunch_checks.csv`与`local_prelaunch/prelaunch_summary.json`。


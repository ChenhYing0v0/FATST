# H5D `bs16_lr2p4` Main-Table Sync

## Decision

用户于2026-08-13明确要求暂时以`ETTh1__h5d_bs16_lr2p4`固定当前论文结果。该profile通过H5D预注册的mean MSE/MAE双guard，但没有严格提高原H5D primary best-cell objective。因此，本次变更是author-selected paper-row override，不改写H5D原始gate-fail及retain-H5B结论。

## Atomic replacement

- Main I与Main II只替换ISCF-BSCA的ETTh1 H96/H192/H336/H720四个cells。
- ECL、Solar、其余datasets、所有baseline数值与source roles保持不变。
- ETTh1 four-H mean MSE/MAE为`0.391582/0.416836`。
- Main I为`31/56` best、`18/56` second。
- Main II为`30/56` best、`25/56` second；ISCF-BSCA seven-dataset macro MSE/MAE为`0.261911/0.307252`，两项均rank 1。

## Evidence boundary

该结果是single-seed、dataset-level、test-tuned且test-informed的paper-facing system evidence。一个profile同时服务four horizons；没有per-horizon、per-metric或per-cell selection。它不提供BSCA或decoder的matched mechanism attribution。

Machine audit为`h5d_bs16_lr2p4_main_table_sync_audit.json`；两张表分别由各自freeze manifest记录artifact与PDF hashes。

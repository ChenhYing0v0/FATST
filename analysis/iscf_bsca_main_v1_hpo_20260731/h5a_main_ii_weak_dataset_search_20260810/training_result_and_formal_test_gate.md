# H5A Training Result and Formal-Test Gate

## 1. Training completion

H5A full queue于2026-08-12 20:24:03结束。冻结的48个train/validation jobs全部完成：

- ECL=`16/16`，Solar=`16/16`，ETTh1=`16/16`；
- checkpoint、training log、four-H validation metrics、effective config、initialization
  contract、model diagnostics和environment artifacts均为`48/48`；
- training阶段official-test artifacts=`0/48`；
- OOM/NaN/Inf/Traceback/RuntimeError=`0`；
- remote queue commit=`7544f76d`，config SHA256=
  `fb1fb57927033b50e04cdc85dc05f500cf2c6a9d11baf9cfff59c83dd09c1034`，
  search-space SHA256=`e7164c22853435308103e8f3d5212ff1e9a678b09802861fae09a79d3392066d`。

## 2. Artifact and selector audit

Generic analyzer重新物化全部48个frozen jobs并生成48-row ledger。逐trial checker核对了
effective identity、H720 training与four-H validation、`best-val` checkpoint policy、
architecture/HPO hashes、effective hyperparameters、best epoch、parameter count和checkpoint
SHA256。48个checkpoint hashes全部唯一且test前未变更。

导出的four-H metrics与training log aggregate通过不同float32 reduction path写出。最大绝对差
为`1.5339e-7`（`ETTh1__h5a_lr4e4`，MSE约1.138），其余47项更小；best epoch identity
不变。这是sub-ULP serialization difference，不是selector drift。Checker使用冻结的
`2e-7` absolute tolerance，仍比table/display和HPO selection margin严格多个数量级。

Training-artifact gate=`pass`。

## 3. Immutable checkpoint manifest

Manifest=`analysis/iscf_bsca_main_v1_hpo_20260731/h5a_checkpoint_manifest.csv`：

- rows=`48`；dataset distribution=`16/16/16`；seed=`2021`；
- unique trial IDs=`48`；unique checkpoint hashes=`48`；
- SHA256=`ee5940c8f66aceab5710f17a4bc8ce2efb9ae3c44fa9cec1459fcd9589fe6643`；
- test target和temporary artifacts在freeze时均为0。

Formal-test contract=
`configs/iscf_bsca_main_v1_hpo_main_ii_h5a_test_audit.json`，固定48 checkpoints × four
standard horizons=`192` rows，同时保存dense 1--720 diagnostic metrics。Test过程中禁止
checkpoint retraining/mutation、partial selection、per-H/per-metric/per-cell/seed selection。

## 4. Authorization and next action

用户于2026-08-13明确要求在remote training完成后继续formal test。该授权只覆盖一次完整
H5A test audit；H5B、extra seeds、selected-profile confirmation以及自动修改Main I/Main II
仍未授权。

Decision=`H5A_training_complete_48_unique_checkpoint_manifest_frozen_formal_test_authorized`。

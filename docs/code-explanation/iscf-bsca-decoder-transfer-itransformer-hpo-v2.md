# iTransformer-style Decoder-HPO v2 代码说明

## 1. 修改目的

v1把SIFF decoder固定为5个scope、4维future coordinate、32×64 allocation MLP和dataset-matched mode rank。Formal结果显示replacement readout在iTransformer-style memory上整体落后native direct decoder。本次修改只把这些decoder构造量变成HPO输入，不改变旧protocol默认行为。

## 2. Tensor flow

iTransformer encoder保持：

1. `x [B,L,C]` 经RevIN normalization；
2. 转置为variates并编码为 `memory [B,C,1,D]`；
3. flatten得到 `hidden [B,C,D]`。

SIFF decoder保持相同forward path：

1. `hidden [B,C,D]` 与 `mode_weight [S,D_q,D,K]`产生component modes `M [B,C,S,D_q,K]`；
2. 每个scope使用其grouped future coordinates和共享synthesis tensors生成 `arms [B,C,S,720]`；
3. direct allocation policy从`hidden`与future coordinates生成 `weights [B,C,720,S]`；
4. scope加权得到full trajectory `[B,C,720]`，请求H只返回前H步。

其中 `S=len(pcsd_scales)`；independent SIFF的`scale_components`在构造时自动等于S。因此修改scope列表不会破坏component与scope的一一对应。

## 3. 代码变化

### `baselines/timealign_official/train_repo.py`

- 新增CLI参数`--pcsd-scales`，默认仍为`1,48,144,360,720`；
- `build_official_args`把scope列表传入模型config；
- `--pcsd-fixed-scale`由静态choices改为运行时检查其必须属于scope列表；
- 全局检查scope必须唯一、递增、为720的正因子；
- 只有protocol profile `iscf_bsca_decoder_transfer_itransformer_hpo_v2_20260816`允许搜索coordinate与policy dimensions，其他protocol继续强制4、32、64。

### `baselines/timealign_official/models/TimeAlign.py`

- SIFF构造读取`configs.pcsd_scales`；
- 对`scale_basis_mode=independent`，令`scale_components=len(pcsd_scales)`；
- 其余readout、encoder与denormalization flow不变。

### Experiment tooling

- `configs/iscf_bsca_decoder_transfer_itransformer_hpo_v2.json`冻结14 profiles、70 runs、test-tuned selector与rollback；
- `scripts/remote/run_iscf_bsca_decoder_transfer_itransformer_hpo_v2.sh`负责三GPU training-only queue；
- `scripts/check_iscf_bsca_decoder_transfer_itransformer_hpo_v2_prelaunch.py`验证结构、optimizer group、prefix与runner；
- `scripts/check_iscf_bsca_decoder_transfer_itransformer_hpo_v2_artifacts.py`在test access前验证70 artifacts、70 hashes、four-H validation selector与dataset-level matched encoder initialization。

## 4. Code-theory consistency

预期理论是：iTransformer variate-token representation可能需要不同的decoder capacity、future basis resolution、allocation capacity或scope geometry，才能被ISCF decoder有效读取。代码确实在`hidden -> SIFF modes -> scope forecasts -> allocation fusion`路径内改变这些量，而没有修改encoder或引入native residual shortcut。

仍然只是proxy的部分：HPO能够发现兼容配置，但不能单独证明哪个轴是机制性原因；多个参数可能存在interaction，且test-tuned selection不能作为untouched generalization evidence。若selected BSCA超过Original，仍需用同profile训练matched +ISCF，才能区分decoder compatibility与BSCA objective贡献。

可证伪条件：完整70-profile test scorecard中没有dataset-level shared profile改善v1 BSCA，或best-tuned profile仍无法在macro MSE/MAE和3/5 dataset MSE wins上超过Original Decoder。

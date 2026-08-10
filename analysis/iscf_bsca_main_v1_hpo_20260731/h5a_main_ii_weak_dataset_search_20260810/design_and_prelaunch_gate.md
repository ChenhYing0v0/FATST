# H5A Main II Weak-Dataset HPO：Design and Prelaunch Gate

## 0. Decision summary

用户于2026-08-10显式重启`ISCF-BSCA-MAIN-v1` HPO，并把范围限定为
`ETTh1/ECL/Solar`，目标是提高Main II中的best-setting数量。H5A不改architecture、
objective、scope set或inference graph；它是single-seed、dataset-level、
`test_tuned/test_informed` hyperparameter search。冻结矩阵为48个from-scratch joint
training jobs（每dataset 16个），训练阶段official test=`0/48`。用户同时授权remote
resource smoke、完整training，以及48/48 immutable checkpoint manifest通过后的一次
完整formal test；H5B、extra seeds和自动修改Main I/Main II表格均未授权。

Decision=`H5A_48_profile_targeted_HPO_frozen_remote_training_and_post_manifest_formal_test_authorized`。

## 1. Problem and current evidence

冻结Main II表使用一个H720 checkpoint裁剪四个prefix。按共同三位小数规则，当前
ISCF-BSCA在三个目标datasets的8个MSE/MAE cells中分别为：

| Dataset | Best | Second | 主要缺口 |
| --- | ---: | ---: | --- |
| ETTh1 | 1/8 | 3/8 | 只有H720 MSE best；四个MAE均非top-2 |
| ECL | 0/8 | 8/8 | 全部cells为second，最接近形成稳定best增量 |
| Solar | 4/8 | 3/8 | 四个MAE best；四个MSE均未best，H720 MSE最弱 |

全局Main II当前为24/56 best、27/56 second。本轮不选择性丢弃任何horizon、metric
或negative trial，也不允许为某个cell单独选择profile。

对H1--H4M 189-trial scorecard的重审给出两个重要边界：

- `Solar__h4j_patch4_lr2e4`已有5/8 displayed best，且four-H mean MSE/MAE仅比当前
  profile差0.225%/0.095%，因此在新0.5% aggregate guard内可重新竞争；
- `ECL__h2_lookback336`可得到1/8 displayed best，但mean MSE退化超过0.5%，因此即使
  增加一个rounded tie也不可选。

## 2. Hyperparameter impact and frozen matrix

历史结果显示，本轮扩大的是已观察到高影响或明显欠搜索的轴，而不是机械Cartesian
product：

- **ETTh1**：learning rate呈明显U形，`3e-4`邻域最好；小capacity显著优于大capacity；
  context/patch interaction仍欠搜索。因此16个profiles覆盖`lr=2.5e-4--4e-4`、
  `dropout=0--0.15`、`L=336/512/720`与patch interaction、`d_model/d_ff`下边界、
  weight decay和LayerNorm control。
- **ECL**：`d_model=512,d_ff=2048,dropout=0.5,lr=5e-4,rank=64`是当前强anchor；
  patch granularity几乎未搜索。16个profiles优先覆盖`patch=2/4/8/12`、
  context×patch、`lr=7e-4/1e-3`、dropout/rank/weight-decay边界和budget extension。
- **Solar**：patch×learning-rate是最强影响轴，rank可能改善MSE但会牺牲MAE。
  16个profiles覆盖`patch=2/4`、`lr=1.5e-4--3.5e-4`、rank32/64/128、
  dropout、context和capacity interactions，并保留当前MAE frontier的budget control。

完整机器合同为
`configs/iscf_bsca_main_v1_hpo_main_ii_h5a.json`。所有jobs使用seed2021、H720
training、four-H validation mean MSE选择checkpoint、最多90 epochs/patience18；一个
dataset最终只能选择一个profile共同服务H96/192/336/720。

## 3. Main II selector and gates

Primary selector在完整official-test结果上执行：

1. 相对冻结的七个external systems，按共同three-decimal display计算每个trial的
   8个best metric cells；
2. 只有four-H mean MSE和mean MAE都不超过当前profile的`1.005×`才eligible；
3. 按best cells降序选择，依次以top-2 cells、normalized regret、mean MSE、mean MAE、
   validation MSE、parameter count和profile ID打破平局；
4. 若没有eligible profile提高best count，保留当前profile。

最低目标为ETTh1/ECL/Solar分别达到`2/8,1/8,5/8`，即目标三dataset从5提高到至少
8个best cells；若其他datasets不变，全局由24/56提高到至少27/56。Solar四个MAE best
不得减少。以上是ambitious HPO gate，不是结果保证或SOTA claim。

## 4. Execution, resources and rollback

- Estimated compute：40--80 GPU-hours；三张3090预计14--28 wall-hours。
- Queue：先ECL，再Solar，最后ETTh1；三个workers动态取队列，避免慢dataset让GPU空闲。
- Storage budget：16 GiB；默认root=
  `/home/yingch/exp_outputs/r-2026-fatst/iscf_bsca_main_v1_hpo/h5a`。
- Prelaunch顺序：remote `nvidia-smi`/disk/git audit -> 48/48 resource smoke ->
  48-job no-test training -> artifact analyzer -> immutable manifest -> complete formal test。
- 任一OOM/numeric/artifact failure：只修复runtime/profile feasibility并重跑完整affected
  block；不得用partial结果选profile。
- 若没有eligible best-count improvement：保留当前dataset profile并关闭H5A，不修改表格。
- 若需要architecture/objective change：建立新candidate并回到Step 4--6 narrative/design gate。

## 5. Prelaunch verification

本地checker验证了：48 jobs、16/16/16 dataset distribution、frozen source hashes、当前
`1/0/4` best counts、189 historical trials、one-profile-per-dataset selector、training
test=0、formal-test manifest gate和generic runner dry-run。训练artifact checker与manifest
builder已经落地，但只有远程48/48训练完成后才可执行。

Local gate=`pass`；remote resource gate=`pending`；training=`not_started`；formal
test=`blocked_until_48_of_48_manifest`。

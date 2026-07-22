# ISCF-FRSC Step7B 执行链说明

## 1. 运行矩阵与边界

`configs/stage_c_iscf_frsc_step7b.json`冻结四个new-training arms：

- `frsc_scope_a055`：canonical scope projector，$\alpha=0.55$；
- `frsc_global_a055`：same-alpha global control；
- `frsc_global_a045`：frozen-diagnostic选择的best global control；
- `frsc_random_a055`：random scope-binding control。

五个数据集与seed2021形成20个new runs。历史`SPS identity`的5个checkpoints只作为hash-pinned effectiveness
reference，因此完整analysis surface为25 runs和100个dataset-arm-horizon validation rows。runner硬限制
`EVALUATION_SPLIT=val`；formal test、confirmation seeds与modern baselines均未开放。

## 2. Runner 的tensor与配置流

`run_stage_c_iscf_frsc_step7b.sh`只设置FRSC专属的config、output root、protocol profile和run label，然后复用
SPS Step7B runner。每个job把dataset profile中的`patch_num/d_model/d_ff`、dataset-matched `mode_rank`以及arm的
`projection_mode/partition/conditioning_strength`传给`train_repo.py`。

model forward保持既有ISCF路径：history encoder产生`hidden [B,C,R]`；五个raw arms为`[B,C,S,T]`；FRSC对第$s$
个arm施加$Q_s=P_s+(1-\alpha)(I-P_s)$，得到conditioned arms `[B,C,S,T]`；既有direct policy权重
`[B,C,T,S]`融合为forecast `[B,T,C]`。$\alpha<1$保证minimum eigenvalue为$1-\alpha>0$，不会删除future direction。

## 3. Evaluator artifacts

validation checkpoint evaluator保存：

- `probe_sps_raw_arms [N,S,T]`、`probe_sps_projected_arms [N,S,T]`与
  `probe_sps_removed_arms [N,S,T]`；FRSC语境下`removed`表示conditioning delta，不表示被删除的rank；
- `probe_arms [N,S,T]`、`probe_fused [N,T]`、`probe_targets [N,T]`和
  `probe_direct_policy [N,T,S]`；
- standard-horizon MSE/MAE、future-bin arm errors、prefix consistency以及checkpoint/config hashes。

`trained_invariants.json`新增`frsc_conditioning_strength`、`frsc_minimum_operator_eigenvalue`和`frsc_full_rank`。
readout contract同时核对CLI strength、$1-\alpha$与strictly positive eigenvalue。

## 4. Analyzer统计与决策

analyzer将四个new roots与历史identity root统一审计，定义：arm pairwise normalized RMS、oracle headroom、policy
normalized entropy、future-bin winner count以及conditioning-delta/raw RMS。performance gate比较candidate与identity；
mechanism attribution分别比较best global、same-alpha global和random binding。internal health不能覆盖negative
effectiveness，random negative也不能单独方向级否定ISCF。

## 5. Code-theory consistency

代码实现了预注册的full-rank conditioning与matched controls，且没有新增loss、router或requested-H input。当前仍只是
Step7B execution contract；只有完整validation artifacts返回后，才能判断FRSC-v0 effectiveness与scope-specific
attribution。validation positive也不等于formal-test或paper-core pass。

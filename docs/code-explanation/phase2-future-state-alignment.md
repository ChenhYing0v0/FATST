# Phase2 Future-State Alignment 代码说明

`PatchEncoderFutureStateAlignment` 是在
`baselines/patch_encoder_target_set_decoder` 中实现的 Phase2-A 候选。它复用
`PatchEncoderPrefixRiskWeighted` 的 target-set carrier，不改变 inference prediction path；
新增部分只在 training/diagnostic 中使用 ground-truth future。

## Forward 数据流

基础预测路径保持 R.3 不变。输入 history：

$$
x \in \mathbb{R}^{B \times L \times C}
$$

经 RevIN、patch embedding 与 Transformer encoder 得到：

$$
Z=E_\theta(x)\in\mathbb{R}^{(BC)\times N\times d}.
$$

给定 requested horizon $H$，模型构造 target segment queries：

$$
Q_T\in\mathbb{R}^{(BC)\times J\times d}.
$$

target-to-history cross-attention 得到 target-side state：

$$
U_T=D_\theta(Q_T,Z)\in\mathbb{R}^{(BC)\times J\times d}.
$$

随后 $U_T$ 生成 FiLM 参数：

$$
(\gamma_j,\beta_j)=F_\theta(U_j),
$$

并调制 dense history readout：

$$
r=R_\theta(\operatorname{Flatten}(Z)),
$$

$$
\hat{Y}_{a_j:b_j}
=
O_\theta(r\odot(1+\gamma_j)+\beta_j).
$$

这一条路径不读取 `future_y`。因此 evaluation/inference 只依赖 history `x` 与 requested
target horizon。

## Future Teacher Branch

当 `future_y` 被传入且 `--future-teacher-layers > 0` 时，模型额外构造 training-only
future teacher。首先使用当前 RevIN 统计把 ground-truth future 归一化：

$$
Y^{norm}_{1:H}=\frac{Y_{1:H}-\mu_X}{\sigma_X}.
$$

然后按同一个 `segment_len` 切分：

$$
Y^{norm}_{seg}\in\mathbb{R}^{(BC)\times J\times S}.
$$

teacher state 由三部分相加后进入轻量 Transformer encoder：

1. `future_segment_embedding(Y_segment)`;
2. `future_feature_embedding(q_j)`;
3. `future_pos_embedding[j]`.

得到 future-side teacher state：

$$
S^Y_T\in\mathbb{R}^{(BC)\times J\times d_s}.
$$

同时从 inference-time target state 得到 student state：

$$
S^X_T=P_\theta(U_T)\in\mathbb{R}^{(BC)\times J\times d_s}.
$$

teacher branch 还输出 normalized future reconstruction：

$$
\tilde{Y}^{norm}_{1:H}=R_\psi(S^Y_T).
$$

该 reconstruction 用于训练 teacher，使 teacher state 不是随机 anchor。

## Loss

主预测 loss 仍是 R.3 的 prefix-risk weighted MSE：

$$
\mathcal{L}_{pred}
=
\frac{1}{BHC}
\sum_{b,t,c}
w_t(\hat{y}_{b,t,c}-y_{b,t,c})^2.
$$

Phase2-A 额外加入三项：

Local alignment:

$$
\mathcal{L}_{local}
=
1-\cos(S^X_T,\operatorname{sg}(S^Y_T)).
$$

Relation alignment:

$$
\mathcal{L}_{rel}
=
\left\|
\operatorname{sim}(S^X_T)-
\operatorname{sg}(\operatorname{sim}(S^Y_T))
\right\|_F^2.
$$

Teacher reconstruction:

$$
\mathcal{L}_{recon}
=
\operatorname{MSE}(\tilde{Y}^{norm}_{1:H},Y^{norm}_{1:H}).
$$

总 loss：

$$
\mathcal{L}
=
\mathcal{L}_{pred}
+
\lambda_{local}\mathcal{L}_{local}
+
\lambda_{rel}\mathcal{L}_{rel}
+
\lambda_{recon}\mathcal{L}_{recon}.
$$

第一版候选使用：

- `--step-loss-weighting prefix_risk`;
- `--step-loss-alpha 0.5`;
- `--future-teacher-layers 1`;
- `--future-align-weight 0.02`;
- `--future-relation-weight 0.01`;
- `--future-recon-weight 0.001`.

## Leakage Boundary

`future_y` 只影响 `future_*` components 和 future losses。预测张量 `prediction` 在 teacher
branch 之前已经由 $U_T$、$\gamma$、$\beta$ 和 history readout 得到；后续 teacher branch
不会回写这些张量。

训练脚本在每个 horizon 的 test split 写出 `future_leakage_audit.json`。audit 比较三种
调用方式：

1. 不传 `future_y`;
2. 传真实 `future_y`;
3. 传 shuffled `future_y`;
4. 传 zero `future_y`.

若这些调用得到的 `prediction` 存在最大绝对差异大于 `1e-7`，该候选直接失败。

## Artifact 语义

新增文件：

- `h{H}/future_alignment_stats.csv`
  - `future_local_alignment_loss`: local teacher/student cosine loss；
  - `future_relation_alignment_loss`: target segment relation alignment loss；
  - `future_reconstruction_loss`: normalized future reconstruction MSE；
  - `teacher_student_cosine`: teacher/student cosine；
  - `prediction_leakage_max_abs`: leakage audit 汇总最大值。
- `h{H}/future_leakage_audit.json`
  - `true_future_prediction_max_abs`;
  - `shuffled_future_prediction_max_abs`;
  - `zero_future_prediction_max_abs`;
  - `prediction_leakage_max_abs`.

`training_log.csv` 新增：

- `train_prediction_loss`;
- `train_future_local_alignment_loss`;
- `train_future_relation_alignment_loss`;
- `train_future_reconstruction_loss`.

## Code-Theory Consistency

[Intended theory] Phase2-A 要验证 target-set decoder 的 $U_T$ 是否应被 training-only
future state 校准。核心 claim 不是“用 future 做辅助任务”，而是“future teacher 作用在真正
控制 decoder readout 的 target-side state 上”。

[Code realization] `future_student_projection` 读取 `target_states`；`future_teacher_encoder`
读取 normalized future segments 和同一 target segment feature；alignment loss 用
stop-gradient teacher 约束 student，reconstruction loss 训练 teacher。

[Proxy] teacher state 仍是 learned latent，不是可直接解释的物理 future state。若
alignment loss 降低但 MSE/MAE 不改善，则它只是 auxiliary proxy。

[Falsification] 若 `prediction_leakage_max_abs > 1e-7`、prefix mismatch 不再是数值零级别、
或 vs R.3 没有稳定提升，则 Phase2-A 不满足 paper-core candidate pass，应回退到
future-aware problem definition 或 covariance-aware objective。

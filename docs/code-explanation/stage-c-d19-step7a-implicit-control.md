# Stage C D19 Step 7A Implicit Control Code Explanation

## 1. Forward tensor path

入口仍是A6 natural carrier：

$$
X[B,720,C]\xrightarrow{Normalize}X_n[B,720,C]
\xrightarrow{A6\ Encoder}M[B,C,P,D]
\xrightarrow{flatten}h[B,C,R].
$$

`baselines/timealign_official/layers/ImplicitForecast.py`新增两类readout。

### ImplicitFrequencyReadout

1. `history_spectrum()`：
   `X_n[B,720,C] -> rFFT -> amplitude/phase[B,C,361]`；
2. `polar_spectrum()`：
   amplitude head读取`[h,A_x]`，sine/cosine heads读取`[h,Phi_x]`；
3. amplitude使用`abs(leaky_relu(..., 0.5))`；
4. phase使用`atan2(tanh(sine), tanh(cosine))`；
5. `full_forecast()`：
   `polar -> complex spectrum[B,C,361] -> irFFT(n=720) -> [B,C,720]`；
6. `forward()`只在full synthesis之后crop，并返回`[B,H,C]`。

`use_input_spectrum=false`不会删除任何module，只把history amplitude/phase替换为zero，因此IF/no-skip可共享
exact decoder initialization。

### DirectNonlinearMatchedReadout

1. 同样从720-point normalized history得到amplitude/phase；
2. 拼接`[h,A_x,Phi_x]`；
3. two-layer GELU MLP直接输出`[B,C,720]`；
4. full output之后crop为`[B,H,C]`。

## 2. Model integration

`baselines/timealign_official/models/TimeAlign.py`新增三种mode：

- `implicit-frequency-readout`；
- `implicit-frequency-noskip-control`；
- `implicit-direct-nonlinear-matched`。

它们复用相同`memory -> hidden` Encoder path，将同一`normalized_history`显式交给readout，然后继续走既有
`Normalize(..., "denorm")`。requested horizon不进入head或frequency construction，只用于最终crop。

## 3. Training and CLI integration

`baselines/timealign_official/train_repo.py`：

1. 将三种readout加入active prefix readouts；
2. 增加IF hidden width、matched-direct width、dropout与FFT norm配置；
3. 强制D19使用`seq_len=pred_len=720`、IF width2048、dropout0.1与orthonormal FFT；
4. 根据`patch_num * d_model`强制direct width为4143、4659或5164；
5. initialization artifact新增IF/no-skip/direct decoder hashes；
6. diagnostics记录history/spectrum bins、params、width、dropout、FFT norm和skip状态。

训练objective不新增特殊分支。D19继续使用既有非coupling
`pcc_objective_mode=measure_only`路径，因此和`A6_MEASURE`共享相同harmonic-L1 risk。

## 4. Gate outputs

`scripts/check_stage_c_d19_if_control_step7a.py`输出：

- `manifest.csv`：15个未来training jobs；
- `local_gate_cases.csv`：每个case的来源、值、threshold与pass；
- `gate_summary.json`：category与authorization摘要。

每个新统计量的定义：

- `prefix gap`：单独forward的H-prefix与同module full720 output前H行的maximum absolute difference；
- `prediction NRMSE`：
  $\sqrt{\operatorname{mean}(Y_a-Y_b)^2}/\sqrt{\operatorname{mean}(Y_{IF}^2)}$；
- `gradient norm`：指定parameter group所有gradient squared sum的平方根；
- `parameter gap percent`：
  $100|N_{IF}-N_{direct}|/N_{IF}$；
- `phase radius minimum`：
  $\min\sqrt{\hat s^2+\hat c^2}$，只用于near-zero numeric probe。

## 5. Code-theory boundary

代码已经实现matched information、matched initialization、full-trajectory projectivity与capacity control；
尚未证明IF、skip或wave synthesis提高forecast accuracy。Step7B必须先通过real-batch finite/resource smoke和
formal matrix audit，才能授权remote experiment。

## 6. Step 7B formal evaluation path

`configs/stage_c_d19_if_control_step7b.json`把正式矩阵冻结为5 datasets × 4 arms × seed2021，
其中15个D19 arms重新训练，5个`A6_MEASURE`复用已按相同four-horizon validation selector训练的reference。
正式scorecard为80个dataset-arm-horizon test cells；test不参与checkpoint选择。

`scripts/evaluate_stage_c_pcsd_cf_checkpoint.py`在不修改checkpoint的test forward中新增：

- `probe_if_amplitude[Q,361]`；
- `probe_if_phase_sine[Q,361]`；
- `probe_if_phase_cosine[Q,361]`；
- `probe_fused[Q,720]`与逐H1..H720累计MSE/MAE。

`scripts/analyze_stage_c_d19_if_control.py`分四层判定：

1. `paper_facing_effectiveness`：IF是否超过A6；
2. `matched_mechanism_attribution`：IF是否同时超过parameter-matched direct与no-skip；
3. `internal_mechanism_health`：finite、projectivity、paired initialization、prediction deformation、
   amplitude与phase是否非退化；
4. `failure_attribution`：映射到Step2/4、Step4或Step6/7A rollback。

即使四层全部通过，D19仍固定为`control_only`；Implicit Forecaster prior禁止其被直接晋升为论文方法。

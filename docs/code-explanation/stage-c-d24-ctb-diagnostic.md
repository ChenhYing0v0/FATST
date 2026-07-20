# Stage C D24 CTB Diagnostic 代码说明

## 1. 实现边界

`scripts/analyze_stage_c_d24_conditional_trajectory_bias.py`只读取D23冻结checkpoint并执行validation inference。
它不训练或修改模型，不读取official test，不实现router、loss、adapter或paper method。

配置文件为`configs/stage_c_d24_conditional_trajectory_bias.json`。运行级输出包含`metrics.csv`与
`metadata.json`；aggregate输出包含cell、summary、run audit、decision与Markdown report。

remote runner为`scripts/remote/run_stage_c_d24_conditional_trajectory_bias.sh`。它在启动前拒绝
`remote_training_authorized=true`、`official_test_access_authorized=true`或非validation split，记录commit、
config hash与GPU状态，再对2 arms × 5 datasets执行冻结checkpoint inference。

## 2. Tensor与统计流

每个validation batch：

```text
batch_x                      [B, 720, C]
target                       [B, 720, C]
frozen forecast              [B, 720, C]
history_rows                 [B*C, 720]
future residual              [B*C, 15, 48]
marginal features            [B*C, 4]
ordered history means        [B*C, 24]
residual block sum / SSE      [B*C, 15]
```

`history_rows`先逐row标准化，再按24个连续30-step blocks求mean。`sorted_history`逐row排序同一24 values，
保留marginal multiset并销毁temporal order。channel one-hot、marginal、recent、ordered与sorted features均从
forecast origin前的history构造。

## 3. Chronological contract

所有channels按forecast origin分组：

- first third拟合ridge map；
- middle third不参与任何拟合或评价；
- last third计算MSE；
- official test path不存在。

这比256-row interleaved probe更严格，但仍只属于validation diagnostic。

## 4. Exact error update

map预测每个48-step future block的constant correction。脚本不保存full predictions，而用residual sum与SSE精确
恢复corrected MSE：

$$
\operatorname{SSE}'=\operatorname{SSE}-2a\sum r+48a^2.
$$

H96/H192/H336/H720分别聚合2/4/7/15 blocks。`target_shuffled`在forecast-origin层面打乱fit residual，
保持feature width与ridge capacity。

## 5. Code-theory consistency

- Intended theory：检验strong fixed trajectory carrier是否遗漏raw-history可识别的coarse output freedom。
- Code realization：同一frozen forecast上比较ordered raw-history map与global、marginal、recent、sorted及
  target-shuffled controls。
- Proxy boundary：48-step constant deformation与linear ridge只是低成本probe，不等价于完整
  history-conditioned trajectory operator。
- Falsification boundary：negative只能关闭exact probe；positive也只能返回Step4，不能promotion为method。

## 6. Verification

最小验证包括：

- config JSON parse；
- `python -m py_compile`；
- synthetic ridge recovery；
- exact corrected-SSE identity；
- aggregate missing-run/refusal与official-test-disabled contract。

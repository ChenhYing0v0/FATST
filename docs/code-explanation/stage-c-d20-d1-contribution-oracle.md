# Stage C D20-D1 Contribution Oracle 代码说明

## 1. 诊断边界

`scripts/analyze_stage_c_d20_contribution_oracle.py`只读取D20 official-test probe，不训练、不修改checkpoint，也不
产生method gate。它属于`test_informed diagnostic_only_posthoc`，用于区分summary contribution的方向与幅度问题。

## 2. Tensor construction

每个SPEC/RANDOM run读取：

```text
probe_fused                           [256, 720]
probe_targets                         [256, 720]
probe_history_prediction_contribution [256, 720]
```

在同一个jointly trained model内部定义：

$$
\hat y_{base}=\hat y_{fused}-c,\qquad
\hat y(\alpha)=\hat y_{base}+\alpha c.
$$

其中$c$是summary coefficient path经learned basis生成的prediction contribution。$alpha=1$是实际模型；
$\alpha=0$只移除summary path。该base与A6不是同一个模型，只能作co-adapted within-model attribution。

## 3. Oracle alpha

每个dataset和future region上，以official-test label计算closed-form MSE-optimal scale：

$$
\alpha^*=\frac{\langle c,y-\hat y_{base}\rangle}{\langle c,c\rangle}.
$$

- actual gain比较$\alpha=1$与$\alpha=0$；
- oracle gain使用$\alpha^*$，只表示headroom；
- clipped gain使用$\alpha\in[0,1]$，检查简单shrinkage是否足够；
- cosine衡量contribution与base residual的方向一致性。

test label参与$\alpha^*$，所以oracle数值禁止用于选择新模型、超参数或paper effectiveness，只能做failure
attribution。

## 4. Interpretation

- actual gain为正：summary path在其co-adapted base上有条件增益；
- actual gain为负但$0<\alpha^*<1$：方向有用但实际注入过强，怀疑scale miscalibration；
- $\alpha^*\leq0$：该region的contribution方向错误；
- contribution有益但完整SPEC仍差于same-run A6：问题落在base/Encoder co-adaptation、冗余shortcut或整体
  generalization，而非简单删除summary path即可修复。

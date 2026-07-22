# Stage C ISCF-SCC D0B Diagnostic Code Explanation

## 1. 功能边界

`scripts/analyze_stage_c_iscf_scc_d0b.py`只读取D0 frozen validation NPZ，不加载或修改checkpoint，不调用forecast
training loop，也不访问test。ridge只属于function-level information-access probe，不是candidate router实现。

## 2. Tensor flow

输入为`arms [R,5,720]`、`policy [R,720,5]`、`fused/target [R,720]`。脚本先按D0公式构造
`coalition_credit [R,720,5]`与`standalone_credit [R,720,5]`。

target-free features为`features [R,720,18]`：五个coordinate-normalized arm deviations、五个policy weights、
`log_arm_dispersion`、`fused_forecast`、policy entropy、normalized position与两组sine/cosine position basis。

前60% probe rows拟合固定ridge，后40% rows评价。预测值clip到nonnegative simplex，再与held-out arms重组为forecast。

## 3. Controls and outputs

standalone-credit使用完全相同的feature、split和ridge。16个shuffle controls在每个future position内置换training-row
credit vector，保留horizon marginal和scope joint distribution，只破坏feature-credit binding。

输出`run_metrics.csv`、`dataset_summary.csv`和`decision.json`；每个新列的source、计算与gate语义在
`d0_result_and_d0b_information_access_plan.md`第3–5节定义。

## 4. Code–theory consistency

- intended theory：train-only coalition credit若要校准inference policy，必须有target-free可预测分量；
- realized code：只用现有forward tensors和position做blocked held-out linear probe；
- proxy：linear ridge低估nonlinear learnability，256 rows也不是完整validation distribution；
- falsification boundary：negative只拒绝当前feature/intervention的直接policy calibration，不否定ISCF architecture或所有
  cooperative training designs。

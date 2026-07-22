# Stage C ISCF-SCC D0B Diagnostic Code Explanation

## 1. 功能边界

`scripts/analyze_stage_c_iscf_scc_d0b.py`只读取D0 frozen validation NPZ，不加载或修改checkpoint，不调用forecast
training loop，也不访问test。ridge只属于function-level information-access probe，不是candidate router实现。

## 2. Tensor flow

输入为`arms [R,5,720]`、`policy [R,720,5]`、`fused/target [R,720]`。脚本先按D0公式构造
`coalition_credit [R,720,5]`与`standalone_credit [R,720,5]`。

target-free features为`features [R,720,18]`：五个coordinate-normalized arm deviations、五个policy weights、
`log_arm_dispersion`、`fused_forecast`、policy entropy、normalized position与两组sine/cosine position basis。

约前60% probe rows拟合固定ridge，后40% rows评价；边界向下对齐dataset channel count，保证同一multivariate
source-sample的channel rows不跨probe sides。当前15 runs均为147/109 rows。预测值clip到nonnegative simplex，再与
held-out arms重组为forecast。

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

## 5. SCC-v0 Step7A training objective

`layers/PCC.py`新增`scope_coalition_credit`与`scope_coalition_credit_shuffled`两种objective mode。输入沿用
`fused [B,T,C]`、`arms/policy [B,C,T,5]`和`target [B,T,C]`，closed-form removal输出
`route_credit [B,C,T,5]`。signed gain与normalized positive credit均stop-gradient；因此route KL只更新existing policy，
arms仍只通过fused harmonic L1更新。

all-nonpositive coordinate回退uniform。SHUFFLED mode用独立`torch.Generator`逐coordinate置换scope axis，保持credit
values/simplex不变且不消费data/model global RNG。`train_repo.py`只负责创建seeded generator并记录effective contract；
model forward与inference文件未修改。

`scripts/check_stage_c_iscf_scc_step7a.py`检查exact formula、无individual arm loss、route-gradient boundary、uniform
fallback、shuffle marginal preservation/reproducibility与global-RNG isolation。

training loop在backward后、optimizer step前记录independent `mode_weight/mode_bias`的五个per-scope gradient norms。
该日志只观测existing arm gradient path，不改变loss或optimizer；用于验证至少三个scope在E2E training中持续获得非零更新。

## 6. RSCC-v1 reliability-preserving modes

SCC-v0 Step9证明删除equal-skill会破坏arm reliability，因此新增`equal_scope_coalition_credit`与对应shuffled mode。
它们只改变objective composition：`skill_kind=equal`保持与parent完全相同的uniform individual arm L1，
`route_kind=coalition`复用同一detached SCC credit和`.1` route schedule。model/inference仍不变。

Step7A checker新增两项等价检查：RSCC `skill_loss`必须逐值等于EQUAL parent；SHUFFLED必须同时保持该skill loss与
coalition credit marginals。由此确保下一轮只测试“reliability-preserved coalition binding”，不重新改变architecture。

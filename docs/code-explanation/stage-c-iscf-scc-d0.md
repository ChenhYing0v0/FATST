# Stage C ISCF-SCC D0 Diagnostic Code Explanation

## 1. 功能边界

本次只新增frozen-checkpoint validation replay与offline analysis，不修改model、training objective或checkpoint。
remote runner硬编码`evaluation_split=val`，配置同时保持`new_training_authorized=false`和
`formal_test_access_authorized=false`。

## 2. Artifact construction

`scripts/remote/run_stage_c_iscf_scc_d0.sh`读取15个existing ISCF-v0 checkpoints，使用已有
`evaluate_stage_c_pcsd_cf_checkpoint.py`在validation split保存：

- `probe_arms [256,5,720]`；
- `probe_fused [256,720]`；
- `probe_targets [256,720]`；
- `probe_direct_policy [256,720,5]`。

新artifact写入repo-external D0 root，source checkpoint前后执行SHA256 nonmutation检查。三张GPU只负责forward
evaluation，不发生optimizer step。

## 3. Analysis flow

`scripts/analyze_stage_c_iscf_scc_d0.py`先验证
$\hat y=\sum_sp_sa_s$的reconstruction gap，再计算renormalized leave-one-scope-out forecast：

$$
\hat y_{-s}=\frac{\hat y-p_sa_s}{\max(1-p_s,\epsilon)}.
$$

primary coalition credit为parent-matched L1 risk difference
$\Delta_s=|\hat y_{-s}-y|-|\hat y-y|$。脚本输出run-level metrics、eight-bin × five-scope profiles、
dataset summaries、three-seed topology stability与machine-readable decision。

controls包括standalone arm error、uniform fusion和32个deterministic non-identity scope-label permutations。
所有新CSV列与gate语义已在
`analysis/stage_c_post_d21_unconstrained_reset_20260720/iscf_post_frsc_step26_20260722/step2_6_innovation_portfolio_and_scc_gate.md`
第7节定义。

## 4. Code–theory consistency

- intended theory：coalition contribution应不同于individual arm accuracy，并与ISCF dense fusion原生耦合；
- realized code：使用同一forward中的arms、policy、fused和target精确构造leave-one-out counterfactual；
- proxy：只有256 validation probe rows，且target-visible credit仍是oracle diagnostic，不证明history可预测；
- falsification：headroom/nondegeneracy、standalone distinction、cross-seed stability或shuffle specificity任一gate失败，
  均阻止SCC进入method implementation。

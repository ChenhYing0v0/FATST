# StageC D21 Evidence-Validity Surface Diagnostic

## Purpose

D21不是预测模型更新，而是Step2/3 problem gate。它检验D14中已确认的sample × future-region oracle headroom能否
由inference-visible past预测，并从official validation迁移到official test。requested horizon、future label和test
performance不进入probe feature。

## Artifact flow

`evaluate_stage_c_d21_evs_checkpoint.py`对每个frozen D14 checkpoint读取一个split：

1. `batch_x [B,720,C]`与`batch_y`来自sequential loader；
2. frozen model生成`output [B,720,C]`；
3. error转为`error_rows [B*C,720]`，按short/mid/long计算
   `row_bin_mse [N,3]`；
4. 所有arms共用由split长度确定的4096个evenly spaced `probe_indices`；
5. anchor arm `c_s1`额外把`history_rows [n,720]`映射为
   `history_features [n,192]`；
6. 每个split输出NPZ与checkpoint hash/invariant JSON，checkpoint只读且不修改。

192维descriptor由4个raw statistics、60个12-step pooled means、48个recent normalized values、64个low
Fourier real/imag coefficients、8个autocorrelations与8个recent-window statistics组成。它是diagnostic feature，
不是未来candidate module。

`analyze_stage_c_d21_evs.py`把五个canonical arms堆为`losses [N,3,5]`，仅在validation拟合：

$$
r_{i,b,s}=\log(L_{i,b,s}+\epsilon)-\operatorname{mean}_j\log(L_{i,b,j}+\epsilon).
$$

primary ridge与sensitivity HistGradientBoosting分别构造global fixed、region fixed、history-global、additive
history+region、full interaction、permuted-history和oracle policies。所有realized MSE只用对应test rows的actual arm
loss计算；oracle只报告upper bound。

## Remote execution

`run_stage_c_d21_evs.sh`读取D14 seed2021的2 carriers × 5 datasets × 5 canonical arms，在val/test各导出一次，
共100个只读checkpoint evaluation jobs。三GPU按strided worker并行。launch记录GPU状态、commit、source/output
root与split role。`sync_stage_c_d21_evs_results.sh`只同步NPZ、JSON、log与launch text。

## Code-theory consistency

- Intended theory：可发表的问题必须是past × future-coordinate interaction，而不是generic sample routing或静态
  region bias。
- Code realization：interaction predictor按future bin分别拟合risk；additive/history-global/region-only controls明确
  去除该interaction。
- Proxy boundary：D14 arms是independently trained decoder strategies；route selection也不等于optimal mixture。
  因此D21只能证明problem existence，不能证明最终architecture或training principle。
- Falsification：interaction不能跨split超过region/additive/permuted controls，或仅由一个弱readout pathology决定。
  exact descriptor probe失败不得自动方向级否定所有representation-level EVS。

## Verification

`check_stage_c_d21_evs_step7a.py`验证192维descriptor、finite值、synthetic past × bin preference recovery、七种policy
顺序以及authorization flags。Python files另执行`py_compile`，remote/sync scripts执行`bash -n`与100-job dry-run。

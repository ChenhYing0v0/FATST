# Stage C D19 Step 6 Control Design Tooling

## 1. Config

`configs/stage_c_d19_if_control_step6.json`冻结D19 control identity、source provenance、five-dataset profiles、
four arms、shared training、IF defaults、per-profile parameter contracts、Phase-A matrix、hard gates与authorization。

关键边界：

- `role=control_only`；
- `method_claim_authorized=false`；
- Step7A local=true；
- remote/test/paper method=false。

## 2. Static checker

`scripts/check_stage_c_d19_if_control_step6.py`读取config并重新计算：

1. candidate与control-only identity；
2. five datasets与four arms集合；
3. prediction/spectrum/iFFT均固定720；
4. `measure_only`与four-H checkpoint rule；
5. IF三head参数；
6. direct MLP参数与relative gap；
7. 15 new + 5 reused = 20 artifacts / 80 test cells；
8. authorization boundary。

IF参数公式：

$$
3\left[(R+49)W+W+W\cdot361+361\right].
$$

Direct参数公式：

$$
(R+98)W_d+W_d+W_d\cdot720+720.
$$

checker不实现模型、不加载dataset、不访问test，也不授权remote。它只证明Step6 frozen design内部一致。

## 3. Verification

最小验证：

```bash
python -m py_compile scripts/check_stage_c_d19_if_control_step6.py
python scripts/check_stage_c_d19_if_control_step6.py \
  --config configs/stage_c_d19_if_control_step6.json \
  --output analysis/stage_c_post_ccsf_step24_reset_20260719/d19_step6_static_gate.json
```

只有`checks_passed == checks_total`且`overall_pass=true`时，Step7A local implementation才保持授权。

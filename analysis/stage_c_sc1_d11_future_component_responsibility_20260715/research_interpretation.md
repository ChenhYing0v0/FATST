# SC1-D11 Future-Component Responsibility Result

## Decision

- `decision`: `transform_generic_pressure_sc2_only`；
- strict directional conflict datasets: `0/5`；
- support-specific component conflict datasets: `2/5`；
- generic component pressure datasets: `3/5`；
- magnitude imbalance datasets: `2/5`；
- invariant pass: `true`；method/test/SC2 remain false。

## Dataset Gates

| Dataset | directional | target/seeds | support-specific | seeds | generic | magnitude |
| --- | --- | --- | --- | ---: | --- | --- |
| ETTh1 | False | coeff_tensor/0 | False | 0 | True | False |
| ETTh2 | False | coeff_tensor/0 | True | 3 | True | False |
| ETTm1 | False | coeff_tensor/0 | False | 0 | False | False |
| ETTm2 | False | coeff_tensor/0 | True | 2 | True | True |
| Weather | False | coeff_tensor/0 | False | 0 | False | True |

## Canonical Validation MSE Means

| Dataset | RGNB JS | RGNB comp-neg | RGNB cancel | DCT JS | random median JS |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 0.072415 | 0.194444 | 0.432825 | 0.011074 | 0.001072 |
| ETTh2 | 0.065007 | 0.246032 | 0.376942 | 0.016608 | 0.001278 |
| ETTm1 | 0.039251 | 0.238095 | 0.407898 | 0.003895 | 0.000835 |
| ETTm2 | 0.070848 | 0.214286 | 0.448666 | 0.014251 | 0.000801 |
| Weather | 0.030840 | 0.416667 | 0.306820 | 0.004007 | 0.001073 |

## Boundary

D11是checkpoint-local、validation-primary diagnostic。它区分strict negative-gradient conflict、component cancellation、generic transform pressure与magnitude imbalance；任何positive decision只返回Step4，不直接授权decoder、loss、optimizer或SC2。

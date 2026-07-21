# Stage C ISCF-v1-CPSI Step7A Code Explanation

## 1. Forward computation

### 1.1 Parent ISCF modes

`TimeAlign.Model`先把normalized history编码为hidden：

```text
history [B,T,C]
-> encoder memory [B,C,P,d_model]
-> hidden [B,C,R], R=P*d_model
```

`CPSIReadout`复用independent SIFF contract：

```text
hidden [B,C,R]
-> component_history_modes
-> modes [B,C,S,D,K]
```

其中`S=5`、`D=4`，各dataset的`K`仍为ISCF-v0 frozen matched rank。

### 1.2 Pre-synthesis interaction

`interact_modes`将最后两维flatten：

```text
modes [B,C,S,D,K]
-> values [B,C,S,L], L=D*K
-> common [B,C,1,L]
-> private [B,C,S,L]
-> latent [B,C,S,r]
-> message [B,C,S,L]
-> updated modes [B,C,S,D,K]
```

`common_projection`与`private_projection`shape均为`[r,L]`；`interaction_output`为`[L,r]`。candidate使用
`GELU(common) * GELU(private)`；SELF、LINEAR、COMMON只替换latent构造，parameter tensors和shape不变。

更新后的每个scope mode才进入原`_scope_forecast`：

```text
updated mode_s [B,C,D,K]
-> pooled states [B,C,G_s,K]
-> identity + GELU nonlinear synthesis
-> arm_s [B,C,T]
```

### 1.3 Post-synthesis control

POST先生成原始arms `[B,C,S,T]`，再在forecast dimension执行同构common/private product。其matrices为
`[r_post,T]`、`[r_post,T]`、`[T,r_post]`，之后仍由原direct policy `[B,C,T,S]`融合。

## 2. Initialization contract

parent constructor先执行，interaction parameters后创建。这样相同seed的encoder、mode maps、scope synthesis和policy不会因
control type改变。两组input matrices使用`xavier_uniform_`；output matrix严格zero init。

zero init时：

```text
message = 0
updated modes/arms = parent modes/arms
forecast = ISCF-v0 forecast
```

first backward只更新output projection；output projection离开零点后，gradient才传播到两组input projections。

## 3. Configuration and CLI routing

`TimeAlign.py`将五个`iscf-v1-cpsi*` modes加入`COUPLING_READOUTS`，因此继续复用existing arm/policy diagnostics和
equal-skill training path。`train_repo.py`新增`--cpsi-rank`，当前active contract只允许`32`，并强制direct policy。

initialization artifacts新增：

- `cpsi_parent_initialization_hash`；
- `cpsi_input_initialization_hash`；
- `cpsi_output_initial_max_abs`；
- interaction mode/rank/effective rank。

model diagnostics新增interaction width、parameter count与三组matrix norms。

## 4. Code-theory consistency

### Intended theory

ISCF learned scopes同时具有common与private response。CPSI让common state在scope-specific synthesis前非线性调制private
deviation；SELF/LINEAR/COMMON/POST分别隔离capacity、linear sharing、common-only和placement解释。

### Code realization

- common由scope mean精确定义；
- private由`values - mean`定义，跨scope和为零；
- no-bias product保证candidate在private为零时message为零；
- shared matrices保证scope-permutation equivariance；
- zero output matrix保证exact ISCF containment；
- POST直接作用于forecast arms，不使用frozen representation replacement。

### Remaining proxies

- local synthetic gradient不等于real optimization health；
- common/private algebraic decomposition不等于唯一latent factorization；
- nonzero product/message不等于forecast benefit；
- POST只有near-matched added parameters，不是exact equality。

### Falsification evidence

- CPSI test materially低于ISCF-v0：exact v1 performance fail；
- SELF解释收益：`capacity_control_explains`；
- LINEAR解释收益：nonlinear product necessity fail；
- COMMON解释收益：private modulation necessity fail；
- POST解释收益：pre-synthesis placement necessity fail；
- NaN、permanent zero gradients或message collapse：numeric/optimization pathology，先修design再作方向判断。

## 5. Verification

`scripts/check_stage_c_iscf_v1_cpsi_step7a.py`在conda `r2026-fsa`中执行81个cases，覆盖readout、equivariance、
semantics、two-stage gradients、production model、CLI和profile parameter formulas，结果81/81通过。

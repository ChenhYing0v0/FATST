# StageC SC1-JAPO Step 7A 代码说明

## Forward Computation Flow

### 1. Encoder memory

`TimeAlign.Model._encode_normalized_history`保持A6路径：

```text
x [B,L,C]
  -> patch_emb_x / encoder
  -> memory [B,C,P,D]
  -> flatten
  -> hidden [B,C,R], R=P*D
```

该flatten是bijective reshape。JAPO没有pooling、patch retrieval或atom-to-patch attention。

### 2. Expert bank

`layers.PLGO.JAPOReadout.expert_latents`通过两个独立`nn.Linear(R,256)`得到：

```text
hidden [B,C,R]
  -> expert_latents [B,C,E=2,K=256]
```

`expert_coefficients`再与`atom_basis [E,T=720,K]`相乘：

```text
expert_latents [B,C,E,K]
  x atom_basis [E,T,K]
  -> expert_coefficients [B,C,T,E]
```

`atom_basis`初始化标准差为`sqrt(E/K)`；两个experts从同一initialization class独立采样，禁止复制或warm-start。

### 3. Router

`gates`按arm选择真实active path：

- `joint`：`LayerNorm(hidden) -> Linear -> tanh -> RMS`与
  `descriptor -> Linear -> tanh -> RMS`逐维相乘；
- `history`：只使用history context；
- `atom`：只使用atom context；
- `uniform`：固定`[0.5,0.5]`，router modules不进入forward。

所有learned variants只在expert axis执行softmax，产生`gate [B,C,T,E]`。不存在atom-axis normalization。

### 4. Projective synthesis

`active_indices(H)`只选择`atom_start < H`的RGNB atoms。`forward`执行：

```text
expert coefficients [B,C,N,E]
  x gate [B,C,N,E]
  -> mixed coefficients [B,C,N]
  x basis_rows [N,H]
  -> output [B,H,C]
```

requested $H$不进入Linear、LayerNorm、tanh或softmax，仅选择active rows与输出columns。

## Training And Artifact Modules

- `train_repo.initialization_contract`在optimizer step之前保存Encoder/expert/basis/descriptor hashes；
- `model_diagnostics`分别报告total decoder parameters与各control的active-forward parameters；
- `patch_interface_diagnostics`把两个expert projections重写为`[E,K,P,D]` blocks，验证flatten等价性，并记录
  trained routing entropy/usage；
- `check_stage_c_sc1_japo_checkpoint_invariants.py`重载checkpoint，审计prefix、from-scratch、patch与expert
  contracts；
- `analyze_stage_c_sc1_japo_e2e.py`先要求每个seed的35/35 artifacts与paired hashes完整，再执行冻结的
  immediate-fail、provisional-pass或inconclusive gate；seed2022返回后先对每个dataset/arm求two-seed metric
  mean，再原样复用provisional-pass threshold，禁止事后改变gate；
- `remote/run_stage_c_sc1_japo_e2e.sh`固定五datasets、七arms、validation-only与每seed 35-job matrix；seed2022
  完成后只读取seed2021/2022并执行冻结two-seed mean gate。

## Code-Theory Consistency

- 理论目标：history-dependent atomwise operator不能被吸收到单个fixed temporal table；
- 代码对应：free expert maps提供operator freedom，history-atom product只控制per-atom expert mixture；
- 仍是proxy：soft mixture是否形成有用specialization必须由remote artifacts证明；
- falsification：JOINT被UNIFORM/HISTORY/ATOM/PERM/RANDOM任一完整解释，或prefix/hash/gradient contracts失效。

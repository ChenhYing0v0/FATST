# ISCF-BSCA Efficiency Result and Table Audit

## 1. Decision

- [Fact] 5 systems × 7 datasets = 35/35 service units完成；77/77 frozen
  checkpoint objects在测量后再次通过SHA256校验。
- [Fact] 所有35个unit的finite与round-CV gate通过；最大all-horizon
  round CV为`0.0368`，低于预注册阈值`0.10`。
- [Fact] profiler仅使用synthetic standardized inputs；未实例化test loader，未读取
  test labels，也未新增训练。
- [Decision] Efficiency表支持的是明确的deployment trade-off，而不是单向的
  compute-efficiency优势。ISCF-BSCA用一个具有architectural CHPC guarantee的模型
  替代四个horizon-specific models，并相对TimeAlign/QDF减少deployed parameters与
  checkpoint storage；但当前full-domain implementation的training time和latency并不领先。
- [Decision] 不允许写成“ISCF-BSCA is uniformly more efficient”或宣称尚未实现的
  prefix-bounded speedup。允许写成“consolidates four horizon-specific models into one
  prefix-consistent service, with a measured storage/parameter versus compute trade-off”。

## 2. Frozen measurement contract

| Item | Frozen value |
| --- | --- |
| Hardware | one exclusive NVIDIA GeForce RTX 3090 |
| Precision / batch | FP32 / batch size 1 |
| Input | synthetic standardized tensor；dataset-specific channels与checkpoint-specific lookback |
| Timing | CUDA events；30 warmups；5 rounds × 100 iterations |
| Unit statistic | median of five round means；另存iteration p95与round CV |
| Single request | four requested-horizon latency的算术均值 |
| All-$H$ one-model service | one $H=720$ forward + prefix views |
| All-$H$ fixed-$H$ family | sequential native $H=96,192,336,720$ forwards |
| Peak memory | fresh process；resident model weights + all-$H$ service total peak |
| Training time | frozen native logs中的per-epoch training time之和；不含validation/test/unlogged orchestration |
| Test access | 0 |

Remote measurement window=`2026-08-14T23:09:41+08:00`至
`2026-08-14T23:17:20+08:00`。启动前GPU 0/1/2均为`18 MiB / 0% util`；
完整矩阵只使用GPU 0，未与Decoder-Transfer或其他GPU workload并发。

## 3. Seven-dataset macro result

| System | Models | Params (M) | Ckpt. (MiB) | Logged train (GPU h) | Single (ms) | All-$H$ (ms) | Peak (MiB) | CHPC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **ISCF-BSCA** | 1 | 2.926 | 17.68 | 2.028 | 10.306 | 10.318 | 38.8 | Architectural |
| TimeAlign | 4 | 10.741 | 95.43 | 0.290 | 0.928 | 3.582 | 109.1 | No |
| QDF | 4 | 5.337 | 20.38 | 0.204 | 1.141 | 4.426 | 31.9 | No |
| DLinear-$H720$-prefix | 1 | 0.485 | 1.85 | 0.021 | 0.405 | 0.429 | 11.1 | Service protocol |
| PatchTST-$H720$-prefix | 1 | 3.198 | 12.23 | 1.110 | 3.933 | 3.882 | 58.0 | Service protocol |

## 4. Interpretation and self-critique

### 4.1 Supported system-level value

- [Strong Evidence] 相对TimeAlign的four-model service，ISCF-BSCA的deployed parameter
  count低`72.8%`、checkpoint storage低`81.5%`，peak memory低`64.4%`，并提供
  architectural CHPC guarantee。
- [Strong Evidence] 相对QDF的four-model service，ISCF-BSCA的deployed parameter
  count低`45.2%`、checkpoint storage低`13.3%`，同时把4个model objects合并为1个。
- [Fact] DLinear/PatchTST H720-prefix也只需一个模型，并通过service protocol得到exact
  prefixes；因此“one model”或CHPC本身不能被写成ISCF-BSCA独有的latency优势。

### 4.2 Negative compute boundary

- [Fact] ISCF-BSCA的all-$H$ latency为`10.318 ms`，分别是TimeAlign、QDF、DLinear
  prefix与PatchTST prefix的`2.88×`、`2.33×`、`24.04×`与`2.66×`。
- [Fact] ISCF-BSCA的logged training GPU-hours为`2.028`/dataset，高于其余四个
  services。本数值只覆盖native per-epoch logs，不应解释为end-to-end wall clock。
- [Fact] 当前reference implementation会materialize完整$T=720$ future field；本实验没有
  实现或测量prefix-bounded execution，因此不能用architecture possibility替代实测结果。
- [Self-critique] Synthetic-input microbenchmark隔离了model forward cost，但不包含data
  loading、host-to-device transfer、concurrent serving或batching effects。它适合比较冻结
  model services的GPU inference graph，不等同于production end-to-end throughput。
- [Self-critique] 各native systems的epoch budgets与early-stopping paths并不matched；logged
  GPU-hours适合描述这些已部署artifacts的实际训练记录，不能单独归因于architecture的
  intrinsic training efficiency。

## 5. Claim and manuscript consequence

1. Section 5.4保留完整表，不隐藏ISCF-BSCA较慢的latency与training结果。
2. 正向措辞限制为model consolidation、lower deployed parameter/storage relative to
   four-model TimeAlign/QDF families、architectural CHPC。
3. 总结必须同时写出compute trade-off；不得使用`faster`、`lower training cost`、
   `uniformly efficient`或`prefix-bounded speedup`。
4. Efficiency不补救Decoder-Transfer的PatchTST negative block，也不补救Core-Ablation中
   Target-Adaptive Allocation control的失败。

## 6. Canonical artifacts

| Artifact | SHA256 |
| --- | --- |
| `efficiency_35_service_units.csv` | `a60067f25065e59e80aae94fd8d833469076709b89071a549b62cae897d0e694` |
| `efficiency_system_macro_means.csv` | `cff7c6efd5abe92d8e379ecb38a63278a2880993b006b9fd0c25c44f62a6d9c0` |
| `efficiency_result_summary.json` | `9153f95d652412974a31bb248c70e5b9aeb3f9baf2a9574e031217992366e49b` |
| `immutable_checkpoint_manifest.json` | `057bb7a02ef5a86dc69241ed53fbf676495288ee479d693289da2c74beeac90f` |
| `table/table_iscf_bsca_efficiency.tex` | `66bd3ecf58bdf995457b66283733261508328432c60687316713bb512b309730` |
| `table/table_iscf_bsca_efficiency_standalone.tex` | `f0775bf7c90c9e55e5b677be6398d0fa16409e8b37b5b4e075b301c43d0357ab` |
| `output/pdf/iscf_bsca_efficiency_20260814.pdf` | `aef36ae1715be19acec42a01ac46c42374abab2d201b068938f818f4f707b224` |

Decision=`efficiency_complete_tradeoff_supported_no_uniform_compute_advantage`。
下一实验cursor=`freeze Figure 5 mechanism-diagnostic and illustrative-case contract`；不自动
新增training或formal test。

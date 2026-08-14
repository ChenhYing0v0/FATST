# Efficiency Table：design and prelaunch gate

## 冻结对象

当前可完整闭合的最小subset是ISCF-BSCA、TimeAlign fixed-H family、QDF fixed-H family、DLinear H720-prefix与PatchTST H720-prefix。覆盖7 datasets，共35个system–dataset service units与77个checkpoint objects。旧9-system/45-unit草案依赖当前不存在或不完整的arms，已排除而不沿用。

## 测量合同

同一空闲RTX 3090、FP32、batch=1、`torch.inference_mode()`、synthetic standardized input；不创建test loader，不访问labels。每个unit先warm-up 30次，再执行5 rounds × 100次CUDA-event timing，报告round-mean的median，并保留p95与round CV。ISCF/DLinear-H720/PatchTST-H720的all-horizon service为一次H720 forward加prefix views；TimeAlign/QDF为四个native fixed-H model顺序forward。

参数量按deployed inference graph统计，QDF train-only learned loss不计入；storage报告actual checkpoint bytes；training GPU-hours只从已完成native logs累计。Peak memory在fresh process测量all-horizon service总峰值，appendix另存activation increment。CHPC区分`architectural guarantee`、`service-protocol guarantee`与`no guarantee`。

## Gate与排程

77/77 checkpoint与对应timing logs已只读确认存在；正式测量前仍要写入immutable hash manifest。Efficiency不新增训练、不访问formal test。Profiler实现、checker与CPU dry-run可和Decoder-Transfer准备同步，但正式GPU latency/memory不能与训练并发；因此测量排在独占GPU窗口，任何unit缺失或CV>0.10都阻断整表而非插值。


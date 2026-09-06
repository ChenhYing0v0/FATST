# ETTm1真实前缀案例（2026-09-06）

current_step=6→8→9，problem=保留前缀分歧证据，同时减少720步主图的周期拥挤；用户明确指定ETTm1并要求可视化。原ETTh1图保留。nature-figure/Python，沿用183×135mm总图+完整前96步inset，不平滑、移位或截去后程。

UVHF使用既有冻结Main-I/II appendix profile ETTm1__h2_table5_capacity/seed2021；checkpoint SHA256=2e8c7e7c7d7545a4edfbe81be857299b6fea0dc7e944464803f273257ecb6948。L720，H720统一轨迹独立请求96/192/336/720验证prefix identity，不训练UVHF。

服务器native TimeMixer仅有ETTh1 checkpoints，故补齐ETTm1四个horizon-specific模型。训练前已读取官方TimeMixer_ETTm1_unify.sh，固定L96、e_layers2、d_model16、d_ff32、avg下采样3层/window2、batch16、lr.01、seed2021、10epochs/patience10。其余native defaults与原生优化路径一致；沿用既有train_timemixer.py移除test loader/test evaluation并导出有序validation。checkpoint为最小native mean-batch validation MSE；完整validation不drop末batch。所有H取同一10801个预测origin，原始future始于34560+origin。15min/step，图中不得误标hours。

narrative_gate=仅selected validation illustration，不是matched机制归因或population性能估计。seq_len按用户已授权作为可调超参数。training不读取test；保留所有运行配置、hash和失败结果。若发散排除该实现结果，不能据此制造比较优势。

筛选沿用已有hard gate：四H MSE均优于baseline，common96 UVHF MSE低于四baseline，GT train-scaled std>=.25；full R2>=.35，tail337–720 R2>=.25/corr>=.70，last192 R2>=0，tail amplitude .5–1.5、bias/std<=.35；prefix visibility96>=.075。先检查已有256个HUFL候选（历史按UVHF视觉fit预筛），若不足则扩大全部10801×7，保留完整记录。通过后结合上一轮visible_net及四H gain、周期密度排名，审阅完整图，不仅看inset。

三个3090 GPU初始均18/24576MiB。GPU0先H720后H96，GPU1 H336，GPU2 H192；环境moe，输出/home/yingch/exp_outputs/r-2026-fatst/uvhf_prefix_ettm1_20260906。训练脚本扩展dataset/seq_len/batch_size参数，ETTh1默认不变。代码先commit/push，再remote git同步后启动。预计时间待首epoch测量，不提前猜测。

启动审计修正：第一次bundle导入指定main失败（bundle仅有HEAD），未启动训练；随后按HEAD完成ff-only同步00fac229后启动。native Exp_Basic._acquire_device覆盖CUDA_VISIBLE_DEVICES，前三个模型实际共用GPU0，显存约1.3GB且训练正常；保留这三项，不重训或改其优化参数。设备选择修正为尊重launcher mask的cuda:0映射，H96安排空闲GPU1，同时终止仅负责排队H96的GPU0父shell，保留其正在训练的H720子进程。此变更只影响设备选择，不改模型/损失/数据/checkpoint选择。

展示候选排序固定为visible_net降序、min_gain降序、visibility96降序，按origin至少间隔96取最多5个完整图审阅。统计沿用上一轮定义。若原256池无通过例，按原计划扩展全变量，不降低门槛。

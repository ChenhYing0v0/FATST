# Origin8752后移144步的指定样本

严格采用用户指定origin8896=8752+144、ETTm1/LUFL；没有在origin8944上继续偏移，也没有重选候选。旧图future step144为新图step0，旧图145为新图step1。0点时间2017-09-26 15:45:00，预测2017-09-26 16:00:00至2017-10-04 03:45:00，共720步。

## 构造与审计

build_shifted.py从已有validation全样本缓存读取新origin的真实预测，使用新历史，非旧预测裁剪/平移。UVHF输入新起点前720步，TimeMixer前96步；source_data.csv保留完整720步history，图中沿用既有history展示宽度。断言新history[-143:0]=旧GT[1:144]，新GT[1:576]=旧GT[145:720]。无新训练、无新test访问，旧版全部保留。

check_final_case.py独立加载冻结UVHF checkpoint并验证SHA256、原始GT/history、缓存与重放一致、四个target_prefix公共前缀差异均0，及四个TimeMixer缓存与图中数据一致。check_figure_exports.py复算指标，核验实际inset数据和完整主轨迹、无遮挡与标签冲突、183×135mm、PNG300dpi/TIFF1000dpi、可编辑SVG/PDF字体。复用此前通过静态检查的Python绘图源。见numeric_audit.json、figure_qa.json。

## 结果与限制

完整H96/H192/H336/H720 MSE分别降低96.07%、89.14%、91.56%、79.59%。10个独立分段与有效horizon组合MSE/MAE均改善，最小分别67.15%/44.32%；最后192步MSE改善80.17%。segment_metrics.csv含全部11个比较。gain=1-error_UVHF/error_TimeMixer，使用同变量原始尺度。

UVHF全720步R2=0.8084，尾337–720步R2=0.8069，最后192步R2=0.7907。TimeMixer全720步R2仅0.0612。因此虽然全程低谷优势和prefix分歧在视觉上强烈，这个用户指定起点重新带来此前担心的baseline过弱问题；不称为平衡样本，不将其默认为最终论文选择。

图形完整保留720点与1–96 inset，不平滑、不截掉失败区间。前缀第69步TimeMixer四模型极差约0.73原始单位；六对完整重叠区间平均绝对差再平均约0.23单位。UVHF保留尖峰、后部极深低谷预测不足等误差。

数据/呈现内部检查通过；代表性与baseline能力审阅不通过平衡案例目标。该图是用户指定的诊断展示，n=1窗口，seed2021，validation，非总体估计。与旧样本高度重叠，且仅距历史LUFL8897一个origin，不是独立证据。TimeMixer仍为visualization专用checkpoint，非Main-I来源。保留结果供用户比较，不擅自换样本。

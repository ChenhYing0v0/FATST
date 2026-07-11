# Configs

保存实验和模型配置。

配置文件变更后，至少运行对应格式解析检查。实验配置应记录 seed、dataset、
horizon、model、trainer、output path 等关键字段。

- `stage_c_mechanism_control.json`：StageC `SC0-MCP` validation-only standardized carrier calibration；
  winner冻结前禁止读取 test metrics进行配置选择。

# CPU MuJoCo＋NPU π0.5＋Mac GPT 复现指南

本 Wiki 面向使用相同实验环境的同事，说明如何复现 LIBERO-PRO 单臂实验。代码位于 [gpt-as-policy-repro](https://github.com/Xxc33128/embodied-ai/tree/main/gpt-as-policy-repro)。

CPU 容器负责仿真和图像采集，NPU 容器负责 π0.5 推理；Direct 与 Hybrid 通过 Mac 上的模型会话接入 GPT。

## 先了解交付边界

仓库包含评测代码、配置和轻量验收记录。模型权重、NPU 镜像、数据资产与账号凭据不在 Git 中。相同环境应先核对这些依赖，再执行实验。

本次检查发现，服务器上的开发实验脚本与公开仓库并不完全一致；仅克隆旧版本不能保证复现昨天的视频。复现入口应以本 Wiki 的完整性检查和已归档脚本说明为准。

## 阅读顺序

1. [[完整性检查|Audit]]
2. [[环境与资产|Environment]]
3. [[权重转换与推理|Weights]]
4. [[Mac 接入与实验运行|Experiments]]

运行结果须区分环境检查、开发实验和正式评测。单次成功不代表整套任务验收通过。

- [[在另一台 NPU 服务器复现|Transfer]]
- [[Mac 监听程序|Mac-bridge]]

- [[下载 LIBERO 仿真文件|Assets]]

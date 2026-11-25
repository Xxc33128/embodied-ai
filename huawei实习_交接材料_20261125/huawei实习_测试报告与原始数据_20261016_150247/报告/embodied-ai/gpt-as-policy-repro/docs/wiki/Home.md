# CPU MuJoCo＋NPU π0.5＋Mac GPT 复现指南

本 Wiki 面向使用相同实验环境的同事，说明如何复现 LIBERO-PRO 单臂实验。代码位于 [gpt-as-policy-repro](https://github.com/Xxc33128/embodied-ai/tree/main/gpt-as-policy-repro)。

CPU 容器负责仿真和图像采集，NPU 容器负责 π0.5 推理；Direct 与 Hybrid 通过 Mac 上的模型会话接入 GPT。

## 先了解交付边界

仓库包含评测代码、配置和轻量验收记录。模型权重、NPU 镜像、数据资产与账号凭据不在 Git 中。相同环境应先核对这些依赖，再执行实验。

**2026 年 9 月 20 日在服务器上实际运行的四组实验脚本已上传到 GitHub：** [实验代码](https://github.com/Xxc33128/embodied-ai/tree/main/gpt-as-policy-repro/experiments/20260920)。每组包含运行、录制和视频整理脚本，并附源码 SHA256 清单。更新仓库后，按 [实验运行步骤](https://github.com/Xxc33128/embodied-ai/wiki/Experiments) 使用对应目录即可；复跑时使用新的输出目录。

## 阅读顺序

1. [[完整性检查|Audit]]
2. [[环境与资产|Environment]]
3. [[权重转换与推理|Weights]]
4. [[Mac 接入与实验运行|Experiments]]

运行结果须区分环境检查、开发实验和正式评测。单次成功不代表整套任务验收通过。

- [[在另一台 NPU 服务器复现|Transfer]]
- [[Mac 监听程序|Mac-bridge]]

- [[下载 LIBERO 仿真文件|Assets]]

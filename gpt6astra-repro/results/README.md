# results — 各模型标准批次结果索引

同一协议（`robosuite-bowl-paired20-v1`）、同一冻结初态库（seeds 2000–2019）、同一提示词；唯一自由变量为决策模型。控制变量与复跑方法见[上级 README](../README.md)。

| 批次目录 | 模型 | 状态 | env_success | 盲评 stage=4 | 备注 |
|---|---|---|---|---|---|
| [`glm-5.3-flash/`](glm-5.3-flash/) | glm-subagent（GLM-5.3-Flash） | 完成 + 已盲评 | 0/20 | 0/20 | stage 分布 0×11，1×6，2×2，3×1（最高 stage3） |
| [`deepseek-v4.1-flash/`](deepseek-v4.1-flash/) | deepseek-v4.1-flash-subagent | 完成，盲评未跑 | 1/22 | — | 22 attempts 含基础设施失败重跑 |
| qwen3.8-flash | qwen3.8-flash-subagent | **进行中** | — | — | 批次未终止，完成后另次同步 |
| gpt-6-astra | — | 未完成 | — | — | 仅 trial-01 有效，中断现场本机保留 |

## 读结果的顺序建议

1. 各批次 `README.md`（结论摘要）→ 2. `summary.csv`（逐 trial 数字）→ 3. 抽查 `trials/trial-NN/video.mp4` 对照 `review.json` 的盲评依据。

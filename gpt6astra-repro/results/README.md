# results — 各模型标准批次结果索引

各批次共享协议（`robosuite-bowl-paired20-v1`）、冻结初态库（seeds 2000–2019）和任务提示词。运行工具、并发、额度中断及worker重启存在差异，不能称为只有模型一个变量；控制条件与限定见各批次报告及[上级 README](../README.md)。

| 批次目录 | 模型 | 状态 | env_success | 盲评 stage=4 | 备注 |
|---|---|---|---|---|---|
| [`glm-5.3-flash/`](glm-5.3-flash/) | glm-subagent（GLM-5.3-Flash） | 完成 + 已盲评 | 0/20 | 0/20 | stage 分布 0×11，1×6，2×2，3×1（最高 stage3） |
| [`deepseek-v4.1-flash/`](deepseek-v4.1-flash/) | deepseek-v4.1-flash-subagent | 完成 + 已盲评 | 1/20 | 1/20 | stage 分布 0×14，1×3，2×1，3×1，4×1；22 attempts（2 次基础设施无效已整 trial 重跑） |
| [`qwen3.8-flash/`](qwen3.8-flash/) | qwen3.8-flash-subagent | 完成 + 已盲评 | 4/20 | 4/20 | stage 0×6，1×8，2×1，3×1，4×4；平均最高阶段 1.45；trial-19 边界样本保守读法 3/20 |
| [`gpt-6-astra/`](gpt-6-astra/) | GPT-6 Astra / medium | **5个有效trial阶段报告已完成，原20次未完成** | 5/5 | 已评2/5：均3–4边界待定 | 五个policy合计7,484,353 tokens；trial06已中断，不计成绩；[总结报告](../../weeks/2026_0914-0920_GPT6Astra评测复现/2026-09-16-GPT6-Astra五次任务总结与用量.md) |

## 读结果的顺序建议

1. 各批次 `README.md`（结论摘要）→ 2. `summary.csv`（逐 trial 数字）→ 3. 抽查 `trials/trial-NN/video.mp4` 对照 `review.json` 的盲评依据。

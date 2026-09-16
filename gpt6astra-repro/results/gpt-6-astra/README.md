# GPT-6 Astra：五个有效trial阶段总结

更新：2026-09-16。原20次计划目前完成5次，覆盖共享scene seeds 2000–2004；第六轮基础设施中断未计入成绩。

- **自动入碗：5/5**；61次机器人决策、2,161控制步，记录完整性5/5 PASS。
- 平均每轮 **12.2次决策、21.61秒仿真、5.85分钟runner墙钟**。
- 五个有效policy的Codex会话计数：**7,484,353 tokens**，其中缓存输入7,144,576；金额unknown。另有中断、调度、复核及作废尝试，分开列账。
- 已审核trial01、02均为 **3–4边界待定**：确认释放入碗，终态稳定证据不足；另三轮未完成盲评，不宣称严格审核5/5。
- 每trial一个无历史GPT-6 Astra/medium子agent连续完成。trial04起由root直接管理，不设中间调度agent。trial02使用同初态重跑的首个有效attempt。

完整结论及与GLM/DeepSeek/Qwen的口径对照见[五次任务总结与用量](../../../weeks/2026_0914-0920_GPT6Astra评测复现/2026-09-16-GPT6-Astra五次任务总结与用量.md)。

| 产物 | 内容 |
|---|---|
| [tokens-and-timing.csv](tokens-and-timing.csv) | 五轮实测时间、输入/缓存/输出tokens |
| [subagent-usage.json](subagent-usage.json) | 19个相关子agent的分类用量、数值事件来源及哈希 |
| [decision-notes.md](decision-notes.md) | 61次动作的目标、观察与简短理由原文；不是内部思维链 |
| [decision-trace.csv](decision-trace.csv) | 动作与控制步、等待、实测位置及私有核验的关联 |
| [paired-first5.csv](paired-first5.csv) | 共同前五seed的跨模型原始结果对照 |
| [summary.csv](summary.csv) / [index.html](index.html) | 逐trial结果与录像入口 |
| [rebuild_metrics.py](rebuild_metrics.py) | 从原始日志和数值token事件复算 |

本机原始批次中的录像、请求/响应和物理轨迹保持原样。`trials/`为已有公开镜像；完整物理轨迹仍在本机，本次新增逐动作核验摘要。本报告没有修改原始评分；`result.json`中的桥接API usage仍为null，新增表中的Codex运行时计数是另一个来源。

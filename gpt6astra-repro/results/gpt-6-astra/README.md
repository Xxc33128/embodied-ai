# GPT-6 Astra：五个有效trial阶段总结

更新：2026-09-16。原20次计划目前完成5次，覆盖共享scene seeds 2000–2004；第六轮基础设施中断未计入成绩。

- **自动入碗：5/5**；61次机器人决策、2,161控制步，记录完整性5/5 PASS。
- 平均每轮 **12.2次决策、21.61秒仿真、5.85分钟runner墙钟**。
- 五个有效policy的Codex会话计数：**7,484,353 tokens**，其中缓存输入7,144,576；金额unknown。另有中断、调度、复核及作废尝试，分开列账。
- **独立评审（2026-09-16）：5/5 达 stage 4**。基于 `physics.jsonl` 逐步物理真值逐条复核：五轮末帧全部满足环境三判据（距碗心 3.8–6.6mm < 35mm、方块顶面低于碗口 ≥17mm、速度 <0.15 m/s）；正式盲评的"终态稳定证据不足"实为"成功即终止"协议的结构性结果（录像必然止于落地反弹帧，该口径下 stage 4 构造性不可达），回弹高度仅 0.2–0.4mm、方块低于碗口 ≥17mm，物理上不可能出碗。全文见 [independent-review.md](independent-review.md)。
- 正式盲评仍如实记录：已审核 trial01、02 均 stage_max=3（3–4边界待定，确认释放入碗）；另三轮未出分，但独立复核认定其物理形态与前两轮完全同构。评分修订应走口径修订（见 independent-review.md 建议），不改原始评分。
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
| [independent-review.md](independent-review.md) | 独立评审全文：5/5 达 stage 4 的物理证据、stage3/4 争议裁决、决策质量与横向对照 |
| [rebuild_metrics.py](rebuild_metrics.py) | 从原始日志和数值token事件复算 |

本机原始批次中的录像、请求/响应和物理轨迹保持原样。`trials/`为已有公开镜像；完整物理轨迹仍在本机，本次新增逐动作核验摘要。本报告没有修改原始评分；`result.json`中的桥接API usage仍为null，新增表中的Codex运行时计数是另一个来源。

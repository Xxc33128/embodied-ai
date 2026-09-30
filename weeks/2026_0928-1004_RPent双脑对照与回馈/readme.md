# 2026-09-28 ~ 10-04：RPent 双脑对照与回馈

> 2026-10-01 更新：Astra/Sol medium三套件120集全部审定，99成功、21失败；Astra50/60、Sol49/60。

## 本周候选方向（2026-09-28 预填）

1. **RPent 双脑对照（主线候选）**：GLM-5.3 vs GPT-6 Astra 在同一 RPent/npunew 栈上各跑 Object Swap 20 集——衔接上周 [RPent 复现与扩展](../2026_0921-0927_同机仿真与云边验证/2026-09-27-RPent扩展-FlashMode与libero_goal.md)（GLM-5.3 首跑 90%/retry-once 100%），产出"GPT-6 大脑 + 国产昇腾执行层"的组合数据（官方 Astra object_swap 99%，但其数字出自官方自建基建）。前置：GPT-6 API key 与 RPent api planner 的 OpenAI 兼容口接法。
2. **补丁/发现回馈**：四份产出提给上游/同事仓库——sam3 双 shim（get_device_properties major=None）、molmo shim、flash 记忆路径文档漂移（HF 数据集已无 libero/flash）、httpx 无超时导致 GLM 连接可无限挂死。
3. **早停+重跑策略实测（可选）**：上周回测（纯早停误杀 5 翻盘胜局、重跑 3.4M token/胜）的结论来自回顾性分析；可在新批次上实装"前兆截断+重跑一次"验证前瞻效果。

## 遗留（上周 → 本周）

- npunew 三服务常驻（VLA@dev5 / SAM3@dev6 / Molmo@dev4），不用时停。
- 若扩官方 100 集/套件口径：先修 httpx 超时问题再跑长批。

## 过程文档索引

- [2026-09-29 Codex双模型试跑与协议](2026-09-29-RPent-Codex双模型试跑与协议.md)：无API key接入、独立官方资产快照、120集范围与首对试跑记录。[E1/E2]

- [GPT 何时介入具身任务](2026-09-29-LLM-VLA-WAM融合架构与后训练启发.md)：新 agent 重写，围绕任务预期与实际进展的本地监测，说明何时请求 GPT、怎样校准触发与验证调用价值；不依赖固定底层架构。[E1/E3]

- [2026-10-01 双模型120集结果](2026-10-01-RPent-Codex双模型120集结果.md)：最终分套件成功率、耗时口径、例外审定与脱敏验收清单。[E2/E3]

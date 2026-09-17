# 2026/9/14–9/20：GPT-6 Astra on robot arms（RoboCurve 评测）仿真复现

## 本周目标

> 9/17 新增 GPT-as-Policy 工作：以[完整链路执行计划 v2.0](2026-09-17-GPT-as-Policy在MuJoCo与昇腾上的复现计划.md)为准，保留原任务、机器人、权重和 agent 链路；下方 RoboCurve/bowl 为另一项历史实验，不能替代该计划。

复现 [RoboCurve 官方评测](https://openai.robocurve.org/gpt-6-astra/)的技术路线：用开源 harness [inspect-robots](https://github.com/robocurve/inspect-robots) + agent policy（`move_to` 末端位姿 + 机器人/仿真后端的控制器 + 回合制 ≤20 次 LLM 调用），在仿真 embodiment 上跑 bowl 任务（抓红块放碗）20 trials。

> 9/14 原始来源复核：后续范围以[原始评测复核与复现范围](2026-09-14-原始评测复核与复现范围.md)为准。下方历史条目中的“与原评测同版/版本完全对齐”须结合原始 transcript 的 0.57.1 与报告概要 0.58.0 冲突阅读；自研 Panda 控制器不是必经前置。

- **复现的是协议（E1/E2）**：harness、agent loop、评测协议、指标采集（阶段评分 0–4 / token / 成本 / 时长 / rrd+视频留痕）。
- **定性对照的是数字（E3）**：仿真 embodiment 替代真机 I2RT YAM 双臂，与 RoboCurve 真机结果（19/20=95%、$0.94/次、2.5 min）只做定性对照并归因差异。
- 模型：GLM-5.3-flash（智谱，OpenAI 兼容）调试全链路 → gpt-6-astra 跑 20 trials；若做两模型对照，则冻结同一环境后各跑 20 次。
- 仿真：后端应优先复用现成机器人控制器；现有 MuJoCo v3 与 Isaac 路线均待验证，mock 只作链路自检。用户已明确无截止时间，旧 Day3 切换规则撤销。
- 关联背景：本周新闻剪藏的"GPT-6 Astra 具身能力/通才 LLM vs 专用 VLA"讨论。

## 本周文档

| 日期 | 文档 | 结论 |
|---|---|---|
| 9/17 | [GPT-as-Policy 完整链路执行计划 v2.0](2026-09-17-GPT-as-Policy在MuJoCo与昇腾上的复现计划.md) | 合并原计划与对抗性审查；F1–F3 转为明确规格，细化 W0–W12、app-server/API 双路线和固定案例统计；只完成计划，迁移与正式实验尚未实施 |
| 9/14 | [原始评测复核与复现范围](2026-09-14-原始评测复核与复现范围.md) | 核对原报告、5 次 Astra 运行与上游 YAM 适配器；澄清模型/控制器职责，纠正版本、wire、步数和力矩观测口径；收敛必要交付 |
| 9/14 | [复现启动与agent链路验证](2026-09-14-复现启动与agent链路验证.md) | 环境就绪（harness 0.58.0 与 RoboCurve 同版）；mock LLM 服务器零成本跑通 agent 全链路（3 trials，含预算耗尽路径）；agent policy 参数源码定位；Kaggle 侦察/安装 notebook 成稿 |
| 9/14 | [Motus2调研](2026-09-14-Motus2调研.md) | 生数×清华自进化世界模型（v1 直系续作）：一模型三接口（policy/simulator/evaluator）+ 失败数据监督路由 + DiffusionNFT MBRL；13 万小时 egocentric 数据与 scaling law；真机 84%、MBRL+规划 65→75%、触觉 +12.5pt；**开源仅 README，复现为时过早**；v1.1 增补 v1 源码级解剖（§3，E1 锚点 f771216）与两代机制对差（§2.4）；v1.2 文风修订、v1.3 校准（可读性改动保留，表达回归文档口径） |
| 9/14 | [LingBot-VA调研](2026-09-14-LingBot-VA调研.md) | 蚂蚁灵波 AR 视频-动作世界模型：v1 **代码+权重全开**（Apache-2.0）→ 源码级审计（单序列掩码布局、槽位式 KV cache、半程去噪、attn_mode 坑；released=共享骨干 vs 论文双流版待核）；RoboTwin 92.9/91.6、LIBERO 98.5、真机全面超 π0.5；v2.0（MoE-13B-A1.9B、225Hz）论文公开但**未放码**；v1.1 文风修订、v1.2 校准 |

## 按日结论

- **9/15（二）**：MuJoCo bowl 环境两代迭代。①v2 浮动执行器版：腕载裁剪相机/yaw 维/执行噪声/reset 随机化全部落地，确定性+噪声回归 10/10+10/10（E1）；②保真度审计成文（10 项差距矩阵 + 5 条机制发现）；③v3 Panda 关节臂版（Menagerie CAD 模型 + DLS IK + 腱驱动夹爪改造）：**交接点状态 = 确定性回归抓取 6/10、入碗 3/10**，失败三分类（闭合挤出/搬运蠕变/释放落沿）与已试路线记录在 gpt6astra-repro/AGENTS.md，待下一人接手。
- **9/14（一）**：Phase 0 完成 + Phase 1a 完成。①harness 版本对齐（0.58.0）；②mock 全链路 2/3 成功、第 3 个复现 20 次调用预算耗尽（E1）；③isaacsim 插件动作空间为 `joint_pos`——与 RoboCurve eef `move_to` 的协议差异已定位（E1）；④阻塞项：智谱 key / OpenAI key / Kaggle notebook 执行。**⑤（调研）Motus2 文献级调研成稿**：三接口世界模型 + MBRL 自进化闭环，数字均为厂商真机口径（E3），代码未放；可作 ① 报告附录 C 案例。**⑥（调研）LingBot-VA 源码级审计成稿**：v1 代码+权重+数据全开（本工作区锚点 `vla_wam_framework_src/LingBot-VA`），单序列因果掩码、槽位式 KV cache、半程去噪等已定位到行；v2.0 论文公开、代码未放。

## 过程文档/工具索引

- [GPT-6 Astra五次任务总结与用量（9/16）](2026-09-16-GPT6-Astra五次任务总结与用量.md)：五个有效trial自动入碗5/5，平均12.2次决策、5.85分钟runner墙钟；从Codex数值事件追回policy用量7,484,353 tokens（含缓存），额外中断/调度/复核另列；附61次动作说明及共同前五seed跨模型对照。已审两轮终态3–4边界待定，原20次尚未完成。
- [bowl 标准测试流程（9/15）](2026-09-15-bowl标准测试流程.md)：各模型共享2000–2019二十种子，每trial新上下文；统一提示、20调用/900步、视频/帧/指令/评分留痕。[配置与提示词](bowl-eval-v1/)；标准已制定，正式运行尚未开始。
- [Codex 交互视觉抓放成功（9/15）](2026-09-15-Codex交互视觉抓放成功.md)：当前助手依据三相机与本体状态，12次move_to、276步成功入碗；未加沉降，作为探索演示，不计正式模型成绩。
- [robosuite 适配器独立复核（9/15）](2026-09-15-robosuite适配器独立复核.md)：底层脚本10/10、契约8/8复测通过；模型接口仍有夹爪反向、yaw四元数约定错误、碗视觉隐藏三项阻塞，区别物理可用与模型闭环可用。
- [Task3–4 独立复核](2026-09-14-Task34独立复核-诊断缺陷与因果边界.md)：实际接触面、IK可达性反例、接触参数混合、CSV错位及失败阶段复核；修正Task3/4 v1.0因果结论的适用范围。
- [v3 核查与 20 trials 执行规划](2026-09-14-v3核查与20trials执行规划.md)：当前源码复测、接口/模型组装问题与分阶段验收；本轮用户明确无截止时间，旧 Day3 降级安排不作为新计划硬约束。
- [qwen3.8-flash 标准20次子agent测试结果（9/15）](2026-09-15-qwen3.8-flash标准20次子agent测试结果.md)：同场景同预算下复核 stage=4 **4/20**（平均最高阶段 1.45）；其中 12 行为单 worker 全程同口径（3/12=25%），8 行因 provider 429 限流暂停被换 worker 单独列。实测持久 worker 单 trial 7–8.5M tokens（94% cache 重发）→ 该 token plan 装不下 20 个持久 trial；主表只认盲评（seed 2002 自动判据与盲评不一致已实证该纪律必要）。
- [GLM 标准20次子agent测试结果（9/15）](2026-09-15-GLM标准20次子agent测试结果.md)、[DeepSeek V4.1 标准20次子agent测试结果（9/15）](2026-09-15-DeepSeek%20V4.1标准20次子agent测试结果.md)：同库同预算的两个对照模型（0/20 与 1/20，均已经盲评收口）。
- 工作目录（本机，非本仓库）：`gpt6astra-repro/`（venv、脚本、Kaggle notebook 源）
- 代码与结果镜像（本仓库根目录）：[`gpt6astra-repro/`](../../gpt6astra-repro/)（脚本全量 + 各批次精简结果，图像大块未上传，取舍见其 README）
- inspect-robots 源码 clone（工作区根目录）：`inspect-robots/`
- RoboCurve 评测页：<https://openai.robocurve.org/gpt-6-astra/>
- 框架文档：<https://docs.inspectrobots.org/>

## 遗留 → 下周

（周五收口时补全；候选：拼图精细任务、π0 系 VLA 对照（xpolicylab）、MuJoCo/MJWarp embodiment 挂 npu 机）

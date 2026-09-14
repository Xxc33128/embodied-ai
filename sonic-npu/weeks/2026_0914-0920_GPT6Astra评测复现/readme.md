# 2026/9/14–9/20：GPT-6 Astra on robot arms（RoboCurve 评测）仿真复现

## 本周目标

复现 [RoboCurve 官方评测](https://openai.robocurve.org/gpt-6-astra/)的技术路线：用开源 harness [inspect-robots](https://github.com/robocurve/inspect-robots) + agent policy（`move_to` 末端位姿 + 框架侧 IK + 回合制 ≤20 次 LLM 调用），在仿真 embodiment 上跑 bowl 任务（抓红块放碗）20 trials。

- **复现的是协议（E1/E2）**：harness、agent loop、评测协议、指标采集（阶段评分 0–4 / token / 成本 / 时长 / rrd+视频留痕）。
- **定性对照的是数字（E3）**：仿真 embodiment 替代真机 I2RT YAM 双臂，与 RoboCurve 真机结果（19/20=95%、$0.94/次、2.5 min）只做定性对照并归因差异。
- 模型：GLM-4.6V（智谱，OpenAI 兼容）调试全链路 → gpt-6-astra 跑 20 trials 正式对照。
- 仿真：Kaggle T4 ×2（Isaac Lab 路线，含死线与降级：A=自写 MuJoCo embodiment，B=cubepick mock 保底）。
- 关联背景：本周新闻剪藏的"GPT-6 Astra 具身能力/通才 LLM vs 专用 VLA"讨论。

## 本周文档

| 日期 | 文档 | 结论 |
|---|---|---|
| 9/14 | [复现启动与agent链路验证](2026-09-14-复现启动与agent链路验证.md) | 环境就绪（harness 0.58.0 与 RoboCurve 同版）；mock LLM 服务器零成本跑通 agent 全链路（3 trials，含预算耗尽路径）；agent policy 参数源码定位；Kaggle 侦察/安装 notebook 成稿 |
| 9/14 | [Motus2调研](2026-09-14-Motus2调研.md) | 生数×清华自进化世界模型（v1 直系续作）：一模型三接口（policy/simulator/evaluator）+ 失败数据监督路由 + DiffusionNFT MBRL；13 万小时 egocentric 数据与 scaling law；真机 84%、MBRL+规划 65→75%、触觉 +12.5pt；**开源仅 README，复现为时过早**；v1.1 增补 v1 源码级解剖（§3，E1 锚点 f771216）与两代机制对差（§2.4）；v1.2 文风修订（humanizer-zh） |
| 9/14 | [LingBot-VA调研](2026-09-14-LingBot-VA调研.md) | 蚂蚁灵波 AR 视频-动作世界模型：v1 **代码+权重全开**（Apache-2.0）→ 源码级审计（单序列掩码布局、槽位式 KV cache、半程去噪、attn_mode 坑；released=共享骨干 vs 论文双流版待核）；RoboTwin 92.9/91.6、LIBERO 98.5、真机全面超 π0.5；v2.0（MoE-13B-A1.9B、225Hz）论文公开但**未放码**；v1.1 文风修订（humanizer-zh） |

## 按日结论

- **9/15（二）**：MuJoCo bowl 环境两代迭代。①v2 浮动执行器版：腕载裁剪相机/yaw 维/执行噪声/reset 随机化全部落地，确定性+噪声回归 10/10+10/10（E1）；②保真度审计成文（10 项差距矩阵 + 5 条机制发现）；③v3 Panda 关节臂版（Menagerie CAD 模型 + DLS IK + 腱驱动夹爪改造）：**交接点状态 = 确定性回归抓取 6/10、入碗 3/10**，失败三分类（闭合挤出/搬运蠕变/释放落沿）与已试路线记录在 gpt6astra-repro/AGENTS.md，待下一人接手。
- **9/14（一）**：Phase 0 完成 + Phase 1a 完成。①harness 版本对齐（0.58.0）；②mock 全链路 2/3 成功、第 3 个复现 20 次调用预算耗尽（E1）；③isaacsim 插件动作空间为 `joint_pos`——与 RoboCurve eef `move_to` 的协议差异已定位（E1）；④阻塞项：智谱 key / OpenAI key / Kaggle notebook 执行。**⑤（调研）Motus2 文献级调研成稿**：三接口世界模型 + MBRL 自进化闭环，数字均为厂商真机口径（E3），代码未放；可作 ① 报告附录 C 案例。**⑥（调研）LingBot-VA 源码级审计成稿**：v1 代码+权重+数据全开（本工作区锚点 `vla_wam_framework_src/LingBot-VA`），单序列因果掩码、槽位式 KV cache、半程去噪等已定位到行；v2.0 论文公开、代码未放。

## 过程文档/工具索引

- [v3 核查与 20 trials 执行规划](2026-09-14-v3核查与20trials执行规划.md)：当前源码复测、接口/模型组装问题与分阶段验收；本轮用户明确无截止时间，旧 Day3 降级安排不作为新计划硬约束。
- 工作目录（本机，非本仓库）：`gpt6astra-repro/`（venv、脚本、Kaggle notebook 源）
- inspect-robots 源码 clone（工作区根目录）：`inspect-robots/`
- RoboCurve 评测页：<https://openai.robocurve.org/gpt-6-astra/>
- 框架文档：<https://docs.inspectrobots.org/>

## 遗留 → 下周

（周五收口时补全；候选：拼图精细任务、π0 系 VLA 对照（xpolicylab）、MuJoCo/MJWarp embodiment 挂 npu 机）

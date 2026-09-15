# 2026/8/24–8/30：环境收口 + Sim2Sim 审查链启动

## 本周目标

环境问题永久归档；启动三项目（humanoid-lab / humanoid-gym / GR00T SONIC）Sim2Sim 一致性实验，用门禁化审查链定位双引擎分叉根因。

## 按日动作与结论

### 8/24（一）环境收口
| 文档 | 结论 |
|---|---|
| [环境配置排障复盘](环境配置排障复盘-2026-08-24.md) | 43 坑 + 8 session 时间线永久归档；三大根因（驱动 610.88 / 16GB 内存天花板 / 版本矩阵）；4–2048 env 吞吐扫描：**1024 env 为甜点（4460–5652 steps/s）**，2048 临界点吞吐腰斩 |
| [GR00T-Isaac-MuJoCo环境搭建与项目进展报告](GR00T-Isaac-MuJoCo环境搭建与项目进展报告-2026-08-24.md) | 环境冻结口径：Isaac 5.1 / torch 2.7.0+cu128 / 驱动 580.88；**物理参数修改量 = 0**；唯一主线卡点 = Isaac benchmark CLI 挂起 |

### 8/26（三）调研 + 实验启动
| 文档 | 结论 |
|---|---|
| [humanoid-lab调研与仿真-会话总结](2026-08-26-humanoid-lab调研与仿真-会话总结.md) | humanoid-gym 训练在 sm_120 **不可行**（Isaac Gym 只编到 sm_80 无 PTX）；humanoid-lab 走 IsaacLab 2.3.2 + IsaacSim 5.1 无此坑；60 天异常关机 28 次 = 系统性问题，非 sim 代码责任 |
| [三项目Sim2Sim两日实验执行计划](2026-08-26-三项目Sim2Sim两日实验执行计划.md) | 审查链母计划 v3：G0–G3 分层门禁设计（接口→资产→动力学→结论） |

### 8/27（四）审查链前三环
| 文档 | 结论 |
|---|---|
| [步骤1-4产物审查](2026-08-27-Sim2Sim步骤1-4产物审查与后续修订计划.md) | 裁决 **G0 接口门禁 FAIL**（qpos_rel max 0.048 rad、启动 obs 差 15.8、q_des 差 0.359 rad；"20ms 延迟"实现实为 200ms），结果冻结为 `AS_SHIPPED-G0-FAIL`，禁止用于物理归因 |
| [R1-R3反馈审查](2026-08-27-R1-R3反馈审查与下一步执行计划.md) | cycle 0 接口对齐 PASS（最大差 4.03e-7）；cycle 1 起右肘率先分叉（dq 差 11.32 rad/s）；**armature 确立为首要假设** |
| [R3-interface-gate-v2审查](2026-08-27-R3-interface-gate-v2审查与N4-N5修订.md) | 启动接口正式 **PASS**；N2 揪出 3 个 P0：joint-id 整体错位、30→14 body 映射错误、`mj_objectVelocity` rot:lin 顺序错 |

### 8/28（五）裁决密集日——从撤回到 74.8%
| 文档 | 结论 |
|---|---|
| [R4-actuator-isolation-v2审查](2026-08-28-R4-actuator-isolation-v2审查与后续计划修订.md) | 29 关节中 **17 个 armature 不一致**（右肘 0.0685 vs 0.01）；MuJoCo 单变量干预证实 armature 显著改变瞬态；"改善 55%"因 Isaac 前置激励+时间错相被**撤回** |
| [smoke-v1验收](2026-08-28-R4F-N5R-smoke-v1验收与全量运行前修订.md) | 裁决 **NO-GO**：拒绝启动 18 次全量（hip 隔离漂移 3.24× 超线、gate 脚本擅自放宽阈值）——负结果纪律样本 |
| [full-v1审查](2026-08-28-R4F-N5R-full-v1审查与后续修订计划.md) | 18 次全量标记 **`INVALID_FOR_CAUSAL_CONCLUSION_ASSET_MISMATCH`**：隔离 URDF 重复施加 root 世界变换，elbow/hip 等效惯量虚大 90.8×/11.5×；新设 G0 资产等价门禁 |
| [StepAB-G0审查](2026-08-28-R4F-N5R-StepAB-G0审查与方向把控.md) | Step A 模型修复 PASS、G0 物理等价 PASS（COM 误差 5.06e-8 m）；同时 HOLD：揪出 5 个 fail-closed 门禁漏洞 |
| [N5R2终验](2026-08-28-N5R2终验与最终行动清单.md) | **本周核心结论**：右肘 armature 0.0685→0.01 对齐后，隔离 MAE **0.006752→0.001704 rad（↓74.8%）**，t50 0.16329 vs Isaac 0.16382 s；hip 负对照逐点差 0 |
| [B1-Lab10s结果审查](2026-08-28-R4F-B1-Lab10s结果审查与最终收口计划.md) | 74.8% 在 B1 补丁+Lab 10s 全身场景复确认；视频/台账四项质量问题转入收口 |

## 过程文档（备份于 [../../archive/](../../archive/)）

- [B1验收与两日计划收口执行清单](../../archive/2026-08-28-B1验收与两日计划收口执行清单.md) — B.1 补丁验收+排程的时间盒执行清单，结论已被同日 Lab10s 审查收录。
- [Win端本周总结报告生成提示词-v2](../../archive/2026-08-28-Win端本周总结报告生成提示词-v2.md) — 周报生成指令（工具类）；**其产物周报已归档于 [../../sim2sim-final/r4f-b31/09_report/](../../sim2sim-final/r4f-b31/09_report/)**。

## 证据包

`R4F_B1_Lab10s_N5R2_B31_20260831.zip`（70MB）→ [../../packages/](../../packages/)，报告子集 → [../../sim2sim-final/r4f-b31/](../../sim2sim-final/r4f-b31/)。

## 遗留 → 下周

包内 4 组不一致待 B.2/B.3 同步；视频/台账质量问题待收口 → 下周（8/31–9/6）。

# GLM 标准 20 次子 agent 测试报告（bowl-eval-v1）

- 批次：`glm-standard20-20260915-120407`；接入方式：persistent-subagent（每 trial 一个全新无历史子 agent 连续完成，保留本 trial 记忆；与 GPT persistent 批次同组定义，已披露与独立 API 条件的差异）。
- 决策模型标签：glm-subagent（确切快照 unknown；编码工具后端自报 builtin:bigmodel/GLM-5.3-Flash，子 agent 后端无法独立核实）。
- 协议：bowl-eval-v1 `robosuite-bowl-paired20-v1`，20 调用 / 900 步 / effort=medium 请求 / max_speed_frac=0.25 / image_horizon=2；共享冻结初态库 seeds 2000–2019（与 GPT 批次同一 bank，逐 trial 初态精确核对全部 matched）。
- 调度披露：最多 6 个 trial 并发（推理等待期物理暂停、trial 间环境隔离、初态逐 trial 核对不受影响）；**wall_seconds 不可作串行比较**。

## 主结果（评分=盲评 stage_max；成功率按 stage=4 计）

| 指标 | 值 |
|---|---|
| 审核成功率（stage=4） | **0 / 20** |
| env_success（自动判定） | 0 / 20 |
| 平均最高阶段 | 0.65 |
| 阶段分布 | stage0 ×11，stage1 ×6，stage2 ×2，stage3 ×1，stage4 ×0 |
| 调用数 | 平均 18.5（上限 20） |
| 控制步数 | 平均 623（上限 900） |
| 终止原因 | {'give_up': 12, 'done': 5, 'max_steps': 3} |
| 记录检查 | 20/20 PASS（audit_standard_trials.py） |

最高阶段 trial：trial-03（seed 2002）stage 3——第 269 步夹起、提升 ~7cm、持物至碗轴 0.085m，松爪时掠过碗口后磕壁弹出，最终贴碗外壁静置（第二盲评独立确认 stage 3 非 4）。次高：trial-07、trial-08 stage 2（真夹取+离桌提升，未到碗口上方）。

## 值得记录的现象（E2，本冻结场景集）

1. **模型信念与物理的分歧**：trial-14 模型自称腕镜确认方块已在碗底并以 done 结束，物理轨迹显示夹爪全程未接触方块（stage 0，终距碗轴 0.273m 为全批最远）；trial-03 模型自称入碗并压坐，实际弹到碗外（stage 3）。盲评不看模型 notes 的设计有效防止了被文字带偏。
2. **执行时序怪癖被模型利用**：多 trial 独立发现"位置型 move_to 只走 2-4 步、夹爪行程型调用走 39-80 步"，并用中间夹爪目标（0.5/0.6）延长插值窗口完成长距离移动——与 SOP §4 预先披露的"重复指定 gripper 改变 chunk 长度"行为一致，所有模型可用同一工具，属协议内行为。
3. **夹取是主要瓶颈**：11 个 trial 连接触碰都没有或仅擦碰（stage ≤1 主因：腕相机斜装+TCP 参考点高于指垫，视觉对中系统性偏差 1-3cm）；stage≥2 的 3 个 trial 都靠多次重试闭合才建立抓持。
4. **预算耗尽模式**：20 次调用普遍用满（均值 18.5），900 步耗尽 3 次；trial-11 在持物悬于碗口上方时调用预算耗尽，差一次松爪。

## 用量与成本

- 模型 API usage/cost：null（子 agent 接入无 API usage 字段，按协议不伪造、不填 0）。
- 编码工具侧子 agent token 消耗（含读图与邮箱协议开销，**不可与 API usage 互比**）：合计 97.4M tokens，均值 4.87M/trial。
- 逐 trial 子 agent 用量（tokens / 工具调用数 / 运行时长 / agent id）已记录在 `subagent-usage.csv` 与 `subagent-usage.json`；该计数是编排工具对整个 worker 运行（含读图、邮箱协议轮次）的统计口径，不是逐次 policy 调用的模型 API 用量。最高 trial-08 6.71M（20 次决策用满），最低 trial-12 1.93M（12 次决策即放弃）。

## 与 GPT 批次的关系

- 本批次结果保存在独立 glm 目录（本目录），GPT 批次 `logs-persistent-subagent20-20260915-113640/` 未做任何改动；其 trial-02 中断记录冻结于 `gpt-trial02-freeze-20260915-1153/`（含续跑说明）。
- 两模型共享同一冻结初态库（SHA256 汇总 50565a9e7bb064b7…），具备按 seed 配对比较的条件；但 GPT 批次尚未完成 20 个有效 trial，**当前不存在可下的两模型对照结论**。
- 本结果限于该 20 场景冻结集与 persistent-subagent 接入组，不外推泛化，不与 Astra 真机 95% 直接比较。

## 评分流程

20 个匿名证据包（去模型 notes/自动成绩/身份）由 4 个盲评 agent 按 0–4 阶段标准独立评分；边界样本（trial-03）追加第二独立盲评并经物理轨迹交叉核验，两轮一致。映射仅在 `blind-review-map.json`，评审全程不可见 trial 身份。

# DeepSeek V4.1 标准 20 次子 agent 测试报告（bowl-eval-v1）

- 批次：`deepseekv4.1-standard20-20260915-1435`；接入方式：persistent-subagent（每 trial 一个全新无历史子 agent 连续完成，保留本 trial 记忆；与 GPT/GLM 批次同组定义，已披露与独立 API 条件的差异）。
- 决策模型标签：deepseek-v4.1-flash-subagent（确切快照 unknown；opencode-go 子 agent 后端无法独立核实）。
- 协议：bowl-eval-v1 `robosuite-bowl-paired20-v1`，20 调用 / 900 步 / effort=medium 请求 / max_speed_frac=0.25 / image_horizon=2；共享冻结初态库 seeds 2000–2019（与 GPT/GLM 同一 bank，逐 trial 初态精确核对全部 matched）。
- 调度披露：先 5 路并发、后补满至 15 路（推理等待期物理暂停、trial 间环境隔离、初态逐 trial 核对不受影响）；**wall_seconds 不可作串行比较**。
- 基础设施重试：trial-07 / trial-16 首跑因子 agent TLS 证书错误无效（原 attempt 保留于 `trial-XX-infra-void-attempt1/`），按 SOP 以同一初态同一配置整 trial 重跑，采用 attempt 2；最终 20/20 有效（22 attempts）。

## 主结果（评分=盲评 stage_max；成功率按 stage=4 计）

| 指标 | 值 |
|---|---|
| 审核成功率（stage=4） | **1 / 20**（trial-09） |
| env_success（自动判定） | **1 / 20**，与盲评完全一致，无假阳/假阴 |
| 平均最高阶段 | 0.60 |
| 阶段分布 | stage0 ×14，stage1 ×3，stage2 ×1，stage3 ×1，stage4 ×1 |
| 调用数 | 平均 18.1（上限 20） |
| 控制步数 | 平均 715（上限 900；7 个 trial 步数耗尽） |
| 终止原因 | {'give_up': 12, 'max_steps': 7, 'success': 1} |
| 盲评覆盖 | 20/20（7 个边界样本经第二轮独立盲评，两轮一致） |
| 记录检查 | 20/20 PASS（audit_standard_trials.py） |

最高阶段 trial：**trial-09（seed 2008）stage 4** —— 第 419 步建立抓持、举升至底高 0.904m，自 514 步起持物位于碗心 3cm 内（末段 6–9mm），第 599 步释放后静置于碗内（终态距碗心 r≈8mm、底高 0.795m），腕镜帧确认方块卧于蓝碗中。次高：trial-02（seed 2001）stage 3 —— 夹举到 1.16m、持物经过碗口上方（块心距碗心最近 4.4cm，属边界；两轮盲评均判 3），释放偏出 8.8cm 落至碗外壁；trial-11 stage 2（两次抓举成功、从未接近碗 0.19m 内）。

## 值得记录的现象（E2，本冻结场景集）

1. **盲评防幻觉有效（双向偏离）**：trial-01 模型自述“已夹住并提升”，盲评 stage 1（仅推挪 2cm，`grasped` 全程 false、底高未离桌）；trial-12 自述“方块位于指间、执行闭合”，盲评 stage 0（全程无接触、位姿逐位不变）；反向地 trial-09 模型自述“无法确认是否入碗”，物理与两轮盲评均确认 stage 4。逐 trial 见 `review.json` 与 `blind-review/` 原始评审。
2. **执行时序怪癖跨模型复现**：多个 trial 独立发现“位置型 move_to 仅走 1–3 步、夹爪开合转变走 ~79–80 步”，并被主动用于延长插值窗口——与 SOP §4 预先披露、GLM 批次同样观测到的“重复指定 gripper 改变 chunk 长度”行为一致，全模型同工具，属协议内行为。
3. **抓取建立是主要瓶颈**：14 个 trial 从未接触方块（夹爪闭合于方块旁 2–6cm 的空气、测距读数 0.016 全闭）；3 个 stage1 均为“推动/蹭到而非抓取”（推挪 0.8–4cm）。腕相机斜装 + eef 参考高于指垫造成 1–3cm 系统性对中偏差。
4. **释放精度是第二瓶颈**：进入 stage≥2 的 3 个 trial 中，仅 trial-09 干净入碗；trial-02 越过碗口后释放偏出，trial-11 未到碗口上方。
5. **预算行为**：调用均值 18.1/20（give_up 12、步数耗尽 7、success 1）；14 个 stage0 trial 平均仍消耗约 18 次调用，探索成本高。

## 用量与成本

- 模型 API usage/cost：null（子 agent 接入无 API usage 字段，按协议不伪造、不填 0）。
- 编码工具侧子 agent token 消耗（含读图与邮箱协议开销，**不可与 API usage 互比**）：20 个有效 trial 合计 **155.5M** tokens，均值 7.78M/trial（2.63M–15.22M；最高 trial-07 为 15.2M）。逐 trial（tokens / 工具调用数 / 运行时长 / agent id）见 `subagent-usage.csv` 与 `subagent-usage.json`；含两次无效 attempt、盲评与编排会话的全量口径另见 `usage-tool-side.json`（合计约 185M）。
- 时间：单 trial 墙钟 1188–4334s（均值 2450s；并发执行，不可作串行比较）；361 次决策等待均值 135s；仿真时长合计 715s。逐项见 `summary.csv`、`call-XX-started|ended.json`、`physics.jsonl`。
- 时间去向（20 trial 合计）：**99.2% 在等模型决策**（其中约 100% 是 LLM 往返时延：每次决策平均 2.57 次往返、单次中位 34.7s），物理执行仅 0.7%（~23 ms/控制步）；逐 trial 分解与瓶颈分析见 `timing-analysis.md`，机器可读表 `timing-breakdown.csv`。

## 与其它批次的关系

- 本批次结果保存在独立目录（本目录，本机 `logs-deepseekv4.1-subagent20-20260915-1435/`）；GPT 批次与 GLM 批次目录未做任何改动。
- 与 GLM 批次（0/20，平均 0.65）按 seed 配对（同库同协议同接入组，n=20）：DeepSeek 更高 5 个 seed（01、02、09、11、16），GLM 更高 7 个 seed（03、07、08、15、17、19、20），8 个持平；聚合几乎同分，差异集中在个别 trial 抓取建立与否。**单批 20 场景、成功率低，不构成模型优劣结论。**
- 本批是标准系列（GLM/DeepSeek/GPT/Qwen 各批次）中首个出现完整“抓取→举升→入碗静置”的 trial（trial-09）。
- 本结果限于该 20 场景冻结集与 persistent-subagent 接入组，不外推泛化，不与 Astra 真机 95% 直接比较。

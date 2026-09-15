# gpt-6-astra — 标准批次结果（按用户决定以现有 5 条收口）

- 接入方式：persistent-subagent（每 trial 一个全新 GPT-6 Astra / medium 子 agent 连续完成，保留本 trial 记忆）；与其它模型共享同一冻结初态库
- **env_success（自动判定）：5/5**（5 条全部 `termination=success`，429–447 步 / 10–14 次决策）
- 盲评已出 2/5：trial-01、trial-02 均 **stage_max=3**（`stage_upper_bound=4`、`boundary_pending=true`——自动判成功但末帧静置证据不足，按口径不写成模型失败也不直接确认）；其余 3 条盲评未出分
- trial-02 原 attempt 基础设施失败（第 14 决策等待超时），已从同一冻结初态整 trial 重跑（`trial-02-attempt-02`），原 attempt 不计成绩
- 收口时 trial-06 仍在运行，未计入本批；权威逐 trial 表见 `summary.csv`，复核口径见 `RUN_STATUS.md`
- 视频：`trials/trial-NN/video.mp4`；每步轨迹与指令：`harness/` 与 `incoming-*.json`；子 agent 终报：`*-agent-final.txt`

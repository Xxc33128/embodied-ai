# deepseek-v4.1-flash — 标准批次结果（20 trials + 2 基础设施重试 attempt）

- decision_model（manifest 标签）：`deepseek-v4.1-flash-subagent`；与其它模型共享同一冻结初态库（`experiment.json` / `initializations/`）
- 盲评（20/20 已评）：**stage=4 成功率 1/20**（trial-09）；stage 分布 `{'0': 14, '1': 3, '2': 1, '3': 1, '4': 1}`
- env_success 与盲评 stage=4 **完全一致**（均仅 trial-09），无假阳/假阴分歧
- trial-02（stage=3）为边界样本，经第二轮独立评审确认一致；见 `summary.csv` 的 `reviewed_stage` / `review_boundary` 列
- trial-07 / trial-16 各有 1 个基础设施失败 attempt（`trial-*-infra-void-attempt1`，0 次决策），已从同一冻结初态整 trial 重跑，不计成绩
- 盲评依据：`blind-review/<clip>/evidence.json` + `blind-review-map.json`（评审为 deepseek-v4.1-flash 子 agent，对模型身份/顺序/notes/自动分盲；复核口径见 `RUN_STATUS.md`）
- 逐 trial 明细：`summary.csv`（权威表，含 `reviewed_stage` / `recording_checks`）；视频在 `trials/trial-NN/video.mp4`；每步轨迹与指令在 `trials/trial-NN/harness/` 与 `incoming-*.json`

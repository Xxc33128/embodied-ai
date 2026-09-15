# glm-5.3-flash — 标准批次结果（20 trials）

- decision_model（manifest 标签）：`glm-subagent (GLM-5.3-Flash)`；scene seeds 与其它模型共享同一冻结初态库（`experiment.json` / `initializations/`）
- **success_at_end：0/20**；盲评 stage 分布：`{0: 11, 1: 6, 2: 2, 3: 1}`（20/20 已盲评）
- 逐 trial 明细见 `summary.csv`；视频在 `trials/trial-NN/video.mp4`（三相机渲染）
- 每步轨迹与指令：`trials/trial-NN/harness/`（transcripts / actions / wire calls.jsonl）与 `incoming-*.json`（模型逐轮决策）

# deepseek-v4.1-flash — 标准批次结果（22 trials）

- decision_model（manifest 标签）：`deepseek-v4.1-flash-subagent`；scene seeds 与其它模型共享同一冻结初态库（`experiment.json` / `initializations/`）
- **success_at_end：1/22**；**盲评尚未运行**（`review.json` 缺失；批次级 blind-review 目录为预备材料）
- 逐 trial 明细见 `summary.csv`；视频在 `trials/trial-NN/video.mp4`（三相机渲染）
- 每步轨迹与指令：`trials/trial-NN/harness/`（transcripts / actions / wire calls.jsonl）与 `incoming-*.json`（模型逐轮决策）

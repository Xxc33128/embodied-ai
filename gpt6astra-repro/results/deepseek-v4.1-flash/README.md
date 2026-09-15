# deepseek-v4.1-flash — 标准批次结果（20 trials）

- decision_model（manifest 标签）：`deepseek-v4.1-flash-subagent`；scene seeds 与其它模型共享同一冻结初态库（`experiment.json` / `initializations/`）
- **env_success：1/20；盲评 stage=4：1/20**；盲评 stage 分布：`{0:14, 1:3, 2:1, 3:1, 4:1}`（平均最高阶段 0.60；20/20 已盲评，7 个边界样本经第二轮独立盲评确认一致）
- 唯一成功 trial-09（seed 2008）：抓取 → 举升 → 碗心上方 6–9mm → 释放 → 静置于碗内（r≈8mm）；trial-02 止步 stage 3（持物经过碗口上方时块心距碗心 4.4cm，属边界，两轮盲评均判 3；释放后落碗外 10cm）
- 执行 22 attempts：trial-07 与 trial-16 首次因基础设施错误（子 agent TLS 证书失效）无效，原 attempt 保留于 `trial-XX-infra-void-attempt1/`，按 SOP 整 trial 重跑后采用 attempt 2（见 `infrastructure-attempts.json`）
- 逐 trial 明细见 `summary.csv`；视频在 `trials/trial-NN/video.mp4`（三相机渲染）
- 每步轨迹与指令：`trials/trial-NN/harness/`（transcripts / actions / wire calls.jsonl）与 `incoming-*.json`（模型逐轮决策）
- 盲评依据：`trials/trial-NN/review.json`；匿名材料与两轮评审原始输出在 `blind-review/`

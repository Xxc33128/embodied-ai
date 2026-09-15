# qwen3.8-flash — 标准批次结果（20 trials，已收口）

- decision_model（manifest 标签）：`qwen3.8-flash-subagent`；与其它模型共享同一冻结初态库（`experiment.json` / `initializations/`）
- 记录检查 20/20 PASS，error attempt 0 个
- 盲评两轮完成（复核模型 qwen3.8-max，对模型身份与 notes 盲）：**stage=4 成功率 4/20**，平均最高阶段 1.45，stage 分布 `{'0': 6, '1': 8, '2': 1, '3': 1, '4': 4}`
- env_success 与盲评无分歧；trial-19 为边界样本（第二复核按包内轨迹记 4，仅看画面为 3，保守读法 3/20 已在 `review.json` 标注）
- 运行内分组注记（见 `RUN_STATUS.md`）：单 worker 全程 12 行 = 3/12；429 后换 worker 8 行 = 1/8——结论只建立在同口径子组上
- 批次期间 runner 被 standard 路径修复改动过候选列表，协议字节经逐次 SHA256 校验未变（详见 `RUN_STATUS.md` 收口节）
- 逐 trial 明细：`summary.csv`（权威表）；视频 `trials/trial-NN/video.mp4`；轨迹与指令 `harness/` + `incoming-*.json`；与 glm/deepseek 同 seed 对照见 `paired-by-seed.csv`
- 大件（模型原图请求、导出帧、wire blob、盲评包、中止 attempt）留本机，清单见 `LOCAL_ONLY_MANIFEST.json`

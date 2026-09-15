# DeepSeek v4.1 标准批次执行状态

用户指示（2026-09-15 下午）：按 bowl 标准测试流程跑 20 次；每 trial 一个全新子 agent 连续完成整个 trial；结果放本目录，不触碰既有 GPT/GLM 批次。

- 模型：deepseek-v4.1-flash（opencode-go 子 agent 后端；快照不能独立验证，见 experiment.json）
- 接入方式：persistent-subagent（每 trial 一个无历史子 agent，保留本 trial 记忆；wire 只带最近 2 组图像，但已看过的图无法遗忘 — 与严格 API 等价的差异已披露，与 GPT/GLM 批次同组口径）
- 协议：robosuite-bowl-paired20-v1，seed 2000–2019，20 调用 / 900 控制步 / max_speed_frac 0.25 / effort medium；提示词与物理未改
- 场景 bank：复制自 logs-persistent-subagent20-20260915-113640/initializations，逐文件 diff 一致（aggregate sha256 见 experiment.json）
- 公开请求（模型可见）：/tmp/bowl-deepseek41-20260915-1435/trial-XX；私有记录：本目录 trial-XX
- 记录：每控制步 20fps 三视图 video.mp4（front|side|wrist）；initial-/final- 三路 PNG；公开请求中即模型所见图原图；物理轨迹 physics.jsonl；请求/回复 outgoing/incoming；harness 原始 log
- 并发：波次执行（每波 5 个 trial，顺序 seed 2000+i），与 GLM 批次观察到的并发度一致；墙钟受机器负载影响，按相对时长解释

进度：
- [ ] wave1 trial-01..05
- [ ] wave2 trial-06..10
- [ ] wave3 trial-11..15
- [ ] wave4 trial-16..20
- [ ] audit_standard_trials.py → summary.csv / recording-audit.json / index.html / frames
- [ ] prepare_blind_review.py → 匿名复核包（评审不接触模型身份与 notes）
- [ ] 匿名阶段复核（stage_max 0–4）并归档 review.json

收口说明：trial 完成事件追加到 dispatch.jsonl；自动 env_success 与人工复核 stage 分开；usage/cost 无法取得时填 null。

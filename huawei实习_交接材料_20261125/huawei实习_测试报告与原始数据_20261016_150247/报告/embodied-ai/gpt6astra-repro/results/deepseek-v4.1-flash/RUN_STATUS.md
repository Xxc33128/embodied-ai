# DeepSeek v4.1 标准批次执行状态

**状态：完成（20/20 有效 trial）**，2026-09-15 14:35 启动，20:00 前收口。

用户指示：按 bowl 标准测试流程跑 20 次；每 trial 一个全新子 agent 连续完成整个 trial；结果放本目录，不触碰既有 GPT/GLM 批次。

## 结果摘要

| 指标 | 值 |
|---|---|
| env_success | **1/20**（trial-09，seed 2008） |
| 盲评 stage 分布 | 0×14 / 1×3 / 2×1 / 3×1 / **4×1** |
| 盲评成功率 | **1/20 = 5%**，平均最高阶段 0.60 |
| 记录检查 | 20/20 PASS（视频/帧/请求/动作/物理/终止原因逐项通过） |
| 边界样本 | 7 个（stage≥1 或与模型自述冲突）经第二独立盲评，两轮一致 |

- stage 1：trial-01（推挪 2cm）、trial-07（推挪 ~4cm）、trial-16（末步触碰 8mm）
- stage 2：trial-11（两次抓起升到 ~0.9m，但从未接近碗 0.19m 内）
- stage 3：trial-02（抓举到 1.16m，碗口上方经过时距碗心 4.4cm——边界，两轮均判 3；释放后落碗外 10cm）
- stage 4：trial-09（抓取→举升→碗心正上方 6–9mm→释放→静置于碗内，r=8mm）
- 主要失败模式：14/20 从未接触方块（闭合发生在空间旁/上方几 cm），抓取建立是瓶颈

## 规程与披露

- 模型：deepseek-v4.1-flash（opencode-go 子 agent 后端；快照不能独立验证，见 experiment.json）
- 接入方式：persistent-subagent（每 trial 一个无历史子 agent 连续完成；wire 只带最近 2 组图像，但已看过的图无法遗忘——与严格 API 等价的差异已披露，与 GPT/GLM 批次同组口径）
- 协议：robosuite-bowl-paired20-v1，seed 2000–2019，20 调用 / 900 控制步 / max_speed_frac 0.25 / effort medium；提示词、物理、评分未改
- 场景 bank：复制自 logs-persistent-subagent20-20260915-113640/initializations，逐文件 diff 一致（aggregate sha256 见 experiment.json）；每 trial 首步前精确匹配检查全部通过
- 并发：先 5 路（trial-01..05 波次），后续 15 路补满（用户要求保持 5+ 在跑）；墙钟受机器负载影响，不代表串行时长
- 基础设施重试：trial-07、trial-16 首次 attempt 因子 agent TLS 证书错误无效（见 dispatch.jsonl `attempt_void`），原 attempt 目录保留（`trial-XX-infra-void-attempt1/`），按 SOP 以同一初态同一配置整 trial 重跑，采用 attempt 2
- 用量：批次日志内 usage_tokens/cost 按规程填 null（子 agent 接入无 API usage 字段）。coding-tool 侧记账两份：`subagent-usage.csv/json`（20 个有效 trial 会话，口径对齐 GLM 批次；合计 155.5M tokens，单 trial 2.6–15.2M）与 `usage-tool-side.json`（全量原始：含 2 次无效 attempt、盲评 5 会话 18.8M、编排会话 10.4M；policy 22 会话合计 155.9M，其中 cache read 148.0M；opencode 记录参考成本 ~$3.05）
- 时间：单 trial 墙钟 1188–4334s（均值 2450s；361 次决策等待均值 135s、最长 1198s；仿真时长合计 715s）；逐项见 summary.csv、call-XX-started|ended.json（决策等待秒数）与 physics.jsonl（每控制步 wall_s）

## 记录产物（SOP §7）

- 每 trial：`video.mp4`（每控制步 20fps 三视图 front|side|wrist）、`frames/`（每次调用最新 3 路 PNG + index.json sha256）、`initial-*.png` / `final-*.png`、`outgoing/incoming-XX.json`（逐请求逐回复）、`physics.jsonl`（逐步物理真值轨迹）、`call-XX-started/ended.json`（时序）、`harness/`（原始 harness log）、`result.json`、`review.json`
- 根目录：`summary.csv`（逐 trial）、`recording-audit.json`、`index.html`（可播放录像的报告）、`dispatch.jsonl`（调度事件）、`blind-review/`（匿名包生成器 + 四组主评审 + 第二评审原始输出）、`blind-review-map.json`
- 匿名复核包副本：`/tmp/bowl-blind-review-ds41-20260915/`（身份/顺序/notes/自动分已去除）

## 复核口径

- 评审为 deepseek-v4.1-flash 子 agent（与 policy 同模型族；对模型身份、trial 顺序、policy notes、自动分盲）；trial-02 的 stage 3 与 trial-09 的 stage 4 为边界样本，均经第二轮独立评审确认一致；trial-02 附 boundary_note
- env_success 与盲评 stage 4 完全一致（均仅 trial-09），无假阳/假阴分歧
- 结论限本 20 冻结场景集，不泛化；仿真对照不可与原报告真机成功率直接比较

# 更正：本批策略 worker 的实际推理档是 xhigh，不是 medium

发现时间 2026-09-15 23:0x（用户追问"到底调用的是中还是高"后代查）。本文件是权威更正，
原先写有 `thinking=medium` 的 `experiment.json` / `RUN_STATUS.md` / `report.md` 段落与
20 个 `trial-NN/manifest.json` 内的 `decision_model_snapshot` 字符串**保持原样不改**（留痕），
以本文件为准。

## 证据链（E1，读源码定位到行）
1. `.pi/agents/bowl-policy.md` 的 frontmatter 写了 `thinking: medium` —— 这是我认为生效的设置。
2. workflow 的 agentType 解析器 **不绑 thinking**：
   `~/.pi/agent/npm/node_modules/@quintinshaw/pi-dynamic-workflows/dist/agent-registry.js:19`
   「Bound today: `tools` (allowlist), `disallowedTools` (denylist), `model`, and the markdown body (`prompt`)」；
   全文件对 `thinking` 零引用 → 该字段被静默忽略。
3. 只有 model spec 带 `:level` 后缀时，workflow 才会把 thinkingLevel 显式传下去：
   `dist/agent.js:700` `resolvedThinkingLevel = resolved.thinkingLevel;` → `:755` `...(resolvedThinkingLevel ? { thinkingLevel: resolvedThinkingLevel } : {})`。
   我的定义里 model 是 `qwen-token-plan-cn/qwen3.8-flash`（无后缀）→ 不传。
4. 于是落到 pi 的启动解析：`pi-coding-agent/dist/core/sdk.js:122-127`
   「If thinkingLevel === undefined && model → perModel = settingsManager.getModelThinkingLevel(...)」，
   而 `~/.pi/agent/settings.json` 里有 `modelThinkingLevels: {"qwen-token-plan-cn/qwen3.8-flash": "xhigh"}`（用户在设置里定的每模型档）。
5. `dist/core/sdk.js:137 clampThinkingLevel(model, "xhigh")` 且该模型 `thinkingLevelMap.xhigh = "xhigh"`、
   `compat.supportsReasoningEffort = true`、`thinkingFormat: "qwen"` → **xhigh 真的被下发到端点**，不是被夹回 medium。

## 结论与影响
- 本批 20 行 = **qwen3.8-flash + reasoning effort xhigh**（持久 worker 接入组），不是我记录的 medium。
- 与 SOP §4「统一请求 medium」的偏离方向是**更高算力**，因此 4/20（同口径 12 行 3/12）不能当作 medium 档成绩与 GLM/DeepSeek 行直接并列：
  那两个批次的后端推理档「未暴露」（父 harness 无法确认），本批则确认是 xhigh → 三行只能各自标注档位，不做等算力断言。
- 顺带：盲评复核 worker 是 `qwen3.8-max`，`modelThinkingLevels` 无该键 → 走 `core/defaults.js:1 DEFAULT_THINKING_LEVEL = "medium"`（其 frontmatter 的 `thinking: low` 同样被忽略）。
- 若将来要真按 medium 跑：显式 `agent(prompt, { model: 'qwen-token-plan-cn/qwen3.8-flash:medium' })`
  （或把该模型的每模型档设为 medium）；这是唯一可靠的降档方式，写 frontmatter 无效。

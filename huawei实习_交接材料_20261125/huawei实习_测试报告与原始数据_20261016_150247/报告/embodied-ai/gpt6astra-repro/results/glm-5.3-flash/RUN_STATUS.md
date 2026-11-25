# 本批执行状态（GLM）

- 20/20 trial 完成，20/20 记录检查 PASS，初态核对 20/20 matched。
- 每 trial 一个全新 persistent 子 agent（fork_turns=none 等价：新会话无父对话/其他 trial 信息），串行逻辑 seed 2000–2019，实际调度最多 6 并发（已披露，墙钟不作串行比较）。
- 盲评：4 名盲评 agent × 各 5 包，边界样本 trial-03 双评 + 物理交叉核验；review.json 已嵌入各 trial，summary.csv/index.html 已含 reviewed_stage。
- 用量：API usage null；子 agent 侧合计 97.4M tokens（口径见 report.md）。
- 已知披露：接入方式=persistent-subagent 组；模型快照 unknown；effort=medium 为请求值，后端实际档位不可确认。
- GPT 批次与本批完全隔离；其 trial-02 冻结见 gpt-trial02-freeze-20260915-1153/FREEZE_NOTE.md。

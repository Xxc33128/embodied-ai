# Embodied AI — 工作记录区

个人工作日志/总结仓库（fork 自 versatile-ai/embodied-ai，本人维护分支：Xxc33128/embodied-ai）。

## sonic-npu/ — SONIC（GR00T WholeBodyControl）→ 昇腾 NPU 迁移研究

实习期主要工作，**按周时间线组织**：每周一个目录，readme 讲本周故事，旁边是原始文档。

| 周次 | 主题 | 一句话结论 |
|---|---|---|---|
| [8/10–8/16](sonic-npu/weeks/2026_0810-0816_环境搭建与Isaac排坑/) | 环境搭建与 Isaac 排坑 | 双引擎环境跑通；scenedb 崩溃定位到驱动版本 |
| [8/17–8/23](sonic-npu/weeks/2026_0817-0823_Phase7接口门禁与跨引擎审计/) | Phase7 接口门禁与跨引擎审计 | Interface gate PASS 14/14；物理对齐预注册路线确定 |
| [8/24–8/30](sonic-npu/weeks/2026_0824-0830_环境收口与Sim2Sim审查链启动/) | 环境收口 + Sim2Sim 审查链启动 | 右肘 armature 不一致确立为分叉首要因素：隔离 MAE ↓74.8% |
| [8/31–9/6](sonic-npu/weeks/2026_0831-0906_科学验收收口与主报告定稿/) | 科学验收全链收口 + 主报告定稿 | P1 迁移成立 / P2 无增量 / P3 VALID；final v1 科学结论全过 |
| [9/7–9/13](sonic-npu/weeks/2026_0907-0913_修订与归档/) | 修订与归档 | ② v1.7 补数值；工作成果整理入本仓库 |

## 成果区（不按周拆分）

- [`sonic-npu/reports/`](sonic-npu/reports/) — 三份主报告现行版（② 原版解析 v1.7 / ③ 分支审计 v1.4 / ① B 路线 v1.4）+ `00_总览.md` 导航。**推荐阅读顺序：② → ③ → ①**（懂原版 → 盘现状 → 评新路）。
- [`sonic-npu/lectures/`](sonic-npu/lectures/) — 讲解提纲 + 三份讲稿（版本对应表在内）。
- [`sonic-npu/sim2sim-final/`](sonic-npu/sim2sim-final/) — Sim2Sim 最终交付报告子集（最终版报告、claim 矩阵、validation、指标图、代码快照）。
- [`sonic-npu/packages/`](sonic-npu/packages/) — 9 个实验反馈包（Git LFS）+ SHA256SUMS。
- [`sonic-npu/notes/`](sonic-npu/notes/)、[`sonic-npu/survey/`](sonic-npu/survey/)、[`sonic-npu/handover/`](sonic-npu/handover/) — 迁移实证笔记、技术综述、交接类文档。
- [`sonic-npu/MANIFEST.md`](sonic-npu/MANIFEST.md) — 完整内容索引 + 未上传大文件（1.75GB final zip 等）的 SHA256 与保存位置。

## 收录规则

周目录只收录**结论/报告/预注册类**文档（审查裁决、验收结论、调研结论、根因归档、实验计划）；执行指令与交接类不入时间线（交接文档在 `handover/`，工具类仅在周 readme 留索引行）。

## 历史

- 原 DreamZero NPU 训练交付物已于 2026-09-08 清理：报告保留在本仓库 git 历史（`git show b62887c:dreamzero/final_report.html` 可找回）；代码改动在独立仓库 zhangqin200182/dreamzero。

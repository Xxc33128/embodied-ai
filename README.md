# Embodied AI — 工作记录区

个人工作日志/总结仓库（fork 自 versatile-ai/embodied-ai，本人维护分支：Xxc33128/embodied-ai）。

## 项目索引

### sonic-npu/ — SONIC（GR00T WholeBodyControl）→ 昇腾 NPU 迁移研究

实习期主要工作：原版训练体系解析、zhangqin 分支 NPU 适配现状审计、MuJoCo Warp 全量迁移 B 路线可行性研究，以及三项目 Sim2Sim 实验链。

**推荐阅读顺序**（同一条决策链：懂原版 → 盘现状 → 评新路）：

1. `sonic-npu/SONIC原版训练体系深度解析.md`（② 基准知识，v1.7）
2. `sonic-npu/SONIC_NPU适配深度报告_zhangqin分支.md`（③ 现状审计，v1.4）
3. `sonic-npu/MuJoCo_Warp架构与昇腾NPU全量迁移B路线报告.md`（① 前瞻决策，v1.4）

导航入口：`sonic-npu/00_总览.md`；完整内容索引与大文件校验和：`sonic-npu/MANIFEST.md`。

其他子目录：`lectures/`（讲解材料）、`survey/`（技术综述）、`worklog/`（里程碑工作文档）、`env-docs/`（环境/周报）、`sim2sim-final/`（实验交付报告子集）、`notes/`（迁移实证笔记）、`packages/`（实验反馈包，Git LFS）。

## 历史

- 原 DreamZero NPU 训练交付物（final_report.html、eval 对比图、code 子模块）已于 2026-09-08 清理：报告仍保留在本仓库 git 历史（`git show b62887c:dreamzero/final_report.html` 可找回）；代码改动在独立仓库 zhangqin200182/dreamzero，不受影响。

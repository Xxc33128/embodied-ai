# MANIFEST — sonic-npu 上传清单

> 生成日期：2026-09-08。本文件是本目录内容的权威索引，包括已上传内容与**有意未上传**的大文件（含校验和与保存位置）。

## 目录结构

| 路径 | 内容 | 说明 |
|---|---|---|
| `weeks/` | **周时间线主轴**：每周一目录（8/10–9/13 共五周，9/7–9/13 周含 VLA/WAM 调研与远程 NPU 接入文档），readme 为周摘要 + 该周结论/报告/预注册类文档 | 收录规则与周索引见 `weeks/README.md`；执行清单类不上传、周 readme 留索引行 |
| `reports/` | 报告套件（同目录互链） | 00_总览 + ② 原版解析 v1.8 / ③ 分支审计 v1.4 / ① B 路线 v1.5 + ② 的配图 |
| `lectures/` | 讲解提纲 + 3 份讲稿 | 版本对应关系见 `lectures/README.md` |
| `survey/` | 知识底稿：技术综述（8/20）、SONIC 论文精读（8/14）、Humanoid-Gym / Humanoid-Lab 代码解析（9/4） | 自 Obsidian 笔记库迁入（2026-09-11） |
| `handover/` | 交接类文档（不入时间线） | 交接文档-Isaac环境搭建（历史快照，被 8/24 排障复盘取代） |
| `archive/` | 过程文档备份区（执行清单/工具类，2026-09-11 由本地迁入） | 清单见 `archive/README.md`；结论均已收录于各周文档 |
| `sim2sim-final/` | 最终交付报告子集 | 14_final_delivery（最终版报告+claim矩阵+validation）、report_v2（可读版）、r5_1_overlay（P3 指标）、code_snapshot、r4f-b31（B3.1 包报告） |
| `notes/` | mujoco_npu_migration_notes.md | A 路线实证记录（原为 GR00T-WholeBodyControl 仓库 untracked 文件，此为备份副本） |
| `packages/` | 9 个实验反馈 zip（Git LFS） | 清单见下，校验和见 `packages/SHA256SUMS.txt` |

**脱敏说明**：排障复盘与 `archive/Phase7-进度总结` 的副本均已删除泄露 PAT 断片；`weeks/2026_0907-0913_调研与归档/远程NPU接入` 为脱敏版（隐去 IP/姓名/节点归属），含真实信息的原件仅存本地。

## packages/ 清单（9 个 zip，Git LFS 存储）

| 包 | 大小 | 对应实验 |
|---|---|---|
| P1_fullbody_elbow_transfer_v1.zip | 46MB | P1 全身右肘 armature 单变量迁移（SUPPORTED_TRANSFER） |
| P2_fullbody_arm_only_v1.zip | 45MB | P2 ARM-ONLY v1（被 v1.1 取代，留作过程证据） |
| P2_fullbody_arm_only_v1_1.zip | 55MB | P2 ARM-ONLY v1.1（NO_INCREMENTAL_ARM_GROUP_EFFECT） |
| P3_lab_delay20_v1.zip | 23MB | P3 实验室 20ms 延迟基线 |
| P3_R1_R2_gate_feedback_v1.zip | 0.6MB | P3 R1/R2 门禁反馈 |
| P3_R21_R3_isaac_d20_10s_v1.zip | 9.4MB | Isaac 20ms/10s 正式仿真（R3_ACCEPTED） |
| P3_R4_final_gate_v1.zip | 35MB | R4 终验（VALID_P3，负测试/fail-closed） |
| P3_R5_metrics_media_v1.zip | 9MB | R5 指标与媒体（VALID_R5） |
| R4F_B1_Lab10s_N5R2_B31_20260831.zip | 70MB | B3.1 最终状态包（elbow MAE 降 74.7696%） |

## 本地清理记录（2026-09-11）

**已删除**（先验证后删）：
- 11 个实验 zip：9 个与 `packages/` LFS 内容逐字节一致（SHA256SUMS 复核）、`R4F_..._20260831 2.zip`（哈希不符的重复下载件）、`Sim2Sim_week_final_R5_1_R6_R7_v1.zip`（1.75GB，与其解包目录内容一致；SHA256 `befe09ed7d32140a91027c32482183ba38edd5d30fa498c427af7dfc8cb45cee` 留作历史校验值）
- `R4F_B1_Lab10s_N5R2_B31_20260831/` 解包目录（LFS 包 + `sim2sim-final/r4f-b31/` 报告子集已覆盖）
- `_archive/2026-08-sim2sim-review-history/`（被取代的中间审查轮产物，结论已收录于周文档）
- `artifacts.7z`（`artifacts/` 目录的重复压缩件）、6 个过程文档原件（已入 `archive/`）

**仍存本地**：`Sim2Sim_week_final_R5_1_R6_R7_v1/`（1.7GB 解包目录；`full_output_archive/` 为唯一未上传的原始数据）、`artifacts/`（20MB 跨引擎 trace）、`backup_20260903/`（主报告旧快照）。

## 同步说明

- 本目录为 2026-09-11 快照；主报告后续修订以各文件头部版本号为准。
- worklog 中的审查对象 zip 引用（如 `xxx.zip + SHA256`）多数对应 packages/ 内文件，个别中间反馈包（R1–R3、smoke、full 等）未上传，引用为文字描述不断链。

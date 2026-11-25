# Sim2Sim 周产出总览导读（2026-09-02 追加层说明）

本文件夹现已包含本周全部产出。共 4 层，**各自独立的哈希范围，互不混称**；
追加层不改动已交付 ZIP（`1ccc9180…`）的任何文件。

```
Sim2Sim_week_final_R5_1_R6_R7_v1/
├── 14_final_delivery/          # ① 最终交付核心（报告/索引/claim/validation）
│   └── FINAL_SHA256.txt        #    哈希范围：本目录
├── r5_1_overlay/               # ② R5.1 展示补丁（指标/图/媒体/门禁脚本，无大 trace）
├── code_snapshot/              # ③ 代码快照（39+2 文件）        → SHA256_CODE_SNAPSHOT.txt（41 项）
├── full_output_archive/        # ④ 全量产出存档 = 周根目录完整复制（去 __pycache__）
│   └── SHA256_FULL_OUTPUT_ARCHIVE.txt                            # 3181 项，LF，sha256sum -c 通过
└── SHA256_FEEDBACK_PACKAGE.txt # 交付 ZIP 内容清单（108 项，范围=ZIP，与③④无关）
```

## 溯源锚点（谁说了才算）

- 结论级锚点：`14_final_delivery/FINAL_CLAIM_EVIDENCE_MATRIX.csv`（CL-01…10 → 证据路径+数值+限定）。
- 包级锚点：`full_output_archive/` 内 7 个源 ZIP 实测哈希与 `14_final_delivery/README.md` 表格逐一相符
  （B31 `cbf8a6fd…` / R3 `3dd2acb8…` / P1 `8bb8ba96…` / P2 v1.1 `a48749cb…` /
  R21_R3 `2fea92f3…` / R4 `9a4ebdf3…` / R5 `4f0c55f6…`），另有最终包 `1ccc9180…` 同目录。
- 代码快照与全量存档是**便利副本**；需要哈希级复现时以 ZIP 内副本 / R4 树 `SHA256_FULL_TREE.txt` 为准。

## 重要产出速览（按阅读优先级）

1. 最终报告：`14_final_delivery/report/本周三项目Sim2Sim调研与实验总结_最终版.md` + 一页摘要
2. claim→证据矩阵：`14_final_delivery/FINAL_CLAIM_EVIDENCE_MATRIX.csv`；交付索引 `FINAL_DELIVERY_INDEX.csv`（FD-01…10）
3. P3 指标与图：`r5_1_overlay/04_analysis/`（p3_metrics.json / long / tidy CSV / 11 图 / P3_result_summary.md）
4. 四格 10s 视频：`r5_1_overlay/05_media/p3_delay_four_panel_10s.mp4`（来源与复用链 `media_manifest.json`）
5. 门禁与负测试记录：`14_final_delivery/validation/`（P3 VALID_P3/VALID_R5、P0 双模式、P2 R6、最终 11 项 + 负测试 5/5+1）
6. 原始证据（trace/视频/审计 JSON 全量）：`full_output_archive/` 各编号目录与 7 个源 ZIP
   - N5-R2 18 条隔离 trace：`full_output_archive/04_isolation_N5R2/`（或 B31 ZIP）
   - P1 6×10s / P2 9×10s / P3 4 条件 10s trace：`10_/12_/13_` 目录
   - readback 对照表：`full_output_archive/04_actuator_isolation/actuator_runtime_contract.csv`
   - Gym 100Hz 基线与历史视频：`full_output_archive/05_gym/`；SONIC 四格台账：`06_sonic/`
   - 历史/被取代版本（P3 v1 frozen、P2 v1、G0 之前各轮）也在 ④ 内，状态见各目录 STATUS/README

## 校验命令

```bash
# 各层哈希范围分别复核（WSL）
cd <本目录>/full_output_archive   && sha256sum -c SHA256_FULL_OUTPUT_ARCHIVE.txt --quiet
cd <本目录>/code_snapshot         && sha256sum -c SHA256_CODE_SNAPSHOT.txt       --quiet
cd <本目录>/14_final_delivery     && sha256sum -c FINAL_SHA256.txt               --quiet
cd <本目录>                        && sha256sum -c SHA256_FEEDBACK_PACKAGE.txt   --quiet
# 最终 validator（需在周根目录环境跑，见 14_final_delivery/README.md；④ 即完整周根目录副本）
```

# 最终交付 README（R6/R7，2026-09-02）

本目录是三项目 Sim2Sim 本周工作的最终交付索引与报告。原两日计划的基础交付以本目录验证通过为关闭条件。

## 内容

```
14_final_delivery/
├── FINAL_DELIVERY_INDEX.csv        # FD-01..FD-10：路径/SHA256/payload/校验命令/等级/supersedes
├── FINAL_CLAIM_EVIDENCE_MATRIX.csv # CL-01..CL-10：结论→证据→数值→限定
├── FINAL_SHA256.txt                # 本目录全部文件哈希（LF）
├── README.md                       # 本文件
├── validate_final_delivery.py      # 最终 validator（embedded / full-provenance）
├── validation/                     # P0 双模式、P2 R6 负测与重算、P3 R4/R5 gate、最终验证日志
├── report/                         # 最终报告 + 一页摘要
└── assets/                         # 报告引用的图/帧/媒体清单（p3 R5.1 版、n5、p1、p2、lab）
```

## 需要同时交付的源 ZIP（full-provenance 依赖，不重复嵌入本包）

| 文件 | SHA256 |
|---|---|
| feedback/B31_20260831/output/R4F_B1_Lab10s_N5R2_B31_20260831.zip | cbf8a6fd92e7947bf1f5136804e30a770ba7ba35f02fc5f193a54986fa7d476f |
| feedback/R3_interface_gate_v2.zip | 3dd2acb8e90cee51f0f4ddf860a35f8ed0a9737ccab5908509f9aaeeb5f1161f |
| P1_fullbody_elbow_transfer_v1.zip | 8bb8ba9658c81160c7d6542e2b418637b7b736688b8a226ba885327b6d8b6824 |
| P2_fullbody_arm_only_v1_1.zip | a48749cb4513df25c1a74f6d52bd803f6c97e15ab7c7d8cbb290bfab4095398a |
| P3_R21_R3_isaac_d20_10s_v1.zip | 2fea92f341557855fabf9e346754d81fc929baae1190d6ac7162bd0bc860f864 |
| P3_R4_final_gate_v1.zip | 9a4ebdf3c75afa72bbf25d108cbce3d75d34eb5fff00aa36aa973493ee64a3e6 |
| P3_R5_metrics_media_v1.zip | 4f0c55f6725289fb82c6085a8b2ef83543f21fe67a39f97d947e4c0cb7cbe1d7 |

哈希范围命名约定：完整树 = `SHA256_FULL_TREE.txt`（`13_p3_delay20_r4_build/13_p3_delay20/`，108 项）；小反馈包 = 各自 `SHA256.txt`（R5 包 32 项）；本目录 = `FINAL_SHA256.txt`。三者分别报告，不混称。

计划文档中的 B31 `2f4e8c15…` 与 R3 `c1a1011b…` 占位哈希从未交付过对应文件；最终索引以实际文件重算为准（`11_p0_evidence_closure/00_sources/HASH_RECONCILIATION.md`），payload 数 R3=47 与计划一致。

## 校验

```bash
# embedded（只查本目录内部一致性）
python validate_final_delivery.py --root . --mode embedded
# full-provenance（含源 ZIP 哈希与 master/claim 路径解析）
python validate_final_delivery.py --root . --source-root <week_root> --mode full-provenance
# 期望 decision=VALID_FINAL_DELIVERY exit 0
```

## 阅读顺序

1. `report/本周三项目Sim2Sim一页摘要_最终版.md`
2. `report/本周三项目Sim2Sim调研与实验总结_最终版.md`
3. `FINAL_CLAIM_EVIDENCE_MATRIX.csv`（每条结论的证据与限定）
4. `FINAL_DELIVERY_INDEX.csv`（包级证据与复现命令）

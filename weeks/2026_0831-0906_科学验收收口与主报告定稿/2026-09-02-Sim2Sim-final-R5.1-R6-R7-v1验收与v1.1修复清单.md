# Sim2Sim Final R5.1/R6/R7 v1 验收与 v1.1 修复清单

日期：2026-09-02  
审查对象：`Sim2Sim_week_final_R5_1_R6_R7_v1.zip`  
对象 SHA256：`1ccc9180987df72cf99d47c1650c60525e73140021a5d4feae5109aea78059e8`

## 1. 验收结论

当前结论：**R5.1 指标与媒体修订通过，P2 R6 科学结论通过；R7 最终交付包暂不关闭。**

阻塞原因集中在最终包的独立复现能力：从一个空目录新鲜解压本 ZIP 后，执行 README 声明的 embedded 命令会返回 `INVALID_FINAL_DELIVERY`。包内保存的 `VALID_FINAL_DELIVERY` 日志来自 Windows 完整周目录，运行时读取了最终包之外的文件，因此不能代表当前 ZIP 自身通过。

这次不需要启动 Isaac Kit、MuJoCo 或新仿真。v1.1 只修最终封装、validator 语义、源包清单和报告中的两处文字事实。

| 检查项 | 结果 | 说明 |
|---|---|---|
| ZIP CRC | PASS | 压缩内容可读取 |
| `SHA256_FEEDBACK_PACKAGE.txt` | PASS | 108/108 |
| `14_final_delivery/FINAL_SHA256.txt` | PASS | 68/68 |
| R4 门禁独立复跑 | PASS | `VALID_P3`，Isaac shift=4、MJ shift=10 |
| R5.1 独立重算 | PASS | `VALID_R5`，crosscheck mismatch=0 |
| R5.1 长表单位 | PASS | 120 行，无 `mixed`；单位按指标拆分 |
| R5.1 图、方向角和媒体语义 | PASS | 图 08/09/10/11 与 media manifest 修订正确 |
| P2 R6 validator 独立重算 | PASS | validity 全项 true，decision=`NO_INCREMENTAL_ARM_GROUP_EFFECT` |
| 最终报告结构 | 基本 PASS | 主线清楚，仍有 2 处内容和 1 处路径要修 |
| fresh-extract embedded | **FAIL** | 18 条错误，见 §3 |
| full-provenance 独立复现 | **未完成** | 缺少索引指定的 B31/R3 精确字节源包 |
| 最终负测试复现 | **未完成** | 包内只有 JSON 结果，没有负测试 runner |
| macOS `unzip` 兼容性 | WARN | 中文文件名编码触发解压错误；`ditto` 可解 |

## 2. 已确认可保留的科学结果

### 2.1 P3 R5.1

在空临时目录中重新叠加 R4、R5 和本包 `r5_1_overlay` 后：

```text
R4 decision = VALID_P3
R5.1 decision = VALID_R5
crosscheck mismatches = 0
long CSV rows = 120
mixed unit rows = 0
```

方向角已经覆盖 pelvis、左右 ankle、左右 wrist。图 08 只画真实 body orientation；图 09 按 m/rad 分面；图 10 拆分 torque RMS、peak 和 saturation；图 11 只呈现无量纲敏感度比。媒体清单也已区分容器播放时长与 500 control-cycle/10 s 仿真覆盖。

因此，P3 的结论继续保留：

- 0/20 ms 四条件均完成 500 周期、无 fall、无 reset；
- requested tracking 上升，effective tracking 接近 D00；
- world body gap、pelvis-relative body gap 和 orientation gap 呈现 tradeoff；
- 20 ms 下的存活事实成立；本周没有预注册鲁棒性阈值，不能给出通用鲁棒性 PASS/FAIL。

### 2.2 P2 R6

使用本包中的 `p2_validate_r6.py` 对本机收到的原始 `P2_fullbody_arm_only_v1_1.zip` 新鲜解压目录重算，结果为：

```text
decision                  NO_INCREMENTAL_ARM_GROUP_EFFECT
validity_pass             true
mean_rel                  -0.0500718152
mean_abs                  -0.0022478428 rad
n_improved_joints         5 / 13
median_improvement        -0.0012236834 rad
right_elbow_preservation  true
system_guard              false
```

`input_hash/asset/config/runtime/repeat/time/joint/shape/nan` 九项 validity 全部为 true。P2 的科学结论不需要改：右肘-only 仍是当前最小有效修改；把其余 13 个臂关节都设为 0.01 没有新增整体收益，并带来系统指标代价。

## 3. 当前最终包的阻塞问题

### P0-1：embedded 模式无法在本 ZIP 内独立运行

复现方式：

```bash
# 在一个空目录新鲜解压 ZIP
cd <fresh_extract>/14_final_delivery
python validate_final_delivery.py --root . --mode embedded
```

实际结果：

```text
decision = INVALID_FINAL_DELIVERY
errors = 18
```

错误由四类组成：

1. `FINAL_DELIVERY_INDEX.csv` 中 7 个外部源 ZIP 在 embedded 模式下仍被强制检查；
2. `11_p0_evidence_closure/` 没有放入 `14_final_delivery`；
3. CL-02～CL-10 的证据路径依赖外部周目录或源 ZIP，embedded 模式仍把它们当必需文件；
4. P2 elbow XML 默认从最终目录的父目录读取，当前 ZIP 没有该文件。

因此，`validation/final_validation_embedded.json` 中保存的 `VALID_FINAL_DELIVERY` 不能由本 ZIP 新鲜解压后复现。

### P0-2：full-provenance 指定的两个源 ZIP没有随本次反馈提供

最终索引要求：

```text
B31 cbf8a6fd92e7947bf1f5136804e30a770ba7ba35f02fc5f193a54986fa7d476f
R3  3dd2acb8e90cee51f0f4ddf860a35f8ed0a9737ccab5908509f9aaeeb5f1161f
```

本机此前实际收到的同名文件为：

```text
B31 2f4e8c15fdb48a1dfd46df6407a06d68a26ebbc3a0ae9c3e5df1ba3d29e9e985
R3  c1a1011b963a2a0917e2178accf4615b71a97aa677f906c8af4b32e946252b3e
```

其余五个源 ZIP 与最终索引完全一致。当前无法从收到的文件验证 `cbf8…/3dd2…` 是否存在、内容是什么、payload 数是否分别为 207/47。

这里必须冻结一套真实字节源包。禁止只修改 CSV 中的哈希来适配当前目录。应将索引声称权威的两个 ZIP 一并交付，或者明确选择此前已收到的两个 ZIP 作为权威源，并从这些字节重新生成 P0/full-provenance 全链。两条路线只能选择一条。

### P0-3：最终和 P2 负测试缺少可执行 runner

当前包只有：

```text
validation/final_negative_tests.json
validation/p2_negative_tests_r6.json
```

没有生成这些结果的脚本。独立审查者只能阅读历史结果，不能在新鲜解压目录执行相同篡改测试。

至少补入：

```text
validation/run_final_negative_tests.py
validation/run_p2_negative_tests_r6.py
```

两个脚本都必须只接受显式 `--root/--source-root`，在临时副本中完成篡改，不能读取历史绝对路径。

### P1-1：最终报告的 Gym history 写错

报告和一页摘要分别出现了：

```text
12 DOF × 5 history
12 DOF × 5 帧
```

`05_gym/sim2sim_record.py` 与 `run_manifest.json` 的真实配置为：

```text
NUM_SINGLE_OBS = 47
FRAME_STACK = 15
NUM_OBSERVATIONS = 705
control_dt = 0.01 s
history window = 0.150 s
first-to-last = 0.140 s
```

统一改成：

```text
观测为 15 帧 × 47 维 = 705 维；100 Hz 下样本窗为 0.150 s，首末样本间隔为 0.140 s。
```

其中 12 是动作/关节维数，不是 history 帧数。

### P1-2：报告中的残余差异句缺少对应口径

当前文字：

```text
残余 ~0.013 rad 量级的孤立差异指向 frictionloss/PD 时序等待查项
```

这句话没有说明是 MAE、RMSE、peak 还是 native-vs-matched 点差，claim matrix 也没有绑定该数字。改为：

```text
armature 对齐后，隔离实验仍有 0.001704 rad 的 |MJ−Isaac| MAE。现有实验没有继续分解这部分差异；frictionloss、PD 执行时序、physics dt 和 solver 仅列为后续候选机制。
```

### P1-3：最终报告仍写入 Windows 绝对根路径

删除：

```text
文中相对路径以 E:\sim2sim-week-2026-08-26\ 为根。
```

改成：

```text
报告内 `../assets/` 路径相对 `14_final_delivery/report/`；源证据路径按 `FINAL_DELIVERY_INDEX.csv` 与 `FINAL_CLAIM_EVIDENCE_MATRIX.csv` 解析。
```

同时删除 `validate_final_delivery.py` 对“为根”绝对路径的白名单。最终报告应通过统一的绝对路径禁止规则。

### P1-4：中文文件名 ZIP 的跨平台兼容性

本 ZIP 在 macOS `/usr/bin/unzip` 下出现中文文件名编码错误，`ditto -x -k` 可以正常解压，CRC 与哈希均通过。最终 v1.1 请使用 Python 3 `zipfile` 重新打包，确保非 ASCII 文件名写入 UTF-8 标志。打包后至少用以下两种方式各做一次新鲜解压：

```bash
python -m zipfile -e Sim2Sim_week_final_R5_1_R6_R7_v1_1.zip <empty_dir_1>
unzip Sim2Sim_week_final_R5_1_R6_R7_v1_1.zip -d <empty_dir_2>
```

若 Win 自带 `unzip` 不便测试，至少使用 Python `zipfile` 与 7-Zip 测试，并在 README 写明 UTF-8 打包方式。

## 4. v1.1 推荐目录结构

让 embedded 真正只依赖 `14_final_delivery/`：

```text
14_final_delivery/
├── README.md
├── FINAL_DELIVERY_INDEX.csv
├── FINAL_CLAIM_EVIDENCE_MATRIX.csv
├── FINAL_SHA256.txt
├── validate_final_delivery.py
├── report/
├── assets/
│   └── p3/
│       └── p3_metrics_long.csv
├── embedded_evidence/
│   ├── p0/                         # 完整 11_p0_evidence_closure overlay
│   ├── n5/elbow_gates.json
│   ├── p1/p1_final_gate.json
│   ├── p2/
│   │   ├── p2_final_gate_recomputed_r6.json
│   │   └── l7_29dof_neck_fixed_elbow_matched.xml
│   ├── p3/
│   │   ├── p3_r4_final_gate.json
│   │   └── p3_metrics.json
│   └── sonic/sonic_II_IM_MM_MI_ledger.csv
└── validation/
    ├── validate_evidence_closure_r6.py
    ├── p2_validate_r6.py
    ├── audit_fullbody_arm_assets_r6.py
    ├── run_p2_negative_tests_r6.py
    ├── run_final_negative_tests.py
    └── validation outputs...
```

如不希望复制整个 P0 overlay，至少需要把 CL-02～CL-10 的最终证据摘要复制到 `embedded_evidence/`，并在索引中明确 `embedded_evidence_path` 与 `source_evidence_path`。推荐复制完整 P0 overlay，体积小、证据链更直观。

## 5. validator 修改要求

### Step 5.1：区分 embedded 条目和 external 条目

在 `FINAL_DELIVERY_INDEX.csv` 增加字段：

```text
storage_scope = embedded | external
```

规则：

- embedded 模式只强制验证 `storage_scope=embedded`；
- external 行必须有 64 位哈希和 `required_for_full_provenance=true`，embedded 模式将它们记为 `EXTERNAL_SOURCE_NOT_CHECKED`；
- full-provenance 模式同时验证 embedded 与 external；
- full-provenance 缺任一外部源 ZIP、哈希不一致或包内精确路径不存在时 exit 1。

### Step 5.2：claim matrix 使用双路径

增加：

```text
embedded_evidence_path
source_evidence_path
```

embedded 模式验证前者；full-provenance 模式同时验证两者。包内路径必须按完整规范化路径匹配，不能只取 basename 后用 `endswith()`。一条 claim 含多个证据路径时，每个必需路径都要验证，不能只检查分号前第一项。

### Step 5.3：P2 elbow 默认读取包内证据

embedded 默认路径：

```text
embedded_evidence/p2/l7_29dof_neck_fixed_elbow_matched.xml
```

继续固定：

```text
SHA256 = ab1e1862367be20a91edeefca12c6faedbc3e99a151d662cf42b4c56da98e7f9
```

`--p2-elbow-xml` 只作为显式复现/负测试参数，不允许从父目录或历史绝对路径自动回退。

### Step 5.4：R5.1 长表放入 final 内部

把 R5.1 长表复制为：

```text
assets/p3/p3_metrics_long.csv
```

embedded 默认读取该文件并执行单位检查。长表缺失必须 fail，不能只写 note 后继续 VALID。

### Step 5.5：P0 的两个模式都要真实重算

最终 validator 不应只读取 `p0_embedded_validation.json` 的历史 decision。embedded 运行时应调用：

```bash
python validation/validate_evidence_closure_r6.py \
  --root embedded_evidence/p0 \
  --mode embedded \
  --output <temporary_output>
```

full-provenance 再调用：

```bash
python validation/validate_evidence_closure_r6.py \
  --root embedded_evidence/p0 \
  --source-root <source_root> \
  --mode full-provenance \
  --output <temporary_output>
```

临时输出不能覆盖包内冻结 JSON。

### Step 5.6：修正索引中的权威源包叙述

`FINAL_DELIVERY_INDEX.csv` 和 README 当前写“2f4e8c15/c1a1011b 从未交付”。本机确实收到过这两个字节文件，这种表述与交付历史冲突。改成中性事实：

```text
本最终构建选用的冻结源为 <sha>；其他同名 ZIP 属于不同打包版本，不参与本构建的 full-provenance。
```

然后附上本构建实际使用的七个精确源 ZIP。不要把不同打包版本称为“未交付”。

## 6. 源包冻结与 full-provenance 操作

### Step 6.1：创建只读源包集合

建议保持索引中的相对目录：

```text
source_bundle/
├── feedback/B31_20260831/output/R4F_B1_Lab10s_N5R2_B31_20260831.zip
├── feedback/R3_interface_gate_v2.zip
├── P1_fullbody_elbow_transfer_v1.zip
├── P2_fullbody_arm_only_v1_1.zip
├── P3_R21_R3_isaac_d20_10s_v1.zip
├── P3_R4_final_gate_v1.zip
└── P3_R5_metrics_media_v1.zip
```

### Step 6.2：从文件字节生成索引

逐个重算 SHA256 与 ZIP payload 数，将结果写入索引。不要手填计划文档中的旧值。生成后再次重算，并断言两次输出完全一致。

### Step 6.3：完整复现

```bash
python 14_final_delivery/validate_final_delivery.py \
  --root 14_final_delivery \
  --source-root source_bundle \
  --mode full-provenance \
  --output <temporary_output>/final_full.json
```

必须满足：

```text
decision = VALID_FINAL_DELIVERY
errors = []
7/7 source ZIP hash matched
P0 external_source_status = EXTERNAL_SOURCES_VERIFIED
```

## 7. 负测试要求

### 7.1 embedded 负测试

至少包含：

1. 删除报告图片 → `INVALID_FINAL_DELIVERY`；
2. 删除 embedded P0 文件 → `INVALID_FINAL_DELIVERY`；
3. 删除 embedded P2 elbow XML → `INVALID_FINAL_DELIVERY`；
4. 篡改 elbow XML 一个字节 → `INVALID_FINAL_DELIVERY`；
5. 把 R5.1 长表某行单位改成 `mixed` → `INVALID_FINAL_DELIVERY`；
6. 把 claim 的 embedded 路径改成不存在 → `INVALID_FINAL_DELIVERY`；
7. 未改动 CONTROL → `VALID_FINAL_DELIVERY`。

### 7.2 full-provenance 负测试

至少包含：

1. 缺 B31 源 ZIP；
2. 缺 R3 源 ZIP；
3. 篡改任一源 ZIP 一个字节；
4. claim 包内路径只保留同 basename 的错误文件，精确路径缺失；
5. 未改动 CONTROL。

所有负测试必须在临时副本上执行，并保存 command、returncode、decision、matched error。最终包内同时交付 runner 和 JSON 结果。

## 8. 最终报告修改要求

只改以下内容，不重写科学结果：

1. Gym observation/history 改为 `15×47=705`；
2. 删除未绑定口径的 `~0.013 rad`，换成 matched MAE `0.001704 rad` 与候选机制限定；
3. 删除 Windows 绝对根路径，改用 final 索引解析说明；
4. 报告中的 `assets/...` 若希望在 Markdown 直接展示，改成相对图片语法 `![说明](../assets/...)`；
5. 一页摘要的三张 P3 图片建议各占一行并加短标题，避免同一列表行连续放三张大图；
6. 检查全文不出现“不是……而是……”句式；
7. 保留 G0-I/G0-A、N5-R2、P1、P2、P3 的现有主线和限定。

## 9. v1.1 最终验收门禁

以下条件全部满足后，才可以宣布“原两日计划基础交付关闭”：

### Gate A：新鲜解压 embedded

```bash
cd <empty>/14_final_delivery
python validate_final_delivery.py --root . --mode embedded
```

要求：

```text
exit 0
decision = VALID_FINAL_DELIVERY
errors = []
external_source_status = EXTERNAL_SOURCE_NOT_CHECKED
```

运行目录的父目录不能预先放置 `11_p0_evidence_closure`、P2 XML、R5 build 或七个源 ZIP。

### Gate B：新鲜 source bundle full-provenance

```bash
python <final>/validate_final_delivery.py \
  --root <final> \
  --source-root <source_bundle> \
  --mode full-provenance
```

要求：exit 0、七个源包精确哈希匹配、P0 full-provenance 真实重算通过。

### Gate C：负测试

```text
embedded 6 个篡改测试全 fail-closed + CONTROL valid
full-provenance 4 个篡改/缺失测试全 fail-closed + CONTROL valid
P2 两个 elbow 测试全 fail-closed + CONTROL 保持 NO_INCREMENTAL_ARM_GROUP_EFFECT
```

### Gate D：内容和路径

```text
Gym = 15×47=705
matched residual MAE = 0.001704 rad
报告无 Win/WSL/macOS 绝对路径
全部 Markdown 图片相对路径存在
全文无“不是……而是……”句式
```

### Gate E：ZIP 与哈希

```text
ZIP CRC PASS
Python zipfile 新鲜解压 PASS
unzip/7-Zip 新鲜解压 PASS
FINAL_SHA256 100%
反馈包 SHA256 100%
```

## 10. 下一次反馈只需提供

1. `Sim2Sim_week_final_R5_1_R6_R7_v1_1.zip`；
2. 索引声称权威、但本次未提供的精确 B31 与 R3 ZIP，或者包含全部七个源包的 `source_bundle`；
3. embedded/full-provenance/negative-test 三类运行摘要。

不需要重跑 N5-R2、P1、P2、P3，也不需要新增实验。v1.1 通过后可以正式关闭原两日计划的基础交付；frictionloss、PD 时序、dt、solver、更多延迟和其他动作进入下一阶段研究，不阻塞本周总结。

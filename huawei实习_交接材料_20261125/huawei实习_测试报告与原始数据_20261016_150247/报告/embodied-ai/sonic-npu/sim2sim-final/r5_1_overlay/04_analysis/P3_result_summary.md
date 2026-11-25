# P3 结果摘要（R5，2026-09-02）

口径：dance_9 / seed=42 / 10 s（500 control cycles）/ gain=1.0 / 当前 policy·motion·canonical·model / 显式 position-target FIFO / 0 ms 与 20 ms 两档。
上游：R4 `VALID_P3`；本摘要全部数值来自 `04_analysis/p3_summarize_r2.py --root .` 对四条冻结 trace 的重算（输入哈希见 `p3_metrics.json`，交叉检查 0 mismatch，`p3_r5_validation.json=VALID_R5`）。

## 1. 存活与延迟语义（硬事实）

| 条件 | cycles | fall | reset | measured delay（trace 复算） |
|---|---:|---|---|---|
| Isaac D00 | 500 | null | 未观察到（计数器独立推断） | 0 ms |
| Isaac D20 (R2 正式) | 500 | null | 未观察到 | **4 physics steps = 20.0 ms 唯一识别**（次优 shift 差 0.419 rad ≫ 阈值） |
| MJ D00 (elbow-matched) | 500 | null | 未观察到 | 0 ms |
| MJ D20 (elbow-matched) | 500 | null | 未观察到 | **10 physics steps = 20.0 ms 唯一识别**（control shift=1，fifo_used=true） |

⇒ **BOTH_SURVIVE_20MS_ON_DANCE9**（受控等时延注入，非 Isaac 原生 DelayBuffer 验证）。

## 2. 引擎内轨迹敏感度（D20 vs D00，trajectory sensitivity，非 PASS/FAIL）

| 指标 | Isaac | MuJoCo | 说明 |
|---|---:|---:|---|
| 29-joint q | 0.030929 rad | 0.048023 rad | MJ 敏感度约为 Isaac 的 1.55 倍 |
| q_des requested | 0.033234 rad | 0.049301 rad | 同上 |
| body position（world） | 0.026946 m | 0.051533 m | |
| body position（pelvis-relative） | 0.009635 m | 0.017089 m | 两引擎内相对构型扰动均较小 |
| body orientation（geodesic） | 0.052106 rad | 0.104533 rad | MJ 约为 Isaac 2 倍 |
| root height / roll / pitch | 0.006496 / 0.011006 / 0.009607 | 0.008846 / 0.016085 / 0.014021 | m / rad |
| 分组 q：LEGS12 | 0.034088 | 0.057339 | 12/3/14 分组由 body_mapping 固定并断言覆盖 |
| 分组 q：WAIST3 | 0.012869 | 0.019398 | |
| 分组 q：ARMS14 | 0.030822 | 0.043342 | |

## 3. requested vs effective tracking（分开报告）

| RMS(q_des−q) | D00 requested | D20 requested | D20 effective |
|---|---:|---:|---:|
| Isaac | 0.131353 | 0.145288 (+10.6%) | **0.130011（≈D00 水平）** |
| MuJoCo | 0.122301 | 0.139712 (+14.2%) | **0.123062（≈D00 水平）** |

解释：20 ms 使**策略请求目标**的跟踪误差上升，但执行器实际收到的**有效目标**跟踪与 0 ms 基本持平——符合“执行目标被整体相移一个 control cycle”的预期，PD 仍在跟踪它真正看到的目标。

## 4. 跨引擎 gap（Isaac vs MuJoCo，同 delay 配对）

| 指标 | D00 | D20 | 变化 |
|---|---:|---:|---|
| q gap | 0.055946 rad | 0.067324 rad | **+20.3%** |
| body position gap（world） | 0.104610 m | 0.074709 m | −28.6% |
| **body position gap（pelvis-relative）** | 0.017400 m | 0.023317 m | **+34.0%** |
| body orientation gap（geodesic） | 0.104374 rad | 0.150575 rad | +44.3% |
| pelvis position gap | 0.103292 m | 0.070367 m | −31.9% |
| pelvis orientation gap | 0.040751 rad | 0.084816 rad | +108.2% |
| anchor position gap（feet/hands，m） | 0.1004–0.1112 | 0.0725–0.0904 | 普遍下降（跟随 pelvis 平移收敛） |
| anchor orientation gap（rad，R5.1 补充） | L-ankle 0.096911 / R-ankle 0.107014 / L-wrist 0.126893 / R-wrist 0.151783 | 0.150586 / 0.219933 / 0.162242 / 0.168811 | 踝 +55%/+106%，腕 +28%/+11%——与 pelvis orientation 同向增加 |
| body linear / angular velocity gap | 0.138428 / 0.436388 | 0.143096 / 0.439748 | 基本持平 |

**必须分开表述（R5 修订）**：world-frame body gap 的下降主要来自 root/pelvis **平移**在 20 ms 下更接近（pelvis gap −31.9%），而**身体相对构型并未同步改善**（pelvis-relative gap +34.0%，orientation gap +44.3%）。因此不能概括为“20 ms 让整体 Sim2Sim 更一致”；这是指标间 tradeoff。

## 5. 其他必算项

- MuJoCo applied torque：RMS 18.22 → 18.83 N·m（+3.4%），peak 188.5 → 187.0 N·m，饱和比例 0.58 → 0.573（fraction of |raw|>|applied|，腕/踝限位主导；raw PD RMS 18.36 → 19.10，qfrc_actuator 单独报告不与 applied 混称）。**Isaac torque：UNAVAILABLE**。
- action / q_des variation：Isaac 0.1302 → 0.1325、MJ 0.1395 → 0.1429（每周期差分 RMS），20 ms 未显著改变指令平滑度。
- 参考动作跟踪（诊断用，未对齐世界系，不进结论）：见 `p3_metrics.json.task_tracking_note`。
- task 主指标（机器人对机器人）以 §2–§4 为准；per-condition 值在 json 中。

## 6. 图表与视频

- 11 张图（`04_analysis/plots/`，每张配 `data_*.csv`，含 metric 标签、单位、聚合方式与坐标系列），标题含口径，world/pelvis-relative 分开（图 07、09 同时给绝对值与相对变化，图 09 每面板单一单位），requested/effective 分开（图 03、04）。
- 图 08（R5.1 替换）：左=各 cycle 的 D20-vs-D00 14-body orientation geodesic RMS（Isaac/MJ 两线）；右=各 cycle 的 Isaac-vs-MuJoCo geodesic RMS（D00/D20 两线）；全部来自 `actual_body_quat_w`，robot-vs-motion 未对齐量只留在 JSON 的 `diagnostic_unaligned_*` 字段。
- 图 11（R5.1）：无量纲 MuJoCo/Isaac sensitivity ratio（5 指标，ratio=1 参考线）。
- 四格视频 `05_media/p3_delay_four_panel_10s.mp4`：250 帧 × 25 fps，1280×720，黑帧 0，first/mid/last 非黑（`media_manifest.json` 含四源 SHA256、逐源 `container_playback_duration_s` 与 `covered_simulation_time_s=10.0` 的区分、映射假设 `uniform index-proportional`、`has_per_frame_sim_timestamp=false`）。画面内 t 标注为 nominal simulation time；源视频无逐帧仿真时间戳，视频仅定性观察，定量结论全部来自冻结 trace。
- MJ D00 画面复用 P1 正式 `mj_matched_r1/video.mp4`（源 trace 与 R4 MJ D00 trace **逐字节同哈希** ba9276f7…，复用理由与链路写入 `media_manifest.json.mj_d00_reuse`），未伪装为 R4 原始文件。

## 6b. R5.1 修订记录（2026-09-02，展示与元数据，不改主指标）

1. `p3_metrics_long.csv` 全部行带确定单位（`WITHIN/CROSS/TORQUE/TASK_UNITS` 映射 + 断言），无 `mixed`；anchor position/orientation 嵌套展开进 long CSV；`p3_metrics_wide.csv` 更名为 `p3_metrics_engine_tidy.csv`（诚实的 tidy 三列+单位）。
2. 补 `anchor_orientation_gap`（pelvis/2×ankle/2×wrist × D00/D20，geodesic rad），position 与 orientation 分栏。
3. 图 08 替换为有效比较；图 09 拆为 4 个单单位面板 + 相对变化；图 10 拆 RMS/peak/saturation 三面板；图 11 改无量纲 ratio。
4. 全部 `data_*.csv` 加 metric/单位/聚合/坐标系标签列。
5. 媒体 manifest 区分容器播放时长与 10 s 仿真覆盖；视频标签改 `nominal sim time` 并重编码（新 SHA 见 manifest）。
6. 代码卫生：finite 检查扩展至 dq/D20 requested/effective/body velocity/torque 字段；survival/fall/reset 由 trace 计数器+summary+上游 gate 交叉推断，不再硬编码；确认 `p3_condition_index.json` 只读取一次。
7. 哈希范围命名：完整树 = `SHA256_FULL_TREE.txt`（不再叫 SHA256.txt），小反馈包 = 各自 `SHA256.txt`/`SHA256_FEEDBACK_PACKAGE.txt`，避免 101/32 混淆。

## 7. R4 遗留三小项（已随本包补齐）

1. `00_inputs/R3_FREEZE_PROVENANCE.json`：上游 ZIP `2fea92f3…` + R3 六项不可变文件相对路径与 SHA256 + status。
2. `requirements-r4-r5.txt` + `python_environment.json`：Windows 分析/Isaac 环境（py3.11.9 numpy1.26.0 yaml6.0.2 ort1.29.0 mpl3.10.3 imageio2.37.0 ffmpeg0.6.0 cv2 4.11.0）与 WSL MuJoCo 运行环境（py3.10.12 numpy1.26.4 yaml6.0.3 mujoco3.2.7）。
3. `runtime_dependency_ledger.json`：`mimic_dance_9.yaml` 已改为 direct runtime participant=true（Isaac runner `load_contract` 实际加载），revision r5.1。

## 8. 结论（限定范围）

在该条件下（dance_9、seed 42、10 s、gain 1.0、当前资产、显式 position-target FIFO、0/20 ms 两档）：
1. 两侧在 20 ms 显式位置目标延迟下均完成 10 秒无摔倒无 reset；延迟语义实测唯一（Isaac 4×5 ms、MJ 10×2 ms）。
2. 20 ms 使 MuJoCo 的关节/姿态轨迹敏感度高于 Isaac。
3. 两侧 requested tracking error 上升（+10.6% / +14.2%），effective tracking 接近各自 0 ms 水平，符合执行目标相移预期。
4. 跨引擎 q gap 与 body orientation gap 增加；world-frame body position gap 下降主要来自 pelvis/root 平移更接近，pelvis-relative body gap 同时增加——指标间存在 tradeoff，不能用单一 world position 指标概括 Sim2Sim 一致性。
5. 不外推到全部动作、全部延迟值、Isaac 原生 DelayBuffer 或其他模型资产。

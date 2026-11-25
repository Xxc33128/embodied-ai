# 仿真后端核验：robosuite（新口径 §5 第一步）

> v1.0，2026-09-14。依据《2026-09-14-原始评测复核与复现范围.md》§5"核验一个已有机器人控制器、可用物理抓放示例的后端"执行。零付费 API；在隔离环境验证，未触碰 inspect-robots 主环境与任何历史实验产物。
> 证据口径：E1=源码实证；E2=本机运行复测；E3=本轮建议。

## 1. 结论

**robosuite 1.5.2（Panda + Lift 任务 + OSC_POSE 控制器）通过核验：独立脚本抓放 10/10，每集 ~170–185 控制步（20Hz，约 9 秒仿真），墙钟约 2.1–2.3s/集。** 三项核验点全部满足：

| 核验点（新口径 §4.1） | 结果 |
|---|---|
| 关节机械臂 + 夹爪 | ✅ Panda 7 关节 + 二指夹爪，控制器为 robosuite 现成 OSC_POSE（本方零自研控制代码） |
| 可用物理抓放 | ✅ 独立脚本（悬停→下探→闭合→抬升）10/10 抬升成功，环境自带成功判定 |
| 相机 + 本体观测 | ✅ frontview / robot0_eye_in_hand 各 224×224；joint pos/vel/acc、eef pos/quat、gripper qpos/qvel、50 维 proprio-state |

安装（隔离 venv `.venv-robosuite/`）：robosuite 1.5.2 + robosuite-models 1.0.0 + mujoco 3.3.0（robosuite 1.5.2 声明要求 ≥3.3；3.13 有 binding 断言不兼容，3.2.7 低于下限——**钉 3.3.0**）。macOS arm64 离屏渲染直接可用。

## 2. 接口清单（适配评估用，E1/E2）

- **动作**：7 维归一化 [−1,1] = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]。OSC_POSE 默认 delta 输入、每控制步输出限幅 ±0.05m / ±0.5rad。
- **坐标系注意**：OSC delta 默认在机器人基座系（与世界系存在旋转，实测 y 轴反向）；控制器配置支持 `"input_ref_frame": "world"` 切世界系（本次核验已用世界系跑通）。
- **夹爪符号**：robosuite Panda 约定 **+1=闭合、−1=张开**（与常见 Gym 约定相反，已实测标定）。
- **观测**：`cube_pos/cube_quat/gripper_to_cube_pos/object-state` 为物体真值——按新口径 §4.2，适配层必须过滤，不得进入模型观测；`robot0_proprio-state`（50 维）含关节力矩类量，与原评测观测中的 joint_eff 对应能力待逐项核对。
- **种子**：`robosuite.make(..., seed=N)` 原生支持；每集可复现。
- **周期**：control_freq=20Hz（原评测 control_hz 待核对；适配层换算即可）。

## 3. 到 bowl 任务的适配清单（E3，工作量评估）

1. **bowl 化场景**：Lift arena 中加入碗体（robosuite 支持 arena 组装/自定义 object）+ 成功判定改为"方块在碗内静置"（沿用本项目已有的判定几何：中心距 + 高度 + 低速）。中等工作量。
2. **inspect-robots embodiment 适配器**：`move_to`（绝对世界系末端位姿）→ OSC 世界系 delta 序列（护栏插值已有）；gripper 0/1 → −1/+1；三路相机帧映射；state 暴露 eef_pos/quat/yaw、joint pos/vel/eff、实测 gripper；**过滤物体真值**。与 v3 适配器同构，且无需自写 IK/夹爪控制器。
3. **评分对齐**：按原报告 0–4 历史最高阶段记分（阶段含义见评测页），自动评分须标注近似并抽查视频。

## 4. 与 v3/v2 的关系

- v3（自研 Panda 后端）按新口径降级为研究路线，其 Task 1–4 产物与诊断记录保留不动；若 robosuite 路线跑通模型闭环，v3 不再是正式评测前置。
- v2 浮动夹爪保留作链路对照。
- 本核验不删除、不覆盖任何历史成绩。

## 5. 复现命令

```bash
uv venv .venv-robosuite --python 3.11
uv pip install --python .venv-robosuite/bin/python robosuite robosuite_models "mujoco==3.3.0"
.venv-robosuite/bin/python verify_robosuite_backend.py
```

产物：`verify_robosuite_backend.py`、`verify_robosuite_result.json`（success=10/10）。

# 本周三项目 Sim2Sim 调研与实验总结（最终版，R7）

日期：2026-09-02
范围：Humanoid Lab（主实验）、Humanoid Gym（参考实验）、SONIC（经验总结）
本文替代 `09_report/本周三项目Sim2Sim调研与实验总结_20260828.md`（B3.1 版，保留为历史）。
所有精确数字都给出证据路径或 claim_id（见 `../FINAL_CLAIM_EVIDENCE_MATRIX.csv`）；包哈希与校验命令见 `../FINAL_DELIVERY_INDEX.csv`。
文中相对路径以 `E:\sim2sim-week-2026-08-26\` 为根。

---

## 1. 执行摘要

研究了什么：把在 Isaac（PhysX）家族训练的运动策略搬到 MuJoCo 回放时，两侧在接口、资产和执行语义三个层面的差异有多大、能否量化、哪些差异会改变闭环行为。

为什么比较三个项目：Humanoid Lab、Humanoid Gym、SONIC 代表三条 Sim2Sim 路线（Isaac Lab→deploy MuJoCo、Isaac Gym→MuJoCo、Isaac Lab→MuJoCo/C++ ROS2）。只有 Lab 一条具备"同一机器人、同一 ONNX、同一动作、同一初态、两侧可运行"的受控配对条件，因此作为定量主线；另外两条提供接口形态与经验边界。

本周获得的有效证据（按 claim_id）：

1. **CL-02/CL-03**：接口 G0-I（cycle0 观测/动作/状态逐字段 <1e-4）与资产 G0-A（下游 COM 5.1e-8/5.2e-7 m、目标关节 I_eff 相对差 1e-7）两道门禁先后通过，为后续一切闭环比较建立了"输入相同"的前提。
2. **CL-04→CL-05→CL-06**：右肘 armature（Isaac runtime 0.01 vs MuJoCo XML 0.0685，N4 实测 6.85×）在单关节隔离下把 |MJ−Isaac| 阶跃 MAE 降低 74.8%；迁移到全身 dance_9 保留 25.7% 的肘部改善；把其余 13 个臂关节一并改成 0.01 后没有增量收益且系统指标变差（`NO_INCREMENTAL_ARM_GROUP_EFFECT`）。
3. **CL-07/CL-08/CL-09**：0/20 ms 受控等时延注入下，两侧各完成 500 周期 10 秒、无摔倒、无 reset；延迟语义从 trace 实测唯一识别（Isaac 4×5 ms，MuJoCo 10×2 ms）。requested tracking 误差上升而 effective tracking 持平（执行目标相移）；跨引擎指标呈现 tradeoff：world-frame body gap 下降主要来自 pelvis/root 平移收敛，pelvis-relative body gap 与 orientation gap 同时上升。

最重要三条结论：

1. Sim2Sim 差异必须按"接口→资产→执行语义"分层门禁度量，混在一起会得到错误因果（本周曾因此撤回一次 armature 结论，见 §4）。
2. 隔离实验的有效参数迁移到全身闭环会衰减（74.8%→25.7%），继续扩大参数组可能得到负收益（P2）；参数逐关节、全身闭环验证缺一不可。
3. 延迟实验必须记录 actuator 实际接收的 effective target 并固定 prehistory 语义；只凭配置推断延迟不算证据。20 ms 下"两侧都活着"是存活事实，跨引擎一致性需按指标分解陈述。

适用范围：全部定量结论限定于 L7 29-DOF、dance_9、seed=42、10 秒、gain=1.0、当前 policy/motion/canonical/模型资产、显式 position-target FIFO、0/20 ms 两档。向其他动作、其他延迟档位、Isaac 原生 DelayBuffer 行为或其他机器人外推均不受本周证据支持。

---

## 2. 三个项目分别怎样实现 Sim2Sim

### 2.1 Humanoid Lab（主实验，定量链）

- 训练引擎：Isaac Lab / PhysX（Windows，`isaaclab_env51` (Windows venv)，Isaac Sim 5.1.0；Isaac Lab 5c2ec81，dirty）。
- 回放引擎：MuJoCo 3.2.7（WSL `~/.venv_lab_sim`）+ 两侧直接 ONNX runner。
- policy：`deploy/policy/dance_9/motion_anchor_obs_model_23500.onnx`（29 关节，metadata 携带 joint_names/default_q/kp/kd/action_scale/body_names）。
- 观测：770 维 = 5 帧 history ×（curr_motion 58 + anchor_ori 6 + base_ang_vel 3 + qpos_rel 29 + qvel 29 + action 29），`deploy/src/era_rl_controller/era_rl_controller/rl_interfaces/mimic_rl_interface.py` 与两侧 runner 内 `MimicObsBuilder` 同实现；history repeat-first 预填。
- 控制链：action×scale+offset→q_des→PD（Isaac 隐式 drive / MuJoCo 每物理步显式 PD）→torque。
- 步长：Isaac physics dt=0.005、decimation=4、control dt=0.02；MuJoCo dt=0.002、ppc=10、control dt=0.02。
- 资产：URDF `l7_29dof_neck_fixed.urdf`（c2aa3837…）与 MJCF `l7_29dof_neck_fixed_elbow_matched.xml`（ab1e1862…，右肘 armature=0.01 干预版）。
- 关键代码：Humanoid Lab 仓库 `scripts/experiments/run_isaac_onnx.py`、`run_mujoco_onnx.py`；P3 执行源码冻结于 `13_p3_delay20_r4_build/13_p3_delay20/00_inputs/executed_sources/`。
- 证据等级：A（双引擎配对）。基线：`02_lab_dance9/`，root 高度 RMSE 8.60 mm、q RMSE 0.0623 rad（`02_lab_dance9/lab_10s_metrics.csv`）。

### 2.2 Humanoid Gym（参考实验，单引擎基线）

- 训练引擎：Isaac Gym Preview 3（历史训练产物）。
- 回放引擎：MuJoCo，脚本 `humanoid/scripts/sim2sim_record.py`（自包含 12-DOF 常量、obs scale、Kp/Kd、MJCF、TorchScript JIT `policy_example.pt`，705→12）。
- 观测：12 DOF × 5 history（15×0.01 s 窗口 0.150 s，first-to-last 0.140 s）；基线指令 vx=0.4 m/s。
- 步长：physics dt=0.001、control 100 Hz；action scale=0.25；obs noise 关闭（patched XML 记哈希）。
- 证据：`05_gym/xbot_walk_existing.mp4`（历史视频，grade C）+ `05_gym/mj_100hz/` 10 秒 1000 步带日志基线（grade B，单引擎；x 3.73 m、y −0.78 m，`05_gym/mj_100hz/`）。
- 限制：Gym 与 Lab 机器人不同，绝对 q/torque/contact 数值与 Lab 不可直接互比（铁律 5）；本轮无 Isaac Gym 侧配对。

### 2.3 SONIC（经验总结，四格台账）

- 训练引擎：Isaac Lab；部署：MuJoCo/C++（ROS2）。
- 证据形态：`06_sonic/sonic_II_IM_MM_MI_ledger.csv` 四象限台账（Isaac-Isaac / Isaac-MuJoCo / MuJoCo-MuJoCo / MuJoCo-Isaac），B.3 重做后缺失项一律标 unavailable。
- 等级：MM/MI 为 C（历史日志与视频），II/IM 为 D（静态代码分析）；无本周新实验（claim CL-10）。

### 2.4 三项目对比表

| 维度 | Humanoid Lab | Humanoid Gym | SONIC |
|---|---|---|---|
| 训练→回放 | Isaac Lab/PhysX → MuJoCo | Isaac Gym → MuJoCo | Isaac Lab → MuJoCo/C++(ROS2) |
| policy 载体 | ONNX(metadata 29 关节契约) | TorchScript JIT(705→12) | ONNX + C++ 部署栈 |
| 观测 history | 5 帧×154=770 | 5 帧×(12×…) 100Hz | 部署侧独立实现 |
| 配对条件 | 同机器人/同权重/同初态/同动作 | 无训练侧回放配对 | 部分环节缺日志 |
| 本轮角色 | 定量主线(A) | 参考基线(B/C) | 经验台账(C/D) |

---

## 3. 为什么实验分五层设计

| 层 | 控制变量 | 测试证据 | 允许结论 |
|---|---|---|---|
| 1 接口 G0-I | 同 ONNX、同 canonical、cycle0 静态注入 | `01_lab_interface_gate/aligned_v1/g0_report_v2/`（CL-02） | 观测/动作/关节顺序/坐标语义逐字段一致 |
| 2 资产 G0-A | 隔离 URDF/MJCF 拓扑、FK/COM/I_eff | `04_G0/asset_equivalence_*.json` + P1 全身审计（CL-03） | 干预前资产等价 |
| 3 单关节执行器参数 | 固定基座、0.5 s 阶跃、仅动一个参数 | `04_isolation_N5R2/`（CL-04） | 参数在该关节的局部因果 |
| 4 全身闭环迁移 | dance_9 10 s×3 重复、单变量 | `10_p1_fullbody_elbow/`、`12_p2_arm_only_v1_1/`（CL-05/06） | 局部效应是否保留、有无系统代价 |
| 5 延迟鲁棒性 P3 | 0/20 ms 受控等时延、双侧同协议 | `13_p3_delay20_r4_build/`（CL-07/08/09） | 存活事实 + 连续指标变化 |

每层失败都定义了 fail-closed 门禁（G0 不过不调参数；哈希不同不比结果；无 effective 记录不下延迟结论），这是本周多次纠偏后固化的流程。

---

## 4. 实验纠偏过程（按因果顺序）

1. **URDF root 变换重复写入**：N5-R v1 隔离 URDF 生成器把 canonical root 世界变换折叠进每个 joint origin，右肘下游等效惯量被放大约 91 倍（Isaac 6.83 vs MJ 0.075 kg·m²）、hip 约 11.5 倍、最远 link 推到 ~24.5 m（`04_actuator_isolation/` 审查记录）。
2. **错误 armature 结论撤回**：基于上述资产的"armature 仅改善 0.17%、非 armature 机制主导"两条结论作废；N6-R 暂停。
3. **修复与新增门禁**：urdf_isolation V3（仅局部冻结、root 由 init_state 一次性设置，FK 自检 4.4e-16 m）；新增资产 G0-A（独立实现 `asset_equivalence_check.py`，不 import 生成器）与 fail-closed 门禁（G1 真检查、G4 预注册最小效应、G6 逐点全字段+manifest 哈希）。
4. **N5-R2 恢复有效局部因果**：三重复 18 条 trace 全门禁 PASS，elbow MAE 0.006752→0.001704（−74.8%），hip 阴性对照 0.0（`04_isolation_N5R2/elbow_gates.json`、`hip_gates.json`）。
5. **P1/P2 检验全身迁移**：P1 右肘单变量 −25.7% 成立（`10_p1_fullbody_elbow/04_analysis/p1_final_gate.json`）；P2 14 臂关节扩展无增量且 29 关节/body_quat 变差（`12_p2_arm_only_v1_1/04_analysis/p2_final_gate.json`，decision `NO_INCREMENTAL_ARM_GROUP_EFFECT`）。
6. **P3 延迟可观测性修复**：v1 的 Isaac 4-step 延迟只有配置推断、无 effective target 记录、无 first-q-des prefill 证据，结论降级为 `OBSERVED_BOTH_20MS_RUNS_SURVIVED_10S`（`13_p3_delay20_v1_frozen/STATUS.md`）。R1/R2.1 引入 instrumented actuator（显式 per-physics-step deque FIFO + requested/effective 子步日志 + ownership/coverage fail-closed + physics index 来自 compute 计数器 + manifest measured=null），R3 重跑 Isaac 20 ms 正式 10 秒，R4 自包含门禁 `VALID_P3`（`13_p3_delay20_r4_build/13_p3_delay20/04_analysis/p3_final_gate.json`，8 真实负测试 8/8）。
7. **R5 指标分解**：world-frame 与 pelvis-relative body gap 拆成两套后，发现 world 下降与相对构型上升并存，修正了"20 ms 让 Sim2Sim 更一致"的错误概括方向（`13_p3_delay20_r4_build/13_p3_delay20/04_analysis/p3_metrics.json`，claim CL-09）。
8. **R6 证据链收口**：P2 validator 删除绝对路径 fallback 并把 elbow 哈希绑定进 validity（两个新负测试 fail-closed，重算 decision 不变，`validation/p2_negative_tests_r6.json`）；P0 overlay validator 改为 `--root/--source-root/--mode` 双模式（embedded 不读外部包并显式 `EXTERNAL_SOURCE_NOT_CHECKED`；full-provenance 重算 7 个源 ZIP 哈希并从 P1 ZIP 内 trace 复算 body/anchor 指标一致，`validation/p0_full_provenance_validation.json`）。

---

## 5. 有效结果

### 5.1 基线与实现链

- Lab 双引擎 10 秒基线：双侧 500 周期无摔倒，root 高度 RMSE 8.60 mm、q RMSE 0.0623 rad（`02_lab_dance9/`，视频 `assets/lab/sidebyside_10s_labeled.mp4`）。
- Gym 100 Hz 带日志基线：10 秒 1000 步，x 3.73 m、y −0.78 m，history 窗口 0.150 s（`05_gym/mj_100hz/`）。
- SONIC 台账四格定级（`06_sonic/sonic_II_IM_MM_MI_ledger.csv`，CL-10）。

### 5.2 N5-R2 右肘隔离（CL-04）

固定基座 0.5 s 阶跃、3 重复：|MJ−Isaac| MAE 0.006752→0.001704 rad（−74.8%），t50 0.177→0.163 s；hip 阴性对照不受影响（0.0）。图：`assets/n5/elbow.png`、`assets/n5/hip.png`。

### 5.3 P1 全身迁移（CL-05）与 P2 分组扩展（CL-06）

- P1：仅右肘 `armature 0.0685→0.01`，全身 dance_9 右肘误差 0.0682→0.0507（−25.7%），3/3 重复同向；伴随 body pos +0.0048 m、pelvis +0.0055 m 小代价。图：`assets/p1/elbow_q_active_window.png`、`assets/p1/native_matched_metric_bars.png` 等。
- P2：其余 13 臂关节也设 0.01 后，E13 0.04489→0.04714（−5.0%，变差），29 关节 +0.00844、body_quat +0.022 超 10% guard；右肘收益保留（+0.96%≤10%）。decision `NO_INCREMENTAL_ARM_GROUP_EFFECT`。图：`assets/p2/arm13_group_rmse_0_2s.png`、`assets/p2/system_tradeoff_relative_to_elbow.png`。
- 解读：armature 是有力候选原因，非唯一来源；残余 ~0.013 rad 量级的孤立差异指向 frictionloss/PD 时序等待查项（后续可选，§7）。

### 5.4 P3 0/20 ms 延迟（CL-07/08/09）

四条件（I00/I20/M00/M20）均 500 周期、fall=null、reset 由计数器独立推断为未发生。延迟实测：Isaac 唯一 physics shift=4（20.0 ms，次优差 0.419 rad）、MuJoCo 唯一 shift=10（20.0 ms）、两侧 repeat-first prehistory。

引擎内 D20 vs D00 轨迹敏感度（`13_p3_delay20_r4_build/13_p3_delay20/04_analysis/p3_metrics.json`）：

| 指标 | Isaac | MuJoCo |
|---|---:|---:|
| q sensitivity (rad) | 0.030929 | 0.048023 |
| q_des sensitivity (rad) | 0.033234 | 0.049301 |
| body world sensitivity (m) | 0.026946 | 0.051533 |
| body pelvis-relative sensitivity (m) | 0.009635 | 0.017089 |
| body orientation sensitivity (rad) | 0.052106 | 0.104533 |
| LEGS/WAIST/ARMS q sensitivity (rad) | 0.034088 / 0.012869 / 0.030822 | 0.057339 / 0.019398 / 0.043342 |

requested/effective tracking（rad，RMS(q_des−q)）：Isaac 0.131353→0.145288（requested，+10.6%）而 effective 0.130011≈D00；MuJoCo 0.122301→0.139712（+14.2%）而 effective 0.123062≈D00。执行目标被相移一个 control cycle，PD 跟踪自己实际收到的目标，符合预期机制。

跨引擎（Isaac vs MuJoCo，同延迟配对）：

| 指标 | D00 | D20 | 变化 |
|---|---:|---:|---|
| q gap (rad) | 0.055946 | 0.067324 | +20.3% |
| body world gap (m) | 0.104610 | 0.074709 | −28.6% |
| body pelvis-relative gap (m) | 0.017400 | 0.023317 | +34.0% |
| body orientation gap (rad) | 0.104374 | 0.150575 | +44.3% |
| pelvis position gap (m) | 0.103292 | 0.070367 | −31.9% |
| pelvis orientation gap (rad) | 0.040751 | 0.084816 | +108.2% |
| anchor orientation gap，右腕 (rad) | 0.151783 | 0.168811 | +11.2% |

MuJoCo torque：applied RMS 18.22→18.83 N·m（+3.4%），peak 188.5→187.0 N·m，饱和比例 0.58→0.573（腕/踝限位；raw PD 与 qfrc_actuator 分列于 schema）；Isaac torque 记 UNAVAILABLE。

图：`assets/p3/02_q_sensitivity_all_and_groups.png`、`assets/p3/03_q_tracking_requested_effective.png`、`assets/p3/04_q_des_requested_vs_effective_selected_joints.png`、`assets/p3/07_body_position_world_vs_pelvis_relative.png`、`assets/p3/08_body_orientation_geodesic.png`、`assets/p3/09_cross_engine_gap_d00_vs_d20.png`、`assets/p3/11_metric_tradeoff_summary.png`。四格 10 秒视频来源与哈希见 `assets/p3/media_manifest.json`（画面为定性观察，帧内 t 为 nominal simulation time）。

**表述纪律**：world body gap 的下降与 pelvis/root 平移收敛同向发生，身体相对构型与姿态差同时增大；两者必须并列呈现。单一 world position 指标不能概括 Sim2Sim 一致性。sensitivity 描述的是轨迹对延迟扰动的位移幅度，属于观测量；本周未预注册鲁棒性阈值，因此不输出 PASS/FAIL 型鲁棒性判定。

---

## 6. 结论、限制与不能外推的范围

**已被实验支持**：CL-02…CL-09 全部条目（各带证据路径与限定）。

**当前只观察到**：20 ms 下两侧存活（BOTH_SURVIVE_20MS_ON_DANCE9）；MJ 敏感度高于 Isaac；指标 tradeoff 的方向。

**当前不可判断**：更大延迟（>20 ms）行为；其他动作/机器人；Isaac 原生 DelayBuffer 在同样本下的鲁棒性（本周为受控等时延注入）；P2 变差的机制（frictionloss/PD 时序/dt 未扫）；SONIC 各环节精确数值（缺日志）。

**后续可选实验**（不影响本周交付关闭）：WAIST_ONLY、N6/17 关节完整消融、gain 大矩阵、frictionloss/dt/solver 机制扫描、Gym 50 Hz、20 秒长视频、其他动作与延迟档位。

**P3 结论限定**：dance_9 / seed=42 / 10 s / gain=1.0 / 当前 policy·motion·canonical·model / 显式 position-target FIFO / 0 与 20 ms 两档。

---

## 7. Sonic 后续建议与 Sim2Sim 经验

给 SONIC 下一轮的最小检查流程（按本周教训排序）：

1. 先跑接口 G0-I（cycle0 静态注入逐字段 <1e-4），再谈任何动力学差异；
2. 资产等价用独立实现审计（FK/COM/I_eff/轴/质量），生成器与审计器不共享代码；
3. 隔离实验必须固定基座+阴性对照关节+3 重复+逐点一致性门禁（G6 类）；
4. 全身闭环迁移单独验证，禁止把隔离百分比直接写成全身预期；
5. 延迟/执行语义实验必须记录 requested 与 effective 两份目标、固定 prehistory 语义、哈希绑定 runner/actuator 源码，measured 值从 trace 复算；
6. 指标按坐标系分解（world vs pelvis-relative vs orientation），避免单一数字概括一致性；
7. 视频只作观察，定量结论全部来自冻结 trace + fail-closed validator + 真实篡改负测试。

---

## 8. 附录

### 8.1 最终证据索引

见 `../FINAL_DELIVERY_INDEX.csv`（FD-01…FD-10，含哈希、payload 数、校验命令、supersedes 关系）。哈希范围命名约定：完整树用 `SHA256_FULL_TREE.txt`（R4+R5.1 树 108 项），小反馈包用各自 `SHA256.txt`/`SHA256_FEEDBACK_PACKAGE.txt`（R5 包 32 项），两者不混称。

### 8.2 环境与 commit

`13_p3_delay20_r4_build/13_p3_delay20/00_inputs/repo_state.json`（Humanoid Lab c68c1e63、dirty=true 88 项；Isaac Lab 5c2ec81）；`python_environment.json`（Windows 分析/Isaac 环境 py3.11.9 等；WSL MuJoCo 环境 py3.10.12/mujoco3.2.7）。结论依赖 artifact SHA256，不依赖 clean-commit 假设。

### 8.3 指标公式

`13_p3_delay20_r4_build/13_p3_delay20/04_analysis/p3_metric_schema.json`（R5.1：每指标含公式、单位、聚合维度、坐标系、是否主指标；四元数一律 geodesic 2·acos|dot|）。

### 8.4 术语表

G0-I 接口门禁；G0-A 资产门禁；requested/effective target；trajectory sensitivity；world-frame / pelvis-relative body gap；受控等时延注入；fail-closed；repeat-first prehistory；E13（13 臂关节增量指标）。

### 8.5 复现命令

```bash
# P3 最终门禁（R4 包解压根目录）
python 04_analysis/p3_validate.py --root .
python 06_checks/run_p3_negative_tests.py --root . --python <py3.11>
# P3 指标与图（R4+R5.1 overlay）
python 04_analysis/p3_summarize_r2.py --root .
python 05_media/make_four_panel_r5.py --root .
# P2（R6 fail-closed 版）
python p2_validate_r6.py --root <P2包根>
# P0 overlay 双模式
python validate_evidence_closure_r6.py --root 11_p0_evidence_closure --mode embedded
python validate_evidence_closure_r6.py --root 11_p0_evidence_closure --source-root . --mode full-provenance
# 最终交付
python validate_final_delivery.py --root . --source-root <源包目录> --mode full-provenance
```

### 8.6 已知限制

`13_p3_delay20_r4_build/13_p3_delay20/KNOWN_LIMITATIONS_R3.md`（执行器旧注释与 velocity/effort 仍走原生缓冲、reset_count 常量的独立推断替代、Isaac 回归 runner 64365052 与正式 de872655 的 checkpoint-only 差异及 0.0 轨迹佐证、MJ 子步恒等记录、manifest 内历史绝对路径、dirty 仓库、受控注入定性）；`11_p0_evidence_closure/00_sources/HASH_RECONCILIATION.md`（计划占位哈希 2f4e8c15/c1a1011b 从未交付，实际 cbf8a6fd/3dd2acb8 为准）。

---

*报告版本：R7 最终版（2026-09-02）。历史版本 B3.1 及之前保留于 `09_report/` 不再更新。*

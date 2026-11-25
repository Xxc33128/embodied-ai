# R3 interface gate v2 审查与 N4/N5 修订

> **2026-08-28 更新：**R4 已执行到 N5，但阶跃采样协议存在前置激励、时间戳错相及隔离条件不足。本文第 5–6 节的 N5/N6 顺序已由 [R4 审查与后续计划修订](/Users/xerxes3/Documents/huawei实习/2026-08-28-R4-actuator-isolation-v2审查与后续计划修订.md) 取代；N3.1 与 N4 的历史要求仍保留。

> 审查日期：2026-08-27  
> 审查对象：`R3_interface_gate_v2.zip`  
> ZIP SHA256：`c1a1011b963a2a0917e2178accf4615b71a97aa677f906c8af4b32e946252b3e`  
> 审查原则：README 和 test_results 只作提交说明；结论以原始 NPZ、manifest、脚本及独立复算为准。

---

## 1. 验收结论

这版可以确认两件事：

1. **N1 原始证据补包已基本完成。**关键 NPZ、ONNX、motion、XML、配置、旧/新脚本、manifest、stdout 和环境信息已经随包提供，SHA256 清单全部校验通过。
2. **当前这次运行的启动接口可以判为 PASS。**G0a/G0b/G0c 已和 R4-A 动力学诊断分开；原始数据支持 cycle 0 接口对齐、cycle 1 开始动力学分叉。

但 **N2 只能判为部分通过**。新 runner 中增加的 runtime 参数表和 14-body 日志存在三个索引/语义错误，会直接污染 N4 参数表和 N6/R5 body 曲线。因此执行决定是：

```text
保留并接受当前 N1/N3 结果
  → 先做 N3.1 小修，不重跑原 3 秒 G0
  → 跑 N4 runtime 参数表
  → N4 满足停止条件后再跑 N5 单关节实验
  → 暂不进入 N6/5 秒基线
```

---

## 2. 我独立确认过的证据

### 2.1 文件与指纹

- ZIP 可完整解压，共 47 个文件；
- `raw_evidence/SHA256.txt` 中全部文件校验通过；
- Isaac/MuJoCo policy SHA256 均为 `30e1fded...7bd659`；
- motion SHA256 均为 `bee3fe34...23fb`；
- canonical initial state SHA256 均为 `7fdf68af...6745cf`；
- mode、history、start frame 均为 `aligned / repeat-first / 0`。

### 2.2 NPZ 可读性

包内 7 组主要 NPZ 均可用 `np.load(..., allow_pickle=False)` 加载。没有必要在 comparator 中继续使用 `allow_pickle=True`。

### 2.3 G0b runtime adapter

- 两侧各有严格 24 个 key；
- key 集合完全一致；
- 全部字段全局最大差为约 `1.49e-8`；
- 四状态 × 六字段的 G0b PASS 可以接受。

### 2.4 共同初态与 cycle 0

独立从两侧 `policy_trace.npz` 复算：

| 字段 | cycle 0 最大差 |
|---|---:|
| q | 0 |
| dq | 0 |
| root position | 0 |
| root quaternion | 0 |
| root linear velocity | 0 |
| root angular velocity | `2.38e-7` |
| observation | `2.38e-7` |
| action | `4.02e-7` |
| q_des | `1.27e-7` |
| actual target | `1.27e-7` |

以上均远小于 `1e-4`。虽然当前 comparator 没有把 root/hash 全部纳入 G0c，当前这一次数据经人工交叉检查后仍可判定启动接口通过。

### 2.5 R4-A 动力学前缀

原始数组支持：

- cycle 1 首次动力学分叉；
- right elbow dq：Isaac 约 `-4.19`，MuJoCo 约 `-15.51 rad/s`；
- dq 差约 `11.32 rad/s`；
- action 首坏项是 `right_ankle_pitch_joint`；
- q、dq、q_des、target 的主要首坏项是 `right_elbow_pitch_joint`。

因此 `overall_interface_gate_pass=true` 与 `r4a_closed_loop_prefix_equal=false` 可以同时成立。

---

## 3. 进入 N4 前必须修的三个 P0 问题

### P0-1：MuJoCo runtime armature 的 joint id 整体错位

当前代码：

```python
_jid = hinge_names.index(_nm)
_did = m.jnt_dofadr[_jid]
```

`hinge_names` 已经把 free joint 过滤掉，而 `m.jnt_dofadr` 的索引仍是原始 `jntid`。本模型的 `floating_base_joint` 是 jntid 0，第一个 hinge 是 jntid 1，所以当前表会整体向前错一项。

README 中“right hip pitch armature=0.088”正是该错误的表现；XML 中 hip pitch 的正确 class armature 是 `0.0968`，`0.088` 属于 hip yaw。

修法：

```python
hinge_joint_ids = [
    jid for jid in range(m.njnt)
    if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
]
hinge_names = [
    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    for jid in hinge_joint_ids
]
joint_id_by_name = dict(zip(hinge_names, hinge_joint_ids))

_jid = joint_id_by_name[_nm]
_did = int(m.jnt_dofadr[_jid])
```

修后至少断言：

```text
right_hip_pitch_joint  -> 0.0968
right_elbow_pitch_joint -> 0.0685
```

N4 不得使用当前 README 中的 `hip_pitch=0.088`。

### P0-2：MuJoCo 14-body reference 实际会写入 30 个 body

打包的 `dance_9.npz` 中 body 数组 shape 是 `(1485, 30, ...)`。新 runner 当前直接执行：

```python
motion["body_pos_w"][ref_t]
```

却把字段命名为 `reference_body_pos_w(14,3)`。这不会自动筛成 policy 的 14 个 body；实际会得到 30 个 body。

对当前 L7，14-body 对应 motion 全身 body order 的索引为：

```text
[0, 1, 4, 6, 7, 10, 12, 15, 17, 19, 22, 24, 26, 29]
```

但不要把这串数字无说明地硬编码。正确做法是：

1. 在 motion sidecar/config/manifest 中保存完整 `motion_body_names`；
2. 用 `policy.aux_body_names` 按名称生成 `motion_body_indices`；
3. 断言 14 个名字逐项匹配；
4. reference 四类数组都使用同一索引；
5. manifest 同时保存 names 和 indices。

修后 smoke 必须断言：

```text
reference_body_pos_w.shape      == (25, 14, 3)
reference_body_quat_w.shape     == (25, 14, 4)
reference_body_lin_vel_w.shape  == (25, 14, 3)
reference_body_ang_vel_w.shape  == (25, 14, 3)
```

当前反馈包没有提供 README 所称的 N2 0.5 秒 smoke 原始 trace，因此该 smoke 声明暂不能验收。

### P0-3：`mj_objectVelocity` 的 frame 与分量顺序均使用错误

当前代码使用：

```python
mujoco.mj_objectVelocity(..., mjOBJ_BODY, ..., tmp6, 0)
linear = tmp6[0:3]
angular = tmp6[3:6]
```

MuJoCo 3.2.7 的接口定义是 `rot:lin`，即 `tmp6[0:3]` 为角速度、`tmp6[3:6]` 为线速度；而 `mjOBJ_BODY` 使用 body inertial frame，当前位置/姿态日志使用的是 `d.xpos/d.xquat` 对应的 regular body frame。源码定义见 [MuJoCo 3.2.7 engine_support.c](https://github.com/google-deepmind/mujoco/blob/3.2.7/src/engine/engine_support.c#L1277-L1325) 和 [mujoco.h](https://github.com/google-deepmind/mujoco/blob/3.2.7/include/mujoco/mujoco.h#L492-L498)。

应改为：

```python
mujoco.mj_objectVelocity(
    m, d, mujoco.mjtObj.mjOBJ_XBODY, body_id, tmp6, 0
)
angular_w = tmp6[0:3]
linear_w = tmp6[3:6]
```

并用一个已知纯平移、一个已知纯转动状态做最小测试，不能只检查 shape。

---

## 4. 建议一并修正的 P1 问题

### P1-1：Isaac runtime readback 不应采用“全有或全无”

当前 `runtime_readback_ok` 要求 Kp、Kd、armature、effort limit、velocity limit 五类全部可用；只要一类 API 不存在，manifest 就把其余已经成功读取的字段也全部写成 `UNAVAILABLE`。

应逐字段独立保存：

```text
kp_runtime_readback_status
kd_runtime_readback_status
armature_runtime_readback_status
effort_limit_runtime_readback_status
velocity_limit_runtime_readback_status
```

每个数组单独检查 shape、finite 和 joint order。N4 的关键停止条件只依赖 Kp/Kd/armature/effort，不能因为 velocity limit 不可读而丢掉其他证据。

### P1-2：T5 并没有真的验证 28-name metadata 会失败

当前 T5 只检查：

- `METADATA_JOINT_COMPAT_TABLE` 为空；
- 正常 29-name policy 可以加载。

它没有构造 28-name metadata 并调用同一条 validation path，因此“14/14 PASS”不能证明 28-name 一定被拒绝。

建议把 metadata 维度校验提取成纯函数，例如：

```python
validate_policy_metadata(names, default_q, kp, kd, action_scale, compat_id=None)
```

loader 和 T5 都调用它。T5 传入 28 names + 29 arrays，必须捕获预期 `ValueError`。

### P1-3：测试包不能离开原 Windows 仓库独立执行

打包后的 `checks/run_gate_tests.py` 仍按原仓库目录推导 import 和资产，并把 T7 golden 写成 `E:/...` 绝对路径。在当前 ZIP 根目录执行时，无法找到根目录 `scripts/compare_interface.py`；在没有 E 盘时 T7 会直接跳过。

应改为全部使用包内路径：

```text
scripts/
raw_evidence/configs_used/
raw_evidence/isaac_g0_golden/
```

T7 缺输入时应 FAIL，不应打印 skip 后仍让测试套件通过。`test_results.txt` 也应统一保存为 UTF-8，当前文件中文已乱码。

### P1-4：G0a 的 policy time input 仍未真正复现 runner

runner 实际传入 motion frame index：`0,1,2,...`；新版 comparator 使用 `reference_time`：`0,0.02,0.04,...` 秒。

本模型的 time sensitivity 探针为 0，所以不会改变这次 PASS 结果，但 gate 应验证真实调用契约。下一版 golden 增加 `policy_time_input`，直接记录实际送入 ONNX 的值，comparator 原样重放，不再推断单位。

### P1-5：G0c 脚本还应纳入 root 与 hash

本次人工复算已经确认 root state 和三类关键 hash 一致，但 comparator 自身只比较六个 trace 字段。建议给 comparator 增加：

- `--canonical-initial-state`；
- `--isaac-manifest`；
- `--mujoco-manifest`；
- cycle 0 root position/quaternion/linear/angular velocity；
- policy、motion、initial-state hash；
- `history_contract_pass`，由 G0a startup 770 observation 支撑。

### P1-6：整机 torque 数组必须声明并统一顺序

MuJoCo 的 `tau_pd_raw(29)`、`applied(29)`、`qfrc_actuator(29)` 当前主要是 sim2sim/raw hinge order，而 q/dq/action/q_des 是 policy order。后续按 joint index 画曲线会错配。

N5 可以直接按 joint name 记录标量；N6/R5 前必须二选一：

- 全部转换为 policy order；或
- 字段名明确带 `_s2s_order`，并在 NPZ/manifest 保存 order。

推荐统一转换成 policy order，同时保留一份 raw 调试字段。

### P1-7：repeat-first 的 `history_valid_length` 字段语义仍不准确

repeat-first 在 cycle 0 已预填 5 个槽，但当前记录值仍是 1。建议改为：

```text
history_slots_initialized = 5
observed_unique_cycles = min(cycle + 1, 5)
history_init = repeat-first
```

这不影响本次 observation PASS，只是防止后续读日志的人误解。

---

## 5. 修订后的下一步

## N3.1：修 runner 与测试，不重跑原 3 秒 G0

预计 30–45 分钟。

必须完成：

1. 修正 MuJoCo joint id/dof id 映射；
2. 修正 motion 30-body 到 policy 14-body 的按名映射；
3. 改用 `mjOBJ_XBODY`，并纠正 velocity 的 rot:lin 顺序；
4. Isaac runtime 字段逐项 readback；
5. T5 走真实 metadata validation path；
6. 测试脚本改为 ZIP 内相对路径，T7 不允许静默 skip；
7. torque 数组增加 order 声明，至少保证 N5 的被测关节按名字取值。

最小验收：

```text
M1 right_hip_pitch armature == 0.0968
M2 right_elbow_pitch armature == 0.0685
M3 14-body 四类 reference shape 严格为 14
M4 已知纯平移时 linear 非零、angular 约 0
M5 已知纯转动时 angular 非零，linear 符合选定 frame
M6 28-name metadata 走真实 loader validation 并 FAIL
M7 从 ZIP 根目录运行测试，T1–T7 无 skip、UTF-8 可读
```

这些修改只影响参数/日志/测试，不改变已有 3 秒原始动力学，因此不需要重新启动 Isaac 重跑旧 G0。修改后的 comparator 可用已有数据离线重跑一次。

## N4：runtime 执行器参数表

N3.1 通过后执行一次短初始化：

- Isaac：创建环境、reset、读取参数、立即退出；
- MuJoCo：加载模型、读取参数、立即退出；
- 不需要执行 0.5 秒动作；
- 输出 29 行、policy order 的 `actuator_runtime_contract.csv`。

除原计划字段外，增加：

```text
isaac_*_readback_status
mujoco_joint_id
mujoco_dof_id
mujoco_actuator_id
mujoco_ctrl_limited
mujoco_force_limited
parameter_source
```

N4 停止条件：

- Kp/Kd/armature 任一侧读不到；
- requested Kp/Kd 与 runtime Kp/Kd 不一致且原因不明；
- right elbow/hip 的 joint/dof id 不可追溯；
- effort/ctrl clamp 语义不能确认；
- right elbow armature 并非预期的 Isaac `0.01`、MuJoCo `0.0685`。

出现任一停止条件，就先反馈 N4，不执行 N5。

## N5：固定基座单关节实验

只有 N4 无停止项时继续。

在原计划的右肘主实验和右髋对照实验前增加一个 `N5-0`：

1. 不推进物理；
2. 给两侧完全相同的 q、dq、q_des；
3. 对比 unclipped PD torque、clipped command、applied torque；
4. 先证明 actuator 输入一致，再开始 step response。

随后运行：

- right elbow：NATIVE + ARMATURE_MATCHED；
- right hip pitch：NATIVE + ARMATURE_MATCHED 对照；
- base fixed、0 ms delay、metadata Kp/Kd；
- 其余关节保持 q_des=q0，并记录其最大漂移；
- 每 physics step 记录，不只记录 50 Hz policy tick；
- 官方 XML 不覆盖，干预 XML 单独 hash。

如果 N5-0 已显示 torque/clamp 不一致，应停止在 N5-0，先解决执行器输入语义，不要用 armature 修改解释后续曲线。

---

## 6. 是否可以继续

可以继续，但顺序应是：

```text
N3.1 小修
  → N4 短初始化参数表
  → 检查 N4 停止条件
  → N5-0 torque parity
  → N5 elbow/hip step
  → 反馈
```

目前不要进入 N6、5 秒 baseline、20 秒、delay 或 gain sweep。

下一次反馈包应包含：

```text
R4_actuator_isolation_v2.zip
├── scripts/
├── checks/N3_1_tests.txt
├── actuator_runtime_contract.csv
├── n5_0_torque_parity/
├── elbow/
├── hip/
├── plots/
├── manifests/
└── result_summary.md
```

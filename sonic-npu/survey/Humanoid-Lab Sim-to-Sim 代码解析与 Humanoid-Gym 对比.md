---
title: Humanoid-Lab Sim-to-Sim 代码解析与 Humanoid-Gym 对比
aliases:
  - Humanoid-Lab Sim2Sim
  - Lab 与 Gym 的 Sim-to-Sim 对比
tags:
  - humanoid
  - reinforcement-learning
  - sim-to-sim
  - Isaac-Lab
  - MuJoCo
  - ROS2
date: 2026-08-31
updated: 2026-09-04
source: https://github.com/roboterax/humanoid-lab
commit: b5b1092
---

# Humanoid-Lab Sim-to-Sim 代码解析与 Humanoid-Gym 对比

> [!abstract] 本文解决四个问题
> 1. Humanoid-Lab 的 Sim-to-Sim 闭环究竟怎样运行？
> 2. 它怎样对齐 Isaac Lab 与 MuJoCo 的策略接口？
> 3. 哪些物理参数实际上没有严格对齐？
> 4. 它与 Humanoid-Gym 的 Sim-to-Sim 有什么本质区别？

## 1. 核心结论

Humanoid-Lab 的 Sim-to-Sim 是：

> 在 Isaac Lab/PhysX 中训练策略，将冻结策略导出为 ONNX，再在独立的 MuJoCo 模型中重建同样的观测、动作和 PD 控制闭环。

它不是重新训练策略，也不是把 PhysX 模型自动转换为等价的 MuJoCo 模型。

整个迁移依赖三件事：

```text
策略接口一致
    +
机器人名义模型大体同源
    +
训练时的 Domain Randomization
    ↓
冻结策略能够在 MuJoCo 中直接运行
```

其中最重要的判断是：

> [!important]
> Humanoid-Lab 做得较完整的是**策略接口对齐**，不是严格的**跨引擎物理对齐**。

它对齐了：

- 29 个策略关节的名称和动作语义；
- actor 使用的观测项、顺序和 5 帧历史；
- 默认关节姿态、动作缩放和名义 Kp/Kd；
- 50 Hz 策略更新周期；
- ONNX 推理前后的输入输出协议。

它没有严格对齐：

- PhysX 和 MuJoCo 的底层求解器；
- physics timestep；
- 接触与摩擦求解语义；
- 部分关节 armature；
- 训练执行器延迟与 MuJoCo 部署延迟；
- 同状态下的逐步动力学轨迹。

因此，Lab 的 Sim-to-Sim 更准确地说是一套“独立 MuJoCo 部署验证”，而不是已经完成的 Isaac–MuJoCo 数值等价证明。

---

## 2. 一条完整的 Sim-to-Sim 数据链

Humanoid-Lab 把策略控制器和 MuJoCo 机器人拆成两个 ROS 2 节点：

```text
┌───────────────────────────────────────────────┐
│ Isaac Lab / PhysX                             │
│ 训练 actor + observation normalizer           │
└──────────────────────┬────────────────────────┘
                       │ 导出 ONNX
                       ▼
┌───────────────────────────────────────────────┐
│ EraRLController，50 Hz                        │
│                                               │
│ motor/IMU feedback                            │
│   → 关节名称重排                              │
│   → 构造当前观测                              │
│   → 更新 5 帧历史                             │
│   → ONNX Runtime 推理                         │
│   → action × scale + default_q                │
│   → 发布 q_des、Kp、Kd                        │
└──────────────────────┬────────────────────────┘
                       │ /rl_controller/policy_inference
                       ▼
┌───────────────────────────────────────────────┐
│ EraRobot / MuJoCo，名义 500 Hz                │
│                                               │
│ 显式 PD → actuator torque → mj_step           │
│   → 发布 joint position/velocity              │
│   → 发布 base quaternion/angular velocity     │
└──────────────────────┬────────────────────────┘
                       └────────反馈到控制器──────►
```

对应代码：

| 模块 | 代码位置 |
|---|---|
| ONNX 导出 | `source/.../utils/exporter.py` |
| 控制器主节点 | `deploy/src/era_rl_controller/.../era_rl_controller_node.py` |
| locomotion 推理 | `.../rl_interfaces/locomotion_rl_interface.py` |
| mimic 推理 | `.../rl_interfaces/mimic_rl_interface.py` |
| 历史观测 | `.../rl_interfaces/hist_observations.py` |
| 关节顺序转换 | `.../utils/utils.py` |
| MuJoCo 与 PD | `deploy/src/era_robot/era_robot/era_robot_node.py` |
| MuJoCo 模型 | `deploy/l7_29dof_neck_fixed/l7_29dof_neck_fixed.xml` |

运行时先启动控制器，再启动 MuJoCo 节点：

```bash
cd deploy

# mimic
ros2 run era_rl_controller era_rl_controller_node \
  --config src/era_rl_controller/configs/mimic_dance_9.yaml \
  --mode sim2sim

# 或 locomotion
ros2 run era_rl_controller era_rl_controller_node \
  --config src/era_rl_controller/configs/loco_walk_1.yaml \
  --mode sim2sim

# 新终端启动 MuJoCo
ros2 launch era_robot era_robot_launch.py
```

这种结构的工程意义是：RL 控制器不直接依赖 MuJoCo API。切换真机时，理论上保留观测构造、ONNX 推理和动作后处理，只替换反馈与命令后端。

---

## 3. 关键源码走读

第 2 章的数据链落到代码上是六个文件、合计约 700 行。本章按数据流顺序逐段走读，作为第 4 章"对齐六层"与第 5 章"未对齐清单"的证据基础：

| 文件 | 行数 | 职责 |
|---|---:|---|
| `deploy/src/era_robot/era_robot/era_robot_node.py` | 133 | MuJoCo 后端：500 Hz 仿真 timer、PD、反馈发布 |
| `deploy/src/era_rl_controller/era_rl_controller/era_rl_controller_node.py` | 208 | 控制器主节点：50 Hz timer、关节重排、命令拼装 |
| `.../rl_interfaces/locomotion_rl_interface.py` | 137 | 观测构造、ONNX 推理、动作后处理、键盘命令 |
| `.../rl_interfaces/hist_observations.py` | 51 | 按观测项分通道的环形历史缓冲 |
| `.../utils/utils.py` | 7 | `convert_joint_order` 按名称重排 |
| `source/.../utils/exporter.py` | 191 | ONNX 导出与 metadata 附加 |

### 3.1 MuJoCo 后端：`era_robot_node.py`

**PD 公式与 Gym 逐字相同**（模块级函数只有这一个）：

```python
def pd_control(target_q, q, kp, target_dq, dq, kd):    # 16
    return (target_q - q) * kp + (target_dq - dq) * kd  # 17 τ = Kp(q_des−q) + Kd(q̇_des−q̇)
```

与 Gym 的 `sim2sim.py:85` 同名同形——两个项目共享 legged_gym 血统的直接痕迹。

**初始化（`__init__`，21–61 行）：**

```python
        self.max_duration = 200.0  # seconds #                          # 25 仿真时长上限（按墙钟计）
        self.imu_publisher = self.create_publisher(Imu, "/imu_feedback", 10)      # 27
        self.robot_state_publisher = self.create_publisher(Float64MultiArray, "/motor_feedback", 10)  # 29 发布 29q + 29dq = 58 维

        qos_profile = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)  # 31
        self.motor_command_subscriber = self.create_subscription(       # 32 订阅策略命令；TRANSIENT_LOCAL 让晚启动的对端
            Float64MultiArray, "/rl_controller/policy_inference", self.motor_command_callback, qos_profile  # 33   仍能收到最后一条消息
        )

        # Load MuJoCo model
        self.m = mujoco.MjModel.from_xml_path(xml_path)                 # 38
        self.d = mujoco.MjData(self.m)                                  # 40
        self.m.opt.timestep = 0.002  # Simulation timestep              # 42 名义 500 Hz
        self.d.qpos[7:] = np.zeros(self.m.nq - 7)                       # 43 关节角清零
        self.d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])              # 44 基座四元数 = 单位四元数
        self.d.qpos[:3] = np.array([0.0, 0.0, 0.977054])                # 45 基座高度——须与 Isaac 侧 spawn 高度一致

        self.N_JOINTS = self.m.nq - 7  # Assuming first 7 are base      # 47 仍按"自由基座占前 7 维"截取
        self.q_des = np.zeros(self.N_JOINTS)                            # 48
        self.qd_des = np.zeros(self.N_JOINTS)                           # 49
        self.Kp = np.zeros(self.N_JOINTS)                               # 50
        self.Kd = np.zeros(self.N_JOINTS)                               # 51
        self.torque = np.zeros(self.N_JOINTS)                           # 52
        self.motor_cmd_msg = Float64MultiArray(data=[0.0] * (5 * self.N_JOINTS))  # 53 5×29 布局：[q_des, qd_des, Kp, Kd, torque]

        self.viewer = mujoco.viewer.launch_passive(self.m, self.d, show_left_ui=False, show_right_ui=False)  # 55 只显示不驱动
        self.viewer_dt = 0.02                                           # 56
        self.sim_timer = self.create_timer(self.m.opt.timestep, self.simulation_callback)  # 61 物理步挂 ROS wall timer（非硬实时，见 8.4）
```

补充两点：收到第一条策略命令前 Kp=Kd=0，机器人以零力矩自由下落——要与控制器侧 sim hold / `move_to_init_pose` 配合才安全；初始高度 0.977054 若与 Isaac 侧不一致，首帧观测即带阶跃。

**主回调 `simulation_callback`（67–104 行）：**

```python
        if self.viewer.is_running() and time.time() - self.start_time < self.max_duration:  # 70 双终止条件：窗口被关 / 墙钟超 200 s（超时后空转，不退出进程）
            joint_positions = self.d.qpos[7:].copy()   # 72 关节角：qpos 前 7 维是基座（3 平移 + 4 四元数）
            joint_velocities = self.d.qvel[6:].copy()  # 73 关节速度：qvel 前 6 维是基座速度（3 线 + 3 角）

            robot_state_msg = Float64MultiArray()                                   # 76
            robot_state_msg.data = np.concatenate([joint_positions, joint_velocities]).tolist()  # 77 29+29 拼一维
            self.robot_state_publisher.publish(robot_state_msg)                     # 78 → /motor_feedback（无时间戳/序号，风险见 8.3）

            imu_msg = Imu()                            # 81
            imu_msg.orientation_w = self.d.qpos[3]     # 82 基座四元数按 w,x,y,z 字段显式命名（qpos[3:7]）
            imu_msg.orientation_x = self.d.qpos[4]     # 83
            imu_msg.orientation_y = self.d.qpos[5]     # 84
            imu_msg.orientation_z = self.d.qpos[6]     # 85
            imu_msg.angular_vel_x = self.d.qvel[3]     # 86 基座角速度（qvel[3:6]）
            imu_msg.angular_vel_y = self.d.qvel[4]     # 87
            imu_msg.angular_vel_z = self.d.qvel[5]     # 88
            self.imu_publisher.publish(imu_msg)        # 89 → /imu_feedback

            num_joints = self.N_JOINTS                                       # 91
            self.q_des = np.array(self.motor_cmd_msg.data[:num_joints])                       # 92 目标位置（5×29 第 1 段）
            self.qd_des = np.array(self.motor_cmd_msg.data[num_joints : 2 * num_joints])      # 93 目标速度（第 2 段，恒 0）
            self.Kp = np.array(self.motor_cmd_msg.data[2 * num_joints : 3 * num_joints])      # 94 Kp（第 3 段，随每条命令重发）
            self.Kd = np.array(self.motor_cmd_msg.data[3 * num_joints : 4 * num_joints])      # 95 Kd（第 4 段）
            self.torque = np.zeros(num_joints)  # Ignore RL torque for steady standing        # 96 前馈力矩恒置零：纯位置目标 + PD

            self.d.ctrl[:] = (                                                                # 98 ctrl 即 motor 力矩；代码不另 clip，上限由 MJCF ctrlrange 决定
                pd_control(self.q_des, self.d.qpos[7:], self.Kp, self.qd_des, self.d.qvel[6:], self.Kd) + self.torque
            )                                                                                 # 100
            mujoco.mj_step(self.m, self.d)                                                    # 101 物理前推 2 ms；两次策略命令间 q_des 不变 = Zero-Order Hold
            if self.sim_count % (self.viewer_dt / self.m.opt.timestep) == 0:                  # 102 0.02 / 0.002 = 10
                self.viewer.sync()                                                            # 103 每 10 步渲染一次（50 FPS，修掉 Gym 的 1000 Hz 渲染问题）
            self.sim_count += 1                                                               # 104
```

与 Gym 的同构与差异一眼可见：状态截取同思路（`q[-12:]` vs `qpos[7:]`）；四元数顺序从"数组内隐式约定"升级为"消息字段显式命名"；渲染降频 20 倍。

### 3.2 控制器主节点：`timer_callback` 与三处关节重排

50 Hz 由 YAML 的 `control_dt: 0.02` 经 `create_timer` 建立（era_rl_controller_node.py:75）。

```python
        if self.imu_msg is None or self.motor_msg is None:      # 99 启动期保护：任一反馈未到则跳过本周期
            return
        self.measured_q = self.motor_msg.data[: self.N_JOINTS]          # 102 拆反馈：前 29 = 位置
        self.measured_qd = self.motor_msg.data[self.N_JOINTS : 2 * self.N_JOINTS]  # 103 后 29 = 速度
        mes_q = convert_joint_order(self.measured_q, self.robot_joint_sequence, self.rl_joint_sequence)    # 104 重排①②：robot → 策略顺序
        mes_qd = convert_joint_order(self.measured_qd, self.robot_joint_sequence, self.rl_joint_sequence)  # 105 （观测必须按 metadata joint_names 排列）

        self.root_quat = np.array(                                     # 107
            [                                                          # 108–114 IMU 四元数按 w,x,y,z 装配，与 3.1 消息字段顺序呼应
                self.imu_msg.orientation_w,
                self.imu_msg.orientation_x,
                self.imu_msg.orientation_y,
                self.imu_msg.orientation_z,
            ],
            dtype=np.float32,
        )                                                              # 115
        self.root_ang_vel = np.array(                                  # 116
            [self.imu_msg.angular_vel_x, self.imu_msg.angular_vel_y, self.imu_msg.angular_vel_z], dtype=np.float32  # 117
        )                                                              # 118

        if not self.enable_rl_infer:                                   # 123 真机模式：先走 3 秒初始化斜坡
            reordered_joint_target_pos = self.move_to_init_pose()      # 124
        else:
            joint_target_pos = self.rl_interface.perform_inference(self.root_quat, self.root_ang_vel, mes_q, mes_qd)  # 126 推理，输出为策略顺序
            reordered_joint_target_pos = convert_joint_order(          # 127 重排③：策略 → robot 顺序再发送
                joint_target_pos, self.rl_joint_sequence, self.robot_joint_sequence)  # 128

        self.send_motor_command(                                       # 131 发送 5×29 命令
            reordered_joint_target_pos,
            [0.0] * len(reordered_joint_target_pos),                   # 133 qd_des 恒零
            self.Kp, self.Kd,                                          # 134 构造时已按 rl→robot 预重排（35–36 行）
            [0.0] * len(reordered_joint_target_pos),                   # 136 torque 恒零
        )
```

三处重排共用同一个 7 行函数：

```python
def convert_joint_order(data, from_order, to_order):                   # utils.py:4
    reordered_data = np.array([data[from_order.index(joint)] if joint in from_order else 0.0 for joint in to_order])  # 5 按名称查表；找不到★静默填 0.0
    return reordered_data                                              # 6
```

**静默填零**是这 7 行最大的隐患：真机 31 关节中颈部关节不在策略 29 关节内，靠这条 `else 0.0` 正常工作；但关节名拼写错误或模型变更同样会被吞掉而不报错。另外两处补充：Kp/Kd 在构造时已按 `convert_joint_order(rl→robot)` 预重排，真机模式还会把颈部两关节覆盖为 `Kp=90, Kd=2`（era_rl_controller_node.py:44-48）；真机特有的 `move_to_init_pose`（139–152 行）做 3 秒线性插值过渡到默认姿态，结束后置 `enable_rl_infer = True` 并发布 `/is_sim_hold_release`，sim2sim 模式则一开始就推理（era_rl_controller_node.py:50-51）。

### 3.3 推理接口：`perform_inference` 逐行

**模型与 metadata 解析（`parse_model`，84–101 行）：**

```python
        self.policy = onnxruntime.InferenceSession(self.config["policy_path"])  # 85 打开 ONNX 会话
        meta_data = self.policy.get_modelmeta().custom_metadata_map             # 86 读自定义 metadata

        for key in meta_data:                                                   # 88 逐项把 CSV 字符串解析回数组
            if key == "joint_names":                                            # 89 策略侧关节顺序
                self.joint_seq = meta_data[key].split(",")                      # 90
            if key == "default_joint_pos":                                      # 91
                self.default_joint_pos = np.array([float(x) for x in meta_data[key].split(",")])  # 92
            if key == "joint_stiffness":                                        # 93
                self.Kp = np.array([float(x) for x in meta_data[key].split(",")])  # 94
            if key == "joint_damping":                                          # 95
                self.Kd = np.array([float(x) for x in meta_data[key].split(",")])  # 96
            if key == "action_scale":                                           # 97
                self.action_scale = np.array([float(x) for x in meta_data[key].split(",")])  # 98

        self.action_buffer = np.zeros((len(self.joint_seq),), dtype=np.float32)  # 100 首帧 action 观测 = 0
        self.time_step = 0                                                      # 101
```

部署端不写任何魔法数字——关节顺序、默认角、Kp/Kd、scale 全部从模型文件自描述读取，这是与 Gym（Python 配置硬编码）的根本区别。

**单周期推理（103–136 行）：**

```python
        command = self.command.copy()                                       # 105 键盘/外部命令的当前值

        gvec2 = quat_rotate_inverse_np(root_quat, np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(  # 107 世界 −Z → 基座系 = projected gravity
            np.float32
        )  # (3,)
        mes_qpos_rel = (mes_q - self.default_joint_pos).astype(np.float32)  # (29,)                            # 110 ★显式减默认角（Gym 省略此步）
        prev_action = self.action_buffer.copy().astype(np.float32)          # (29,)                            # 111 上一次原始动作

        self.obs_hist.push_item("command", command)                         # 113 以下 6 项各自压入独立历史通道
        self.obs_hist.push_item("gvec2", gvec2)                             # 114 3+3+3+29+29+29 = 96 维单帧
        self.obs_hist.push_item("base_ang_vel", root_ang_vel)               # 115
        self.obs_hist.push_item("qpos_rel", mes_qpos_rel)                   # 116
        self.obs_hist.push_item("qvel", mes_qdot)                           # 117
        self.obs_hist.push_item("action", prev_action)                      # 118

        obs = self.obs_hist.build_obs()  # shape = (num_obs, )              # 120 组装 96×5 = 480 维；无任何 scale 乘法（归一化已内嵌 ONNX 图，见 3.5）

        action = self.policy.run(                                           # 122 ONNX Runtime 推理
            ["actions"],                                                    # 123 输出名
            {
                "obs": obs.reshape(1, -1),                                  # 125 输入名 obs，加 batch 维
            },
        )[0]                                                                # 127 取第一个输出

        action = np.asarray(action, dtype=np.float32).reshape(-1)           # 129
        self.action_buffer = action.copy()                                  # 130 存原始 action（未乘 scale）供下帧观测——语义同 Gym obs[29:41]

        joint_pos_des = action * self.action_scale + self.default_joint_pos  # 132 完整后处理 q_des = a×s + q_default；s 逐关节（公式见 4.5 节）

        self.time_step += 1                                                 # 134

        return joint_pos_des                                                # 136
```

`quat_rotate_inverse_np` 按 `[w,x,y,z]` 取 `q_w=q[0]、q_vec=q[1:]`，实现 $q^* \otimes v \otimes q$ 的世界→基座方向变换（math.py:108-118）。键盘命令（57–78 行）是 locomotion 特有交互：`wasd` 增减 `command[0:2]`、`q/e` 增减 yaw、空格清零、步长 0.1——部署时速度命令可变，不像 Gym 写死在类属性里。

### 3.4 历史缓冲：`RingHistory` 逐行

```python
class RingHistory:
    def __init__(self, capacity: int, dim: int):
        self.capacity = int(capacity)                                  # 6
        self.dim = int(dim)                                            # 7
        self.buf = np.zeros((self.capacity, self.dim), dtype=np.float32)  # 8 定长矩阵，固定内存
        self.head = 0                                                  # 9 指向最旧帧
        self.size = 0                                                  # 10 已有帧数

    def push(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32).reshape(                   # 13
            self.dim,
        )
        tail = (self.head + self.size) % self.capacity                 # 16 写入位 = 最新帧的下一格
        self.buf[tail] = x                                             # 17
        if self.size < self.capacity:                                  # 18 未满：size 增长，head 不动
            self.size += 1                                             # 19
        else:                                                          # 20 已满：本次写入覆盖最旧帧，head 前移一格
            self.head = (self.head + 1) % self.capacity                # 21

    def get_matrix(self) -> np.ndarray:
        if self.size < self.capacity:                                  # 24 未满（此时 head 仍为 0）：
            out = np.zeros((self.capacity, self.dim), dtype=np.float32)  # 25
            if self.size > 0:                                          # 26
                out[-self.size :] = self.buf[: self.size]              # 27 有效数据右对齐 = 左侧补零（同 Gym 首帧零填充语义）
            return out                                                 # 28
        return np.concatenate([self.buf[self.head :], self.buf[: self.head]], axis=0)  # 29 满：从 head 切一刀首尾拼接 → [最旧…最新]

    def flatten(self) -> np.ndarray:
        return self.get_matrix().reshape(-1)                           # 32


class ObsHistory:
    def __init__(self, history_cfg):
        self.dims = {name: cfg["dim"] for name, cfg in history_cfg.items()}  # 37
        self.rings = {name: RingHistory(cfg["history_length"], cfg["dim"]) for name, cfg in history_cfg.items()}  # 38 每个观测项一条独立 ring
        self.order = list(history_cfg.keys())                          # 39 YAML 键序 = 展开顺序

    ...

    def build_obs(self) -> np.ndarray:
        parts = [self.rings[name].flatten() for name in self.order]    # 49 按 YAML 顺序取各通道展开
        return np.concatenate(parts, axis=0).astype(np.float32)        # 50
```

与 Gym 的 `deque.append + popleft` 语义等价，但固定内存、无数据搬移，更接近 C++ 部署写法。风险落点：**总维度相同但 YAML 顺序写错时，网络输入语义完全错位**——观测项顺序契约见 4.3 节。

### 3.5 ONNX 导出：normalizer 内嵌与 metadata 附加

locomotion 导出类的核心一行：

```python
    def forward(self, x, time_step):                       # 144
        return (self.actor(self.normalizer(x)),)           # 145 ★计算图 = raw obs → normalizer → actor：归一化器随图一起导出
```

这是"策略自描述"与"配置手工同步"两条路线的分水岭——部署端因此不需要知道任何 scale（3.3 的 `build_obs()` 里没有一次乘法）。mimic 版（33–56 行）的 forward 额外返回参考动作五元组，并用 `torch.clamp(time_step, max=T-1)` 把时间索引钳在最后一帧——"动作结束后停在终态"的行为就是这一行 clamp 的直接结果。`export`（147–163 行）以零张量探图、`torch.onnx.export` opset 11 落盘，输入名 `obs/time_step`、输出名 `actions`。

metadata 附加（`attach_loco_onnx_metadata`，166 行起）：

```python
        metadata = {
            "run_path": run_path,
            "joint_names": env.scene["robot"].data.joint_names,                       # 175 策略关节顺序
            "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),   # 176 Kp——读自当前 env 实例
            "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),       # 177 Kd
            "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),  # 178 默认角
            "command_names": env.command_manager.active_terms,                         # 179
            "observation_names": env.observation_manager.active_terms["policy"],        # 180 ★观测项顺序随图走
            "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),      # 181 逐关节 scale
        }

        # 以下读回 ONNX、逐项写入 metadata_props（CSV 字符串，3.3 的解析与这里键名一一对应），最后保存
        model = onnx.load(onnx_path)
        for k, v in metadata.items():
            entry = onnx.StringStringEntryProto()
            entry.key = k
            entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
            model.metadata_props.append(entry)

        onnx.save(model, onnx_path)
```

注意读取时机：176–178 行的 stiffness/damping/default_joint_pos 取自 env **运行时实例**——若导出时环境正处于随机化后的状态，写入的就是被随机化过的值（8.2 节风险的代码出处）。

---

## 4. Isaac 与 MuJoCo 是怎样对齐的

### 4.1 第一层：机器人拓扑与名义模型

Isaac Lab 使用：

```text
source/.../assets/xbot_description/urdf/l7_29dof_neck_fixed.urdf
```

MuJoCo 使用：

```text
deploy/l7_29dof_neck_fixed/l7_29dof_neck_fixed.xml
```

两边不是运行时共享同一个模型，而是各自加载 URDF 和 MJCF。当前提交中可以确认：

- 都是 L7 neck-fixed 模型；
- 都有 29 个活动关节；
- 关节名称、集合和当前顺序一致；
- 主要质量、惯量、质心和关节安装关系来自同一套机器人描述；
- 关节限位和电机力矩上限大体对应；
- 腿、腰和双臂采用相同的控制自由度划分。

当前 29 关节顺序为：

```text
左腿 6
→ 右腿 6
→ 腰 3
→ 左臂 7
→ 右臂 7
```

例如 Isaac 侧 hip roll/yaw/pitch 的 effort limit 是：

```text
255 / 100 / 350 N·m
```

MuJoCo actuator 的 `ctrlrange` 也使用相应范围。

不过，仓库没有 URDF→MJCF 自动同步脚本，也没有资产参数 diff。两份文件仍然是独立维护的，所以“同源模型”不等于“所有字段完全相同”。

### 4.2 第二层：关节名称与顺序

虽然当前 URDF 与 MJCF 的 29 关节顺序相同，控制器仍然使用名称映射，而不是把顺序一致当成永久前提。

ONNX metadata 保存训练侧 `joint_names`；部署 YAML 保存 MuJoCo 或真机的 `joint_sequence`。控制器用：

```python
convert_joint_order(data, from_order, to_order)
```

重排：

- 关节位置和速度；
- Kp/Kd；
- 策略目标位置。

这层设计主要解决三个问题：

1. 未来 URDF、MJCF 排列发生变化；
2. 真机驱动使用另一套顺序；
3. 真机有 31 个关节，而策略只控制 29 个，额外两个是颈部关节。

当前实现的不足是：找不到某个目标关节时会静默填零。更安全的做法是启动时校验关节集合，受控关节缺失时直接失败。

### 4.3 第三层：观测契约

Sim-to-Sim 最容易出错的不是神经网络本身，而是 MuJoCo 端是否重建了与 Isaac 训练端完全相同的 actor 输入。

需要一致的内容包括：

```text
观测名称
观测顺序
单位和 scale
坐标系
四元数顺序
历史长度
历史展开顺序
首帧填充方式
上一动作的写入时机
```

#### Locomotion 观测

Locomotion 单帧观测为：

| 观测 | 维度 | 含义 |
|---|---:|---|
| `command` | 3 | 前向、侧向和 yaw 速度命令 |
| `gvec2` | 3 | 世界重力在机器人基座坐标系中的投影 |
| `base_ang_vel` | 3 | 基座角速度 |
| `qpos_rel` | 29 | `q - default_q` |
| `qvel` | 29 | 关节速度 |
| `action` | 29 | 上一帧策略动作 |

因此：

```text
单帧：3 + 3 + 3 + 29 + 29 + 29 = 96
5 帧历史：96 × 5 = 480
```

#### Mimic 观测

Mimic 单帧观测为：

| 观测                    |  维度 | 含义                         |
| --------------------- | --: | -------------------------- |
| `curr_motion_data`    |  58 | 参考动作的 29 维关节位置与 29 维关节速度   |
| `motion_anchor_ori_b` |   6 | 参考 anchor 相对机器人当前基座的 6D 朝向 |
| `base_ang_vel`        |   3 | 基座角速度                      |
| `qpos_rel`            |  29 | `q - default_q`            |
| `qvel`                |  29 | 关节速度                       |
| `action`              |  29 | 上一帧策略动作                    |

因此：

```text
单帧：58 + 6 + 3 + 29 + 29 + 29 = 154
5 帧历史：154 × 5 = 770
```

通用训练配置原本还定义了 `motion_anchor_pos_b` 和 `base_lin_vel`，但 L7 专用配置主动将它们关闭：

```python
self.observations.policy.motion_anchor_pos_b = None
self.observations.policy.base_lin_vel = None
```

这不是随意删观测，而是在训练阶段就避免 actor 依赖部署端或真机端难以稳定获取的信号。

#### 历史帧

部署端为每个观测项建立独立 ring buffer：

```text
[最旧帧, ..., 上一帧, 当前帧]
```

历史不足时在左侧补零，再按照 YAML 中观测项出现的顺序展开。该顺序必须与 Isaac Lab observation manager 的输出一致，否则即使总维度相同，网络输入语义也会完全错误。

### 4.4 第四层：Mimic 参考动作和朝向

Mimic 部署额外加载 `dance_9.npz`，其中包含：

- 每帧 29 维参考关节位置；
- 每帧 29 维参考关节速度；
- 各刚体的参考四元数；
- 50 Hz 动作序列。

策略和动作数据都是 50 Hz，所以每次推理前进一个 motion frame。

机器人启动方向通常不等于动作数据的世界 yaw。第一帧会计算：

```text
参考动作初始 anchor yaw → 机器人启动 yaw
```

后续将参考 anchor 姿态转换到当前机器人基座坐标系，再取旋转矩阵前两列作为 6D rotation 输入。

这里只消除初始 yaw 差异，不会删除 roll/pitch，因为 roll/pitch 本身属于平衡状态。

动作结束后，时间索引被钳制在最后一帧，机器人持续追踪最终姿态，不会自动循环或回到站立。

### 4.5 第五层：动作、默认姿态与 PD

Isaac 训练动作采用：

```python
JointPositionActionCfg(
    use_default_offset=True,
)
```

因此 actor 输出的不是绝对角度，而是默认姿态附近的残差：

$$
q_{des,i}=q_{default,i}+s_i a_i
$$

每个关节的 action scale 为：

$$
s_i=0.25\frac{\tau_{limit,i}}{K_{p,i}}
$$

ONNX 导出器将以下信息写入 metadata：

```text
joint_names
default_joint_pos
joint_stiffness
joint_damping
action_scale
observation_names
anchor_body_name / anchor_body_id（mimic）
```

导出的网络还包含 observation normalizer：

```text
raw observation
→ normalizer
→ actor
→ action
```

部署端读取 metadata，恢复相同的动作后处理：

```python
q_des = action * action_scale + default_joint_pos
```

MuJoCo 节点再计算：

$$
\tau=K_p(q_{des}-q)+K_d(0-\dot q)
$$

两次策略推理之间保持最近一次 `q_des`，即 Zero-Order Hold。代码没有再次手动裁剪力矩，最终范围由 MJCF motor 的 `ctrlrange` 限制。

消息格式虽然保留了前馈 torque 字段，但当前 MuJoCo 节点实际将其置零，因此部署是纯位置目标 + PD，不是策略直接输出力矩。

### 4.6 第六层：控制周期

| 层级 | Isaac Lab | MuJoCo |
|---|---:|---:|
| physics dt | 0.005 s | 0.002 s |
| physics rate | 200 Hz | 名义 500 Hz |
| decimation / 每控制周期步数 | 4 | 约 10 |
| policy/control dt | 0.020 s | 0.020 s |
| policy rate | 50 Hz | 50 Hz |

因此，两侧对齐的是 actor 每 20 ms 更新一次，而不是底层物理积分完全相同。

还要注意：MuJoCo 的 2 ms 步进由 Python `rclpy` wall timer 调度。桌面运行中它可能存在抖动，不能直接等同于 Isaac 内部固定 decimation，也不能视为硬实时 500 Hz。

---

## 5. 哪些地方没有严格对齐

### 5.1 部分 armature 不一致

代码中的名义值已经能看到差异：

| 关节族 | Isaac armature | MuJoCo armature |
|---|---:|---:|
| shoulder pitch/roll、arm yaw、elbow pitch | 0.01 | 0.0685 |
| elbow yaw、wrist | 0.01 | 0.03 |
| waist yaw | 0.01 | 0.0685 |
| waist roll/pitch | 0.02 | 0.025101925 |

所以不能说 MJCF 完整复制了 Isaac actuator dynamics。

### 5.2 求解器与接触不同

Isaac 使用 PhysX articulation；MuJoCo XML 使用 PGS solver，并设置 50 iterations。两者在以下方面具有不同语义：

- 接触何时建立；
- 允许多少穿透；
- 摩擦锥如何近似；
- 多接触点之间如何分配力；
- 关节限位与 drive 如何进入约束求解；
- warm start、正则化与稳定化方式。

因此，即使模型质量和控制力矩完全相同，足底滑移、落地冲击和长期姿态仍可能不同。

### 5.3 关节摩擦不同

MuJoCo MJCF 给关节设置了：

```xml
<joint frictionloss="0.1" .../>
```

Isaac 侧使用自己的关节参数，并在训练时随机化 joint friction。两者不是同一种固定摩擦模型。

### 5.4 延迟没有一一复现

训练使用 `DelayedImplicitActuator`，每个环境会随机选择 0～4 个 physics step 的延迟：

```text
0～4 × 5 ms = 0～20 ms
```

MuJoCo 部署没有对应的显式 delay buffer，只存在 ROS 消息和 timer 带来的实际调度延迟。训练延迟是受控随机变量，部署延迟是运行时系统行为，二者不能直接视为相同。

### 5.5 Domain Randomization 不是对齐证据

Isaac 训练随机化了：

- 地面摩擦和 restitution；
- 全身质量与 torso 附加质量；
- torso 质心；
- 关节默认位置；
- joint friction 和 armature；
- Kp/Kd；
- 执行器延迟；
- 外部推力；
- actor 观测噪声。

它的作用是让策略在一族动力学参数上都能工作，从而降低对单一 PhysX 参数的依赖。

正确理解是：

> Domain Randomization 提高跨引擎成功率，但不能证明两个引擎已经对齐。

### 5.6 仓库没有跨引擎数值门禁

当前官方路径没有：

- 相同 state/action 注入；
- Isaac/MuJoCo observation 逐项比较；
- PyTorch/ONNX 同输入输出报告；
- 单关节 response diff；
- raw/clipped/applied torque 对比；
- 全身轨迹 MAE、跌倒率或接触误差阈值；
- CI 自动 pass/fail。

此外，L7 配置中的：

```python
self.use_identify_params = True
self.use_high_waist_stiffness = True
```

在当前提交中只有赋值，没有其他读取位置。它们不会自动触发参数辨识或跨引擎对齐。实际腰部 stiffness 来自 actuator 配置本身。

---

## 6. Locomotion 与 Mimic 的区别

二者共用 ROS 2 控制器、历史观测、ONNX Runtime、关节名称映射和 PD 后端，区别主要在任务输入。

| 项目 | Locomotion | Mimic |
|---|---|---|
| 目标 | 跟踪速度指令 | 跟踪一段固定动作 |
| 外部输入 | `[vx, vy, yaw_rate]` | motion NPZ + time step |
| 单帧观测 | 96 | 154 |
| 历史长度 | 5 | 5 |
| actor 输入 | 480 | 770 |
| 姿态信息 | projected gravity | motion anchor 相对 6D orientation |
| 参考关节 | 无 | 29 q + 29 dq |
| 动作结束 | 不适用 | 停在 motion 最后一帧 |

Locomotion 适合交互式速度控制；Mimic 除了验证动力学，还验证动作数据时间推进、初始 yaw 对齐和参考姿态坐标变换。

---

## 7. 与 Humanoid-Gym 的 Sim-to-Sim 对比

### 7.1 总体区别

| 维度 | Humanoid-Gym | Humanoid-Lab |
|---|---|---|
| 训练框架 | Isaac Gym | Isaac Lab / Isaac Sim |
| 机器人 | XBot-L | L7 neck-fixed |
| 策略任务 | 12-DoF 下肢 locomotion | 29-DoF locomotion + mimic |
| 部署结构 | 单 Python 脚本 | ROS 2 控制器 + MuJoCo 节点 |
| 策略格式 | TorchScript `.pt` | ONNX |
| 配置来源 | Python 中硬编码 | ONNX metadata + YAML |
| 关节映射 | 依赖固定数组位置 | 按关节名称重排 |
| 动作维度 | 12 | 29 |
| actor 输入 | 47 × 15 = 705 | loco 480；mimic 770 |
| policy rate | 100 Hz | 50 Hz |
| MuJoCo rate | 1000 Hz | 名义 500 Hz |
| 真机复用 | 没有形成后端抽象 | 同一 RL 控制器支持 sim2sim/real |

### 7.2 Humanoid-Gym 的最小闭环

Gym 的核心全部集中在：

```text
humanoid/scripts/sim2sim.py
```

数据流是：

```text
MuJoCo state
→ 手写 47 维当前观测
→ 15 帧历史，705 维
→ TorchScript policy
→ 12 维 action
→ q_des = 0.25 × action
→ PD torque
→ mj_step
```

47 维单帧包括：

```text
步态相位 sin/cos       2
速度命令               3
关节位置              12
关节速度              12
上一动作              12
基座角速度             3
基座欧拉角             3
合计                  47
```

Gym 的优点是链路短，适合学习和数值调试；缺点是许多部署契约依赖手工同步：

- 默认姿态；
- 关节数组位置；
- Kp/Kd；
- action scale；
- 观测顺序；
- 模型和策略版本。

### 7.3 Lab 相比 Gym 真正增加了什么

Lab 的进步不是换成 ONNX 这么简单，而是把策略部署从“一个脚本”提升为“一个接口契约”：

```text
policy.onnx
├── actor + normalizer
├── joint names
├── default joint position
├── Kp/Kd
├── action scale
└── observation metadata
```

再用 ROS 2 将控制器与 MuJoCo/真机后端解耦。

代价是引入新的工程风险：

- motor 与 IMU 反馈不同步；
- 消息没有完整时间戳协议；
- timer 抖动；
- 反馈过期时仍可能沿用旧数据；
- 两节点启动、退出和异常恢复更复杂。

所以 Lab 更接近真机部署结构，但不代表它天然比单进程 Gym 更容易做严格数值对比。

---

## 8. 当前实现中最值得注意的风险

### 8.1 观测坐标系缺少显式契约

MuJoCo 节点直接使用 `qvel[3:6]` 作为 base angular velocity，训练端则使用 Isaac Lab 的 `base_ang_vel`。必须确认两者：

- 是否都在 base frame；
- 轴方向是否一致；
- 四元数是否都是 `wxyz`；
- 四元数表示 body-to-world 还是 world-to-body。

仅靠机器人能站住不能证明坐标系完全正确，应使用已知姿态和已知角速度做 golden test。

### 8.2 ONNX metadata 可能固化随机化参数

导出器从当前环境实例读取 stiffness、damping 和 default joint position。训练环境又会随机化这些量，因此同一 checkpoint 在不同随机种子下导出时，metadata 可能不同。

更稳妥的方案是：

- 导出前关闭随机化并 reset 到 nominal config；
- 单独维护 identified deployment config；
- 写入 checkpoint、seed、URDF/MJCF 和配置哈希；
- 对多次导出的 metadata 做 diff。

### 8.3 反馈没有时间戳和数据新鲜度检查

控制器每次使用“最近一次 motor feedback + 最近一次 IMU”，但二者未必属于同一物理时刻。

应补充：

- source timestamp；
- sequence number；
- feedback age；
- approximate/exact sync；
- timeout watchdog。

### 8.4 名义 500 Hz 不等于硬实时

桌面 MuJoCo 演示可以使用 Python timer，但真机高频 PD 应位于电机驱动器、实时控制板或实时 C++ 进程。RL 节点只负责以 50 Hz 更新目标。

### 8.5 缺少状态机和量化评测

当前演示还缺少：

- 跌倒检测与自动 reset；
- motion 结束后的安全过渡；
- feedback 超时保护；
- action rate limit；
- command tracking error；
- base height/roll/pitch 越界统计；
- 足部滑移与力矩饱和比例；
- 多摩擦、质量、延迟和初始姿态的批量成功率。

---

## 9. 怎样把它升级成真正的 Isaac–MuJoCo 对齐验证

推荐按以下顺序进行，而不是直接比较完整舞蹈视频。

### 9.1 Gate 1：资产与接口静态检查

自动比较：

```text
joint/body names
joint axes and limits
mass/inertia/COM
collision geometry
armature/friction
effort limit
Kp/Kd/action scale/default_q
physics dt/control dt
```

输出 URDF/MJCF/ONNX/YAML 的版本和哈希。

### 9.2 Gate 2：Observation Golden Test

1. 在 Isaac 保存一组原始机器人状态；
2. 保存 Isaac observation manager 的逐项输出；
3. 将同一状态输入部署 observation builder；
4. 按观测项逐元素比较；
5. 最后比较 actor 输入和 actor 输出。

报告应类似：

```text
motion command       max error: ...
anchor orientation   max error: ...
base angular velocity max error: ...
joint position       max error: ...
joint velocity       max error: ...
history              first mismatch: ...
actor output         max error: ...
```

这一步先证明接口正确，避免把观测 bug 错判为物理引擎差异。

### 9.3 Gate 3：固定基座单关节阶跃

统一初态、`q_des`、Kp/Kd、力矩上限、physics dt 和 control dt，记录：

```text
q_des
q / dq
tau_raw
tau_clipped
tau_applied
limit state
```

先验证 PD 和 actuator 语义，再逐项加入 armature、frictionloss 和不同 timestep。

### 9.4 Gate 4：无控制动力学与接触

分开测试：

- 无接触自由运动；
- 固定高度落地；
- 固定站立姿态；
- 可重复外力推扰。

收集根节点位姿、接触状态、法向力、足端滑移和能量变化。

### 9.5 Gate 5：冻结策略批量闭环

最后才运行 locomotion 或 mimic，并在多组物理参数下统计：

- 成功率和跌倒时间；
- command tracking error；
- mimic joint/anchor error；
- base 姿态与高度；
- torque saturation；
- action smoothness；
- 接触和足端滑移。

只有形成阈值和自动 pass/fail 后，Sim-to-Sim 才从“能运行的演示”变成“可重复的回归门禁”。

---

## 10. 最终评价

Humanoid-Gym 展示的是最小 Sim-to-Sim 闭环：单进程、12-DoF、TorchScript、手写观测和 PD。它适合快速理解，但部署契约大量依赖硬编码。

Humanoid-Lab 展示的是更接近 Sim-to-Real 的软件分层：29-DoF 全身策略、ONNX 自描述 metadata、按名称映射关节、ROS 2 控制器与机器人后端解耦，并支持 locomotion 与单段动作 mimic。

它当前可以证明和不能证明的内容如下：

| 可以确认 | 不能据此确认 |
|---|---|
| 冻结策略可以通过部署接口在 MuJoCo 中闭环运行 | PhysX 与 MuJoCo 动力学数值等价 |
| actor 观测、动作语义和 50 Hz policy rate 基本复现 | physics dt、接触、摩擦、armature 和延迟完全一致 |
| ONNX metadata 降低了部署配置漂移风险 | metadata 一定来自未随机化的 nominal 参数 |
| Domain Randomization 提高了跨环境鲁棒性 | 随机化已经覆盖全部引擎差异 |
| ROS 2 分层更接近真机软件结构 | 当前 Python 节点具备硬实时和反馈同步保证 |

> [!summary]
> Humanoid-Lab Sim-to-Sim 的核心价值，不是把 Isaac 和 MuJoCo 调成完全相同，而是尽可能保持“策略看到什么、输出什么、多久输出一次”不变，再用同源模型和 Domain Randomization 承受剩余动力学差异。
>
> 如果要进一步声称“Isaac–MuJoCo 已对齐”，还必须补上资产 diff、observation golden test、同状态注入、单关节响应、接触实验和自动化误差门禁。

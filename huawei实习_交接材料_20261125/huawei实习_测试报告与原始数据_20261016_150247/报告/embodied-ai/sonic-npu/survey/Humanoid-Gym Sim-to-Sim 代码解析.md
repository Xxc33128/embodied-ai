---
title: Humanoid-Gym Sim-to-Sim 代码解析
type: 代码阅读笔记
tags:
  - 人形机器人
  - 强化学习
  - Sim-to-Sim
  - Sim-to-Real
  - Isaac-Gym
  - MuJoCo
项目: Humanoid-Gym
仓库: https://github.com/roboterax/humanoid-gym
核心文件: humanoid/scripts/sim2sim.py
创建时间: 2026-08-31
更新时间: 2026-09-04
本版说明: 源码逐行走读（对照本地仓库实测行号）；行内注释解释代码，框外只留补充信息
---

# Humanoid-Gym Sim-to-Sim 代码解析

> 一句话：**Sim-to-Sim 将在 Isaac Gym 中训练并导出的 Actor 策略，放入 MuJoCo 中重新运行；通过更换物理引擎检查策略对动力学与接触模型差异的鲁棒性。**

## TL;DR

- 训练在 **Isaac Gym** 中完成，策略由 PPO 学得；`play.py` 从 checkpoint 中提取 Actor，并导出为 TorchScript 文件 `policy_1.pt`。
- `sim2sim.py` 在 **MuJoCo** 中手工复现训练端的控制闭环：构造 47 维单帧观测、堆叠 15 帧得到 705 维输入、以 100 Hz 调用 Actor、以 1000 Hz 执行 PD 控制。
- Actor 输出 12 维动作，对应左右腿各 6 个关节；动作乘 `action_scale=0.25` 后成为目标关节位置偏移。
- Sim-to-Sim 的关键不是"模型格式转换"，而是确保**观测顺序、归一化、历史帧、关节顺序、动作缩放、PD 增益和控制频率**在两个仿真器中一致。
- 当前实现存在若干需要注意的差异：MuJoCo 端统一使用 ±200 N·m 力矩限制（训练端实际是 URDF effort × 0.85 ≈ 42.5/68 N·m）、默认姿态被假设为 0、关节状态通过"最后 12 项"获取、没有摔倒重置，也没有完整复现训练噪声。

---

## 1. 常用缩写与符号

| 缩写/符号 | 英文全称 | 含义 |
|---|---|---|
| **Sim-to-Sim / Sim2Sim** | Simulation-to-Simulation | 将策略从一个仿真器迁移到另一个仿真器验证 |
| **Sim-to-Real / Sim2Real** | Simulation-to-Reality | 将仿真中训练的策略部署到真实机器人 |
| **RL** | Reinforcement Learning | 强化学习 |
| **PPO** | Proximal Policy Optimization | 近端策略优化，本项目使用的 RL 算法 |
| **Actor** | Policy Network | 策略网络，根据观测输出动作 |
| **Critic** | Value Network | 价值网络，训练时评估状态价值，部署时不需要 |
| **MLP** | Multi-Layer Perceptron | 多层感知机；本项目 Actor/Critic 都是 MLP |
| **PD** | Proportional-Derivative | 比例-微分控制器，将目标关节位置转换为力矩 |
| **DOF** | Degree of Freedom | 自由度；这里主要指 12 个受控腿部关节 |
| **IMU** | Inertial Measurement Unit | 惯性测量单元，提供姿态、角速度等信息 |
| **URDF** | Unified Robot Description Format | Isaac Gym 使用的机器人描述格式 |
| **MJCF** | MuJoCo XML Configuration Format | MuJoCo 使用的模型描述格式 |
| **JIT** | Just-In-Time Compilation | PyTorch 即时编译/导出机制；这里使用 TorchScript |
| **Hz** | Hertz | 频率单位，1 Hz 表示每秒执行 1 次 |
| $q$ | Joint Position | 关节位置，单位通常为 rad |
| $\dot q$ / `dq` | Joint Velocity | 关节速度，单位通常为 rad/s |
| $q_{target}$ | Target Joint Position | PD 控制器的目标关节位置 |
| $\tau$ / `tau` | Joint Torque | 关节力矩，单位 N·m |
| $K_p$ | Proportional Gain | PD 比例增益，决定位置误差产生多大力矩 |
| $K_d$ | Derivative Gain | PD 微分增益，抑制关节速度和振荡 |

---

## 2. Sim-to-Sim 在项目中的位置

完整链路如下：

```text
Isaac Gym 并行训练（4096 个环境）
              │
              ▼
PPO checkpoint：model_xxx.pt
包含 Actor + Critic + 动作标准差 + 优化器状态
              │
              ▼
play.py 加载 checkpoint
              │
              ▼
只提取 Actor MLP，导出为 policy_1.pt
              │
              ▼
sim2sim.py 使用 torch.jit.load()
              │
              ▼
MuJoCo 状态 → 705 维观测 → Actor → 12 维动作
              │
              ▼
1000 Hz PD 控制 → MuJoCo 物理仿真
```

需要区分两种模型文件：

| 文件 | 内容 | 用途 |
|---|---|---|
| `model_3001.pt` | Actor、Critic、探索标准差、优化器状态等 | 恢复训练或在 `play.py` 中加载 |
| `policy_1.pt` | 只有 Actor MLP | MuJoCo、C++ 或其他部署环境推理 |

导出的 Actor 网络结构为（隐藏层维度来自 `actor_hidden_dims = [512, 256, 128]`，`humanoid_config.py:236`）：

```text
705 → 512 → 256 → 128 → 12
```

---

## 3. 入口与运行方式

核心文件：

```text
humanoid/scripts/sim2sim.py
```

先运行 `play.py` 导出策略：

```bash
python humanoid/scripts/play.py \
  --task humanoid_ppo \
  --run_name v1
```

导出的模型默认位于：

```text
logs/XBot_ppo/exported/policies/policy_1.pt
```

然后运行平地 Sim-to-Sim：

```bash
python humanoid/scripts/sim2sim.py \
  --load_model logs/XBot_ppo/exported/policies/policy_1.pt
```

使用 MuJoCo 地形模型：

```bash
python humanoid/scripts/sim2sim.py \
  --load_model logs/XBot_ppo/exported/policies/policy_1.pt \
  --terrain
```

它不再创建 PPO、Critic 或 RolloutStorage——整个脚本只有 193 行（含 30 行许可证头），是一个自包含的最小部署闭环。

---

## 4. 源码总览：193 行的功能地图

逐行讲解之前，先给出整份文件的分段地图。`sim2sim.py` 除许可证头外只有约 160 行有效代码，结构是"三个工具函数 + 一个主循环函数 + 一个入口"：

| 行号 | 代码块 | 职责 | 训练端对应物 |
|---|---|---|---|
| 31–39 | `import` 区 | 引入 MuJoCo、TorchScript、训练侧配置 | — |
| 42–45 | `class cmd` | 硬编码速度命令 | `env.commands`（随机重采样） |
| 48–68 | `quaternion_to_euler_array()` | 四元数 → roll/pitch/yaw | `legged_robot.py` 的 `get_euler_xyz_tensor()` |
| 70–80 | `get_obs()` | 从 MuJoCo `data` 抽取机器人状态 | `post_physics_step()` 中的状态刷新 |
| 82–85 | `pd_control()` | PD 力矩公式 | `legged_robot.py` 的 `_compute_torques()` |
| 98–102 | `run_mujoco()` 初始化段 | 加载 MJCF、设定步长、创建 viewer | Gym 创建 sim 与 env |
| 104–111 | 状态初始化 | 目标、动作、15 帧全零历史 | `obs_history` 初始化 |
| 114–119 | 主循环头 | 1000 Hz 读取状态、截取关节 | — |
| 122–150 | 100 Hz 策略分支 | 47 维观测 → 705 维输入 → 推理 → 目标位置 | `compute_observations()` + `get_inference_policy()` |
| 153–162 | PD 与步进段 | 1000 Hz 力矩计算、`mj_step`、渲染 | `_compute_torques()` + `gym.simulate()` |
| 167–193 | `__main__` | 命令行参数、`Sim2simCfg`、加载策略 | `task_registry` 装配 |

两个嵌套频率是理解主循环的钥匙：外层 `for` 每 0.001 s 转一圈（1000 Hz，做 PD），`if count_lowlevel % decimation == 0` 每 10 圈进入一次（100 Hz，做推理）。

```text
for 循环体（1000 Hz，每 1 ms 一次）
 ├── 读 q / dq（最新状态）
 ├── [每 10 次进入] 构造 47 维 obs → 更新历史 → 705 维输入 → Actor → target_q
 ├── 计算 PD 力矩 tau（用刚读的 q / dq 和最新 target_q）
 ├── data.ctrl = tau
 └── mj_step() + viewer.render()
```

---

## 5. 逐行走读：导入、命令与工具函数（31–85 行）

### 5.1 导入区：复用训练配置是整个脚本的基石

```python
import math                                             # 31 相位 sin/cos 与 π 回折
import numpy as np                                      # 32
import mujoco, mujoco_viewer                            # 33 物理引擎绑定（MjModel/MjData/mj_step）+ 被动渲染窗口
from tqdm import tqdm                                   # 34 主循环进度条（按物理步刷新）
from collections import deque                           # 35 15 帧观测历史
from scipy.spatial.transform import Rotation as R       # 36 get_obs() 里的一次坐标变换
from humanoid import LEGGED_GYM_ROOT_DIR                # 37 拼接 MJCF 绝对路径
from humanoid.envs import XBotLCfg                      # 38 ★ 直接 import 训练配置类——scale 的单一来源
import torch                                            # 39 加载并运行 TorchScript Actor
```

关键一行是 `from humanoid.envs import XBotLCfg`：Sim-to-Sim 不自己定义任何缩放，后文所有 `cfg.normalization.obs_scales.*` 都来自训练侧同一个 Python 类——"归一化一致性"不靠文档约定，而靠**同一份代码同时服务训练与部署**。若把脚本复制到别的仓库独立维护，这层保证立即消失（Lab 用 ONNX metadata 解决同一问题，见对比笔记 3.3 节）。

### 5.2 `class cmd`：写死的速度命令

```python
class cmd:          # 42 部署命令，无任何运行时修改入口（问题见 §12.6）
    vx = 0.4        # 43 前向 0.4 m/s——训练范围 [-0.3, 0.6] 内，分布内测试
    vy = 0.0        # 44 不横移
    dyaw = 0.0      # 45 不转向
```

训练端 `env.commands` 每隔 `resampling_time` 秒在区间内随机重采样；部署端固定为常量。

### 5.3 `quaternion_to_euler_array()`：手写四元数转欧拉

```python
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat                                  # 50 入参已是 [x,y,z,w]（重排在 get_obs 完成）

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)                        # 53
    t1 = +1.0 - 2.0 * (x * x + y * y)                  # 54
    roll_x = np.arctan2(t0, t1)                        # 55

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)                        # 58
    t2 = np.clip(t2, -1.0, 1.0)                        # 59 arcsin 入参可能因浮点误差略超 ±1，clip 防 NaN
    pitch_y = np.arcsin(t2)                            # 60

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)                        # 63
    t4 = +1.0 - 2.0 * (y * y + z * z)                  # 64
    yaw_z = np.arctan2(t3, t4)                         # 65

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])          # 68
```

三个公式是 ZYX 欧拉角的标准解析解，与 Isaac Gym `torch_utils.get_euler_xyz()` 逐项一致；不用 SciPy 的 `as_euler()`，正是为了避免库间角度范围、万向锁约定的细微差异。主循环 126 行那句 `> π` 回折，同样是从训练端 `get_euler_xyz_tensor()`（legged_robot.py:54）逐字抄来的防御行。

### 5.4 `get_obs()`：从 MuJoCo 抽取一帧原始状态

```python
def get_obs(data):                                                          # 70
    '''Extracts an observation from the mujoco data structure'''
    q = data.qpos.astype(np.double)                                         # 73 全部 19 维广义坐标 = 7(基座) + 12(关节)，截取在主循环做
    dq = data.qvel.astype(np.double)                                        # 74 全部 18 维 = 3 线速 + 3 角速 + 12 关节速度
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)  # 75 framequat 传感器 [w,x,y,z] → SciPy 约定 [x,y,z,w]
    r = R.from_quat(quat)                                                   # 76 包装成旋转对象
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)              # 77 世界系线速度 → 基座系（乘 Rᵀ）※算了不用
    omega = data.sensor('angular-velocity').data.astype(np.double)          # 78 gyro 传感器：基座系角速度
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double) # 79 世界重力方向 → 基座系（projected gravity）※算了不用
    return (q, dq, quat, v, omega, gvec)                                    # 80
```

两点补充：

- `v`、`gvec` 计算了但从未进入观测——非对称 Actor-Critic 的直接证据：这两项只喂给 Critic 的 73 维特权观测，Actor 看不到（§9.2）。
- 两个传感器在 MJCF 里自带噪声（`framequat noise=0.001`、`gyro noise=0.005 cutoff=34.9`），且模型顶部 `<flag sensornoise="enable"/>` 已启用——这是部署端仅存的噪声来源，远弱于训练端的高斯观测噪声（§9.1）。

### 5.5 `pd_control()`：一行 PD 公式

```python
def pd_control(target_q, q, kp, target_dq, dq, kd):        # 82
    '''Calculates torques from position commands'''
    return (target_q - q) * kp + (target_dq - dq) * kd     # 85 τ = Kp(q_des−q) + Kd(q̇_des−q̇)，逐元素作用于 12 个关节
```

调用方恒传 `target_dq = 0`，微分项退化为纯阻尼 $-K_d\dot q$。训练端 `_compute_torques()` 的 `p_gains*(actions_scaled + default_dof_pos - dof_pos) - d_gains*dof_vel` 是同一个公式，只是把 `action_scale` 与 `default_dof_pos` 内联在力矩函数里，部署端拆到了 150 行的 `target_q`（§7.3）。

---

## 6. 逐行走读：`run_mujoco()` 初始化段（98–111 行）

```python
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)  # 98 解析 MJCF
    model.opt.timestep = cfg.sim_config.dt                                  # 99 用 Python 覆盖模型自带步长 = 0.001（1000 Hz）
    data = mujoco.MjData(model)                                             # 100 分配运行时状态缓冲（qpos/qvel/sensor/ctrl）
    mujoco.mj_step(model, data)                                             # 101 先步进一次：填充传感器缓冲，避免首帧读到全零
    viewer = mujoco_viewer.MujocoViewer(model, data)                        # 102 被动渲染窗口，只显示不驱动仿真

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)             # 104 PD 目标位置，初始 0
    action = np.zeros((cfg.env.num_actions), dtype=np.double)               # 105 上一次动作，初始 0 → 首轮 obs[29:41] = 0

    hist_obs = deque()                                                      # 107
    for _ in range(cfg.env.frame_stack):                                    # 108 15 帧
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))  # 109 全零初始化 = "历史不足时补零"

    count_lowlevel = 0                                                      # 111 物理步计数器：100 Hz 分频开关 + 相位时钟
```

训练端 `obs_history` 同样从全零开始，"首帧填充方式"两端一致——首轮推理实际输入是 14 个零帧 + 1 个真帧。

---

## 7. 逐行走读：主循环（114–162 行）

### 7.1 循环头与状态读取：1000 Hz 的节拍

```python
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):  # 114 循环 60.0/0.001 = 60000 个物理步

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)   # 117 每个物理步读取最新状态（v、gvec 本循环不用）
        q = q[-cfg.env.num_actions:]                  # 118 截取最后 12 维 = 受控关节（左腿 6 → 右腿 6）
        dq = dq[-cfg.env.num_actions:]                # 119
```

当前 MJCF 的 12 关节顺序：

```text
0   left_leg_roll_joint       6   right_leg_roll_joint
1   left_leg_yaw_joint        7   right_leg_yaw_joint
2   left_leg_pitch_joint      8   right_leg_pitch_joint
3   left_knee_joint           9   right_knee_joint
4   left_ankle_pitch_joint    10  right_ankle_pitch_joint
5   left_ankle_roll_joint     11  right_ankle_roll_joint
```

与 Isaac Gym 侧 URDF 的 DOF 顺序一致，因此动作索引可直接对应。

> [!warning] 隐含假设
> `q[-12:]` 假设受控关节始终位于状态向量最后，而且顺序永远不变。增加腰部、手臂或其他关节后可能失效。更稳健的实现应根据 MuJoCo joint name 查询地址（Lab 的部署端正是这么做的，见对比笔记 §3.2）。

### 7.2 100 Hz 分支：47 维观测逐行构造

```python
        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:                  # 122 分频开关：物理步每 10 步进入一次

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)   # 124 预分配 [1,47]，自带 batch 维
            eu_ang = quaternion_to_euler_array(quat)                        # 125 四元数 → [roll, pitch, yaw]
            eu_ang[eu_ang > math.pi] -= 2 * math.pi                         # 126 >π 回折，照抄训练端 get_euler_xyz_tensor 的防御行

            obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)  # 128 相位 sin，周期 0.64 s 硬编码
            obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt  / 0.64)  # 129 相位 cos
            obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel       # 130 前向命令 ×2.0 → 0.8
            obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel       # 131 侧向命令 ×2.0
            obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel     # 132 偏航命令 ×1.0
            obs[0, 5:17] = q * cfg.normalization.obs_scales.dof_pos         # 133 关节位置 ×1.0（未减默认角，见下）
            obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel       # 134 关节速度 ×0.05
            obs[0, 29:41] = action                                          # 135 上一周期 clip 后的原始动作（未乘 scale）
            obs[0, 41:44] = omega                                           # 136 基座角速度（scale 恰为 1，省略乘法）
            obs[0, 44:47] = eu_ang                                          # 137 基座欧拉角（scale 恰为 1，省略乘法）
```

逐索引对照训练端 `compute_observations()`（`humanoid_env.py:200-244`，走读见 §9）：

| 索引 | 内容 | 部署端 | 训练端 |
|---|---|---|---|
| `0:2` | 步态相位 sin/cos | 物理时间 / 0.64 | `episode_length_buf × 策略周期 / 0.64` |
| `2:5` | 速度命令 | `×[2.0, 2.0, 1.0]` | `commands_scale = [2.0, 2.0, 1.0]` |
| `5:17` | 关节位置 | `q × 1.0` | `(dof_pos − default) × 1.0` |
| `17:29` | 关节速度 | `dq × 0.05` | `dof_vel × 0.05` |
| `29:41` | 上一动作 | clip 后原始 action | `self.actions`（同为原始 action） |
| `41:44` | 角速度 | 不乘（×1.0 等价） | `base_ang_vel × 1.0` |
| `44:47` | 欧拉角 | 不乘（×1.0 等价） | `base_euler_xyz × 1.0` |

三个跨行的坑：

- **相位时钟不同源但等价**：部署端用物理时间 `count_lowlevel × 0.001`，训练端用策略步数 × 策略周期；在 100 Hz 采样点上数值相同。但 0.64 在部署端是**硬编码字面量**，训练端来自 `cfg.rewards.cycle_time`——改训练配置不会自动同步。
- **`q` 没有减默认角**：当前 `default_joint_angles` 12 项全 0（`humanoid_config.py:103-115`），两种写法等价；默认角一旦改为非零站立姿态，这里与 §7.3 的 `target_q` 要一起改。
- **`ang_vel`/`quat` 不乘是"恰好"而非"本来"**：`XBotLCfg.normalization` 把基类的 `ang_vel=0.25` 覆盖为 1.0，并新增 `quat=1.0`（`humanoid_config.py:218-227`）——照抄基类数值就会出错，"不乘"只对最终生效配置成立。

### 7.3 裁剪、历史堆叠与推理：从 47 维到 12 维动作

```python
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)  # 139 整帧夹 ±18：正常行走是无操作，传感器异常时兜底

            hist_obs.append(obs)                                            # 141 压入新帧
            hist_obs.popleft()                                              # 142 弹出最旧帧 → 队列恒 15 帧，索引 0 最旧

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)  # 144 [1,705]
            for i in range(cfg.env.frame_stack):                            # 145 逐帧铺开
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]  # 146 第 i 帧 → [i×47,(i+1)×47)，时间从旧到新
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()  # 147 推理：float32 张量 → TorchScript（确定性均值，无采样）→ [1,12] 取 [0] → detach → numpy；action[:] 原地写保住引用链
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)  # 148 动作夹 ±18；注意 = 重新绑定新数组

            target_q = action * cfg.control.action_scale                    # 150 目标位置 = action × 0.25（默认角全 0，故无 +q_default）
```

展开顺序若弄反（新→旧），网络仍能运行但语义完全错误——调试清单必查项。动作 clip ±18 后乘 0.25，目标位置理论范围 ±4.5 rad，代码**没有**再按关节限位裁剪目标。

> [!warning] 默认姿态假设
> 如果以后把 `default_joint_angles` 改成非零站立姿态，第 150 行要写成 `action × 0.25 + q_default`；同时 §7.2 表中 `5:17` 的观测也应改为 `(q - q_default) × scale`。两处必须一起改。

### 7.4 PD 与物理步进：1000 Hz 的执行

```python
        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)        # 153 目标速度恒 0 → 微分项 = 纯阻尼
        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,                 # 155 每个物理步用最新 q/dq 重算力矩
                        target_dq, dq, cfg.robot_config.kds)                # 156 两次推理间 target_q 不变（Zero-Order Hold）
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # 157 夹 ±200（训练端实为 42.5/68，见 §12.1）
        data.ctrl = tau                                                     # 158 motor 型 actuator（gear=1）：ctrl 值即力矩

        mujoco.mj_step(model, data)                                         # 160 物理前推 1 ms
        viewer.render()                                                     # 161 每物理步渲染（性能问题见 §12.5）
        count_lowlevel += 1                                                 # 162
```

这一段在 `if` 之外、每个物理步都执行。MJCF 里 12 个 motor 统一 `ctrlrange="-200 200"`，与脚本 clip 一致；但与训练端 `effort × 0.85` 明显不一致（§12.1）。

---

## 8. 逐行走读：入口与 `Sim2simCfg`（167–193 行）

```python
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,            # 171 TorchScript 路径（必填）
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')  # 173 平地/地形 MJCF 开关
    args = parser.parse_args()

    class Sim2simCfg(XBotLCfg):                                             # 176 继承训练配置：normalization/env/control 全部沿用

        class sim_config:                                                   # 178 只新增部署专属参数
            if args.terrain:                                                # 179
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-terrain.xml'   # 180
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L.xml'           # 182
            sim_duration = 60.0                                             # 183
            dt = 0.001                                                      # 184 1000 Hz
            decimation = 10                                                 # 185 ÷10 → 100 Hz 策略

        class robot_config:                                                 # 187
            kps = np.array([200, 200, 350, 350, 15, 15, 200, 200, 350, 350, 15, 15], dtype=np.double)  # 188 hip roll/yaw 200、hip pitch/knee 350、ankle 15
            kds = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.double)           # 189 全 10
            tau_limit = 200. * np.ones(12, dtype=np.double)                 # 190 统一 200（与训练端不一致，§12.1）

    policy = torch.jit.load(args.load_model)                                # 192 反序列化 TorchScript（不含 PPO/Critic/优化器）
    run_mujoco(policy, Sim2simCfg())                                        # 193
```

两端频率一致的依据链：训练端 `XBotLCfg.sim.dt = 0.001` 覆盖了基类的 0.005（`humanoid_config.py:130-131`），`control.decimation = 10`（`humanoid_config.py:128`），同样得到 1000 Hz / 100 Hz。Kp/Kd 数值与训练端 `stiffness/damping` 字典（`humanoid_config.py:119-123`）一致。`Sim2simCfg` 只新增两个部署子类、其余全部继承——"单一配置源"设计的精髓所在。

---

## 9. 训练端对照源码：`compute_observations()`

部署端 47 维观测的"真值定义"在训练环境里（`humanoid_env.py:200-260`）：

```python
    def compute_observations(self):

        phase = self._get_phase()                                  # 202 phase = episode_length_buf × self.dt / cycle_time；self.dt = 0.01 s 是策略周期
        ...
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)     # 205
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)     # 206
        ...
        self.command_input = torch.cat(                            # 211 5 维命令块
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)  # 212 commands_scale = [2.0, 2.0, 1.0]

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos   # 214 ★显式减默认角再乘 scale
        dq = self.dof_vel * self.obs_scales.dof_vel                # 215 ×0.05

        ...
        obs_buf = torch.cat((                                     # 237 47 维策略观测——顺序即部署端 obs[0,:] 的顺序
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,    # 12D
            dq,  # 12D
            self.actions,   # 12D                              # 241 上一步原始动作（未乘 scale）
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3   # 242 ×1.0
            self.base_euler_xyz * self.obs_scales.quat,  # 3    # 243 ×1.0
        ), dim=-1)
```

### 9.1 噪声与历史堆叠（训练端）

```python
        if self.add_noise:                                                            # 250 训练默认 add_noise=True
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level  # 251 高斯噪声，noise_level=0.6
        else:
            obs_now = obs_buf.clone()                                                 # 253
        self.obs_history.append(obs_now)                                              # 254 压入 15 帧历史
        ...
        obs_buf_all = torch.stack([self.obs_history[i]                                # 258
                                   for i in range(self.obs_history.maxlen)], dim=1)   # 259
        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K              # 260 展开成 [N, 15×47]，时间维从旧到新
```

部署端完全没有复现噪声——策略在 MuJoCo 中看到的是**干净观测**，相当于比训练时更容易的输入条件。历史堆叠逻辑与部署端等价：maxlen=15 的 deque，stack 后同样按时间维展开。

### 9.2 为什么没有基座线速度与重力方向

`compute_observations()` 同时维护 73 维 `privileged_obs_buf`（humanoid_env.py:219-235），其中包含 `base_lin_vel`、`rand_push_force`、`env_frictions`、`body_mass` 等仿真特权信息。训练采用**非对称 Actor-Critic**：

```text
Actor：只使用部署时容易获得的信息（47 维）
Critic：可使用基座线速度、外力、摩擦、质量等特权信息（73 维 × 3 帧）
```

部署时 Critic 被丢弃，所以 MuJoCo 端既不需要、也不应该复现特权观测——这解释了 `get_obs()` 里 `v` 和 `gvec` 算了不用（§5.4）。

### 9.3 两端 scale 数值速查

| 量 | 训练端数值 | 部署端行为 | 是否一致 |
|---|---:|---|---|
| `lin_vel` | 2.0 | `obs[2:4] = cmd × 2.0` | ✅ |
| `ang_vel` | 1.0（覆盖基类 0.25） | 不乘（×1.0 等价） | ✅ |
| `dof_pos` | 1.0 | `q × 1.0` | ✅ |
| `dof_vel` | 0.05 | `dq × 0.05` | ✅ |
| `quat` | 1.0（新增项） | 不乘（×1.0 等价） | ✅ |
| `clip_observations` | 18.0 | `np.clip(obs, ±18)` | ✅ |
| `clip_actions` | 18.0 | `np.clip(action, ±18)` | ✅ |
| `action_scale` | 0.25 | `target_q = action × 0.25` | ✅ |

一致性完全依赖"`sim2sim.py` 与训练代码 import 同一个 `XBotLCfg`"这一事实；若部署脚本被复制到别的仓库独立维护，上表就会开始漂移（Lab 用 ONNX metadata 解决此问题）。

---

## 10. `play.py` 与策略导出

导出发生在 `play.py`（play.py:80-84）：

```python
    if EXPORT_POLICY:                                                          # 81 入口处 EXPORT_POLICY = True
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')  # 82
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)               # 83 只导出 actor_critic.actor
```

`export_policy_as_jit()` 的实现（helpers.py:248-253）：

```python
def export_policy_as_jit(actor_critic, path):
    os.makedirs(path, exist_ok=True)                             # 249
    path = os.path.join(path, "policy_1.pt")                     # 250 固定文件名
    model = copy.deepcopy(actor_critic.actor).to("cpu")          # 251 深拷贝 Actor 子模块并搬到 CPU（不动 GPU 上的训练模型）
    traced_script_module = torch.jit.script(model)               # 252 script 编译：按源码保留控制流（非 trace）
    traced_script_module.save(path)                              # 253
```

导出物只含前向图和权重，不含归一化参数或元数据——所有部署契约（scale、顺序、默认角）都留在 Python 配置里，这正是 Gym 与 Lab 部署方案的分水岭。

---

## 11. Isaac Gym 与 MuJoCo 对照

| 项目 | Isaac Gym 训练端 | MuJoCo Sim-to-Sim 端 | 是否一致 |
|---|---|---|---|
| 机器人描述 | URDF | MJCF/XML | 表达格式不同 |
| Actor | 705→512→256→128→12 | 同一 TorchScript Actor | ✅ |
| 策略频率 | 100 Hz（sim dt 0.001 × decimation 10） | 100 Hz（dt 0.001 ÷ 10） | ✅ |
| 物理/PD频率 | 1000 Hz | 1000 Hz | ✅ |
| 单帧观测 | 47 维 | 47 维 | ✅ |
| 历史帧 | 15 帧 | 15 帧 | ✅ |
| Actor输入 | 705 维 | 705 维 | ✅ |
| 动作维度 | 12 | 12 | ✅ |
| `action_scale` | 0.25 | 0.25 | ✅ |
| 默认关节角 | 目前全 0 | 直接假设为 0 | 当前一致 |
| PD增益 | 按关节名匹配 | 手写数组 | 当前基本一致 |
| 力矩限制 | URDF effort × 0.85（42.5/68） | 全部 ±200 N·m | ❌ 明显偏大 |
| 观测噪声 | 显式高斯噪声 | 仅 MJCF sensor noise | ⚠️ 不完全一致 |
| 动作延迟/噪声 | 训练时加入 | 未加入 | 有意省略 |
| 摩擦/质量随机化 | 训练时开启 | 固定 MJCF 参数 | 有意改变 |
| 摔倒重置 | 有 | 无 | ❌ |

---

## 12. 当前实现值得注意的问题

### 12.1 力矩限制不一致（量化）

训练端 `_compute_torques()` 末尾 `torch.clip(torques, -self.torque_limits, self.torque_limits)`，其中 `torque_limits = URDF effort × 0.85`（legged_robot.py:293,356）。URDF 实际声明（XBot-L.urdf）：

| 关节 | URDF effort | ×0.85 后训练端上限 |
|---|---:|---:|
| hip roll / hip yaw / ankle pitch / ankle roll | 50 N·m | **42.5 N·m** |
| hip pitch / knee | 80 N·m | **68 N·m** |

MuJoCo 端统一 `tau_limit = 200`，且 MJCF motor `ctrlrange="-200 200"`（sim2sim.py:190）。也就是说部署端可用力矩是训练端的 3～5 倍。正常行走时 PD 输出达不到上限、差异不显现；在大扰动或即将摔倒时，MuJoCo 中的机器人会爆发出训练端从未见过的力矩，**高估**策略的真实救场能力。逐关节对齐训练端 `torque_limits` 才是严格做法。

### 12.2 关节索引依赖模型顺序

`q = q[-12:]` 简洁但脆弱，扩展模型（加腰、臂）即失效，推荐按 joint name 建立显式映射（§7.1）。

### 12.3 归一化参数隐式依赖导入链

当前一致性靠 `import XBotLCfg` 成立（§9.3）。`ang_vel`/`quat` 恰为 1 才允许"不乘"的写法；配置一改，脚本不会报错，只会静默偏移。

### 12.4 没有摔倒检测与重置

固定 60 秒，第 5 秒摔倒也会跑到结束。适合人工观察，不适合批量成功率评测。

### 12.5 每个物理步都渲染

`viewer.render()` 位于 1000 Hz 循环内（sim2sim.py:161）。显示器远低于 1000 FPS，60 秒模拟需要远超 60 秒墙钟时间。可改成每 10～20 物理步渲染一次（Lab 端 `era_robot_node.py` 正是按 20 ms 间隔 `viewer.sync()` 的）。

### 12.6 命令与步态周期写死

`cmd` 三行类属性（sim2sim.py:42-45）与相位周期字面量 `0.64`（sim2sim.py:128-129）都无法从命令行覆盖；后者与 `cfg.rewards.cycle_time` 重复定义，改训练配置不会同步。

---

## 13. 为什么 Sim-to-Sim 对 Sim-to-Real 有帮助

如果策略只在训练仿真器里有效，换一个物理引擎就摔倒，通常说明策略过度依赖：

- 特定接触求解器；
- 精确惯量和质量；
- 理想关节模型；
- 特定摩擦参数；
- 仿真器数值误差。

MuJoCo 与 Isaac Gym 在以下方面存在差异：

```text
接触求解方式
关节阻尼与摩擦
碰撞几何
惯量表达
传感器模型
数值积分与约束求解
```

因此，跨仿真器稳定运行可以作为一种便宜的鲁棒性筛查。

但 Sim-to-Sim 不能完全替代 Sim-to-Real。真实机器人还会引入：

- 通信和计算延迟；
- 电机带宽、饱和与温升；
- 减速器回差和非线性摩擦；
- IMU 漂移和状态估计误差；
- 结构柔性；
- 电池电压变化；
- 地面材料和足底形变。

---

## 14. 调试检查清单

当策略在 Isaac Gym 正常、但在 MuJoCo 中立即摔倒时，建议按以下顺序排查：

- [ ] Actor 输入是否确实为 `[1, 705]`？
- [ ] 47 维单帧观测的字段顺序是否完全一致（§7.2 表）？
- [ ] 15 帧历史是从旧到新排列，还是顺序反了？
- [ ] 关节顺序是否与 Actor 输出顺序一致？
- [ ] 四元数是否正确从 `[w,x,y,z]` 转为 `[x,y,z,w]`？
- [ ] 关节位置是否减去了训练时的默认姿态？
- [ ] `dof_pos`、`dof_vel`、命令、角速度是否使用相同 scale（1.0 / 0.05 / 2.0 / 1.0）？
- [ ] Actor 动作是否乘了相同的 `action_scale=0.25`？
- [ ] 策略频率是否为 100 Hz、PD 是否为 1000 Hz？
- [ ] $K_p$、$K_d$ 是否与训练端逐关节一致（200/350/15 与 10）？
- [ ] 力矩符号、关节轴方向和左右腿映射是否一致？
- [ ] MuJoCo 与 Isaac Gym 的初始姿态和基座高度是否一致？
- [ ] 力矩限制是否过大或过小（±200 vs 42.5/68）？
- [ ] 是否因为 viewer 渲染导致误以为仿真卡死？

---

## 15. 核心结论

Sim-to-Sim 能否成功，主要取决于三个层面的一致性：

```text
观测一致性
  ├── 字段顺序
  ├── 坐标系
  ├── 归一化
  └── 历史帧

动作一致性
  ├── 关节顺序
  ├── action_scale
  ├── 默认姿态
  └── 动作裁剪

控制一致性
  ├── 策略频率
  ├── PD频率
  ├── Kp/Kd
  └── 力矩限制
```

物理引擎可以不同，但 Actor 所理解的"输入含义"和"输出含义"不能不同。换句话说：

> **Sim-to-Sim 的本质不是搬运网络，而是在另一个仿真器中重建训练时的策略接口和控制语义。**

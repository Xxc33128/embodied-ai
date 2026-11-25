# Isaac 仿真迁移到 MuJoCo 并使用昇腾 NPU 训练：讨论总结

## 1. 目标与约束

当前目标不是追求最高训练速度，而是：

1. 在必须使用昇腾 NPU 的条件下，先跑通完整强化学习训练流程；
2. 将现有 Isaac 仿真环境迁移到 MuJoCo；
3. 尽量减小 Isaac–MuJoCo、MuJoCo–Real 之间的差异；
4. 最终让 MuJoCo 中训练的策略能够部署到真实机器人；
5. 参考其他项目的公开实践，而不只依赖本项目自身经验。

推荐的基础架构是：

```text
经典 MuJoCo（CPU，物理仿真与采样）
                  ↓
昇腾 NPU（策略网络前向、反向传播和参数更新）
                  ↓
经典 MuJoCo（加载更新后的策略继续采样）
```

这条路线不要求 MuJoCo 本身运行在昇腾 NPU 上。仿真器与神经网络训练可以解耦：CPU 负责物理仿真，NPU 负责张量计算。

## 2. “速度较慢”与“数值精度较低”不是一回事

此前所说的“经典 MuJoCo 较慢”，准确含义是：

> 经典 MuJoCo 在同时运行数千个环境时，总采样吞吐量通常低于 GPU 上的 MJX-JAX 或 MJX-Warp；它并不一定在单个环境、单个仿真步上更慢。

经典 MuJoCo 默认使用 `double/float64`，而 GPU 后端通常使用 `float32`：

| 数据类型 | 有效数字 | 一般特点 |
|---|---:|---|
| `float64` | 约 15～16 位 | 舍入误差更小，内存和计算开销更大 |
| `float32` | 约 7 位 | 舍入误差更大，更适合 GPU/NPU 大规模并行计算 |

运行速度主要由计算架构和并行规模决定，而不是简单由数值精度决定：

- 经典 MuJoCo：CPU 上顺序或少量并行地运行环境，单环境延迟通常很好；
- MJX-JAX/MJX-Warp：把成百上千个环境组成 batch，交给 GPU 并行处理；
- 单个环境时，GPU 编译、调度和数据组织的开销可能使 MJX 反而更慢；
- 数千个环境时，GPU 的总环境步数吞吐量通常更高。

因此，“MJX 更快”通常指训练期间的总 `environment steps/s`，不是单个机器人每一步一定算得更快。

另外，`float32` 的数值精度较低，不等于物理真实性必然较低。Sim-to-Real 中更主要的误差通常来自：

- 电机、减速器和执行器模型；
- 控制、通信和传感延迟；
- 质量、惯量和质心误差；
- 关节摩擦、阻尼、死区和回差；
- 足底材料、地面摩擦和接触模型；
- 结构柔性；
- 传感器噪声和状态估计误差。

在“经典 MuJoCo CPU + 昇腾 NPU”方案中，MuJoCo 物理计算仍然使用经典 MuJoCo 的数值实现；NPU 上的 `float32` 主要影响神经网络计算，不会把 MuJoCo 的物理求解自动变成 `float32`。

## 3. 经典 MuJoCo、MJX-JAX 和 MJX-Warp 的差别

这三者共享 MuJoCo 的模型概念和大部分物理语义，但不是完全相同的执行实现。

| 实现 | 主要硬件 | 主要用途 | 特征 |
|---|---|---|---|
| 经典 MuJoCo | CPU | 单个或少量环境、验证、控制、部署测试 | 官方 C/C++ 基准实现，默认 `float64`，功能最完整 |
| MJX-JAX | GPU/TPU | JAX 强化学习、大批量环境、可微计算 | 用 JAX/XLA 重新实现相关计算，通常使用 `float32` |
| MJX-Warp | NVIDIA GPU | 大规模并行物理仿真 | 使用 Warp/CUDA 实现，通常使用 `float32`，强调批量吞吐量 |

当前官方文档通常将统一入口称为 MJX，并区分 MJX-JAX 和 MJX-Warp 后端。早期材料中的 MuJoCo Warp、MJWarp 等名称，通常指 Warp 路线，但具体名称和支持范围会随版本变化。

三者产生差异的主要位置包括：

- 碰撞检测算法；
- 接触点的数量、顺序和容量上限；
- 约束与摩擦求解；
- 求解器迭代次数和收敛条件；
- `float64` 与 `float32`；
- GPU 所需的固定形状和预分配数组；
- 不同版本支持的关节、执行器、传动和传感器功能。

所以，即使模型、初始状态和 action 相同，三种实现也不保证长时间轨迹逐位一致。对落脚、碰撞、滑动和跌倒等接触密集行为，初期的微小数值差异还会随时间快速放大。

## 4. 是否可以直接照搬 MuJoCo Playground 的物理参数

可以将 MuJoCo Playground 的模型和参数作为高质量起点，但不能把它们当成无需验证的最终配置。

### 4.1 通常可以优先复用的内容

- link 几何尺寸；
- 质量、惯量和质心；
- 关节轴、关节限位和初始姿态；
- nominal joint position；
- actuator gear；
- 关节阻尼和 armature；
- 摩擦系数的标称值；
- `kp`、`kd`；
- action scale；
- 控制周期、仿真步长和 decimation。

这些参数可以直接迁移为第一版基线，但仍需确认模型字段在目标 MuJoCo 版本中的语义一致。

### 4.2 不能盲目照搬、必须重新验证的内容

- `solver`、`iterations`、`ls_iterations`；
- `integrator`；
- `solref`、`solimp`；
- friction cone；
- collision pair、碰撞 mesh 和接触过滤；
- 接触点与约束容量；
- position、torque 或自定义 PD actuator 的具体实现；
- policy action 到目标位置或力矩的映射；
- reset、termination 和 command generation；
- reward 各项及其权重；
- domain randomization；
- 控制延迟、传感器噪声与状态估计。

尤其需要注意：训练环境不只存在于 MJCF/XML 中。observation、reward、reset、随机化、动作解释和控制逻辑通常位于 Python 代码中。只复制机器人 XML，不等于复制了完整训练环境。

Playground 为提高 GPU 批量训练速度，还可能采用较少的求解器迭代次数或其他性能配置。把这些配置原样放入经典 MuJoCo，不一定是最合适的精度与速度平衡。

## 5. MuJoCo–Real gap 应如何理解

“MuJoCo 已经有 G1 等机器人 zero-shot 上真机案例”说明：

> 使用 MuJoCo 建模、合理的控制接口和充分的 domain randomization，可以实现无需真机微调或只需很少调整的策略迁移。

但它不能推出：

> 任意 MuJoCo 模型训练出的策略，都可以不经验证直接部署并得到良好结果。

MuJoCo–Real gap 不一定天然大于 Isaac–Real gap，也不一定天然更小。最终差距主要由机器人模型、执行器模型、控制接口、随机化范围、真实机器标定和部署一致性决定，而不是只由仿真器名称决定。

要实现较可靠的 Sim-to-Real，至少应保证：

- 训练和部署使用完全一致的 observation 顺序、单位和归一化；
- action 的含义、缩放、限幅和控制频率一致；
- 训练时包含真实可能出现的延迟、噪声和参数偏差；
- 真机的关节方向、零位、gear ratio、`kp/kd` 与仿真匹配；
- 上机前经过安全约束、离线回放和分阶段验证。

## 6. 为什么经典 MuJoCo 与 MJX/MJWarp 不保证完全等同

这里的“不完全等同”需要分层理解：

| 比较层次 | 一致性预期 |
|---|---|
| MJCF/XML 参数及主要物理语义 | 大部分一致 |
| 相同状态和 action 下的单步结果 | 通常接近，但不保证逐位相同 |
| 长时间 rollout 轨迹 | 不保证一致，接触密集任务尤其容易分离 |

### 6.1 它们不是同一份经典 MuJoCo 代码换了硬件

经典 MuJoCo 的核心是 CPU 上的 C/C++ 实现；MJX-JAX 是基于 JAX/XLA 的重新实现，MJX-Warp 则是基于 Warp/CUDA 的重新实现。它们共享模型定义和主要算法目标，但为了适配加速器和大规模并行，会采用不同的数据结构和执行方式。

所以它们的关系更接近：

```text
同一套模型概念和物理目标
          ↓
分别由 C/C++、JAX、Warp 实现
          ↓
追求行为一致，但不承诺逐位一致
```

### 6.2 浮点表示不同

经典 MuJoCo 默认使用 `float64`，MJWarp 等加速器后端通常使用 `float32`。单次舍入误差很小，但一个仿真步包含质量矩阵、约束、摩擦、加速度和积分等大量计算，长时间迭代后误差会累积，也可能改变临界接触判断。

这不代表 `float32` 的物理真实性一定更差，只表示它与经典 MuJoCo 的数值结果不可能天然逐位一致。

### 6.3 碰撞检测可能不同

不同实现可能采用不同的凸碰撞算法。例如 MJX-JAX 对凸 mesh 的处理与经典 MuJoCo 所使用的算法并不完全相同，因此可能产生不同的：

- 首次接触时刻；
- 接触点数量和顺序；
- 接触位置与法向；
- 穿透深度；
- 地面反作用力。

人形行走对足底接触非常敏感。即使只相差一个仿真步，也可能改变机身姿态和下一次策略输出。

### 6.4 求解器实现存在细节差异

即使两边都配置 `Newton` 和相同的迭代次数，内部计算也不保证完全相同。官方文档列出的差异包括：

- warm-start 初始值选择；
- 惯性矩阵分解方法；
- 稠密/稀疏矩阵选择阈值；
- 接触和约束容量；
- 部分求解器、积分器和功能的支持范围。

因此，相同的 XML solver 配置不等于相同的底层数值执行路径。

### 6.5 GPU 并行归约可能不确定

浮点加法不满足严格结合律，GPU 并行线程不同的计算和归约顺序可能产生微小差异。MJWarp 官方文档说明，GPU 原子操作可能导致同一代码不同执行之间存在顺序或细小数值差异。

### 6.6 为什么小差异会变成明显轨迹差异

无接触的短时间关节运动通常会非常接近；但人形机器人是接触丰富且具有敏感性的闭环系统：

```text
落脚时刻出现微小差异
        ↓
接触力发生变化
        ↓
机身姿态和观测发生变化
        ↓
策略输出不同的 action
        ↓
后续轨迹进一步分离
```

长期轨迹不同不一定说明某个实现错误。迁移验收应强调短时响应、接触模式和行为统计一致，而不是要求长时间轨迹逐点重合。

## 7. 是否能用经典 MuJoCo 配置出 Playground 的效果

可以。Playground 本身是在 MuJoCo/MJX 之上构建的开源机器人学习环境，不是另一个不可配置的专用物理引擎。经典 MuJoCo 具备复现其机器人模型、场景、控制和任务定义的能力。

不过，“Playground 的环境”由多个层次组成，并不只是一个 MJCF/XML 文件。

### 7.1 机器人与物理场景

MJCF/XML 主要定义：

- link 几何、质量、惯量和质心；
- 关节轴、限位和初始姿态；
- 碰撞体和地面；
- 摩擦与接触参数；
- actuator；
- timestep、integrator 和 solver。

这部分可以在经典 MuJoCo 中加载和调整。

### 7.2 任务和控制环境

Playground 的大量关键行为位于 Python 代码而不是 XML 中，包括：

- observation 定义、顺序与归一化；
- action 的含义、缩放与限幅；
- policy action 到关节目标或力矩的映射；
- PD 控制器和 decimation；
- reward；
- reset、termination 和 command generator；
- domain randomization。

只复制 G1 的 XML，不能复现完整的 Playground 训练效果。

### 7.3 强化学习配置

还需要对齐：

- PPO 或其他算法的实现；
- 网络结构；
- rollout length 和 batch size；
- learning rate、discount、GAE；
- entropy coefficient；
- observation normalization；
- curriculum。

### 7.4 从 Playground/MJX 到经典 MuJoCo + 昇腾的映射

| Playground/MJX | 经典 MuJoCo + 昇腾 NPU |
|---|---|
| `mjx.step` | `mujoco.mj_step` |
| JAX device array | NumPy、`MjModel`、`MjData` |
| `jax.vmap` 批量环境 | CPU 多线程或多进程 rollout worker |
| JAX/Brax PPO | `torch_npu`、MindSpore 或其他昇腾适配实现 |
| GPU 同时进行仿真和学习 | CPU 仿真，NPU 网络训练 |
| JAX 函数式状态 | 经典 MuJoCo 有状态的 `MjData` |

推荐保留 Playground 的模型、任务定义和超参数作为基线，把仿真调用改成经典 MuJoCo，并把学习器迁移到昇腾支持的框架。

### 7.5 可以达到和不能保证达到的效果

| 目标 | 预期 |
|---|---|
| 相同机器人外形、质量、关节和碰撞体 | 可以复现 |
| 相似的站立、行走和控制行为 | 可以复现 |
| 相似的 reward、任务与 Sim-to-Real 方法 | 可以复现 |
| 完全相同的训练曲线和 checkpoint | 不能保证 |
| 完全相同的逐步轨迹 | 不能保证 |
| 与 MJX 相同的 GPU 批量吞吐量 | 经典 MuJoCo 通常不能达到 |

合理的复现验收标准应包括：

- 相同初始姿态下能够稳定站立；
- 单关节 step response 接近；
- action、observation 的单位、顺序和范围完全一致；
- reward 各项数量级合理；
- 足底接触模式合理；
- 训练能够稳定收敛；
- 行走速度、跟踪误差、滑移和跌倒率接近；
- 策略能够承受延迟、噪声和模型参数扰动。

结论是：经典 MuJoCo 可以复现 Playground 的物理行为、任务设计和 Sim-to-Real 方法，主要损失是 MJX 的大规模 GPU 并行吞吐量，而不是环境配置能力。

## 8. 建议的迁移与验证顺序

### 阶段一：建立经典 MuJoCo 基线

1. 导入或转换机器人模型；
2. 对齐关节名称、顺序、方向、限位和零位；
3. 对齐质量、惯量、碰撞体、执行器和控制周期；
4. 完成站立、自由落体、单关节控制等基础测试。

### 阶段二：迁移强化学习环境

依次迁移并单独测试：

1. observation；
2. action 与低层控制器；
3. reward；
4. reset 和 termination；
5. command generator；
6. domain randomization。

不要在所有模块尚未分别验证时，直接用最终训练曲线判断迁移是否正确。

### 阶段三：接入昇腾 NPU

1. MuJoCo CPU worker 负责 rollout；
2. 将轨迹放入共享内存、队列或 rollout buffer；
3. NPU learner 执行策略网络前向、loss、反向传播和优化；
4. 定期将新权重同步回 CPU worker；
5. 首先以少量环境跑通正确性，再增加 worker 数量。

由于目标是“先跑通”，第一版可以接受采样较慢，不必一开始就重写 MuJoCo 物理后端或追求 GPU/MJX 级吞吐量。

### 阶段四：做一致性测试

建议保留固定 seed、固定初始状态和固定 action 序列，比较：

- 静态 forward kinematics；
- 自由落体与静止接触；
- 单关节 step response；
- 开环 action 下前 1、10、100 步的状态；
- 站立高度、足底接触时间和滑移距离；
- 关节跟踪误差、能耗和跌倒率；
- 策略 rollout 的统计分布。

目标不应是让经典 MuJoCo 与 MJX 的每一步完全相同，而是保证模型语义正确、短时误差可解释、长期行为统计合理。

### 阶段五：缩小 Sim-to-Real gap

优先随机化和标定：

- 质量、质心和惯量；
- 电机强度、`kp/kd`、阻尼和摩擦；
- 控制与观测延迟；
- 编码器和 IMU 噪声；
- 地面摩擦和恢复系数；
- 外部扰动；
- 初始状态与指令变化。

若能获取真机日志，应使用相同控制命令分别驱动仿真和真机，通过关节响应、机身姿态和接触行为反推需要校准的参数。

## 9. 经典 MuJoCo、SONIC 与昇腾 NPU 的兼容性和剩余冲突

### 9.1 总体判断与仓库实证

经典 MuJoCo、SONIC 和昇腾 NPU 之间不存在根本性的硬件冲突。合理的职责划分是：

```text
经典 MuJoCo：CPU 物理仿真、observation/reward 生成
SONIC：策略网络、critic 和 PPO 训练逻辑
昇腾 NPU：PyTorch 网络前向、反向传播和参数更新
```

当前仓库的 `feature/mujoco-npu-experiments` 分支已经实际跑通这条链路。官方 Isaac 训练的 SONIC 37M 权重经 MuJoCo CPU + 昇腾 NPU 微调后完成 11000 迭代，在 MuJoCo 确定性评测中达到约 `0.73 m/s`，参考运动约为 `0.72 m/s`。因此当前问题不是“能否训练”，而是接口一致性、物理迁移和生产化完整性。

### 9.2 SONIC 原环境层绑定 IsaacLab

SONIC 的 PyTorch 网络和 PPO 可以在 NPU 上运行，但原始环境实现依赖 IsaacLab 的 assets、sensors、managers 和 `ManagerBasedRLEnv`。当前仓库使用 `SONIC_MUJOCO_ENV=1` 切换到自定义 `MuJoCoEnvManager`，绕过 IsaacLab。

这意味着 IsaacLab 原来自动管理的以下逻辑必须在 MuJoCo 侧维护：

- observation；
- action mapping；
- PD 控制；
- reward；
- reset 和 termination；
- motion sampling；
- contact/body state；
- domain randomization。

这不是运行阻塞，但形成了两套 MDP 实现，后续修改时存在配置漂移风险。

### 9.3 动作空间是已经验证过的关键冲突

SONIC 官方权重学习的是 Isaac 环境中的 joint order、default position、action scale、effort limit、stiffness 和 damping。如果 MuJoCo 按 XML joint range 直接解释 action，同一个网络输出会映射到不同的目标关节角，进而造成巨大 PD 力矩或立即摔倒。

仓库实验中，修复前策略约每 4 步摔倒；按照 Isaac 公式逐关节对齐后才具备稳定训练基础。当前 `MuJoCoEnv` 已显式实现 Isaac action mapping、armature、effort、`kp/kd` 和 default pose，但更换机器人配置或 checkpoint 时必须重新检查：

```text
joint name/order/direction
default position
action scale 与限幅
torque limit
kp/kd
control frequency 与 decimation
```

### 9.4 策略会绑定训练物理引擎

实验观察到明显的对称现象：

| 权重 | Isaac Sim | MuJoCo |
|---|---|---|
| 官方 Isaac 权重 | 正常运行 | 直接迁移时很快摔倒 |
| MuJoCo 微调权重 | 回传 Isaac 时很快摔倒 | 正常运行 |

这说明策略会适应训练引擎的接触、摩擦、执行器响应、积分器和足底碰撞细节。MuJoCo 中 reward 和步行效果良好，并不能单独证明真机效果良好。这是当前最大的科学和 Sim-to-Real 风险，而不是 NPU 软件兼容问题。

### 9.5 当前成功路线是微调，不是严格的从零训练验证

已验证的路线是：

```text
官方 Isaac SONIC 37M 权重
          ↓
MuJoCo + 昇腾 NPU 微调
          ↓
适应 MuJoCo 动力学
```

它证明了 rollout、SONIC forward、PPO loss、backward、optimizer 和 checkpoint 链路均可工作，但尚不能直接推出“随机初始化后一定能训练出官方同等级的全身控制能力”。从零训练更依赖 curriculum、motion sampling、reward shaping、termination、domain randomization 和运动数据规模。

### 9.6 当前 MuJoCo 环境仍缺少部分完整 SONIC 功能

`MuJoCoEnvManager` 中以下接口目前是空实现或兼容 stub：

- adaptive sampling；
- motion resampling/curriculum；
- domain randomization 重采样；
- 环境状态完整保存和恢复；
- 部分日志与回调数据。

这些功能不阻止最小训练流程，但会影响从零训练、困难动作覆盖、严格续训和 Sim-to-Real。优先级最高的是实现真正的 domain randomization。

### 9.7 CPU 与 NPU 每步存在同步数据搬运

当前数据链路是：

```text
MuJoCo/NumPy → 共享内存 → CPU Tensor → NPU
NPU action → CPU/NumPy → 共享内存 → MuJoCo
```

这不是 CPU 与 NPU 之间的真正零拷贝，每个控制步都需要 observation 的 CPU→NPU 搬运以及 action 的 NPU→CPU 搬运。它会影响吞吐量和同步等待，但不影响正确性。

仓库实测显示，1024 环境、128 worker、单 NPU 已可完成训练；当前瓶颈主要在 CPU 采集而不是 NPU 学习，因此第一阶段没有必要为追求零拷贝而改写架构。

### 9.8 NPU 精度、算子和 checkpoint 工程问题

当前训练入口检测 `torch_npu` 后使用 NPU 设备，并因现有随机采样路径的兼容性关闭 BF16/FP16，采用 FP32：

```text
MuJoCo physics：CPU float64
SONIC network：NPU float32
```

现存注意事项包括：

- 新增网络算子需要在目标 `torch_npu + CANN` 版本上测试；
- 代码中少量 `torch.cuda.*` 调用应逐步改为设备判断；
- NPU 完整 checkpoint 可能包含 `torch_npu` 类型，无法在普通 CPU/Mac 环境直接加载；
- 部署前应在 NPU 服务器导出全部 tensor 已 `.cpu()` 的纯策略 checkpoint；
- NPU storage 在 callback 的 `deepcopy` 中曾出现兼容问题，需要保留现有规避逻辑。

### 9.9 多 NPU 与 MuJoCo 多进程的启动风险

当前单 NPU 路线已经验证稳定。若扩展多卡 HCCL/DDP，需要注意不要让 fork 出来的 MuJoCo worker 继承 NPU/HCCL 状态。建议：

- 使用 `spawn` 或 `forkserver`；
- 或在 HCCL/DDP 初始化前创建 MuJoCo worker；
- 每个 rank 只管理自己的 local environments；
- MuJoCo worker 不初始化或 import NPU runtime；
- 只有在单卡 learner 成为瓶颈后再扩展多 NPU。

### 9.10 GAE、auto-reset 与 terminal observation 需要专项审计

MuJoCo 环境 done 后会自动 reset，返回的 next observation 可能属于新 episode。用于 timeout bootstrap 的 value 应对应 terminal observation，而不能错误使用 reset 后状态。当前 manager 已返回 `terminal_obs`，但仍应构造短回合单元测试，手算 GAE 并与完整 SONIC trainer 的输出逐项比较。

这类问题可能不会让训练立即崩溃，却会使 value target 和 advantage 有偏。

### 9.11 当前最重要的 Sim-to-Real 缺口

应优先接入和验证以下随机化：

- link mass、COM 和 inertia；
- motor strength、armature、`kp/kd`；
- joint damping 和 friction；
- control/observation latency；
- IMU 和 encoder noise；
- ground friction 和 contact parameters；
- external push；
- action latency。

因此当前组合的准确判断是：

| 组合关系 | 判断 |
|---|---|
| 经典 MuJoCo ↔ 昇腾 NPU | 没有直接冲突，CPU/NPU 解耦即可 |
| SONIC 网络 ↔ 昇腾 NPU | 已跑通，当前使用 FP32 |
| SONIC PPO ↔ MuJoCo 环境 | 已通过适配器跑通 |
| 官方 SONIC 权重 ↔ MuJoCo | 需要动作对齐和微调 |
| MuJoCo 权重 ↔ Isaac | 当前不能直接迁移 |
| MuJoCo 权重 ↔ 真机 | 当前结果尚不能单独保证 |
| 多 NPU ↔ 多进程 MuJoCo | 存在 HCCL/fork 和同步风险 |
| 当前环境 ↔ 完整 Sim-to-Real | domain randomization 尚需补齐 |

推荐保持单 NPU + 1024 environments + 128 workers 的已验证组合，优先完成 action/observation/reward 一致性测试、GAE 审计和 domain randomization，再考虑多 NPU、混合精度及吞吐优化。

## 10. 当前结论

1. 使用经典 MuJoCo CPU 采样、昇腾 NPU 训练，是一条可行且便于先跑通流程的路线；
2. 经典 MuJoCo 的主要劣势是大规模并行采样吞吐量，不是单环境精度低；
3. `float32` 和 `float64` 的差别主要是数值舍入精度，不能直接等同于物理真实性；
4. 经典 MuJoCo、MJX-JAX 和 MJX-Warp 共享模型体系，但底层计算路径不同，不保证逐步轨迹一致；
5. Playground 参数可以复用为基线，但必须连同控制、奖励、观测、随机化等环境逻辑一起分析，并重新验证求解器和接触相关设置；
6. MuJoCo 的 G1 zero-shot 案例证明 Sim-to-Real 路线可行，但不意味着仿真策略可以无条件直接上真机；
7. 真正决定 Sim-to-Real 效果的是模型与真机的匹配、控制接口一致性和 domain randomization，而不是单纯选择 Isaac 或 MuJoCo。
8. 经典 MuJoCo 可以复现 Playground 的主要环境和训练逻辑，但需要迁移 XML 之外的 observation、action、reward、reset 和随机化代码；
9. “同一模型”代表物理语义大致一致，不代表经典 MuJoCo、MJX-JAX 与 MJX-Warp 的逐步数值轨迹完全相同。
10. 经典 MuJoCo + SONIC + 昇腾 NPU 的训练链路已被仓库实验验证跑通，剩余问题主要是接口语义、物理引擎迁移和 Sim-to-Real，而不是硬件不兼容；
11. 当前成功结果来自 Isaac 预训练权重在 MuJoCo 中的 NPU 微调，不能直接等同于已验证随机初始化从零训练；
12. domain randomization、GAE terminal observation、多 NPU 进程模型和部署 checkpoint 可移植性是下一阶段的主要工程事项。

## 11. 项目内实证与参考资料

- [SONIC × MuJoCo × 昇腾 NPU 实验总结](EXPERIMENTS.md)
- [NPU 微调与 Mac 渲染全流程指导](NPU微调与Mac渲染指导.md)
- [预训练策略与 MuJoCo 动作空间不匹配分析](Task7_Action_Mismatch.md)
- [MuJoCo CPU + NPU 单机训练设计](NPU_MuJoCo_Training_Design.md)

- [MuJoCo MJX 官方文档](https://mujoco.readthedocs.io/en/latest/mjx.html)
- [MuJoCo 3.2.3 MJX 文档：单场景与大规模并行性能说明](https://mujoco.readthedocs.io/en/3.2.3/mjx.html)
- [MuJoCo Warp 官方文档](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html)
- [MuJoCo Playground](https://playground.mujoco.org/)
- [MuJoCo Playground GitHub 仓库](https://github.com/google-deepmind/mujoco_playground)

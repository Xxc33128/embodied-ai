# GR00T-WholeBodyControl：Isaac–MuJoCo 环境搭建与项目进展报告

> 报告日期：2026-08-24  
> 报告范围：环境搭建、故障排查、跨引擎对齐基础设施、当前进展与后续计划  
> 信息来源：《环境配置排障复盘-2026-08-24》《交接文档-Isaac环境搭建》及 Phase 7/8 对齐计划与执行记录

## 摘要

本项目面向 NVIDIA G1 人形机器人全身控制，目标是在同一套 29-DoF SONIC 控制接口和官方 checkpoint 下，使 MuJoCo 的动力学行为尽可能接近 Isaac/PhysX，并建立可审计、可复现的跨引擎验证流程。

环境建设经历了 WSL Isaac 路线失败、Windows Isaac 4.5 兼容性失败以及 Windows Isaac 5.1 重建三个阶段。最终形成了明确的平台分工：Windows 原生负责 Isaac/PhysX、训练冒烟与双引擎 GPU 验收；WSL 负责 MuJoCo、TensorRT、ROS/部署；Mac 负责纯 Python 开发、审计和轻量单测。

截至 2026-08-24：

- WSL MuJoCo 仿真和 TensorRT 部署链路已经跑通；
- Windows Isaac Sim 5.1 环境已经稳定运行；
- Phase 7 接口门禁通过，Phase 7.5 已完成测试、反向门禁和私库提交；
- Windows 训练入口已完成 4–2048 env 的多档冒烟验证；
- Phase 8A 的 MuJoCo 单项冒烟和 Isaac adapter probe 已通过；
- 当前唯一主线技术卡点是 Isaac benchmark CLI 冒烟卡死；
- 尚未开始修改 MuJoCo 物理参数，相对 Maxwell 基线的物理参数修改量仍为 **0**。

项目目前已经从“搭环境”进入“建立成对物理基线”的阶段，但还不能宣称完成物理对齐。

### 报告口径说明

《交接文档-Isaac环境搭建》主要记录 2026-08-18 至 8 月 19 日早期状态，其中“Windows Isaac 4.5 仍卡在 scenedb”是历史快照；《环境配置排障复盘-2026-08-24》已经记录该路线被 Isaac 5.1 稳定环境取代。因此本报告以 8 月 24 日复盘为当前状态，以较早交接文档补充问题演进和决策依据。

同理，8 月 20 日早期 Phase 7 报告中尚未 push、测试未收口等状态，已被 8 月 24 日 Phase 7.5 的 12 commit、104 项测试和反向门禁结果更新。仍未完成的 PAT 撤销等事项则继续保留。

---

## 1. 项目目标与技术路线

### 1.1 总体目标

项目不是单纯复现 Isaac 或 MuJoCo demo，而是建立一条完整的跨引擎对齐链路：

```text
冻结版本、模型与资产
    ↓
自动导出 Isaac/MuJoCo 运行时参数
    ↓
证明 observation → action → target → torque 接口等价
    ↓
运行无策略的 18-case 隔离物理测试
    ↓
逐参数族优化 MuJoCo
    ↓
用同一官方 checkpoint 做双引擎否决性验收
```

Isaac/PhysX 被作为参考引擎，MuJoCo 被作为待标定引擎。官方 SONIC sim-to-sim 配置作为独立参考基线，而不是被直接视为 PhysX 的等价参数。

### 1.2 计划中的阶段

| 阶段 | 内容 | 当前状态 |
|---|---|---|
| Phase 0 | 冻结代码、环境、资产、checkpoint、motion 和随机种子 | 已完成主要部分 |
| Phase 1 | 导出两个引擎的模型、执行器、碰撞和 solver 参数 | 已实现并完成真实环境审计 |
| Phase 2 / 7 | 100-state 控制接口门禁 | 已通过 |
| Phase 7.5 | 环境收口、非循环验证、测试和私库提交 | 已完成，仍有用户安全操作 |
| Phase 3 / 8A | 6+4+8 共 18 个成对物理基线 case | 前置就绪，尚未完整运行 |
| Phase 4 / 8B | 分参数族优化 MuJoCo | 未开始 |
| Phase 5 / 9 | 官方 checkpoint 双引擎裁决 | 未开始 |
| Phase 10 | 留出动作、扰动与稳健性验证 | 未开始 |

---

## 2. 硬件条件与平台分工

### 2.1 主要硬件与现实约束

Windows 测试机主要配置为：

- AMD Ryzen 9 7845HX；
- NVIDIA RTX 5060 Laptop，8GB 显存，Blackwell `sm_120`；
- AMD Radeon 610M 核显；
- 16GB 物理内存；
- C 盘空间紧张，大型环境、缓存和数据需要放在其他磁盘；
- 国内网络环境，PyPI、Hugging Face 和 NVIDIA/PyTorch 源需要分别配置。

这台机器可以完成单环境 Isaac 采集、短时评测和中等规模训练冒烟，但内存、显存和磁盘余量都不足以支持无约束并发、大规模 LFS 下载或反复启动多个 Kit 实例。

### 2.2 最终平台分工

| 平台 | 主要职责 | 不承担的工作 |
|---|---|---|
| Windows 11 + NVIDIA GPU | Isaac Sim、Isaac Lab、PhysX、runtime exporter、golden trace、18-case 成对验收、训练冒烟 | 不作为日常大规模 Git/LFS 操作节点 |
| WSL2 Ubuntu 22.04 | MuJoCo、TensorRT、ROS、部署链路和 Linux 工具 | 不再运行 Isaac Sim |
| Mac Apple Silicon | schema、mapping、diff、CLI、文档、轻量 MuJoCo 测试 | 不运行 Isaac Sim |

这个分工避免继续强行在 WSL 中解决 Isaac 的 Vulkan/RTX 问题，也降低 Windows 环境被开发依赖反复污染的风险。

---

## 3. 环境路线演进

### 3.1 第一阶段：WSL 部署与 MuJoCo 跑通

项目最初在 WSL2 Ubuntu 22.04 中搭建。完成了：

- WSL 虚拟磁盘迁移到非系统盘；
- TensorRT 10.13 与 C++ 部署程序编译；
- Unitree SDK、CycloneDDS 和符号链接修复；
- MuJoCo 3.2.7 仿真；
- `run_sim_loop.py` 与 `deploy.sh sim` 双进程联调；
- G1 机器人加载策略并播放动作。

这一阶段证明了 MuJoCo 和部署链路是可用的，也形成了后续跨引擎工作的 candidate 侧基础。

### 3.2 第二阶段：尝试在 WSL 运行 Isaac

WSL 中先后完成了 Python、Isaac Sim、Isaac Lab、PyTorch 和 CUDA 依赖组合，但最终遇到 Vulkan 不可用：

- WSL 内 Vulkan 只有软件路径或缺少满足 Isaac RTX 渲染要求的实现；
- OpenGL 可以通过 D3D12 选择 NVIDIA GPU，但不能替代 Isaac Sim 对 Vulkan/RTX 的要求；
- Ubuntu 22.04 上尝试补 Mesa `dzn` 也不能形成可靠的 Isaac 运行路径。

最终结论是：WSL 保留给 MuJoCo、ROS、TensorRT 和部署，不再承担 Isaac Sim。

### 3.3 第三阶段：Windows Isaac 4.5 路线

Windows 原生首先尝试 Isaac Sim 4.5 + Python 3.10 + Isaac Lab 2.1.x。期间解决了：

- RTX 5060 需要 CUDA 12.8 对应 PyTorch wheel；
- torch nightly 缺失 `torch.jit`、`c10.dll` 异常；
- NumPy 被重装升级至 2.x 后与 Isaac Sim 冲突；
- Isaac Sim 自带旧 VC++ DLL 遮蔽系统运行时，导致 osqp/qdldl 原生模块崩溃。

尽管依赖层已经修好，Kit 106.5 仍在 `rtx.scenedb.plugin.dll` 中确定性崩溃。这表明继续修补 4.5 路线的收益很低，因此最终放弃。

### 3.4 第四阶段：Windows Isaac 5.1 稳定路线

环境随后切换到 Isaac Sim 5.1 + Python 3.11 + Isaac Lab 2.3 系列。通过隔离实验发现，跨 Isaac 4.5/5.1 的 `scenedb` 崩溃并非代码问题，而是宿主图形环境问题。

采取以下变更后，原始 smoke 脚本通过：

1. NVIDIA 驱动由 610.88 回退至 580.88；
2. 禁用 Parsec 和 MuMu 虚拟显示适配器；
3. 清理约 10.6GB CrashDumps，恢复系统盘空间；
4. Windows headless 入口强制使用 D3D12；
5. 通过环境变量接受 EULA。

这套环境成为当前正式参考环境。

---

## 4. 当前冻结环境

依据 2026-08-24 的环境冻结记录，当前 Windows 参考环境为：

| 组件 | 当前版本/状态 |
|---|---|
| Isaac Sim | 5.1.0.0 |
| Isaac/PhysX | PhysX 107.3.26 |
| Isaac Lab | 2.3 系列；Python distribution 显示 0.48.0 |
| Python | 3.11.9 |
| PyTorch | 2.7.0 + CUDA 12.8 |
| NumPy | 1.26 系列 |
| NVIDIA Driver | 580.88 |
| GPU | RTX 5060 Laptop 8GB，`sm_120` |
| 渲染后端 | Windows D3D12 |

版本记录中同时出现“Isaac Lab 2.3”和“distribution 0.48.0”，后续实验报告应继续保存 Isaac Lab 源码 commit/tag 和 Python distribution 版本，避免只用单个包版本号产生歧义。

---

## 5. 主要问题、根因与解决方案

### 5.1 GPU 驱动与 RTX 渲染器崩溃

**现象：** Isaac Sim 4.5 和 5.1 均在 RTX/Hydra 初始化阶段发生固定偏移的 access violation；GUI、headless 和不同 experience 均可复现。

**根因：** NVIDIA 610.88 驱动、RTX 5060 Blackwell、虚拟显示设备与 Kit RTX 渲染栈组合不兼容。

**解决：** 回退至 580.88，禁用虚拟显示适配器，清理系统盘 CrashDumps，并保持 D3D12。

**结果：** Isaac Sim 5.1 正常启动，100 个物理步 smoke 通过。

### 5.2 Python、CUDA 与 Isaac 版本矩阵冲突

**现象：** 不同组合出现 `torch.jit` 缺失、`c10.dll` 错误、NumPy 2.x 不兼容、cu126 不支持 `sm_120`、Isaac Sim cp310/cp311 wheel 不匹配。

**根因：** Isaac Sim、Isaac Lab、Python ABI、PyTorch、CUDA 和 RTX 50 系架构具有强耦合关系，直接安装“最新版”会破坏兼容矩阵。

**解决：** 最终冻结 Python 3.11、PyTorch 2.7.0+cu128、NumPy 1.26 和 Isaac Sim 5.1；所有大包安装后检查 NumPy、Torch/CUDA、ABI 和 Isaac pin。

**结果：** GPU 计算、Isaac runtime、采集和训练入口均能运行。

### 5.3 Isaac 自带 VC++ 运行时遮蔽系统 DLL

**现象：** 加载 osqp/qdldl 等原生 Python 模块时发生 access violation。

**根因：** Isaac Sim 目录中的旧版 MSVC runtime 优先于系统新版 DLL 被加载，新编译的 `.pyd` 与旧运行时混用。

**解决：** 在 4.5 环境中用系统新版 DLL 替换相关 VC runtime，并保留原文件备份。

**结果：** 启动期第三方原生模块崩溃消失。该问题与后续 `scenedb` 驱动崩溃被证明是两个独立问题。

### 5.4 WSL Vulkan 路线不可行

**现象：** WSL 内 Isaac 报 Vulkan/GPU 初始化失败，只有软件渲染路径，Isaac RTX 无法工作。

**根因：** WSL 图形栈不能提供当前 Isaac Sim 所需的原生 RTX/Vulkan 能力。

**解决：** 停止在 WSL 中运行 Isaac，改为 Windows 原生 Isaac；WSL 仅保留 MuJoCo、TensorRT、ROS 和部署。

**结果：** 避免继续在不可行路径上反复安装和降级。

### 5.5 Headless 渲染与 URDF importer 卡死

**现象：** 部分 headless 脚本被切换到 Vulkan 后，URDF importer 在原生 C++ 调用处静默卡死；降低 Kit 日志等级也会增加误判和排障难度。

**根因：** Windows headless 的渲染后端没有被统一固定，部分参数注入方式无效。

**解决：** 所有 Isaac Windows 入口显式传入 D3D12 参数，保持默认 info 日志，使用 `py-spy` 观察挂起栈。

**结果：** exporter、capture、训练入口的启动方式得到统一。

### 5.6 Isaac 5.1 URDF 碰撞导入回归

**现象：** 机器人质量、关节和执行器存在，但 audit 显示 collider 数量为 0；历史接触力数据因此不可信。

**根因：** 当前 importer 没有把 URDF `<collision>` 正确生成到 USD stage，且 instance prim 不能直接写入。

**解决：** 取消相关 prim 的 instanceable 状态，解析同一 URDF 并重新 author primitive colliders，再应用 `UsdPhysics.CollisionAPI`。

**结果：** Isaac collider 从 0 增加到 54，真实 G1 落地不再直接穿透；18 个手部 mesh 暂未补齐。

**影响：** 修复前 P0/P1 中的接触力不能用于 Phase 8 物理比较，必须在带碰撞环境中重新采集。

### 5.7 Isaac runtime 与采集脚本问题

代表性问题包括：

- EULA 交互导致后台脚本等待；
- `SimulationContext` 被 warmup 和环境重复创建；
- motion library 只加载 `num_envs` 个 motion，后续索引越界；
- adapter 的局部变量错误被包装异常吞掉；
- adapter 的 `argparse` 吞掉外层 CLI 参数；
- ContactSensor 初始化顺序错误；
- `simulation_app.close()` 在 Windows 收尾挂死；
- 浮点 readback 导致近零 torque 的相对误差失真。

对应解决方式为：

- 入口统一设置 EULA、D3D12 和 `parse_known_args()`；
- 移除冲突的 warmup context；
- 明确加载完整 motion key 集；
- 缩小异常包装范围，保留真实 traceback；
- sensor 在 reset 前构造、reset 后更新；
- 结果先原子落盘，再限时关闭 Kit；
- same-state torque 使用同一精确注入状态计算，readback 仅作为 sanity check。

这些修复最终支持了 100-state 接口门禁和训练入口的稳定运行。

### 5.8 MuJoCo、文本资产与 Windows 跨平台差异

**现象：** MuJoCo 新版本对全局重名更严格；Windows CRLF checkout 导致 XML/URDF 的 SHA-256 与 Git LF 内容不一致；PowerShell 和 Python 默认编码造成测试失败。

**解决：** 修改重复的地面、纹理和材质名称；文本资产哈希前统一换行；所有脚本显式指定 UTF-8；平台测试覆盖 CRLF、GBK 和路径行为。

**结果：** 29-DoF MJCF 可正常编译，正式 benchmark 的资产指纹可跨平台复现。

### 5.9 网络、代理和依赖下载

**现象：** PyPI/Hugging Face 下载失败；Clash TUN 使 WSL 网络进入黑洞；清华源可能下载到不支持 RTX 50 系的 cu126 PyTorch。

**解决：** 普通 Python 包使用可用镜像，Hugging Face 使用镜像 endpoint，PyTorch cu128 使用官方 cu128 源；WSL 使用 NAT 和直接镜像，不依赖 TUN；安装后立即检查 PyTorch build tag。

### 5.10 内存、磁盘与进程风暴

**现象：** 全量 Git LFS、多个 agent、并行 Git、反复启停 Kit 和日志递归导致内存耗尽，出现 Event 26、Event 41 和异常重启；C 盘曾接近 0GB。

**根因：** 16GB 内存不足以同时承载 Kit、浏览器、Node/agent、多层 Git 子进程和大文件解压；日志桥接递归可单进程占用数 GB。

**解决：**

- 禁止无界 `git lfs pull`，只按路径获取资产；
- clone/checkout 使用 `GIT_LFS_SKIP_SMUDGE=1`；
- 同时只运行一个 agent，Git 操作串行；
- Kit 运行前检查可用内存，低于 3GB 停止；
- 关闭 torio/torchaudio 噪声日志，禁止 root logger 使用 DEBUG；
- 大文件、pip cache、临时目录和 WSL 虚拟盘放在非系统盘；
- 定期清理 CrashDumps 和残留 Kit/Python/Git 进程。

**结果：** 后续采集与训练冒烟没有再重复早期的系统级 OOM 模式。

### 5.11 Git 私库、LFS 与凭据

**现象：** 私库 token 权限不足返回 404；Git LFS 安装不完整；LFS lock verify 造成 push 卡住；聊天中曾出现两个明文 PAT。

**解决：** 使用 Git Credential Manager 浏览器授权；仅向个人私库 push；关闭不需要的 LFS lock verify；扫描仓库确认 token 未被提交；禁止提交 checkpoint、motion、NPZ、STL、绝对路径和凭据。

**尚未完成：** 两个已泄露 PAT 仍需要用户在 GitHub 侧手动撤销。这是安全遗留项，不能仅以“仓库中未发现 token”代替撤销。

---

## 6. Isaac–MuJoCo 对齐基础设施进展

### 6.1 已实现能力

当前私有对齐分支已经实现：

- 统一模型审计 schema；
- 29-DoF canonical joint mapping；
- Isaac、MuJoCo 和旧 SONIC 15-DoF runtime exporter；
- 模型、执行器、碰撞、材料和 solver diff；
- 环境与资产 fingerprint；
- 100-state interface trace 与 gate；
- joint-step、free-fall、PD-landing 三类 benchmark；
- 6+4+8 共 18 个冻结 case；
- 13 个 fit 与 5 个 holdout 的冻结划分；
- JSON/NPZ 原子输出和失败记录；
- Windows 环境检查、PowerShell 运行入口和跨平台测试。

### 6.2 SONIC sim-to-sim 的使用方式

旧 SONIC MuJoCo sim-to-sim 已作为独立 baseline 参考。记录的典型设置包括：

- 15 个受控 DoF；
- `physics_dt=0.005 s`；
- decimation 为 4；
- `control_dt=0.02 s`；
- action scale 为 0.25；
- 旧路线的 damping、armature、friction loss 和点式脚底接触。

这些参数可以作为 MuJoCo 候选来源，但旧路线在动作维度、模型、手部和 checkpoint 接口上与当前 29-DoF 路线不同，不能直接覆盖当前配置，也不能宣称与 PhysX solver 数值等价。

### 6.3 接口门禁结果

Phase 7 使用 100 个确定性同状态样本，检查：

- observation block、history 和 930 维 actor observation；
- policy/raw/clipped/canonical action；
- target joint position；
- unclipped/applied torque；
- joint order、reset、reference time 和 substep 时序；
- checkpoint、motion、state-set 与 config provenance。

门禁结果为：

- 100 帧完成；
- 14/14 exact 项通过；
- 12/12 observation blocks 通过；
- torque 最大绝对差约 `2.96e-6`；
- 5 类故意注入的接口错误能够被门禁拒绝。

后续 Phase 7.5 又完成了 104 项测试、Ruff 清理和非循环 gate 反向测试，并将 12 个 commit 推送到个人私库，记录 commit 为 `45ed748`。

### 6.4 物理参数状态

截至本报告：

```text
新增或调优的 MuJoCo 物理参数：0
当前仍使用：Maxwell 29-DoF baseline
impratio：30
solver iterations：200
```

已有工作只是把 default pose、Kp/Kd、effort limit、armature、action scale 和 clipping 集中到共享 control profile，并保持数值不变。当前不能把接口门禁通过表述为物理已经对齐。

---

## 7. 训练与容量验证进展

Windows 训练入口补齐了 EULA 和 D3D12 参数，并处理了本地 motion 路径。`num_envs=1` 暴露出 batch size 必须能被 4 个 mini-batch 整除，因此正式最小并发为 4。

已经验证 4、8、16、32、64、128、256、512、1024 和 2048 env 均能完成短训练冒烟：

- 256/512 env 适合作为日常调试档；
- 1024 env 在 8GB 显存上仍有较好吞吐；
- 2048 env 已接近显存临界并出现吞吐下降；
- 4096 env 不建议在当前硬件上使用。

这项结果证明训练入口和 Windows Isaac 项目集成已经基本打通，但不代表模型完成了正式训练，也不属于物理对齐最终验收。

---

## 8. 当前进度与未完成项

### 8.1 当前进度

| 工作项 | 状态 | 说明 |
|---|---|---|
| WSL、MuJoCo、TensorRT 部署 | 完成 | G1 仿真和部署链路已跑通 |
| Windows Isaac 参考环境 | 完成 | 5.1 环境已冻结，驱动 580.88 |
| 模型与运行时审计 | 完成 | Isaac collider 已从 0 修复至 54 |
| P0 基线采集 | 已采集但需分用途 | 24 个 NPZ、零 NaN；修复碰撞前的数据不能作为接触证据 |
| P1 关节驱动采集 | 已采集但需分用途 | 27 个 NPZ；接口/关节响应可参考，接触相关字段需重采 |
| 100-state 接口门禁 | 完成 | Phase 7 与 7.5 已通过 |
| Windows 训练冒烟 | 完成 | 4–2048 env 多档验证 |
| MuJoCo Phase 8 joint-step smoke | 完成 | 单项冒烟通过 |
| Isaac benchmark adapter probe | 完成 | adapter 基础探针通过 |
| Isaac benchmark CLI smoke | 阻塞 | 当前主线技术卡点，出现挂起 |
| 完整 18-case 成对基线 | 未完成 | 等待 Isaac CLI 修复 |
| MuJoCo 参数优化 | 未开始 | 必须先取得成对基线 |
| 官方 checkpoint 双引擎裁决 | 未开始 | 必须等待候选物理配置 |

### 8.2 当前待处理问题

按优先级排列：

1. 撤销两个已泄露的 GitHub PAT；
2. 定位并修复 Isaac benchmark CLI 卡死；
3. 在碰撞修复后的环境重新采集 Phase 8 接触数据；
4. 明确 18 个手部 mesh collider 是否纳入目标动作的碰撞范围；
5. 运行完整 18-case Isaac/MuJoCo 成对基线；
6. 决定是否准备约 21.9GB 的完整官方 motion 库；
7. 完成 Phase 7 的正式封板声明。

手部 mesh 不应仅为了让 collider 数量相等而盲目补齐。如果 Phase 8 动作没有手部接触或自碰撞，可以在两侧一致关闭并写入 collision contract；如果后续动作涉及手部接触，再建立独立 hand-contact profile。

---

## 9. 下一阶段计划

### 9.1 Phase 8A：零改参成对物理基线

首先保持 Maxwell 配置完全不变，运行：

- 6 个代表性关节 joint-step；
- 4 个 zero-torque free-fall；
- 8 个 fixed-PD landing。

两侧统一初始状态、joint order、控制参数、timestep、重力、case seed、碰撞范围和采样率。第一轮只建立基线，不设未经数据支持的物理 pass/fail 阈值。

主要记录：

- joint rise time、overshoot、settling time、steady error；
- q、qd、qdd 和 torque；
- touchdown time；
- GRF peak 和 impulse；
- penetration depth；
- 足端滑移；
- root 位移和旋转；
- 能量变化与 torque saturation。

13 个 fit case 用于选择参数，5 个 holdout case 只用于候选晋级；看到结果后不能重新划分。

### 9.2 Phase 8B：逐参数族优化

获得完整基线后，按以下顺序一次只修改一个参数族：

1. timestep、control timing、integrator/substep；
2. mass、inertia、CoM；
3. Kp/Kd、armature、damping、friction loss 和限制；
4. 脚底碰撞几何和位姿；
5. friction、restitution、margin/contact offset；
6. `solref`、`solimp`、`impratio`、`cone` 和 iterations。

如果无接触 joint-step 已经不一致，应优先检查控制生效时序和 actuator/passive 参数，而不是直接调 contact solver；如果 joint-step 一致而落脚差异大，再进入碰撞、材料和 solver 标定。

### 9.3 Phase 9：checkpoint 双引擎裁决

每个晋级候选都要使用同一官方 checkpoint 和固定 sample motion 在 Isaac、MuJoCo 中运行。

硬性否决条件包括：

- 任一引擎无法完成动作；
- 跌倒或异常穿透；
- 长期 torque saturation；
- observation/action NaN；
- 物理指标改善但 checkpoint 在任一引擎失效。

### 9.4 Phase 10：稳健性验证

最终候选还需要在未用于拟合的 motion、初始高度、姿态、速度、水平偏移、摩擦扰动和多个 deterministic seed 上验证。

如果不存在一组固定参数能同时改善关节响应、足端冲量、滑移和 root motion，则应停止追求单点精确匹配，转向围绕标定值做物理随机化，或研究低层跨引擎 adapter。

---

## 10. 风险控制与停止条件

后续实验遵守以下规则：

- interface gate 未通过，不调物理；
- 模型、资产或 config SHA 不一致，不比较结果；
- 接触数据来自无碰撞环境，不进入 Phase 8；
- fit 改善但 holdout 退化，不晋级；
- checkpoint 在任一引擎失效，不设为默认；
- 一次实验修改多个参数族，结果作废；
- 原始 NPZ、checkpoint、motion、STL、视频和机器路径不提交 Git；
- Windows 空闲内存不足或出现进程风暴时立即停止 Kit/Git 操作；
- 驱动 580.88 和虚拟显示禁用状态未经新的隔离验证不得改变。

---

## 11. 结论

环境搭建阶段最关键的成果不是单纯“安装成功”，而是形成了稳定、可解释、可复现的平台边界：

1. WSL 跑 MuJoCo/部署，Windows 原生跑 Isaac，Mac 做轻量开发；
2. 固定驱动、CUDA、PyTorch、Python 和 Isaac 版本矩阵；
3. 将环境、模型、资产、控制接口和实验结果纳入统一 fingerprint 和 gate；
4. 通过反向测试证明门禁能够发现 joint order、history、clip、substep 和 quaternion 错误；
5. 明确当前完成的是环境与接口等价基础设施，而不是物理参数优化。

下一里程碑不是继续安装环境，也不是立即调 `solref/solimp`，而是解决 Isaac benchmark CLI 挂起，完成带真实碰撞的 18-case 成对基线。取得这组数据后，项目才能进入真正的 MuJoCo 物理参数标定阶段。

# SONIC NPU 适配讲解讲稿（配《zhangqin 分支 NPU/CPU-MuJoCo 静态源码审计》）

> 对象：给他人讲解 zhangqin 分支做了什么。时长约 20 分钟。
> 立场声明（开场必须说）：本报告是静态源码审计，只能证明 NPU 路由和形状兼容代码存在，不能证明训练适配已经完成。收敛与性能需另附运行证据。
> 讲稿与报告章节一一对应：§0→§8＋附录，问答见末尾。

---

## 0. 开场白（1 分钟）

各位好，今天讲的是 zhangqin 的 `feature/mujoco-training` 分支（锁定提交 `627743e`），对比锚点是原版 `NVlabs` 的 `origin/main@0e35637`。

先定两个框架，对应报告§0：

第一，正交维度——物理环境和学习设备是两个独立维度。物理有 Isaac PhysX、CPU MuJoCo、Stub、PhysX SDK 直连四种；学习设备有 CUDA、NPU、CPU 三种。zhangqin 的主路径是 CPU MuJoCo 乘 NPU。

第二，证据等级——我今天讲的都是源码层面能证明的东西：路由存在、维度对得上、权重差异在哪。训练能不能收敛、多快、多稳，不在今天的证据范围内。

报告里我把改动分成五类，大家记住这个分类，后面不再混着讲：核心 NPU 路由、CPU-MuJoCo 环境适配、Smoke Test、日志评测增强、PhysX 实验功能。只有前两类是 NPU 跑通的核心依赖。

---

## 1. 训练入口（4 分钟，对应报告§1）

请大家翻到§1。入口文件是 `gear_sonic/train_agent_trl.py`，zhangqin 用环境变量做四选一：`SONIC_MUJOCO_ENV`、`SONIC_STUB_ENV`、`SONIC_PHYSX_ENV`，都不置位才是 IsaacSim。

这里要纠正一个容易误传的点：文件开头仍然会尝试 import isaaclab，只是在非 Isaac 模式下允许失败而不退出。不是把 import 推迟到了 Isaac 分支。大家看报告里贴的 33 到 50 行。

NPU 的核心就三行：检测到 `torch_npu` 可用，就把 bf16 和 fp16 关掉，强制 fp32，`Accelerate` 的混合精度设为 no，设备写成 npu 加编号。关于 bf16 我多说一句：代码注释说 NPU 不支持 bf16 的 torch.normal，但我们没有跨版本的最小复现 probe，所以正式口径是特定版本组合下复现失败，不说成硬件普适结论。

DDP 有两处：冻结 encoder 时把 find_unused_parameters 置 true，否则会报错；按 world_size 均分 env 和 worker 数。这里提醒两个健壮性缺口：整除丢余数，以及 worker 数小于卡数时会得到零，要加 guard。timeout 6000 秒是原版就有的，沿用，不是新增。

训练实例化时注意三个细节：模型用 `g1_29dof_v17.xml`，数据用 `sample_data/robot_filtered`，这两个都是容器绝对路径，换环境要改；alive_bonus 在 env 里默认 4.0，但训练入口覆盖成 0.0；MuJoCo 专用的探索噪声收敛在分支内部，不污染全局配置。

还有一个必须讲的缺口：可复现环境。现在 pyproject 里没有锁定 torch、torch_npu、CANN、驱动、固件、Accelerate 版本，NPU 不可用时还是静默继续而不是 fail-fast。正式交付要补兼容矩阵、容器和 lock 文件。

fork 安全也要讲清楚：Accelerator 在 211 行初始化，Manager 在 382 行才建，用的还是默认 mp.Process，Linux 下 fork 会继承已初始化的加速器运行时。提前 import Manager 解决不了这个问题，要显式用 spawn 或 forkserver，加多 rank 乘多 worker 的启停测试。我报告里之前写错过，已经作废，以这版为准。

---

## 2. MuJoCo 环境（6 分钟，对应报告§2，今天的重点）

翻到§2。`mujoco_env.py` 是 552 行的新文件，原版没有。先说维度：930、1645、1761 三个数字只代表 checkpoint 张量形状装得下，叫 shape-compatible，不叫逐字段等价。预训练 37M 权重的正确表述是结构可加载、行为兼容未验证。六项差异都在§2.1 的表里：动作仿射不同、统一 kp kd 对逐关节参数、速度状态定义不同、没有观测噪声、SMPL 全零、奖励终止改过。观测噪声这一点是实测过的：mujoco 文件里 grep 不到 noise，Isaac 的观测配置里全是 AdditiveUniformNoise。

归一化现状用一句话：jm jh 用了 Isaac 真值，解决了 Task7 里用 XML range 会映射错关节的问题；但 kp kd 还是统一的 100 和 5，没有写 dof_armature，没有引入按 effort 和 stiffness 算的 act_scale。solver 只加了 iterations 200。

控制时序：pd target 等于 action 乘 jh 加 jm，物理步是 10 子步、每步 2 毫秒，控制周期 20 毫秒。关键是每个子步前都重算力矩，不能冻结 10 步，否则注释里写的 qacc 会发散到 1e5。

观测三头按维度记住就行，细节在§2.4。提醒一句：future dof 用的是向下截断取前一帧，不是最近帧，更不是 lerp slerp。MotionLib 对位移用 linear、对四元数用 slerp，两边不对齐。

奖励 13 项里重点讲 feet_acc，这是之前报告写错过的地方：MuJoCo 是负 2.5e-9，注释自述比 Isaac 的负 2.5e-6 降了 1000 倍，因为它算的是踝关节角加速度，接触时能冲到 1e8 量级。另外 stub 配置是负 2.5e-6，Isaac 的 terms 是负 2.5e-7，三处互不一致，不能说一致。这是明确的奖励非等价。

数据管线是之前漏掉、这次补上的§2.6：最多 500 条 motion，固定种子洗牌后截断，不是完整 BONES-SEED；无自适应采样。扩数据之前先看这个天花板。

重置就是把参考的根位置、根四元数、关节角赋给 qpos，清历史缓冲。终止透传 _orig_done，设计意图是给 GAE 区分终止和截断，但正确性还要靠单测验证，今天不下正确性的结论。

---

## 3. 并行管理器（4 分钟，对应报告§3）

翻到§3。`mujoco_env_manager.py` 401 行，7 块共享内存：obs、terminal、actions、rewards、dones、timeouts、orig_dones。MuJoCo 的 obs 维是 4336，三头之和；PhysX 是 4365，多 29 维 ref_action。

容量纠正一下：1024 个 env 时 obs 区约 16.94 MiB，terminal 区同尺寸，7 块合计约 34 MiB。我之前只算了 obs 一区，漏了 terminal，以这版为准。而且每步仍有两次搬运：NPU 到 CPU 的 action，CPU 到 NPU 的 observation，所以不能叫 zero-copy。

同步是 Barrier 双击：worker 等、算完再等；trainer 写 action 后等两次再读。60 秒超时。两处坑：terminal 区没有 valid 位，没终止的 env 会读到旧值，要用 dones 或 timeouts 门控；worker 崩了会 respawn 但随即抛 RuntimeError，当前 rollout 作废，训练器不会自动恢复，不能叫自动恢复。

_to_numpy 是通用边界，不是 NPU 专属：输入是 TensorDict 时走 storage 或 to cpu 的回退，否则抛 TypeError。CUDA 和 CPU 下同样经过它。

worker 的随机数也要提一句：motion 加载是固定种子洗牌，采样用全局 numpy 随机，fork 后没有按 worker 派生，多 worker 可能采到重复序列。

接口上 reset 和 step 保持了主要 trainer 接口兼容，但 trainer 本身改了 _orig_done、mask、存档和日志，所以不叫零改动接入。

---

## 4. 收尾三节快讲（3 分钟，对应§4 到§8）

§4 定性一句话：stub 是 Smoke Test 诊断工具，physx 是实验备选，都不是 NPU 核心依赖。

§5 配置：stub_train 模板关 bf16 fp16，future 10 帧 0.1 秒，和环境对齐。

§6 PPO 改动四项：TensorBoard 落盘、无 wandb 时看曲线；_orig_done 对齐，解决 ignore_terminations 下 GAE 和 transformer mask 跨 episode 污染；torch_npu 旧格式 Byte storage 的存档修复；deterministic rollout 开关，开时用均值不采样。表述统一为保持主要接口、trainer 仍有修改。

§7 改个说法：不叫未改动保形，叫保留的文件格式与模型拓扑。数据加载、插值、采样、奖励终止都被重实现过。

§8 数字按锁定提交重计：physx 约 4150 行 10 文件，docs 加 scripts 约 24784 行 132 文件，合计 33188 加 40 减，174 文件。mujoco_env 的定语是实验性 shape-compatible 环境。

---

## 5. 结束语（1 分钟）

收一下：zhangqin 分支证明了 NPU 路由存在、CPU MuJoCo 能以兼容形状接入 PPO；同时留下了明确的非等价清单：奖励、终止、噪声、插值、采样、PD 参数、数据规模。静态审计到此为止，下一步的运行证据不在今天范围内。

溯源命令都在附录 A，任何一行都可以当场 git show 复核。

---

## 6. 预设问答（讲解后用）

问：NPU 适配核心改了哪？
答：训练入口的 torch_npu 加 fp32 加 device 是算力路由，mujoco 两文件提供 CPU 物理，_to_numpy 是通用边界。

问：37M 权重能直接用吗？
答：结构装得下，行为没验证，六项差异见§2.1。

问：feet_acc 为什么差 1000 倍？
答：MuJoCo 算的是踝角加速度，量级不同，权重 accordingly 调小，三处配置本身也不一致，属非等价。

问：1024 env 内存多大？
答：7 块合计约 34 MiB，之前 17MB 只算了 obs 一区。

问：worker 崩了会自动恢复吗？
答：不会，respawn 后抛错，当前 rollout 作废。

问：原版是不是 BF16？
答：现有证据不足，以 resolved 参数和日志为准，不预设结论。

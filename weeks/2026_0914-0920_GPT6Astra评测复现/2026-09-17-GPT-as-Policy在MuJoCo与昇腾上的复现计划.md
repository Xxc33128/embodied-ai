# GPT-as-Policy 原任务完整链路的 MuJoCo / 昇腾迁移计划

> v2.0 · 2026-09-17 · 合并原计划与独立 agent 对抗性审查的权威执行计划；三项审查发现已转成规格和验收要求，尚未实施迁移或运行远程 NPU/付费模型实验。
>
> 以用户本轮最终要求为准：直接推进原任务完整链路，不以 bowl、LIBERO 或自制相似任务替代；阶段验证发生在最终实现上，不另做演示工程。
>
> 公开版只保留平台类别，实际服务器配置、接入信息、账户与凭据不入本文。
>
> 证据口径：E1=源码与行号；E2=配置、文件哈希与公开结果记录；E3=文档、作者报告或工程建议，均明确标记。不使用 E4 作为承诺依据。下文工作包和验收设计均为 E3 建议，非已完成实验。

**架构：** Mac 保留原持久 agent，CPU MuJoCo 执行原任务，昇腾负责原 π₀.₅ 推理。两种模型接入分别实现和评测，共享机器人、任务与学生策略接口。

**技术栈：** 固定上游 Python / OpenPI、MuJoCo + CPU 软件渲染、JAX CPU 噪声与参考推理、PyTorch / torch_npu、原 Codex app-server 与 API／网关适配；具体运行版本由 W0/W1 锁定，不预设未经验证的兼容组合。

**规格来源与执行方式：** 本文 §§1–5、7–9 是实现约束，§6 是工作清单；执行时使用 executing-plans 技能按验收门推进。未勾选项均未完成，计划中的文件和测试是拟建产物，不表示仓库已有实现。

## 1. 目标和不可降低的范围

**目标：在现有工作基础上，保留 GPT-as-Policy 的原机器人、原任务、原权重、观测/动作契约、agent 决策流程与评测案例，将必须依赖 GPU 的执行部分迁移到 CPU MuJoCo 与昇腾 NPU。**

主交付覆盖 RoboDojo 原十任务、50 个指定案例的 Direct 与 Hybrid 链路；匹配初态的 Student-only 是补充对照，单独标记为本次新增实验。报告中的 RoboLab 单臂实验另列完整性清单，不用 RoboLab、旧 bowl 或 LIBERO 的完成代替 RoboDojo 主实验。只有两套均完成，才可称覆盖整份报告的主要实验。

以下项目不作未经用户接受的替换：

- 双 ARX X5 不换成 Panda；原物体和布局不换成自制方块抓放。
- RoboDojo 微调 π₀.₅ 不换成 pi05_base、pi05_libero 或脚本策略。
- GPT Direct 和 Hybrid 保留原模型/推理设置、episode 内记忆与分析工具语义；裸 API 调用不是自动等价的 agent。
- 原任务成功条件、阶段分、演示/交互事件和控制预算不因实现困难而简化。
- 叠衣服、顺序记忆、麻将任务不从总样本中悄悄删除；做不到时明确报告该项未复现，不能用缩小后的分母宣称全部完成。

**严格的原数值复现目前不能承诺。** MuJoCo 与 Isaac 的接触动力学、柔性求解和图像渲染不同；可以争取完整链路与任务语义的高保真迁移，但不能把跨引擎结果直接当成同引擎、同分布复测。这一限制不通过更名或相似 Demo 消除。（E3 工程判断）

## 2. 已做的实质核查

### 2.1 固定的来源

| 对象 | 固定身份 | 核查说明 |
|---|---|---|
| GPT-as-Policy | `8f3d362b077d8efb77e2a7274d5b2c20e2243846` | 已读 Direct/Hybrid、checkpoint、RPC、IK、评测 manifest 与中文报告正文 |
| RoboDojo 原生环境 | `ee67a1468510da7624a089164402359f2afc72c8` | 与评测 panel 所记 commit 相同 |
| XPolicyLab | `432f82b1758c5b1202e42a3dfe014546dbc50871` | RoboDojo 此版本的 submodule；含 `policy/Pi_05/openpi` |
| RoboDojo 资产仓库 | Hugging Face revision `91f76c28d93dd20c5fa46ce6a5a1d96a4f384acd` | 本轮核对布局与公开资产/权重目录 |

`SOURCE.json` 把 `432f82b...` 写在 `openpi_commit` 字段，但该哈希实际也对应 RoboDojo 固定的 XPolicyLab submodule；不能直接把它当作 Physical-Intelligence/openpi 上游 commit。已在该 XPolicyLab 版本的 `policy/Pi_05/openpi/src/openpi/training/config.py` 找到原 `pi05_base_aloha_full_sim_arx-x5_seed_0` 配置及 `arx_x5_sim`。（E1/E2，S6、S15、S16）

### 2.2 输入一致性结果

| 核查项 | 本轮结果 | 能证明什么 / 不能证明什么 |
|---|---|---|
| 原 50 个评测布局 JSON | **下载并逐文件计算 SHA256，50/50 与原 panel 一致** | 能锁定场景布局输入；不能证明跨引擎落地后的像素、接触状态相同 |
| 相关原生源码/配置 | **抽取 75 个文件，75/75 与 panel 一致** | 包括环境、机器人、控制、奖励、相机与所选任务；不是所有源码/资产的全量审计 |
| 支持演示/交互轨迹 | **9/9 远端 LFS SHA256 与 panel 一致** | 包括排序 5 条、麻将 4 条；仅比对元数据，尚未下载并验证文件内容 |
| 原 π₀.₅ 权重 | 找到 `Pi_05/RoboDojo-sim-arx_x5-joint-0/59999` | params 元数据为 16 文件、12,440,988,569 字节；params + assets 共 17 文件的聚合身份已与原记录一致，尚未下载大文件内容或加载 |
| 原归一化 | 找到 `59999/assets/arx_x5_sim/norm_stats.json` | 纳入 17 文件聚合身份；不能据此宣称 NPU 推理等价 |
| 原机器人 | 找到 `Assets/Robots/x5/X5A.urdf`、`ARX.usd`、meshes/configuration | 可以沿用原 ARX X5 来源；USD/URDF 物理属性一致性和 mesh 转换未验收 |

上述核查均为 E2；详细可追溯记录见[输入一致性核查 JSON](2026-09-17-GPT-as-Policy输入一致性核查.json)。17 文件的聚合哈希为 `d15fb8bd1d29cb30b69f01b71c66596cb0293c1a8a94111b343c1580dd3e3e5b`，与原 `previous_load_sha256` 一致，计算依据为大文件远端 LFS SHA256 元数据和小文件实际哈希，见[权重身份元数据核查](2026-09-17-GPT-as-Policy权重身份元数据核查.json)。本轮未下载十余 GB 权重或完整大型资产，未运行上游脚本或反序列化轨迹；75 个源文件仅为原清单 169 个文件的子集。

**结论：目前没有证据要求换模型、换机器人或换任务。** 原链路的关键输入有公开来源；当前困难应按迁移工程解决，不能提前降级为类似实验。（E3，依据上述 E1/E2）

## 3. 必须先说明的限制和未知项

| 项目 | 当前能做的事 | 当前不能承诺的事 |
|---|---|---|
| Isaac 原程序 | 逐模块抽取依赖，移植到 MuJoCo 接口 | 原 Isaac Sim/RTX 链路不能直接在纯 NPU 机器原样启动 |
| 接触动力学 | 读取原惯性、碰撞、驱动、摩擦和求解参数，逐项对齐 | 两种求解器不能保证轨迹逐步一致 |
| 渲染 | 保留原 mesh/纹理/相机内外参，CPU OSMesa 生成图像 | OSMesa 不能复现 RTX 材质/光照的像素等价；这会影响 VLA 分布 |
| 叠衣服 | 原任务源码与衣物关键点判据已定位；评估保留原网格/关键点的 MuJoCo 柔性表示 | 原 `SingleClothPrim` / 粒子物理不能直接移植；同等抓持、弯折、自碰撞与判据保真度未验证 |
| NPU π₀.₅ | 原权重、原配置→兼容的 PyTorch 实现→torch_npu；做分层数值核验 | 昇腾示例跑通不代表此 checkpoint 等价，更不能假设改一行 device 即可 |
| 原模型访问 | 用户可在 Mac 调用已授权模型服务；保持同一 agent 回合语义 | 原模型版本、xhigh、完整工具能力和成本/限流未在目标服务验证 |
| 跨引擎证据 | 公开源码、布局、演示数据和作者视频可作为参考 | 仓库不含原始模型会话/全量实验轨迹；只有 CPU/NPU 时无法现场生成原 Isaac 对照轨迹 |
| 远程执行 | 本地可做源码与资产检查、接口实现和轻量验证 | 本轮只交付计划，不要求提供连接；实际执行者在 W0 配置授权工作目录与运行环境 |

其中 cloth 的 Isaac 依赖为 E1（`env/scene_manager/objects/garment.py:1–24`）；其余引擎差异是 E3 工程判断；访问/远程状态为当前工作状态。

这些项分别标为“已知不能原样保留”“可迁移但待验收”“缺少外部条件”，不把所有未知项笼统写成不可做。任何一项导致改变实验定义，应先报告具体缺口、已尝试方法和残留差异，再决定后续，不自行换题。

## 4. 完整链路的部署与保留边界

| 层 | 实际职责 | 部署与修改原则 |
|---|---|---|
| 原 agent | 看图、计算、维护笔记和上下文、选择动作 | Mac 调用模型；保留上游 skill、gate、上下文结构和动作决策权 |
| 原 orchestration | start→infer→review→execute→observe、错误处理及留痕 | 以原模块为主体，只改传输/运行环境边界；不新写简化的单轮 VLM 循环 |
| π₀.₅ | 当前观测→50×14 候选 | 原 RoboDojo checkpoint、normalizer、预处理；NPU 单卡先验收，多卡用于独立运行扩展 |
| robot-only FK / DLS IK | 候选关节目标→末端预览；EEF 修正→局部关节目标 | 原 NumPy/SciPy 算法已存在，优先直接复用；替换其读取原生 env 的接口 |
| 仿真执行器 | 双臂关节/连续夹爪控制、物理、三路 RGB | CPU MuJoCo；原 ARX X5 几何、驱动目标和布局；软件渲染单独检查 |
| 任务与评分 | 指令、示范、触发状态、阶段分、最终成功/超时 | 移植原 task/reward 逻辑与状态接口，不重写为粗略距离判断 |
| 评测与结果 | 固定 case、版本、动作来源、录像、指标和用量 | 继承原 manifest/配对逻辑；迁移输入另存新身份，不篡改原哈希 |

原 RoboDojo 契约：头部+双腕 RGB、14 维本体信息、两臂 link6 实测位姿；动作坐标为 environment origin、米、单位 wxyz；夹爪学生值 0=闭/1=开。Direct 每段 1–5 控制步；Hybrid Student 前缀 1–15 步、局部修正 1–5 步；EEF 界限 5 cm / 0.35 rad；控制 25 Hz、物理 250 Hz，每个控制步 10 个物理子步；排序和麻将另有原 Franka 支持臂，其动作不并入策略的 14 维输出。（E1，S3–S8、S21–S23）

Hybrid 必须逐段取得新 π₀.₅ 候选；GPT 分开判断上一段执行结果和下一段意图。只有观察到失败或意图偏离才接管，仅不确定不能接管；恢复后交还。FK 仅提供机器人运动学，不允许未来物体物理/成功预览。（E1，S4、S8）

Mac 返回的动作由服务器 CPU 上的仿真控制器执行；NPU 只负责学生模型推理。模型推理和网络等待期间暂停物理步进，这与作者报告的可暂停仿真假设一致；墙钟与仿真秒分开统计。（E3 报告与部署设计，S2）

传输至少保证 episode/step/request 身份、过期候选拒绝、重复请求返回已执行 ACK、执行状态不明时不重复执行；服务器图像须实际传到 Mac，不能假设两台机器绝对路径通用。（E3 设计）

## 5. 原十任务逐项迁移范围

下表任务、控制步预算及标准/随机分配来自原公开 panel/结果（E2）。迁移关注点是据源码提出的 E3 工作项；不是新增任务。

| 原任务 | 控制步预算 | 必须保留的内容 |
|---|---:|---|
| organize_table / 整理桌面 | 1000 | 原物品与指定位置、抽屉关节、阶段完成、双臂收尾 |
| classify_objects_by_language / 按语言分类 | 1100 | 原类别—容器指令映射、对象身份、全部物体判定 |
| imitate_sorting_sequence / 模仿排序 | 1600 | 原演示轨迹、演示观察顺序、记忆与执行顺序状态机 |
| arrange_largest_number / 最大数字 | 1050 | 原数字纹理/对象、排列逻辑、朝向/位置和随机场景 |
| pack_objects_into_box / 装箱 | 1300 | 原箱体碰撞、各物体几何、朝向条件和随机场景 |
| classify_objects / 物体分类 | 1100 | 原对象类别、三个篮子、完整分类及收尾 |
| build_tower / 搭塔 | 1050 | 原结构件、搭建关系、接触与稳定性条件 |
| make_kong / 麻将杠牌 | 600 | 原麻将标记、支持臂轨迹、交互触发与序列判据 |
| fold_clothes / 叠衣服 | 500 | 原衣物拓扑/标记点、柔性接触、自碰撞、折叠与放手/归位事件 |
| put_bottles_into_dustbin / 瓶子入桶 | 700 | 原瓶/桶形状与碰撞、完整入桶、任务收尾 |

每任务 5 个原 case；数字、装箱、叠衣服为标准 2+随机 3，其余标准 5。不能用同 seed 的新随机摆放替代已核对的布局 JSON。（E2，S9、S17）

不能只导入 X5 URDF：原 Isaac X5 配置额外规定驱动 stiffness/damping、关节速度/力矩限制、armature、重力与自碰撞开关；原控制器还包含夹爪渐进开合语义。它们必须进入迁移清单。（E1，原 `env/robot_manager/robot_config/x5.py:9–62`、`control_manager.py:22–41`）

RoboLab 补充实验是另一套契约：Franka / DROID，原公开 checkpoint 配置为 `pi05_droid_jointpos`，内部标识含 `pi05_droid_jointpos_polaris`，horizon=15、action_dim=8；不能沿用 RoboDojo 的 50×14 假装完成。已读该集成源码，尚未核对其全部原始 case/资产/Direct 入口，列为独立待办。（E1，`hybrid_rollout/robolab/pi05_server/checkpoint.py:8–23`；当前核查边界）

## 6. 详细执行计划

### 6.1 用户已确定的边界与交付数量

本轮已确认：**只给出计划，无需现在提供服务器信息；Codex 登录/app-server 与 API／网关两条路线都要；预算暂不设上限。** 因此没有必须继续向用户确认的问题。8 卡 910B 和“64G”作为资源描述，单卡容量、CPU 架构、驱动/CANN 版本由执行者在 W0 实测，不能据未核实配置承诺吞吐。（用户约束；容量解释待核实）

两条接入路线分别完成 Direct 与 Hybrid：RoboDojo 每条路线 50 Direct + 50 Hybrid，共 **200 个正式 episode**；RoboLab 在原始输入和实现缺口闭合后采用相同规模，另 200 个。两套均完成且两接入路线可验证时，主要交付共 **400 个正式 episode**。这只是计划样本数，不包含开发诊断和失败尝试，也不是已承诺能够获取全部外部条件。Student-only 每套 50 个、共最多 100 个，作为新增配对基线单列，不因模型接入方式重复运行。

原报告主复现以 app-server 路线为主；API 路线单列为“agent 接入迁移实验”。即便使用相同模型名，两种服务的隐藏提示、版本、推理预算和工具调度也可能不同，不能默认可合并统计。若原模型或关键工具无法取得，保留该路线为未完成并写出差异；不自行换模型后称原实验复现。

不设费用硬上限不等于无限重试：完整保存用量、请求和失败原因，依照 §8 的重试规则执行。本轮不启动收费调用。

### 6.2 最终工程的目录与边界（拟建，不是现有文件）

计划将自主适配代码放在 `embodied-ai/gpt-as-policy-repro/`；大型上游仓库、资产、权重和运行数据放到执行机器的独立数据目录，公开仓库只保留固定来源、哈希清单、适配代码和脱敏结果。旧 `gpt6astra-repro/` 的 bowl 实验不作为实现基础；只可提取经验证的通用日志/图像传输工具，不继承其任务预算或控制协议。

| 拟建路径（相对新工程根） | 单一职责 | 原实现的保留位置 |
|---|---|---|
| `configs/upstream.lock.json`、`configs/runtime.lock.json` | 原输入身份、实测运行版本；原始与迁移身份分别保存 | §2 固定来源，原 panel 只读 |
| `configs/campaigns/*.json`、`configs/validation.json` | 接入路线、方法、案例、统计协议及冻结验收阈值 | 不修改原 panel 伪造哈希匹配 |
| `src/gap_repro/inputs.py`、`assets.py` | 内容哈希、引用闭包、USD/URDF/MJCF 转换审计 | 每项资产回指原文件与 prim/link |
| `src/gap_repro/sim/environment.py`、`robots.py`、`control.py` | MuJoCo 状态接口、机器人映射、子步调度 | 对接原 session、FK/DLS 读取接口 |
| `src/gap_repro/sim/observations.py`、`support.py`、`cloth.py` | 三相机、第三臂支持轨迹、原衣物表示 | 保留原相机和支持事件语义 |
| `src/gap_repro/tasks/registry.py`、`state_adapter.py` | 十任务入口与原判据需要的状态查询 | 原 task/reward/query 尽可能直接复用；必要补丁集中保存 |
| `src/gap_repro/policy/convert.py`、`noise.py`、`npu.py` | 权重转换、原 JAX 随机流、NPU 推理 | 原预处理/解码、normalizer 和去噪配置 |
| `src/gap_repro/agent/app_server.py`、`api_runner.py` | 两种模型传输与持久上下文 | 原 skill、gate、工具 schema 与 orchestration |
| `src/gap_repro/transport.py` | 远程图像和动作身份、去重与执行回执 | 原 RPC 语义；不承担任务规划 |
| `src/gap_repro/evaluation.py`、`results.py` | 冻结 campaign、attempt 账本与汇总 | 原 session 原始结果字段不覆盖 |
| `tests/`、`patches/`、`docs/acceptance/` | 契约测试、最小上游补丁与验收证据 | 每一差异有来源、理由和测试 |

跨模块的数据契约先固定如下，字段命名是适配设计，不声称原包已有这些新类型：

| 接口 | 输入 → 输出 | 必须满足的不变量 |
|---|---|---|
| 环境 `reset(case_identity)` / `observe()` | 冻结 case → 三相机、14D state、双 link6 EEF、step_id | 原 reset 只允许在 episode 开始；观测不推进物理 |
| 学生 `infer(observation, request_id)` | 当前观测 → 50×14 候选、inference_index、noise_hash | 同 request 幂等；原指令与实际噪声身份可追溯 |
| 执行 `execute(episode_id, step_id, request_id, actions)` | 合法前缀/EEF 转换动作 → 实际 ACK 列表 | 每 ACK 一个控制步；原子记录已执行状态，拒绝陈旧 step |
| agent `run_episode(case, method, route)` | 原任务输入/公开工具 → 工具调用和终止请求 | episode 内持续上下文，方法间隔离；无评分真值 |
| 汇总 `summarize(campaign_manifest, attempt_ledger)` | 固定登记表+所有尝试 → 50 行结果及配对统计 | 无最佳 attempt 选择、无替换 case、无缺分补零 |

这些模块名称是执行约定，可在实现前按真实上游依赖收敛；不得另写一套简化 agent 或评分器来适配自己设计的接口。依赖安装隔离在专用环境，不升级整台服务器。开发代码每个验收门单独 commit，记录未通过项；文档版本与源码版本在 campaign 中关联。

### 6.3 工作包与验收门

下列清单及建议测试名均为 **E3 实施设计**。测试命令须在相应实现和 fixture 建立后运行；本文没有把未来命令当成已执行证据。每包按“原始契约/反例→实现→独立验收→保存证据与 commit”推进。

#### W0：运行条件与两种接入的能力清单

**依赖：** 无。**文件：** `configs/runtime.lock.json`、`docs/acceptance/runtime.md`。

- [ ] 记录 CPU 架构、内存、磁盘、实际卡数/单卡容量、驱动、固件、CANN、Python、PyTorch/torch_npu 和可用 CPU 渲染后端；选择官方支持的版本组合并锁定依赖。
- [ ] 验证 CPU MuJoCo + OSMesa 在目标机产生有效 RGB、JAX CPU 可运行原 PRNG；识别 ARM/x86 wheel 或构建依赖问题。
- [ ] 两种模型接入分别核验实际模型身份、xhigh 参数、图像输入、持久会话、工具调用、用量字段、限流/错误行为；只记录非敏感配置，不读取或发布凭据。
- [ ] 输出“已满足/可实现/外部缺失”的能力表。仅有 CLI 或 HTTP 200 不算模型链路通过；付费验证留到实际执行阶段。

**通过条件：** 明确可运行的依赖组合与两路线差异。外部服务不可用时不阻塞资产/任务移植，但 W9/W11 相关路线不能验收。

#### W1：原始输入全量锁定与资产引用闭包

**依赖：** §2 已有证据；目标下载环境。**文件：** `inputs.py`、`assets.py`、`configs/upstream.lock.json`、`tests/test_input_identity.py`。

- [ ] 固定三份源码及 submodule，补齐原 panel 的 169 个源码文件检查；区分明确不使用项与必须依赖项，不能把 75/75 写成全量通过。
- [ ] 实际下载 checkpoint 的 params/assets 全部 17 文件和 9 条支持轨迹，逐字节验 SHA256，并重算 checkpoint 聚合身份；不只读取 LFS 指针。
- [ ] 从原 50 布局、机器人配置与任务递归解析 USD 引用、纹理、mesh、碰撞、衣物文件和支持臂配置，输出“引用→实际文件→哈希→使用 case”表；下载完整必要闭包。
- [ ] 保存原 50 case 的 task、runtime task name、layout、seed、预算、指令来源和支持轨迹对应关系；标准/随机配置不靠文件名猜测。
- [ ] 在 provenance 确认后按原格式读取轨迹，检查字段/长度/dtype；记录缺项，禁止替换成近似 mesh 或录制新演示冒充原输入。

**通过条件：** 所有原 case 所需依赖可定位且内容验算通过。缺一项则标记受影响 case，不能开始其正式评测。测试须包含缺文件、内容变动、LFS 指针误当资产、错误 revision 四类反例。

#### W2：双 X5、第三 Franka 与场景资产转换

**依赖：** W1 对应资产闭包。**文件：** `assets.py`、`sim/robots.py`、`tests/test_asset_mapping.py`、`tests/test_kinematics.py`。

- [ ] 逐 link/prim 映射视觉网格、碰撞体、质量、惯性、关节轴/限位、驱动和坐标变换；生成 MJCF 及来源表。USD 包含的物理信息不能仅通过 URDF 导入就假定齐全。
- [ ] 双 X5 的 6+1 / 6+1 策略顺序、左右根坐标、link6、夹爪 mimic 和 0闭/1开映射逐项检查；第三 Franka 保持原初态和支持身份。
- [ ] 映射 stiffness、damping、effort、velocity、armature、gravity/self-collision；将“原值、目标表示、是否等价、已验证方式”写入表，不把 PhysX 参数机械复制后宣称物理一致。
- [ ] 用原 `DualKinematics` 对原布局初态、关节边界附近及固定可达姿态验证 FK；位置误差 <2 mm、角度误差 <0.01 rad，沿用原 check 的门槛。（E1，S27）

**通过条件：** 所有必需刚体/机器人属性有映射或显式缺口，FK 达标、夹爪和关节方向正确。姿态通过不能代替接触动力学验收；掉落、抓持和抽屉摩擦在 W5 另验。

#### W3：原衣物和 CPU 渲染的高风险验证

**依赖：** W1；与 W2/W6 可独立推进。**文件：** `sim/cloth.py`、`sim/observations.py`、`tests/test_cloth_contract.py`、`tests/test_cameras.py`。

- [ ] 提取原衣物网格、关键点身份、材料/粒子属性和评分依赖；评估 MuJoCo flex 表示，保存顶点/关键点一一映射以及无法直接映射的参数。
- [ ] 在最终原场景中验证重力落布、桌面接触、抓持不穿透、提拉、双层折叠、自碰撞与放手；用相同初态保存视频和关键点轨迹。测试控制用于诊断，不作为正式策略。
- [ ] 三相机保留内外参、原尺寸/裁剪、RGB 顺序和图像方向；学生输入原 640×480，GPT 缩略图遵循原流程且原图可取，不自行降低学生图像质量。
- [ ] 输出原材质/光照与软件渲染差异表；用几何投影、可见性、纹理身份检查可确认的量，作者视频只能作定性依据，不能用它声称像素误差达标。

**通过条件：** 衣物关键点与原评分可计算，所需物理行为真实存在；相机几何与预处理契约通过。缺少 Isaac 参考时动态等价保持“未证明”。无法支持原衣物行为就将全任务交付标为受阻，继续其他包，不转为刚性衣物或删除此任务。

#### W4：250 Hz 子步、25 Hz 观测与支持臂时序（关闭 F1 的规格缺口）

**依赖：** W2。**文件：** `sim/control.py`、`sim/support.py`、`tests/test_substep_schedule.py`。

- [ ] 每次原 25 Hz ACK 对应恰好 10 个 `dt=0.004 s` 物理步。主臂目标前 8 子步使用 `alpha=(i+1)/9`，即 1/9…8/9，后 2 子步保持目标；不能改成 i/8 或一次瞬移。（E1，S23）
- [ ] 夹爪按原主标量插值、原开合范围裁剪后映射 mimic；核对 MetaControl 的渐进夹爪限制在原调用栈哪层应用，避免适配层重复限幅。
- [ ] 支持臂轨迹含关节/夹爪位置和速度，每物理子步最多消耗一项；对队列长度 0、1、9、10、11 的动作字段和消费序列逐项与原代码比对，保留队列耗尽后的原保持/缺省行为。
- [ ] 验证排序演示结束与麻将交互触发的时刻、支持演示失败的 unstable 标志；reset 沉降不计策略步数，但物理子步另记。
- [ ] 保存一条控制命令的全部 10 个目标和实测状态，再验证连续两条命令的计数、观测与支持臂消费连续性。

**通过条件：** 子步计数、插值、队列、触发时序完全符合原逻辑；Franka 不进入 GPT 14 维动作空间。这里只关闭规格问题，运行通过后才关闭 F1 实施验收。

#### W5：原 session 接口和十任务评分状态机

**依赖：** W2–W4；衣物任务需 W3。**文件：** `sim/environment.py`、`tasks/registry.py`、`tasks/state_adapter.py`、`tests/test_task_contracts.py`。

- [ ] 以原 `RoboDojoSession` 实际调用为边界实现 `reset`、`take_action`、robot/scene/obs/reward manager 查询、`end_flag`、`step_lim`、`unstable_envs`；先做调用清单，不模拟完整 Isaac API。
- [ ] 保留原 `register_native_evaluation`、score/success/trigger 状态机；启动即检查奖励清单非空，防止“空判据全成功”。支持臂的交互查询也必须注册。（E1，S25）
- [ ] 十任务分别建立至少一条应成功与一条应失败的状态/事件序列；覆盖错类别、错数字序列、演示失效、物体落箱外、塔未稳定、麻将顺序错、衣物未折叠、未归位等反例。
- [ ] 将原判据和适配后的同一判据作用于冻结状态 fixture，对比逐事件触发、阶段分、终局与超时。若原模块无法离开 Isaac 导入，提取纯判据并保存最小补丁；该比对只能证明逻辑保留，不能冒称 Isaac 动态对照。
- [ ] 在 50 原布局逐一 reset，检查初态、指令、对象身份、观测、预算和状态机初始化；保存 RGB/本体/EEF 指纹及更完整的内部物理状态摘要，用于检测漂移。

**通过条件：** 每任务有独立验收记录，全部原布局能初始化，原判据与适配状态接口一致。接触/柔性行为另列实测，不把人工构造成功状态计入策略成功率。

#### W6：原 checkpoint 的 CPU 参考与转换

**依赖：** W1 权重；W0 参考运行环境。**文件：** `policy/convert.py`、`tests/test_policy_reference.py`、`docs/acceptance/policy-reference.json`。

- [ ] 从固定 XPolicyLab 的实际配置恢复 π₀.₅ 架构、tokenizer、normalizer、图像变换、state 拼接、action padding、delta/absolute 解码、horizon 和去噪步数。
- [ ] 逐张量记录原参数名、目标参数名、shape、dtype、转置/切分规则及缺失/多余键；不得忽略加载异常或加载 base 权重补齐。
- [ ] 用真实原任务观测、显式相同噪声比较 JAX CPU 与 PyTorch CPU 的关键中间量、未解码动作和最终 50×14 动作；原始观测不足时先用迁移原任务观测，并注明输入域限制。
- [ ] 预注册 `validation.json` 的阈值后才做验收：用同参考实现重复运行/精度变化的校准 fixture 确定数值噪声范围，并由原动作分辨率、FK 门槛约束控制误差；校准与验收 fixture 分离，不能看失败结果后放宽。

**通过条件：** 参数映射无未解释缺项，无 NaN/Inf，按维关节/夹爪和 FK 误差均达到冻结门槛。CPU 内存或版本依赖不足时记录受阻层级，不跳过参考就宣称转换等价。阶段不安排重新训练或量化替代。

#### W7：正式 JAX 随机流与 NPU 数值验收（关闭 F3 的规格缺口）

**依赖：** W6。**文件：** `policy/noise.py`、`policy/npu.py`、`tests/test_policy_rng.py`、`tests/test_npu_equivalence.py`。

- [ ] CPU 使用固定原 JAX 版本、PRNG 配置、seed0、原 key 分裂次序和 dtype，生成真实噪声后注入 Torch/NPU。shape 从模型内部动作维度读取，不能直接填解码后的 14 维。（E1，S26）
- [ ] 每个 episode 重置随机状态，记录 `inference_index`、key 前后状态、noise shape/hash、observation hash、checkpoint ID；前若干次噪声与原代码生成值逐字节对比。
- [ ] 预热使用独立实例/随机状态；重复同一请求返回缓存，不能多消耗随机流；推理前校验拒绝不推进，推理已发生而执行被拒绝则保留已消耗的 index，不回退随机状态。
- [ ] 同观测/噪声做 PyTorch CPU→单卡 NPU eager 验收，沿用 W6 冻结门槛，保存按维误差、夹爪、FK 偏差和异常。不可支持的算子逐项适配；如需 CPU fallback，披露位置和耗时。
- [ ] 正确性通过后再考虑编译、混合精度和独立 episode 多卡分配，每次优化都重新过相同数值门；不默认 8 卡张量并行更合适。

**通过条件：** RNG 身份、噪声注入和数值误差均有证据。只用 `torch.manual_seed(0)` 不通过。无法在同门槛下兼容时保留“原 checkpoint 已定位但 NPU 等价未完成”，不替换模型。

#### W8：远程 RPC、图像传输与故障恢复

**依赖：** W5；学生 RPC 依赖 W7。**文件：** `transport.py`、`tests/test_transport_recovery.py`。

- [ ] 统一请求身份 `{campaign_id, episode_id, request_id, step_id, control_epoch}`；候选另关联 `observation_hash, inference_index`。服务端返回实际执行后的 ACK 和计数。
- [ ] 图像按内容哈希实际传输到 Mac，验证解码后一致；本地路径不作为跨机可读的假设，原图和 agent 缩略图均能追溯。
- [ ] 动作请求写入账本；已执行请求重复到达只返回原 ACK，过期候选拒绝。对执行中断而无法证明执行次数的状态标为 poisoned/incomplete，禁止盲重放。
- [ ] 故障注入覆盖 ACK 丢失、重复包、陈旧 step、缺图、推理超时、进程退出；网络等待期间物理暂停，重连不能增加控制步。

**通过条件：** 故障测试中无重复动作、无隐藏 reset、无错误关联图像。只有账本能证明的状态可恢复；不能承诺在进程崩溃后凭日志完全恢复物理状态。

#### W9：两种原模型接入与三条策略入口

**依赖：** W8；Hybrid/Student 依赖 W7。**文件：** `agent/app_server.py`、`agent/api_runner.py`、`tests/test_agent_contract.py`。

- [ ] **路线 A：** 复用原 `skill/run.py`、transport 和持久 Codex app-server；只改部署/通信边界，保留模型、推理档位、原提示词、工具 schema、图像和 episode 内笔记。（E1，S28）
- [ ] **路线 B：** API／网关适配同一 tool loop，保留完整消息/工具调用结果、文件/图像读取、计算工具、笔记和上下文策略；网关没有执行工具的能力时由 Mac runner 执行，不把模型文本中的“已执行”当成工具结果。
- [ ] 两路线分别实现 Direct 原 EEF 工具与 Hybrid 的 start→infer→review→execute→observe；Student-only 复用相同观测和执行器，明确它是新增对照。
- [ ] 工具可见范围只含当前 episode 的 RGB、本体、原指令和历史；agent 的 shell/文件能力不能读到评测真值、布局答案、未来支持轨迹或其他方法结果。原工具能力中无法安全隔离的部分须记录策略差异，不静默改动。
- [ ] Hybrid 每次执行后获取新观测、新候选；保留“上一段结果/下一段意图”分开评估和原接管 gate。学生始终收到原任务指令，不把 GPT 子目标替换进学生输入。
- [ ] 用固定工具往返 fixture 验证两路线 schema、上下文持续性、错误反馈和图像可读；再在原任务中做少量完整开发 episode，检查真实模型链路。开发 episode 留痕且永不进入正式统计。

**通过条件：** 两路线各有能力矩阵、真实工具轨迹与差异说明；路由同名不等于模型同版。允许同一输入产生不同回答，不要求生成文本逐字相同；要求决策输入、工具权限、动作边界和记忆机制可审计。原模型访问缺失不能由替代模型的成功抵消。

#### W10：冻结验收结果和正式评测协议（关闭 F2 的规格缺口）

**依赖：** W1–W9 对应门全部通过。**文件：** `evaluation.py`、`results.py`、`configs/campaigns/*.json`、`tests/test_result_accounting.py`。

- [ ] 将所有已验收源码、上游补丁、资产、提示词、模型接入身份、checkpoint、RNG、控制和评分配置写入不可变 campaign manifest；原与迁移 manifest 并存。
- [ ] 按 §8 的状态与重试规则生成失败 fixture，检验无效布局不变成普通失败、缺分不补零、不选最佳 attempt、病例不被替换、方法间配对不串行污染。
- [ ] 冻结两路线内相同的 case 顺序与方法交错调度表，每个 episode 独立上下文/RNG；记录硬件/服务差异，避免把不同版本或重跑混在一张主表。
- [ ] 保存所有任务的 W2–W9 验收链接和剩余动态/视觉差异；若任一必需任务未通过，主结果只能标记部分完成，不能发布完整十任务结论。

**通过条件：** 汇总器在构造的成功/失败/无效/缺失集合中产出预期分母、空值和配对数量；冻结文件有哈希和时间。此时才开始正式计分运行。

#### W11：RoboDojo 正式全任务运行与复核

**依赖：** W10。**文件：** campaign 数据目录；仓库只存脱敏 `docs/acceptance/robodojo-results.md` 和结果索引。

- [ ] 两种接入分别完成原 50 Direct + 50 Hybrid，记录全部 attempt；可扩展到多卡独立 worker，但每个 worker 的物理状态、随机状态、agent 上下文互相隔离。
- [ ] Student-only 50 个补充对照使用同一原 case 与迁移环境，单列结果，不冒充作者公布的参考策略同 seed 重跑。
- [ ] 自动汇总后逐项复核异常终止、接管/交还、成功判据和视频是否一致；发现评分 bug 时冻结旧 campaign，修复后建立新 campaign，不只重跑失败 case。
- [ ] 同时输出成功率、阶段分可用数量、逐任务配对、动作来源/控制步、接管次数、token/费用、CPU/NPU/渲染/网络/模型耗时及全部差异。

**通过条件：** 每个原 case 的最终状态和证据可追溯；没有未解释缺项才能称该路线完整运行。达到作者的 48%/26% 不是验收条件，结果更低仍应原样报告。

#### W12：RoboLab 完整性审计、迁移与第二套评测

**依赖：** 输入审计可从 W1 起推进；实际迁移复用已验收公共设施。

- [ ] 先锁定原 RoboLab 源码、Franka/DROID 资产、checkpoint、10 任务×5 case 身份、指令、预算、评分、Direct 和 Hybrid 入口；已查 checkpoint 是 `pi05_droid_jointpos`、H=15、action_dim=8，禁止复用 50×14 契约。
- [ ] 查清原 Direct 入口是否公开及案例身份能否取得。若只有 Hybrid 源码，列出缺失文件/行为；依据公开契约重建时标记“重建实现”，不能声称保留了未取得的原代码。
- [ ] 建独立 `configs/robolab/` 与 `src/gap_repro/robolab/`，将 W1–W10 验收按该源契约重新执行，复用传输/日志时仍验证接口差异。
- [ ] 可验收后两路线分别运行 Direct/Hybrid 各 50 个，再单列 Student-only 50；不得和 RoboDojo 合并成一个总成功率。

**通过条件：** 第二套原始输入、实现和结果完整可追溯。若公开材料不能支持高保真重建，交付缺口清单和已完成部分，明确整份报告尚未完整复现，不用新造 50 case 补齐。

### 6.4 依赖顺序、先做什么与工期

第一轮执行直接处理最终工程：W0 条件清单 → W1 固定输入；随后优先推进 W3 原衣物/渲染、W6–W7 原权重等价，以及 W12 的公开材料缺口审计。这三项最早决定完整复现的可达边界。W2–W5 建成真实原任务环境，W8–W9 连接完整 agent，W10 冻结，W11/W12 正式运行。

独立工作可以并行安排，但不要求另建 Demo，也不通过简化任务拿“阶段成功”。每轮失败记录“假设、证据、单项修改、验证结果、下一步”；软件实现问题持续修复，外部输入缺失、无原模型权限或缺少对照证据则明确披露，同时推进不依赖它的工作。

目前不承诺总周数。W1/W3/W7 首轮结果出来后，依据资产转换数量、真实单次推理/渲染/agent 耗时重估。计划运行量由 §6.1 固定；并发数根据 CPU 渲染吞吐、NPU 内存和服务限流实测决定，不能用“8 卡”直接除总耗时。

### 6.5 执行者的验收命令约定

以下命令在拟建工程根目录和锁定依赖环境中运行；**测试文件尚待 W1–W10 创建，现在不能作为可用脚本交付**。每个测试应断言该工作包描述的行为和反例，不以进程退出码或“能动起来”替代验收。

```bash
python -m pytest tests/test_input_identity.py tests/test_asset_mapping.py
python -m pytest tests/test_kinematics.py tests/test_substep_schedule.py
python -m pytest tests/test_cloth_contract.py tests/test_cameras.py tests/test_task_contracts.py
python -m pytest tests/test_policy_reference.py tests/test_policy_rng.py
python -m pytest tests/test_npu_equivalence.py
python -m pytest tests/test_transport_recovery.py tests/test_agent_contract.py tests/test_result_accounting.py
```

前四组需要固定资产/权重与 CPU 条件，第五组必须在真实 NPU 上验收；第六组的固定往返/故障 fixture 可离线跑，真实模型接入另存真实调用记录。环境缺失时标记 blocked，不得以全部 skip 的绿色退出码算通过。正式运行入口在实现后生成可运行命令及配置哈希，本文不提供虚构的现成 CLI。

## 7. 固定契约与三项审查的收敛结果

| 审查项 | 已合入的规格 | 实际关闭条件 | 当前状态 |
|---|---|---|---|
| F1 支持臂与子步 | W1/W2 纳入第三 Franka；W4 明确 250 Hz、10 子步、前 8 步 1/9…8/9、轨迹队列与夹爪过程 | 原轨迹/子步测试与支持演示有效性通过 | 规格已合入，运行未验证 |
| F2 异常和重试 | §8 固定状态、attempt 选取、分母、缺分、重试与 campaign 更新 | W10 汇总/故障测试通过，正式日志遵守协议 | 规格已合入，运行未验证 |
| F3 正式随机流 | W7 的 CPU 原 JAX PRNG、内部 shape、噪声注入、index/哈希与重试规则 | 原噪声序列匹配，CPU/NPU 数值验收通过 | 规格已合入，运行未验证 |

此外，原 FK/DLS 保留两层动作约束：决策目标最大 5 cm / 0.35 rad；每个执行 ACK 的 DLS 局部推进最大 2 cm / 0.1 rad、单关节最大 0.05 rad。读取 MuJoCo 状态时必须保持 environment-origin 与单位 wxyz 的约定，不能将目标约束和每步约束混为一次限幅。（E1，S7、S27）

原 14 维动作顺序为 `[left_joint6, left_opening, right_joint6, right_opening]`。Hybrid 学生输出 50×14，只执行原允许的 1–15 步前缀；Direct/修正段为 1–5 步。完整原始候选、真正下发动作和实际状态分别保存，FK 预览只推机器人，不调用物体未来仿真。（E1，S3–S8、S27）

## 8. 正式运行的异常、重试与统计规则

以下是本次拟冻结的 **E3 评测协议**；原始状态字段和空值语义依据 E1 的 S24–S25。它比“保留所有 attempt”更具体，但不是声称作者完全采用了这些新规则。

### 8.1 attempt 选取与运行恢复

1. **开发与正式隔离。** W10 前均为开发记录；正式 campaign 冻结后每个 `(case, method, route)` 有唯一预定试验。全部尝试保留，不能按成功优先选取。
2. **只允许明确未开始的初始化重试。** 在首个观测交付策略、首个模型决策、首个策略动作三者均未发生，且错误证明没有策略执行的情况下，基础设施初始化最多额外重试 2 次，共 3 次。最终选择首个进入策略阶段的 attempt；均未进入则记基础设施未完成。原布局 invalid 不适用此重试。
3. **进入策略阶段后不自动重置整集重跑。** 任务失败、支持演示 invalid、模型预算耗尽均不得靠重跑挑好结果。模型输入格式错误可按原工具协议返回错误并让同一 agent 更正，不暗中换策略。
4. **请求恢复和 episode 重跑分开。** 同 request 的可证明幂等重取不增加动作/RNG；执行次数不明就终止为 incomplete。服务限流可在同会话按服务重试提示等待，物理暂停，等待和用量写日志；进程状态丢失不能当作无损续跑。
5. **修改实现建立新 campaign。** 修复物理/评分/模型或提示词后不混用旧成功。需要形成新的完整对照时，在新版本上重跑所比较方法的整个固定 panel；旧结果保留。

### 8.2 终局状态与数值字段

| 汇总状态 | 触发 | success / native score | 分母处理 |
|---|---|---|---|
| `valid_success` | 原判据成功，环境有效且 session 未 poisoned | true / 原 score | 已解决 case |
| `valid_failure` | 有效运行触发原失败或用尽原控制步上限 | false / 原 score | 已解决 case |
| `invalid_native_layout` | 原支持演示或环境有效性检查失败 | null / null | 未解决；保留 case，不补新布局 |
| `budget_censored` | 非原控制步上限的决策/资源预算截断 | null / null | 未解决；不得混成普通任务失败 |
| `infrastructure_incomplete` | 初始化耗尽重试、进程丢失、模型不可访问或执行状态不明 | null / null | 未解决，单列原因 |

**idle timeout 的决定：** 作者部分 Direct 超时经用户裁定计为失败，但没有补造 native score。本次主协议不复制临时人工裁定；idle/RPC 超时归入 `infrastructure_incomplete`。另输出“全部基础设施未完成按失败计”的敏感性统计，明确这是另一种口径，不能冒称 native 成绩。若将来要正式采用作者式超时裁定，必须在新 campaign 前定义触发时钟、阈值和适用状态，不能运行后临时决定。（原事实 E1，S24；本次选择 E3）

原结果 `complete`、`valid_for_success_rate`、`native_success`、`native_score`、`status` 和 `reason` 原样存储；表中状态属于新增汇总字段。原控制步耗尽与外部预算截断必须由不同原因编码，避免把同为 truncated 的运行一概处理。

### 8.3 分母、配对与缺失分数

- 每方法/路线固定登记 50 cases；全部已解决才输出主口径完整 `successes/50`。若成功数 S、未解决数 U，报告已解决数量、`S/(50-U)` 的有效集率（U=50 时该率为 null），并报告固定 panel 的范围 `[S/50, (S+U)/50]`；这是缺失结果上下界，不是统计置信区间。
- Direct/Hybrid 配对只使用两者均已解决的同身份 case，明确配对数和缺失清单；不据缩水样本宣称完整 panel 结论。初始 RGB/本体/EEF 指纹不一致也必须报告，不能重抽 case 消除差异。
- `native_score` 缺失保留 null；完整 panel 均值只在所有 50 分数可用时计算。另可给 `score_available_mean` 与 `n_scored`，不能用可用子集均值冒充全样本均值，不能补零。
- 原主实验参考为 RoboDojo Hybrid 48%、Direct 26%；RoboLab Direct 98%、Hybrid 92%。这些仅作跨引擎外部参考，不作为达标门槛，也不据其排序筛选本次结果。（E2/E3，S1、S2、S9）
- 每任务只有 5 case，报告逐 case 结果和配对列联；如加置信区间或检验，注明方法与小样本局限。作者其他策略的重加权参考与本次新增 Student-only 同环境实验分开。

### 8.4 最低日志与交付物

每集保存输入身份、两类时间（墙钟/物理）、三相机与观测、完整模型工具对话、候选/噪声哈希、动作来源、ACK/控制步、接管及交还、原评分状态、终止原因、token/缓存/费用及视频。服务不提供精确费用时保留“未知”或标为估算，不写 0；订阅调用记录用量/额度而非杜撰单次账单。

原始日志可能含账号/路径等信息，保存在受控运行目录；公开结果只保留脱敏索引、关键帧、指标和必要复核材料。交付至少包括固定输入清单、参数映射与差异清单、可执行适配代码、验收记录、两路线逐 case 结果、失败归因，以及未完成项。

## 9. 当前状态、完成定义与仍不能承诺的部分

**现在完成的是计划与证据审计。** 已有 50 布局/75 源文件内容哈希、9 支持轨迹元数据、17 权重文件聚合身份元数据核对；独立审查的 F1–F3 已全部并入规格。尚未实现上述新工程、下载完整权重/资产、验收 MuJoCo/NPU 或运行正式 episode。

“完整交付”要求：所有原任务及必要资产齐全，控制/评分/随机流验收通过，两种接入分别交付完整 Direct/Hybrid 链路和原 case 结果，所有差异与缺项明示；覆盖整份报告还需 RoboLab 的输入与实现完整性通过。Student-only 是附加证据，不能替代上述任何缺项。

当前四个不能先承诺的点：① cloth 在 MuJoCo 中的真实抓持/折叠保真度；② RTX→软件渲染和接触动力学的跨引擎等价；③ 原 checkpoint 转换到 NPU 的数值通过；④ RoboLab 未核全的 Direct 入口/案例及两服务上的原模型可访问性。前三项需要实施证据，第四项需要公开材料和执行时接入核验；“继续尝试”不能把缺证据写成已解决。

**不再需要用户补充范围、服务器或预算信息才能把计划交给执行者。** 用户已选择两种接入和暂不设预算上限；W0 由执行者记录实际运行条件。若后续发现必须更换原任务、权重、模型或判据才能推进，再提出具体缺口和可选方案；在此之前按现有授权方向持续完成可实施工作。

## 10. 来源与源码锚点

以下 GPT-as-Policy 源码均固定至上述 commit；访问日期 2026-09-17。

- S1：[项目 README](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/README.md)。
- S2：[中文报告正文源](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/report_site/app/src/content/report/article-copy.json)；[用户提供的报告站](https://anonymous-report-421.github.io/public-website/?view=1&lang=zh)。
- S3：[Direct agent 契约](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/robodojo-gpt-only-rollout/SKILL.md)。
- S4：[Hybrid agent 契约](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/skill/SKILL.md)。
- S5：[checkpoint 契约](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/pi05_server/checkpoint.py)。
- S6：[原始依赖与 checkpoint 配置](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/SOURCE.json)。
- S7：[Direct EEF 动作约定](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/robodojo-gpt-only-rollout/context/eef_control.md)。
- S8：[接管 gate](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/skill/gate_prompt.md)。
- S9：[公开逐案例结果](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/public_results/evaluation_cases.json)。
- S10：[MuJoCo 编程文档：OpenGL rendering](https://mujoco.readthedocs.io/en/stable/programming/index.html#using-opengl)。
- S11：[OpenPI：预训练权重、PyTorch 支持与转换](https://github.com/Physical-Intelligence/openpi)。
- S14：[昇腾 π₀.₅ 推理样例](https://github.com/hicann/cann-recipes-embodied-intelligence/blob/master/manipulation/pi05/infer_with_torch/README.md)。

- S15：[RoboDojo 固定版本目录与 XPolicyLab submodule](https://github.com/RoboDojo-Benchmark/RoboDojo/tree/ee67a1468510da7624a089164402359f2afc72c8)。
- S16：[固定 XPolicyLab 的原 π₀.₅ 配置](https://github.com/XPolicyLab/XPolicyLab/blob/432f82b1758c5b1202e42a3dfe014546dbc50871/policy/Pi_05/openpi/src/openpi/training/config.py#L635)。
- S17：[原 50-case 选择清单](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/eval_panels/robodojo_panel50_scope_v2.json)与[源 panel 的布局/轨迹/代码哈希](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/eval_panels/robodojo_panel60_v1.json)。
- S18：[RoboDojo 固定资产与模型 revision](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/91f76c28d93dd20c5fa46ce6a5a1d96a4f384acd)。
- S19：[RoboDojo 安装与资产说明](https://robodojo-benchmark.com/doc/usage/install-and-download/)。文档会变化，具体下载源以固定版本脚本为准。
- S20：[RoboLab 上游](https://github.com/NVlabs/RoboLab)及[原集成的 checkpoint 契约](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robolab/pi05_server/checkpoint.py)。

- S21：[第三 Franka 配置](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/env_cfg/robot/dual_x5_and_franka_competition.yml#L18)，18–27 行；[支持轨迹与演示有效性](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/task/RoboDojo/tasks/imitate_sorting_sequence.py#L47)，47–117 行。
- S22：[250 Hz 物理步](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/env_cfg/sim/sim_config.yml#L2)及[25 Hz 观测配置](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/env_cfg/arx_x5.yml#L10)。
- S23：[支持轨迹逐子步推进与插值](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/src/eval_client/eval_env.py#L500)，500–557 行。
- S24：[原配对统计与 idle failure 裁定](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/paired_evaluation.py#L135)，135–155 行。
- S25：[原 session 评分注册与结果资格](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/robodojo_server/session.py#L178)，结果字段见 178–189 行，注册见同文件开头。
- S26：[固定 OpenPI 的 RNG/noise 注入](https://github.com/XPolicyLab/XPolicyLab/blob/432f82b1758c5b1202e42a3dfe014546dbc50871/policy/Pi_05/openpi/src/openpi/policies/policy.py#L41)，41、65、85、98–103 行；[原采样噪声](https://github.com/XPolicyLab/XPolicyLab/blob/432f82b1758c5b1202e42a3dfe014546dbc50871/policy/Pi_05/openpi/src/openpi/models/pi0.py#L230)；[episode RNG 身份](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/pi05_server/server.py#L29)。
- S27：[原 FK/DLS 与限幅](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/robodojo_server/kinematics.py)。
- S28：[原持久 agent runner](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/skill/run.py)。

## 11. 合并记录

v1.0 计划见 git `fb48ba5`；独立对抗性审查全文与权重元数据复核见 `2e36b03`。v2.0 合并两份正文，修正权重核查进展，吸收 F1–F3，并补齐 W0–W12、两种接入、attempt/统计协议和交付定义。旧审查路径只保留历史索引，不再维护第二份有效规格。

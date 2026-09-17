# GPT-as-Policy 原任务完整链路的 MuJoCo / 昇腾迁移计划

> v1.0 · 2026-09-17 · 原任务复现方案；已做源码/输入一致性核查，尚未运行远程 NPU 或付费模型实验。
>
> 以用户本轮最终要求为准：直接推进原任务完整链路，不以 bowl、LIBERO 或自制相似任务替代；阶段验证发生在最终实现上，不另做演示工程。
>
> 公开版只保留平台类别，实际服务器配置、接入信息、账户与凭据不入本文。
>
> 证据口径：E1=源码与行号；E2=配置、文件哈希与公开结果记录；E3=文档、作者报告或工程建议，均明确标记。不使用 E4 作为承诺依据。下文工作包和验收设计均为 E3 建议，非已完成实验。

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
| 原 π₀.₅ 权重 | 找到 `Pi_05/RoboDojo-sim-arx_x5-joint-0/59999` | params 目录元数据显示 16 文件、12,440,988,569 字节；尚未完整下载、聚合哈希或加载 |
| 原归一化 | 找到 `59999/assets/arx_x5_sim/norm_stats.json` | 文件目录已确认，不能据此宣称 NPU 推理等价 |
| 原机器人 | 找到 `Assets/Robots/x5/X5A.urdf`、`ARX.usd`、meshes/configuration | 可以沿用原 ARX X5 来源；USD/URDF 物理属性一致性和 mesh 转换未验收 |

上述核查均为 E2；详细可追溯记录见[输入一致性核查 JSON](2026-09-17-GPT-as-Policy输入一致性核查.json)。本轮未下载十余 GB 权重或完整大型资产，未运行上游脚本或反序列化轨迹。

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
| 远程执行 | 本地可做源码与资产检查、接口实现和轻量验证 | 本轮未建立可用的 NPU 服务器执行通道；真实适配验收需要可用的连接方式与工作目录 |

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

原 RoboDojo 契约：头部+双腕 RGB、14 维本体信息、两臂 link6 实测位姿；动作坐标为 environment origin、米、单位 wxyz；夹爪学生值 0=闭/1=开。Direct 每段 1–5 控制步；Hybrid Student 前缀 1–15 步、局部修正 1–5 步；EEF 界限 5 cm / 0.35 rad；控制 25 Hz。（E1，S3–S8）

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

## 6. 实施顺序：同一最终工程持续推进

分阶段是隔离问题和累计验收，不是先交付 Demo；每一步产物都必须进入同一原任务完整实现。顺序按风险和依赖安排，不以易成功的替代任务作为阶段终点。（用户要求 + E3 实施方案）

| 工作包 | 直接做什么 | 必须拿到的证据 |
|---|---|---|
| A 原始依赖锁定 | 取得固定源码、原 X5/物体资产、原 checkpoint/norm、原布局/支持轨迹；核对完整输入清单 | 文件内容哈希、引用闭包、配置映射；找出公开缺项，不能用近似件填空 |
| B 高风险可迁移性 | 在原 cloth/抽屉/支持臂对象上验证参数和表示可迁移；同时识别无 GPU 的渲染依赖 | cloth 关键点/拓扑与判据映射，mesh/材质/碰撞缺项表；尽早暴露可能阻止全任务完成的项 |
| C 原环境适配层 | 将 MuJoCo 包装成原 session 需要的 robot/scene/observation/step 接口；复用原 FK/DLS | 关节顺序、坐标、link6、夹爪、25 Hz、动作限制和预算逐项符合；真实物理而非瞬移 |
| D 原任务与评分移植 | 逐个移植十任务与原 reward/trigger/query；包括支持轨迹和柔性点位 | 对冻结状态及事件序列做原判据/适配判据比对；覆盖未抓住、掉落、未归位、错误顺序等反例 |
| E 原学生 NPU 推理 | 同一 checkpoint/normalizer 在兼容 PyTorch 与 torch_npu 上运行 | JAX CPU→PyTorch CPU→NPU 的分层误差记录；真实原观测输入、固定采样噪声、动作解码检查 |
| F 原 agent 三条运行入口 | Direct、Hybrid 与补充 Student-only；保持上游持久上下文、gate、原指令和动作工具 | 每段新观测/新候选/审核/执行/回读可审计；接管和交还正确；不存在隐藏脚本策略 |
| G 原案例批量评测 | 在移植完成的十任务上跑原 50 cases，按方法对齐；保留所有运行尝试 | Direct 50 + Hybrid 50 的完整记录；补充 Student 50 单列；所有缺项和物理/视觉差异可追溯 |
| H RoboLab 补充复现 | 在单独任务/机器人/权重契约下完成报告第二部分 | 独立 10 任务/每任务 5 cases 的身份与结果；不混用两套成功率和配置 |

软件/硬件初步调查和离线校验可并行安排在工程流程中；本轮没有启动后台任务或多 agent。到每个验收点继续解决下一依赖，不因某个动作能运行或某段视频成功就宣称完成。

**不再沿用此前“相似任务约 2–4 周”的估算。** 高保真十任务迁移的关键不确定性是 cloth、原资产物理属性、评分状态机和 NPU 等价性；在 B/E 给出证据前，无法负责任地承诺总工期。可以按工作包记录耗时和更新预计完成时间，不能为了按期交付削减复现范围。（E3）

### NPU 数值验收细则

- 保留训练配置、图像 resize/padding、tokenizer、state 拼接、normalizer、delta/absolute 变换、夹爪连续值、去噪过程与 horizon。
- 固定实际噪声张量和观测比较，不把跨后端相同随机 seed 视为相同采样。
- 先在 CPU 上验证 JAX→PyTorch 转换，再比较 PyTorch CPU→NPU；无法完成哪一层参考就披露哪一层证据缺口。
- 指标同时包括 NaN/Inf、按维误差、反归一化关节目标、夹爪和 FK 位姿偏差；容差根据控制分辨率和参考误差预注册，不看到结果后放宽。
- 单卡 eager 先保证正确性，之后才加编译/加速；正式 policy RNG 不能被预热消耗，预热与 episode 随机状态隔离。
- 昇腾 π₀.₅ 样例只提供算子/部署参考，不将其 LeRobot 模型与 mock 输入结果当作本 checkpoint 已复现。（E3，S11、S14）

### 仿真保真度验收细则

- 资产级：尺寸、惯性、坐标、碰撞体、关节、mesh 与纹理来源可追溯；不能把全部 USD 仅当视觉网格转走。
- 机器人级：原 URDF FK 与 MuJoCo FK 对同一关节向量一致；原 DLS 动作及限幅行为保留。
- 控制级：原驱动/插值/夹爪过程映射；可用原演示或参考记录检查轨迹跟踪，不能只看终点。
- 图像级：相机内外参、RGB 通道、分辨率、翻转、视野/遮挡；明确 RTX→MuJoCo 的视觉差异和潜在学生分布偏移。
- 任务级：保持原触发时序、初态语义、步数上限、支持臂行为、success 与 partial score 的区别。
- 柔性级：衣物关键点、拓扑和原成功条件逐一对应；不能换成刚性“衣服形状”或只用覆盖面积代替折叠。

以上均为 E3 验收设计。缺少原 Isaac 实测参考时，只能报告通过了哪些静态/离线检查，不能声称跨引擎动态保真度已证明。

## 7. 评测与结果口径

保留原两种主要方法的所有 50 cases。原 RoboDojo 参考结果 Hybrid 48%、Direct 26%，但它们不是迁移成功门槛；RoboLab 原排序相反（Direct 98%、Hybrid 92%），所以不能只接受 Hybrid 更好的结果。（E2/E3，S1、S2、S9）

正式协议冻结：模型和推理档位、工具与上下文策略、提示词哈希、源码/资产/checkpoint 哈希、50 cases 身份、初始化语义、动作/控制频率、评分和预算、异常处理。迁移导致的配置变化写进单独的迁移 manifest，不修改原 panel 去冒充原输入一致。

必须记录：环境最终成功/部分分、动作来源、实际控制步数、GPT 审核/接管次数、token 拆分和实际费用、NPU/CPU/渲染/网络/模型耗时、初态与控制过程、候选和动作、终止原因及录像。修正步占比与 GPT 调用占比是不同指标。

模型可读 RGB、本体、原任务说明和本 episode 历史；对象真值、内部 reward、未来物体轨迹及隐藏规划器留在评分/验证侧。开发人员可以用真值诊断，但不能泄漏到正式策略。恢复动作不得偷偷重置环境。

基础设施失败、API/预算中断、物理失败和任务失败分开记账；保留全部 attempt。某任务迁移未完成时报告“未复现”，不输出完整十任务成功率。Student-only 是新增同环境基线，不冒充作者的官方重加权参考。

以上为 E3 协议方案，原控制与 gate 依据 S3–S8。

## 8. 当前工作边界与下一步

已经完成：原链路核读；找到原机器人/权重/normalizer 来源；50 布局与 75 源码文件内容哈希验证；9 支持轨迹的远端哈希验证；重写本高保真迁移计划。

尚未完成：完整资产下载和转换、原权重内容验算与推理、MuJoCo 任务实现、远程 NPU 执行、原模型接入冒烟、付费评测，以及 RoboLab 的完整输入审计。

下一步沿 A/B/C/E 的依赖持续推进：先完善原始输入与难点清单，并在最终适配工程中处理原 X5、原场景/评分和原模型；不另建 bowl/LIBERO 演示。远程实际运行前需取得可用 SSH 配置别名或执行方式及工作目录，不需要在聊天或公开仓库提供明文凭据。

## 9. 来源与源码锚点

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

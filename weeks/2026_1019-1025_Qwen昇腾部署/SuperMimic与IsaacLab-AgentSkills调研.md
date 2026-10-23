# SuperMimic 与 Isaac Lab Agent Skills 调研

调研日期：2026-10-23。方式：网络公开资料（arXiv、项目页、GitHub API 与 raw 文件、官方文档站），未运行任何代码。两个主题的关联：SuperMimic 直接对标本工作区三份主报告的研究对象 SONIC（NVIDIA GR00T WholeBodyControl）；Isaac Lab agent skills 覆盖 SuperMimic 非官方复现的运行底座（Isaac Lab 3 kit-less + Newton/MuJoCo-Warp），也与本工作区自身的 skill 使用体系同构。

证据口径（沿用 00_总览 §3 分级在网络调研下的适配）：**〔E3·一手〕**=直接读取的一手网络来源（arXiv 页面、项目页、仓库 raw 文件、GitHub API）；**〔E3·推断〕**=基于一手的推断；**〔E4〕**=二手转述或 AI overview，未逐条复核，不作定量结论的唯一依据。所有链接抓取时间均为 2026-10-23。

---

## 1. SuperMimic：单可微世界模型扩展人形全身控制

### 1.1 基本信息〔E3·一手〕

| 项 | 内容 |
|---|---|
| 论文 | *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*，arXiv 2610.28480（cs.RO），2026-10-20 v1 |
| Venue | CoRL 2026 **Oral** |
| 作者 | Qiayuan Liao, Yiyang Shao\*, Junfeng Long\*, Kevin Zakka, Roberto Horowitz, Pieter Abbeel†, Koushil Sreenath†（UC Berkeley；\*共同一作，†共同指导） |
| 项目页 | <https://supermimic.github.io> |
| 官方代码 | **截至 2026-10-23 未放出**（项目页无代码链接；非官方复现亦如此声明） |

### 1.2 核心主张

"Compute is no longer the bottleneck"：**单张 H100 训练 3 小时即超过 SONIC 的归一化 return 0.90**；SONIC 对应 checkpoint 为 128×L40 训练 7 天（约 21,000 GPU-hours）〔E3·一手，项目页与 arXiv abstract〕。折算约 **7200× GPU-hours、56× wall-clock**（项目页脚注口径；文档正文表述为 ">1000× fewer GPU-hours、>50× less wall-clock"）。

### 1.3 方法：One loop, two learners

不用可微仿真器——梯度过 **learned world model**、不过 simulator。世界模型与策略在同一个循环里联合训练〔E3·一手，项目页 + alphaXiv 摘要〕：

1. **Collect**：策略在普通接触仿真器（**MuJoCo Warp**）里采集 transitions 进 replay buffer；
2. **Fit**：在 8-step 自回归窗口上拟合**单一共享 world model**（所有 motion 共用），每次迭代重拟合、永不"预训练后冻结"（仅在单次策略更新内部固定）；
3. **Imagine**：在 world model 内做 32-step 短 rollout，策略用 **BPTT 解析梯度**优化，critic 负责超出 horizon 的 return bootstrap；实验表明 16-step + critic 即可匹敌 critic-free 的 32-step。

关键定量佐证〔E3·一手〕：

- world model 在实用 batch size 下比 MuJoCo Warp 快 **10× 以上**；
- 在**同一个 world model 上跑 PPO**（PPO-WM，LAFAN1 walk/run/sprint）作对照，SuperMimic 达到其最优 return 只需约 **1/64 的 imagined steps**——收益来自解析梯度本身，不只是 rollout 快；
- 思路上是"跨 motion 摊销共享物理"（amortize），替代 SONIC 式的 per-motion specialist + 大规模 PPO。

### 1.4 结果 vs SONIC

| 指标 | SONIC | SuperMimic |
|---|---|---|
| 训练算力 | 128×L40 × 7 天（≈21,000 GPU-h） | 1×H100 × 3 h（≈3 GPU-h） |
| 归一化 return | 0.90 | 3 小时即超过 0.90 |
| 失败动作（共 129,514） | 179（含全部 44 个 flip、全部 22 个 side roll） | 3 h/1 GPU：25；2 GPU × 9.8 h：**0** |
| 训练数据 | 611 h | BONES-SEED 264.56 h（公开子集，少 2.3×） |

〔E3·一手，项目页〕基准条件：foot friction 1.0、无 domain randomization、无观测噪声、确定性物理。

### 1.5 Scaling 发现〔E3·一手〕

motion 数据从 1 h 扫到 264 h：world model 预测误差始终平稳在 **0.007–0.011**（归一化），达到目标 return 的 GPU-hours 随数据量增长缓慢。作者据此宣称使 motion-tracking scaling law 的实测成为可能——注意这是对"可测性"的宣称，不是已确立的定律〔E3·推断〕。

### 1.6 Sim-to-real〔E3·一手〕

单一通用策略 **zero-shot** 部署到真实 **Unitree G1**，含空翻、侧滚、跳绳式跳跃等高难动作，无逐动作调参。项目页提供真机视频与浏览器内策略演示（demo/）。

### 1.7 公平性 caveat〔E4，alphaXiv 页 AI overview 归纳〕

- **失败判据不同**：SuperMimic 用"触发训练终止条件"判失败；SONIC 用"高度偏差 > 0.25 m 且不看朝向"。
- 测试 clip 即训练动作，评的是**训练集覆盖（coverage），不是泛化**。
- 硬件（H100 vs L40）与数据量（264 h vs 611 h）均不对齐；作者论证"这些差异解释不了 1000× 的 GPU-hours 差距"，但对比确实无法归因单一因素。

### 1.8 局限（作者自述）〔E3·一手〕

world model 只从仿真 transitions 学习；只在平地训练；地形与物体交互为 future work。

### 1.9 代码生态

- **官方代码未放出**〔E3·一手〕。
- **非官方复现 [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)**（ETH Zurich，Chong Zhang，MIT 协议）〔E3·一手，README〕：
  - 自述"1 小时 vibe-coded"，复现论文 Fig. 4a：Unitree G1（29 DoF）+ 40 个 LAFAN1 clip（重定向后约 2.45 h motion）；
  - 运行底座：**Isaac Lab 3 kit-less（无 Isaac Sim/Kit/PhysX）+ Newton 引擎的 MuJoCo-Warp solver，50 Hz**；actuator/reward 采用 BeyondMimic 风格；
  - 结果：eval return 0.824 @ 1.24 GPU-h（论文：H100 1.2 h 达 0.82）；RTX PRO 6000 Blackwell 上 1.47 h 达 0.829、40 clip 全过；对照组"simulator 内 PPO"18 h / 8 GPU 仍到不了 0.82；
  - 自述偏差：关自碰撞换速度、bf16、解析 reward/observation/termination heads、观测额外加 6D 参考 root orientation、world model 先行 500 步 warm-up；README 明言代码是"slop"、风险自负。
- 已收录于 [awesome-humanoid-robot-learning](https://github.com/YanjieZe/awesome-humanoid-robot-learning)（标注 CoRL 2026）；Qiayuan Liao 在 X 官宣主打"1 GPU 3 小时 vs 128 GPU 7 天"〔E4〕。
- 相关工作锚点：SoftMimic（arXiv 2510.17792）、ResMimic（2510.05070，真机 G1 loco-manipulation 跟踪）、HumanMimic、BeyondMimic（vibed 版组件风格来源）〔E4，检索摘要〕。

### 1.10 与本工作区的关联

1. **三份 SONIC 主报告的算力前提被重新定性**：报告叙事建立在"SONIC 需要 128 GPU × 7 天"的大规模 PPO 前提上；若 1 卡 3 小时可超其 return，该前提及"迁移大规模训练到昇腾"的性价比论述需要更新版本号时一并修订〔E3·推断〕。
2. **数据采集端正是 B 路线研究对象 MuJoCo Warp**：SuperMimic 与 vibed 复现都把 MJWarp 用作采样/求解内核，复现门槛显著低于 SONIC（无需 128 卡集群）。
3. **数据可得性反转**：SONIC 原版 611 h 数据不可得，而 SuperMimic 用公开的 BONES-SEED（264.56 h）——复现 SuperMimic 反而比"复现 SONIC"更可行。
4. **NPU 适配视角**：训练主体是 MLP world model + BPTT，算子负载以矩阵乘为主；与 SONIC 大规模并行仿真+PPO 的适配难点（仿真内核）不在同一处，若做昇腾对照，这一侧迁移阻力理论上更小〔E3·推断〕。

---

## 2. Isaac Lab Agent Skills

### 2.1 背景：develop 分支已删除

调研起点链接 `github.com/isaac-sim/IsaacLab/tree/develop/skills` 已 404：GitHub API 返回 "No commit found for the ref develop"，页面自动跳转 `main`〔E3·一手〕。原因是 **2026-10-19 的 main branch cutover**（PR #8728，commit b9808256）〔E3·一手，commits API〕。`skills/` 目录现位于 main。

时间线〔E3·一手，commits API 按 path 过滤，含 rename 追踪的近似成分〕：

- 2026-07-01 之前无任何触碰 `skills/` 路径的提交 → 约 7 月起步（7 月下旬已有 Newton 迁移指南类提交）；
- 10-12 #8374 重构 agent skills 文档、#8388（DOC-009）物理后端章节重排；
- 10-14 #8387（DOC-008）backends/presets 架构合并；
- 10-18 #8700 文档刷新；10-19 #8728 main cutover；
- 官方文档页 Agent Skills 约在 10-22 更新〔E4，搜索引擎快照时间〕。

### 2.2 是什么

`skills/` 是 NVIDIA 在 Isaac Lab 仓库**官方维护的 agent skills 目录**：SKILL.md 格式的工作流指南，教 Claude Code / Codex / Cursor 等编码 agent 按 Isaac Lab 的规范干活。仓库 README 明确定位为 "guidance assets, **not** runtime Python packages"〔E3·一手〕。Isaac Sim 本体仓库亦同步携带约 25 个同款 skills〔E4，skills.rest 列表〕。

目录结构〔E3·一手，API 实测〕：

```
skills/
├── README.md            # 目录 + 发现机制 + authoring rules + 常用 import 路径表
├── developer/           # 7 个：面向贡献者/CI，按 AGENTS.md 自动加载
└── user/                # 17 个：面向使用者，按 description 自动触发或 /isaaclab-xxx 显式调用
                        # + 1 个 planned：user/import-robot-urdf-mjcf/（占位无描述）
```

### 2.3 机制设计〔E3·一手，skills/README.md 与官方文档页〕

- **发现**：Codex 扫 `.agents/skills/`，Claude 扫 `.claude/skills/`（同一套软链 alias，实测 24 个 symlink 指回 `skills/` 规范目录）。单一维护副本，**禁止拷贝到 agent 全局目录**；alias 名来自 frontmatter `name`，skill 间互链以 name 为稳定标识。
- **契约**：frontmatter 五字段（`name` / `description` / `audience` / `status` / `owners`；name 用 `isaaclab-` 前缀、小写；frontmatter 为受限 YAML 子集，不允许 `|`/`>` block scalar）+ 五个必选章节（`When To Use` / `Workflow` / `Validation` / `Maintenance` / `References`）。user 类还必须有 `evaluations.md`，**至少 3 个评测场景**（样例请求 + 预期行为 + 失败判据/已知失效模式）。
- **生命周期**：`status` ∈ experimental / stable / deprecated，deprecated 必须指向替代品。
- **验证器**：`uv run --no-project python tools/skills/cli.py check`（检查 frontmatter、章节、命名、链接、alias 一致性、evaluations），配套 pytest；CI 仅在 PR 触碰 `skills/`、`tools/skills/` 时跑 gate。
- **单一权威出处**：`docs/source/` 与源码是 source of truth，skill 只做 agent 侧的路由/排序/校验点/决策分支，**不复制文档内容**（文档级细节先改 docs 再链接）。明确 anti-pattern：不 vendor 安装指南、API 目录、生成日志、硬件基准报告、大段教程拷贝。
- 其他纪律：避免时效性措辞（用 "Current Workflow"/"Old Patterns"）；后端相关指引标注 PhysX/Newton；结构性校验之外还要求用代表性请求（含不应触发的反例）实测。

### 2.4 Catalog：24 个 skill 逐项

**Developer（7 个）**

| Skill（frontmatter name） | status | 用途 |
|---|---|---|
| `isaaclab-preparing-pr-workflow` | stable | 提交前按仓库 PR、commit、changelog、验证约定准备可评审 branch |
| `isaaclab-writing-changelog-fragments` | stable | 按仓库 fragment 格式与 bump 规则写/校验包级 changelog |
| `isaaclab-following-coding-style` | stable | 写/评审 API、config 类、导出、docstring 时套用 Python 约定 |
| `isaaclab-auditing-tests` | experimental | 测试守门：拦截低价值、耦合实现、重复覆盖的测试；诊断 CI 慢 |
| `isaaclab-auditing-an-issue` | experimental | 对照当前代码审计单个 issue，判 close/keep 并起草证据化关闭评论 |
| `isaaclab-triaging-issue-backlog` | experimental | 批量 triage 整个 issue backlog，报告可 retire 数量 |
| `isaaclab-updating-environment-docs` | experimental | 环境浏览器（生成文档）与任务注册、preset 选择器保持同步 |

**User（17 个）——安装与迁移**

| Skill | status | 用途 |
|---|---|---|
| `isaaclab-installing-isaac-lab` | experimental | 端到端安装：只读探测系统→选方式（uv checkout / uv 托管环境 / Isaac Sim 源码构建 / wheel / Docker）→一次确认后无人值守执行到验证；支持**中国资产区域 Profile** |
| `isaaclab-migrating-from-isaac-gym` | experimental | Isaac Gym 任务、资产、向量化环境、训练流程移植到 Isaac Lab |
| `isaaclab-migrating-2x-to-3x` | experimental | 2.x→3.0：actuator collection、四元数顺序、ProxyArray、后端迁移、可视化 CLI 等破坏性变更 |

**User——环境与任务搭建**

| Skill | status | 用途 |
|---|---|---|
| `isaaclab-building-environments` | experimental | 创建并注册环境：选 workflow、模板生成外部项目 vs 内部任务、从哪个示例改起 |
| `isaaclab-converting-direct-to-manager` | experimental | 已验证的 direct 环境转 manager-based，换取可复用 obs/reward/command/event/curriculum/termination 组件 |
| `isaaclab-planning-manipulation-tasks` | experimental | 操作任务（reach/grasp/lift/place/insertion/接触丰富）分阶段 gate：scene→reset 几何→action 契约→reward→验证 |
| `isaaclab-randomizing-with-events` | **stable** | domain randomization：event term 实现固定/自适应 DR、reset 随机化移植、ADR 课程 |
| `isaaclab-using-sensors-actuators` | experimental | 加传感器/执行器：camera、contact sensor、足端接触/touchdown/air-time reward、ray caster、IMU、joint wrench、tiled camera、执行器模型 |

**User——训练与调试**

| Skill | status | 用途 |
|---|---|---|
| `isaaclab-training-rl-agents` | experimental | 选 RL 框架、接 agent config、`uv run isaaclab train/play` 启动/恢复、启动问题排查 |
| `isaaclab-training-multi-gpu` | experimental | 多卡/多机训练启动与诊断，专治 NCCL 挂死、deadlock、ProcessGroupNCCL abort |
| `isaaclab-debugging-rl-training` | experimental | 训练行为诊断：reward 曲线异常、policy 训而不成、checkpoint 不兼容、聚焦 ablation |
| `isaaclab-diagnosing-joint-poses` | experimental | 初始关节位姿错误（腕/爪/工具/相机朝向）时从 articulation state 量测修正，不靠截图猜 |

**User——物理后端与 Newton/MJWarp（与 B 路线最相关）**

| Skill | status | 用途 |
|---|---|---|
| `isaaclab-selecting-backends` | experimental | 选并验证物理/渲染后端（PhysX / Newton / OvPhysX），调试后端特定仿真、随机化、传感器问题 |
| `isaaclab-using-presets` | experimental | 用 `PresetCfg` 定义多后端/多渲染器/多域变体，判断任务是否需要 preset |
| `isaaclab-preparing-assets-for-newton` | experimental | 资产迁 Newton（MJWarp）：per-solver 配置、质量/惯量警告、摩擦与接触滑移、armature/damping、solver profile、任务级验证 |
| `isaaclab-transferring-policies-sim-to-sim` | experimental | PhysX↔Newton 双向策略迁移：PresetCfg/checkpoint 兼容性、跨后端关节与 body 顺序、Franka mimic-joint 差异、面向迁移的 DR、四组合评测。**已验证任务含 Isaac-Velocity-Rough-G1**（Unitree G1 走行）、Isaac-Lift-Franka、Isaac-Velocity-Rough-AnymalD |

**User——运维**

| Skill | status | 用途 |
|---|---|---|
| `isaaclab-setup-troubleshooting` | experimental | 安装、验证、启动失败、仿真性能问题的分诊，路由到官方文档与权威命令 |

（status 取自各 SKILL.md frontmatter实测〔E3·一手〕：stable 共 4 个——pr-workflow、changelog-fragments、coding-style、randomizing-with-events；其余 experimental。）

### 2.5 实例解析：`isaaclab-training-rl-agents`〔E3·一手，SKILL.md 原文〕

该 skill 编码的全是"当前正确做法"：用 `uv run isaaclab train/play` 取代已废弃的 `scripts/reinforcement_learning/` 下脚本；不带 `--rl_library` 时加载任务注册的 `default_agent`；`play` 自动套 `play_mode()`（环境数上限 50、关观测噪声/腐蚀）；RL 框架配置不得混用（RSL-RL Python config 与 RL-Games/SKRL/SB3 的 YAML 分开）；先小规模 smoke 再扩并发/horizon/网络；多后端任务先在单后端验证再上 preset。skill 间路由：环境设计→`isaaclab-building-environments`；训练行为/奖励失败→`isaaclab-debugging-rl-training`；分布式挂死在 NCCL→`isaaclab-training-multi-gpu`。

### 2.6 与本工作区的关联

1. **SuperMimic 复现路径直接被后端四件套覆盖**：`select-backends`（选 Newton）、`prepare-assets-for-newton`（vibed 版 README 手动踩的 URDF→MJCF→USD 离线转换坑，此 skill 即官方答案）、`transferring-policies-sim-to-sim`（已验证的 Isaac-Velocity-Rough-G1 可作 G1 对照基准）。把 Isaac Lab checkout 放入工作目录后，编码 agent 自动发现这 24 个 skill。
2. **skill 工程化范式可借鉴**：强制 evaluations.md（≥3 场景含反例）、validator CLI 进 CI、alias 单份维护、文档为权威出处——与三报告的"冗余收敛"原则同构，可作为本机 `.agents/skills/` 与 skill-creator 规范建设的对照参考〔E3·推断〕。
3. **趋势信号**：一线仿真框架把 repo-owned agent skills 当标准配置（Isaac Sim + Isaac Lab 双仓落地），与本工作区 Mac 上的 skills 使用是同一生态〔E3·推断〕。

---

## 3. 参考链接（均于 2026-10-23 抓取）

**SuperMimic**

- 项目页：<https://supermimic.github.io>
- arXiv：<https://arxiv.org/abs/2610.28480>
- alphaXiv（含 AI overview caveat）：<https://www.alphaxiv.org/abs/2610.28480>
- 非官方复现：<https://github.com/zita-ch/supermimic_vibed>
- X 官宣（Qiayuan Liao）：<https://x.com/qiayuanliao/status/2112936035911135695>
- awesome 收录：<https://github.com/YanjieZe/awesome-humanoid-robot-learning>

**Isaac Lab Agent Skills**

- skills 目录（main）：<https://github.com/isaac-sim/IsaacLab/tree/main/skills>
- skills/README.md：<https://github.com/isaac-sim/IsaacLab/blob/main/skills/README.md>
- train-rl-agents 实例：<https://github.com/isaac-sim/IsaacLab/blob/main/skills/user/train-rl-agents/SKILL.md>
- 官方文档：<https://isaac-sim.github.io/IsaacLab/main/source/developer-tools/agent_skills.html>（NVIDIA Docs 镜像：<https://docs.nvidia.com/isaaclab/developer-tools/agent-skills>）
- main cutover commit：b9808256（PR #8728）
- Isaac Sim 同款 skills：<https://docs.isaacsim.omniverse.nvidia.com/latest/development_tools/agent_skills.html>

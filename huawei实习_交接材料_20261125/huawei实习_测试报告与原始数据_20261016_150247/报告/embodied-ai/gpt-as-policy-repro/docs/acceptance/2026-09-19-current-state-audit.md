# GPT-as-Policy 复现工程现状核查

> **范围与时点说明（2026-09-19）：** 本文是HEAD bf60127的全工程历史审计，后续修复已存在；用户现仅保留LIBERO-PRO。RoboDojo/RoboLab专项问题不再阻塞当前交付；共享修复按[单线计划v2.0](../superpowers/plans/2026-09-19-repro-next-execution.md)复验和集成。本文末尾旧执行顺序已被该计划替代。

> v1.0 · 2026-09-19 · 基准：本地 HEAD `bf60127`。
> 范围：本地实现、全部现有测试、最小反例、已有验收材料、目标机轻量检查。未启动正式采集，未重跑大模型数值验收，未修改实现。
> E1=源码位置；E2=配置、测试及运行记录；E3=据此作出的工程判断。下文行号对应本次基准。

## 1. 总判断

**不是只剩 GPT 接口。** 基础软件环境已可用，LIBERO 特定任务学生闭环有证据；完整 RoboDojo 环境、正式批量 runner、统计与故障恢复仍有阻断项。现有测试通过不代表完整任务链通过。（E1/E2）

必须区分四层：依赖可导入 → 组件测试通过 → 实际任务闭环通过 → 冻结协议下完整正式运行。此前 README/交接材料混用了这些完成口径。（E1/E3）

## 2. 本轮检查记录

| 检查 | 结果 | 边界 |
|---|---|---|
| 目标机连接、容器状态 | 两个项目容器运行中 | 不代表每条管线均可执行 |
| RoboDojo π₀.₅ 服务 health | `ok=true, device=npu` | 本轮未发推理请求，不代表 LIBERO 权重服务在线 |
| LIBERO 容器依赖 | Python 3.8.13 / mujoco 3.2.3 / robosuite 1.4.0 / torch 1.11.0 | 当前导入元数据；不是重新部署 |
| 默认 Mac 全套 pytest | 106 passed / 15 errors / 6 xfailed / 1 xpassed / 1 skipped | 首层错误为 `MUJOCO_GL=osmesa` 在 Mac 不受支持 |
| 仅改本进程 `MUJOCO_GL=glfw` | 仍 15 errors | 第二层为未设 `GAP_REPRO_DATA`，并非实际缺少资产 |
| 再设 `GAP_REPRO_DATA=$PWD/data` | **121 passed / 6 xfailed / 1 xpassed / 1 skipped**，退出 0 | LIBERO 整组跳过；tower/kong 缺口仍在 |
| 目标机重跑LIBERO session测试 | **9 passed in 155.97s**，退出0 | 当前容器中的goal/task0会话与指纹测试；未覆盖24条件格 |
| 跨机源码哈希抽查 | environment.py、results.py、run_w11_campaign.py 三个 SHA256 相同 | 本地新增 agent/client.py 在目标机对应目录不存在；不据此推断所有文件已同步 |
| 远端 Git 只读检查 | safe.directory/ownership 阻断；命令级例外也未解锁 | 未修改全局 Git 配置；改用文件哈希核对 |

Mac 可复现命令（工程根）：

```bash
MUJOCO_GL=glfw GAP_REPRO_DATA="$PWD/data" ./.venv/bin/python -m pytest tests/ -q --tb=short -rxX
```

`test_libero_session.py` 在导入时触发 `libero_session.py` 的 `os.environ.setdefault("MUJOCO_GL", "osmesa")`；即使 LIBERO 不存在而整组 skip，也已影响同一测试进程的其他模块。应拆分测试环境，并取消可选模块导入时的全局后端副作用。（E1/E2）

## 3. 正式采集前必须处理的问题

### F01 · P1：汇总器将未完成算成成功，漏日志也未进入未知数

- 位置：`src/gap_repro/results.py:119–126`。（E1）
- 复现：panel=2，一成功一 `infrastructure_incomplete`，`sensitivity_infra_as_failure=1.0`，应为 0.5。（E2）
- 复现：panel=2，只传入一个成功记录，返回 `unresolved=0`、`resolved_only_rate=0.5`、bounds=[0.5,0.5]；实际仍有一个未记录 case，已解决集率应为 1.0，固定 panel 缺失界应为 [0.5,1.0]。（E2）
- 根因：只统计已传入的 unresolved，未从冻结 case 清单补齐缺失；敏感性列分子直接使用 `success_count + U`。（E1）
- 还需明确：invalid_layout / budget_censored 与 infra 的分类不能合并进同一“按失败计”分母。单一 panel_size 不能校验 case 身份；需要冻结 case 清单。（E3）

### F02 · P1：ACK 重取被判陈旧，配对丢掉初始指纹

- `transport.py:65–79` 在查询已存 ACK 前先查 step：执行 5 步后同一请求重发，实测返回 `stale step_id 0 < current 5`，没有返回原 ACK。（E1/E2）
- `results.py:127–144` 的 per_case 没保留 init_fingerprint；不同指纹的同 case 两个成功结果直接计入一对，且未报告差异。（E1/E2）
- 需要区分完全相同的请求重取与复用 request_id 但改 step/动作内容的冲突；应验证身份和 payload，再返缓存。指纹差异必须显式记录，配对资格按预注册口径处理，不能无声配对或重抽布局。（E3）

### F03 · P1：reset 不复位完整物理状态，失稳标记直接报错

- 位置：`sim/environment.py:207–225,312–313`。（E1）
- 复现：把 qvel 全设为 1、time 设为 12，调用 reset(seed=42)，仍有 qvel_max=1、time=12。（E2）
- reset 未使用 seed 读取布局，未完整复位 qvel、ctrl、time 等状态，也未清除全部 episode 标记；当前实现不是固定 case 的完整 reset。（E1）
- `unstable_envs=[]`，`mark_env_unstable()` 却调用 `.add()`，实测 AttributeError。（E1/E2）

### F04 · P1：场景出图不等于具备可操作物理场景

- `sim/scene.py:23–35` 只提取位置，未应用 layout 姿态；`:75–76` 物体 body 无 joint，且统一 mass=0.05。当前组装路径中的物体是固定 body，不是可抓取自由物体。（E1）
- `MuJoCoEnvironment.__init__` 仍调用 `compile_dual_x5`，创建空物体 StateProvider；`register_layout_objects` 只是注入状态，动作后只同步机器人，没有物体位姿到评分器的实时同步。（E1）
- 转换脚本生成了带 freejoint 的 XML，但 scene.py 实际直接加载 OBJ，不消费这些 XML；不能因 XML 模板存在就认为物理已接入。（E1）
- 碰撞形状、质量/惯量、材质/纹理、单位、缩放均需按原资产核查；没有核验依据的默认参数不能作为高保真原任务。（E3）

### F05 · P1：USD 网格变换矩阵的乘法方向有错误

- `scripts/w11_convert_scene_objects.py:40–42` 使用 `row_points @ GetTranspose(matrix)`。（E1）
- 本轮用真实 `pxr.Gf.Matrix4d` 最小反例：平移矩阵 (1,2,3)，原点 `Gf.Transform` 结果 (1,2,3)，脚本相同乘法表达式结果 (0,0,0)。（E2）
- 进一步调用实际 `convert_usdz_to_obj`：临时USD中三角形置于平移(1,2,3)的父Xform下；导出OBJ bbox实际为[[0,0,0],[1,1,0]]，预期[[1,2,3],[2,3,3]]。临时资产运行后已自动清理。（E2）
- 这证明该表达式不保持一般 USD 变换；具体历史资产有多少受影响尚未逐个复查。修复后需重转受影响资产，保留旧产物哈希与新旧 bbox 对比。（E3）

### F06 · P1：主环境只有 3/10 任务判据真正接入

逐个实例化十任务并调用 `run_reward()` 的结果：（E1/E2）

| 可登记判据（并非完整闭环通过） | 抛 NotImplementedError |
|---|---|
| organize_table、classify_objects、put_bottles_into_dustbin | classify_objects_by_language、imitate_sorting_sequence、arrange_largest_number、pack_objects_into_box、build_tower、make_kong、fold_clothes |

- 位置：`sim/environment.py:180–202`。
- `support_arm.py` 只有合成轨迹队列；Franka 实体、原轨迹解析和环境执行接线未完成。
- `state_adapter.py:169,191` functional points / project_plane 仍显式未实现；fold_clothes 仍未完成。
- 本轮 6 xfail 和 1 xpass 涉及 tower 的多实例上下关系以及 kong 的 qpos/axis-up。xfail 不算验收。（E1/E2）

### F07 · P1：视觉未接入原任务观测，正式 runner 仍为脚手架

- `sim/environment.py:257–259`：include_vision=True 明确抛 BLOCKED。
- `scripts/run_w11_campaign.py:40–41`：非 dry-run 明确退出；当前提供的 w11_v1.json 还是 draft_unfrozen，按原命令 dry-run 也不能越过冻结检查。
- 现有图像渲染脚本证明后端与相机可出图；不能替代完整环境→观测→学生/GPT→动作→评分链路。（E1/E3）

### F08 · P1：LIBERO 正式 24 条件格及三方法适配未完成

- `scripts/lp_a2_closed_loop.py:25,131` 固定 max_steps=300、libero_goal/task0；`:49–53` 是验证用 NumPy 噪声。
- `LiberoSession` 可按 suite/task 创建基础环境，但构造接口没有 condition 参数；本轮未找到本地生产代码把 Ori/Obj/Pos/Sem/Task/Env 条件 manifest 接入 session 的入口。
- 现有证据为该任务 6 集设备比较，不能外推为 4 套件×6 条件全覆盖。
- agent/contract.py 当前是 RoboDojo 双臂 14D 契约；LIBERO 学生输出是 7D。不能把前者的“黄金对等”当成 LIBERO Direct/Hybrid 动作适配已通过。（E1/E2/E3）

### F09 · P1：GPT 传输只是配置检查，缺实际请求与事件循环

- `agent/client.py:38–42` 的 request/next_message 均为 NotImplementedError，两个具体子类没有覆盖。
- 还需实现通信、工具路由、图像输入、候选/接管/交还循环、日志与用量、同请求恢复。离线 transport/事件 fixture 可先做；真实模型验收才依赖接入信息。（E1/E3）

### F10 · P1：W6/W7 既有证据不足以关闭正式原随机流要求

- 已有三方固定噪声输出对比：设备 mean≈0.00153、转换 mean≈0.00307，承认其组件证据价值。（E2）
- `scripts/w6_fixed_noise_harness.py:78–95` 使用合成随机图像与 NumPy 噪声；不能证明主计划 W7 的原 JAX seed/key-split 序列、重取不前进 RNG、跨设备同噪声身份。
- 主计划要求真实任务观测、关键中间量、未解码动作、夹爪/FK 误差；当前主线不能仅据单个固定噪声锚点宣称全部完成。
- LP-A0 的 JAX 多 draw 是 LIBERO 权重与输入域；其已通过结果不自动验收 RoboDojo。（E1/E2）

### F11 · P1：LP5 精度计算及配对样本量不一致

- `ceil(z²p(1-p)/w²)` 为正态近似，不是 Wilson 半宽精确反推。（E1，代数）
- 同样 p=0.01、N=16，Wilson 规划半宽≈0.1045，并不是目标0.05；p=0.97、N=45 时≈0.0605。实际经验比例受整数成功数约束，更应由脚本按预定口径核算。（E2）
- p=0 并不会使 Wilson 区间退化为零；0/20 的上界约0.1611。
- 学生某些格只有16/20集，GPT又要求每格≥30且同 case 配对；若不预先统一 case 池/重叠子集，无法同时满足。
- “72格总量”需分方法核算；1,786/1,410 只是现有分层定义的学生行算术和，不是已证实达到目标精度的最终规模。
- §6仍用1,693/1,374旧总量；§4仍有错误 ceil 摊派公式，与§3矛盾。
- 32 worker 的环境并行估计不能当整条管线墙钟：单卡串行推理上界若为17.7h，不会仅靠CPU并行变成4h。需实测队列、批处理/卡数、服务开销。（E1/E2/E3）

### F12 · P2：部署与完成状态需要重新对账

- 本地/远端部分文件相同，agent/client.py 远端缺失；不可以用本地提交号替代已部署文件集合。
- README称“只差接入信息”、W6交接末表称“W0–W8/W10核心全部完成”，与本轮发现冲突。
- 本地预存未跟踪 `scripts/w11_pull_convert_push.sh`：保持原样，不应顺手删除或加入本轮提交。
- 需为每个工作包登记四种独立状态：已实现 / 已单测 / 已目标机集成 / 已正式验收。（E1/E2/E3）

## 4. RoboLab 边界

已有材料以审计记录为主，没有完整迁移与评测实现。历史“无 Direct/无 case”结论来自特定路径检索，本轮未独立穷尽核对所有上游入口，故仅列为待复核缺口，不能把旧推断升级成不可获得的事实。若目标仍是整份论文复现，需要单独工作流与15×8动作契约，不得混用RoboDojo/LIBERO统计。（E1/E3）

## 5. 优先次序

1. 先修统计、ACK、reset、失稳标记等已复现错误，并固化测试入口。
2. 修场景变换/物理与真实状态同步，打通一个原任务的无GPT链路。
3. 同步处理高风险衣物/支持臂和剩余七任务；补真实输入与原RNG验收。
4. 实现正式runner与GPT离线工具循环，再做真实GPT联调。
5. LIBERO补充面板单独补全条件/动作适配，重算冻结方案。
6. 验收达到门槛后正式采集；RoboLab单列可达性与实施计划。

详见 `docs/superpowers/plans/2026-09-19-repro-next-execution.md`。此顺序为E3工程建议，不构成已启动实施或批准正式采集。

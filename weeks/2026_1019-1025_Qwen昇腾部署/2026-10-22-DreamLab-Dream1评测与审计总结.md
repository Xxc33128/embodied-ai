# 2026-10-22 · DreamLab（Dream-1 / RoboDojo）评测与审计总结

- **日期范围**：2026-10-20 — 10-23（实验 10-20—10-21，整理与成文 10-22—10-23；时间均为 Asia/Singapore）。
- **主题**：Dream-1 API 接入 RoboDojo-Sim（ARX X5 双臂）的自定义适配评测、GPT-6 Astra 对照、用户叫停后的两项审计与全字段证据补采。
- **一句话总结论**：在自建 Dream-1→X5 适配与 RoboDojo-Sim 五任务子集上，Dream-1 共 4/128 集成功（3.125%）；停测后审计确认该数字衡量的是"自定义适配配置的成绩"，与官方 RoboDojo-RC 真机 59.3%（mean score）不构成同一协议对照，低分来源（协议差异、视野裁剪、IK 精度未闭环验收）已逐项定位但未做因果分解。〔E2/E3〕

实验现场为工作区 `DreamLab-repo/`（非 git、不上传）；本文证据索引中的路径均为该工作区内的相对路径，不是公开仓库附件。所有批次保留原始成绩与失败集，未因低分重跑或筛选。

## 1 背景与任务

- **对象**：Dream-1 是 dreamlabs 的 VLA 模型，以 `dreamlabs/dream-1-beta` checkpoint 经官方 API 调用；RoboDojo 为双臂操作 benchmark，其 RoboDojo-Sim 变体跑在 Isaac Sim、原生 ARX X5 双臂；RoboDojo-RC Tier 1 是 Robocurve 基于 I2RT YAM 真机改编的六个任务的榜单（官方 Dream-1 记录 mean score 59.3% 出处）。
- **控制链（本工作区自建）**：三相机 320×176 JPEG（4:3 先居中裁 16:9 再 resize，10-20 审计后修正）+ X5 URDF FK 得到的 TCP 状态 → Dream-1 `robot: other_embodiment` 路径输出 EEF 预测（48 行/块，执行前 24 行后重规划）→ 官方 `dream_ik.decode_chunk` 轨迹 IK → 关节命令，25 Hz。网络等待期间仿真暂停，API 等待不消耗控制步。〔E1/E2〕
- **任务动机**：验证 Dream-1 在 RoboDojo-Sim 五任务子集（general_pickup / stack_blocks / push_T / deposit_coin / insert_tubes，原生步上限 200/550/600/300/500）上的表现，并与 GPT-6 Astra 同场景对照；600 集正式批次被用户叫停后，转入协议审计与全字段证据补采。
- **纪律**：全程未改任务重置、布局、评分与步数上限；基础设施中断用同 run_id 断点恢复，不覆盖已完成集。〔E2〕

## 2 各批次结果（按时间）

### 2.1 10-20 四瓶入框 pilot 与控制抖动审计（put_bottles_into_dustbin，原生 700 步）

- 旧接法三集（`per_key` 末端预测直接逐点 IK）：25 / 0 / 25，均分 16.67/100，0/3 成功；90 次 API 调用全部成功，RTT 中位 1.193 s、平均 1.251 s，模型服务侧推理平均 0.587 s。录像可见持续抖动。〔E2〕
- **抖动审计**（control-audit）：`robot: other_embodiment` 路径正确（未把 X5 当 YAM），但旧接法缺平滑/衔接/限速（SDK 自动平滑只重写 `q_*_chunk`，不处理 `per_key`）。同一批缓存动作对照：方向反转计数 324/363 → 官方轨迹 IK 1/0；最大命令跳变 27.167°/12.459° → 5.674°/0.418°；左手净位移 193.75 → 189.77 mm，说明不是以停住换平稳；对原始末端目标的位置误差中位数左臂 1.20 → 7.37 mm（含滤波代价）。已改用官方轨迹 IK + 启动 2 s 限速 0.5 rad/s + 夹爪每步限变 0.05。〔E1/E2〕
- **同日第一审计**（audit-20261021）发现图像比例形变：640×480 直接 resize 到 320×176 使圆形测试图宽高比 1.3729（先居中裁 16:9 后恢复 1.0253）；已在 `pilot/contract.py` 修正并回归测试。〔E1/E2〕
- 修正版三集：10 / 0 / 25，均分 11.67/100，0/3 成功（n=3，不据此宣称改善或下降）。〔E2〕
- Astra 同任务三集（astra-pilot，同原生任务/物理/评分/700 步）：25 / 40 / 40，均分 35/100，0/3 成功。〔E2〕

### 2.2 10-20 五场景 Dream / Astra 对照（multitask_20261020_180709）

原生 score×100，success 用原生判定；Dream 布局 0/1/2 各一次，Astra 仅布局 0，policy seed 均 0，同 X5 与同步数上限。〔E2〕

| 场景 | Dream 三次得分 | Dream 成功 | Astra 得分 | Astra 成功 |
| --- | --- | --- | ---: | --- |
| 杂物抓取 general_pickup | 0 / 100 / 0 | 1/3 | 100 | 是 |
| 堆叠方块 stack_blocks | 0 / 0 / 0 | 0/3 | 100 | 是 |
| 推 T 对齐 push_T | 0 / 0 / 0 | 0/3 | 100 | 是 |
| 投币 deposit_coin | 0 / 0 / 0 | 0/3 | 0 | 否 |
| 插管入架 insert_tubes | 0 / 0 / 0 | 0/3 | 0 | 否 |

结论：Dream 1/15、Astra 3/5。两条控制管线不同（Dream 走上述 EEF→IK 链；Astra 由 GPT-6 依据 RGB、机器人状态与相机标定逐决策选动作、数值 FK/IK 辅助），且 3 次 vs 1 次取样，不能作正式排名。推 T 的 Astra 过程为先单点推转、后夹杆低位滑移，非全程单点自由推。〔E2/E3〕

### 2.3 10-21 128 集原生批次与用户叫停（native_20261021_093945）

原计划 600 集（5 任务 × 3 seed × 每任务 25–50 集，共 15 个任务/seed 作业、每集 3 相机），为官方协议五任务子集 + 自定义 Dream API 接入，不是官方 RC 真机评测、也不是官方 Dream-1 基线：〔E2〕

| 任务 | 每 seed 集数 | seed | 原生步数上限 |
| --- | ---: | --- | ---: |
| general_pickup | 50 | 0,1,2 | 200 |
| stack_blocks | 25 | 0,1,2 | 550 |
| push_T | 25 | 0,1,2 | 600 |
| deposit_coin | 50 | 0,1,2 | 300 |
| insert_tubes | 50 | 0,1,2 | 500 |

10-21 14:23 用户认为超出少量测试需求而叫停，600 集计划取消，保留已完成的 128 集（全部 seed0）：〔E2〕

| 场景 | 有效集数 | 成功数 | 成功率 | 平均原生得分/100 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| 抓取 general_pickup | 50 | 4 | 8.0% | 8.0 | 本 seed 完成 |
| 堆积木 stack_blocks | 25 | 0 | 0% | 0.6 | 本 seed 完成 |
| 推 T 形块 push_T | 25 | 0 | 0% | 0.0 | 本 seed 完成 |
| 投币 deposit_coin | 28 | 0 | 0% | 0.0 | 中途停止 |
| 插管 insert_tubes | 0 | — | — | — | 未开始 |

合计 4/128（3.125%）；样本数不齐，不作等权任务排名；中止时未完成的一集不计入。〔E2〕

- **运行事件（均保留证据、不重跑已完成集）**：① Dream API `DA-2000 NO_CAPACITY (pool_full)` 后端容量中断，以同 run_id `--resume-partial` 断点恢复（已完成 27 集跳过）；② 原生把 layout29 判为 unstable 换官方备用布局，外层"layout 必须 0..N-1"断言误报，修正为"足额、唯一、属官方候选集、缺口由 unstable 计数解释"；③ push_T 4 集原生提前失败（`is_lift` 规则允许提前终止），外层"失败也须跑满步数"断言误报，修正为 `0 < steps ≤ cap`。三次都只改外层 runner/collector 验证逻辑，未动 Dream 适配、任务、评分。〔E1/E2〕
- 逐集 campaign 日志已从 GPU 镜像到本地（四任务 1819 个完成调用事件 + 中止投币的 374 次调用、`state.json`、infra 尝试记录；不含视频），逐集打点与 state 记录抽查一致。〔E2〕

### 2.4 10-21 晚 75 集全字段证据批次（fullfield_20261021_123729）

叫停后由用户明示发起的**新批次**（不解除旧叫停），目的：补齐 128 批缺失的三类数据——逐请求原始 IO、适配器分项计时、逐步命令-实测 trace——并补 seed 1/2；仅 Dream-1。第四次启动后无人值守跑完。前三次启动的失败（存档目录判断、`queue_timeout` 重试白名单、`--eval-num` 配额、`save_dir` 创建、`obs.jsonl` flush）均为 fail-loud 止损，无脏数据入库。〔E2〕

- **结果**：15/15 job、75/75 集回收校验通过；75 集全部失败于原生判据，唯一非零 `deposit_coin seed2 layout2 = 20/100`；每 job API 调用量恰等于 5 集 × 步上限预算（45/115/125/65/105 次），即全部跑满步数、无提前成功。与 128 批 seed0 基线一致（general_pickup 15 集未中成功，真率 8% 时 P≈29%，统计兼容）。insert_tubes（128 批未开始）本批 15 集全 0。〔E2〕

| 任务 | seed0/1/2 平均分 | 成功 | API 调用 |
| --- | --- | --- | ---: |
| general_pickup | 0 / 0 / 0 | 0/15 | 135 |
| stack_blocks | 0 / 0 / 0 | 0/15 | 345 |
| push_T | 0 / 0 / 0 | 0/15 | 375 |
| deposit_coin | 0 / 0 / 4.0 | 0/15 | 195 |
| insert_tubes | 0 / 0 / 0 | 0/15 | 315 |

- **采得的全字段数据**：逐 tick 三相机 JPEG ~135,000 帧（可逐字节重建每请求 9 帧滑窗）；1,365 组逐调用原始 reply/执行动作/timing；32,250 步逐步 trace（命令/实测/物体位姿/阶段计时）；1,365 条网络+IK 诊断事件；225 段三相机视频（含 SHA256 与 ffprobe）；exchanges 全量清单 100,920 文件 / 0.99 GB。本地 `verified_summary.json` 连视频 1.3 GB。〔E2〕
- **计时口径（对 GPT 报告 decision_v2 的指标映射，融自 REPORT_SKELETON）**：VLA 动作 API 无 token/TTFT/TPOT 计量，对应物为 `serve_infer_ms`、`request/response_bytes`、`rows_returned`；总耗时 `rtt_ms`（含 serve/gateway/network 分解）；输入编码 `encode_ms`（逐相机）、本地解算 IK `solve_ms` + `decode_ms`；chunk 墙钟 = get_obs + update_obs + get_action + Σtake_action；三钟分离（仿真钟 25 Hz / chunk 墙钟 / API 钟）。控制步内含插值物理子步（前 80% 插值、后 20% 保持），一步命令 ≠ 一次物理步进。〔E1〕
- **已知偏差（引用本批数据时必须带）**：① 14/15 job 的末集 `obs.jsonl` 因进程退出缓冲尾损 1–8 行（帧文件逐个直写无损失、调用记录完整）；② 适配器为埋点版，SHA256 与 128 批不同（行为不变性：控制调用序列与 API 请求内容未改，只是多写文件多计时）；③ 适配器 episode 计数每集 +2，有效数据在偶数 `ep###` 目录。〔E1/E2〕
- **失败定量分析**（`failure_analysis.json` / `failure_analysis2.json` 汇总，不贴原始条目）：
  - 物体被动过：30/75 集有任一物体位移（insert_tubes 13/15 最多，push_T 仅 2/15，stack_blocks 8/15，general_pickup 4/15，deposit_coin 3/15）；35/75 集出现过夹爪闭合动作。〔E2〕
  - EEF 接近度：以 30 个"物体开始移动"锚点拟合模型 EEF 帧→世界系变换（右臂、仅轴置换/符号/平移、无旋转项），任一臂 EEF 路径到各物体初始位置的最小距离：75 集均值约 16.9 cm，仅 11/75 集进入 5 cm 内、0 集进入 2 cm；按任务均值 general_pickup 最远约 23 cm、insert_tubes 最近约 9.5 cm。变换拟合残差 RMSE 21.6 cm，故这些距离是粗估——但"末端普遍没有接近目标物体"的定性结论不受影响（与 general_pickup 仅 4/15 集动过物体互证）。〔E2 数字 / E3 解释〕
  - 输出重复度：相邻 chunk 首行 EEF 目标的余弦相似度 `rep_mean` 在 75 集均值 0.994、最低约 0.98，即块间输出高度相似，接近重复/停滞模式（stuck-loop 检测指标）。〔E2 数字 / E3 解释〕
- 已按 decision_v2 结构生成单文件离线报告 `fullfield-20261021/report_fullfield_20261021_123729.html`（9.9 MB，内嵌证据、离线可分享）。〔E2〕

### 2.5 10-21 YAM 四瓶移植诊断（yam-bottles，run_3x3600）

- **目的**：绕开 X5 自定义 IK 的不确定性——改走官方 YAM station 路径（`robot.arm=yam`、`exec_head=joint`，官方示例处理关节/夹爪，不用 X5 IK），检验同一模型在"官方关节管线 + 移植场景"下的表现。场景为官方 YAM 示例水瓶的四份复制品（各 0.276 kg，一只横放，按原 RoboDojo 瓶子任务 layout0/1/2 的平面位置排布）+ 新增开口框（桌面加宽、头部焦距 ×0.75）。〔E2〕
- **结果**：3 集均执行满 3600 步（20 Hz、48 行/块、每集 75 次 API 调用、无会话替换），最好/最终入框瓶数均 0/4、进度分 0、完整成功 0/3。9 段视频（20 FPS、3601 帧、180.05 s）与每步关节命令/测量、API 返回动作块已核验 SHA 并全量在本地。〔E2〕
- **口径**：移植诊断场景——瓶子资产与评分（框内碰撞几何+低速+双臂回位）均为自定义，不同于原生 RoboDojo，不能视作官方榜单成绩；未用官方示例的 120 s 墙钟截止，用满步数预算。〔E2/E3〕

## 3 审计结论（停测后，10-21，只读）

### 3.1 官方逐次记录与本地接入对照

- **官方 59.3% 的真实口径**：来自 Robocurve 官方记录导出的 120 条 Dream-1 RC 运行（六任务 × 20 次），mean score 59.25%（官网显示 59.3%）；按 `score==1` 计满分 31/120 = 25.83%——它本来就是平均进度分而非成功率。全部记录 `control=end-effector`、每集 180 s 墙钟限时、yam_arms 真机、adapter `dream-native-v1`、Autograded。〔E2〕

  | RC 任务 | 次数 | 平均分/100 | 满分次数 |
  | --- | ---: | ---: | ---: |
  | Cap Pen | 20 | 31.75 | 0 |
  | Pack And Pour Fruit | 20 | 43.00 | 2 |
  | Sort Objects | 20 | 60.75 | 5 |
  | Stack Bowls | 20 | 95.00 | 17 |
  | Stand Up Bottles | 20 | 55.00 | 4 |
  | Store In Safe | 20 | 70.00 | 3 |

- **3600 步说法的证据边界**：榜单首页统一注明 VLA 上限 3600 控制步，但 Dream 公开运行页实际采用 180 s 限时、`total steps` 显示 0 且无逐步 transcript；只能确认"统一预算 3600 + Dream 记录限时 180 s"，不能断言每集恰好 3600 步或两者严格换算。〔E2/E3〕
- **逐项对照**：checkpoint 字符串相同（服务端权重版本不可证）；机器人（YAM 真机 vs X5 仿真）、任务集（RC 六任务 vs Sim 五任务）、预算（180 s 墙钟 vs 200–600 仿真步）、评分（进度 rubric vs 原生几何/状态规则）、适配器（官方 external adapter vs 自建）均不同。SDK 只接受 `yam` / `other_embodiment` 枚举（无 `robot=x5`）；`other_embodiment` 路径屏蔽模型状态输入，云端并未拿到完整 X5 运动学；XPolicyLab 所 pin 版本只含 `policy/DreamZero`，无可复用的官方 Dream-1→RoboDojo-Sim/X5 适配。〔E1/E2〕
- **命名结论**：本系列实验最准确的名称是"Dream-1 API 的自定义 X5 仿真适配测试"；官方研究页 77.4%（ARX X5）是其自有跨机器人测试中相对 YAM 的归一化值，不是本任务集的绝对成功率。低分可能含跨机器人泛化损失，也受视野裁剪与未验收的闭环执行精度影响，协议差异不能替代因果实验。〔E3〕

### 3.2 低分原因与步数口径

- **550/3600 是两套预算**：`stack_blocks.py:19` 的 `step_lim=550` 等 200/550/600/300/500 为 Sim 原生步数上限，日志实测 25 集堆积木均 550 步、28 集投币均 300 步——不存在"跑到 3600 被截成 550"。〔E1/E2〕
- **三类已证实的混杂因素（未做因果分解，不能把低分全部归因于任一项）**：
  1. **协议口径**：上节全项不同，分数本就不应直接对齐。〔E2/E3〕
  2. **视野裁剪**：16:9 中心裁剪删去原图上下各 60 行（25% 行数）；用原始录像抽帧按冻结代码离线重建 API 输入，投币 layout0 与堆积木 layout24 等例中腕部相机的目标物下部及周围抓取区域被截断；头部相机仍能看全场景，受损的是腕部近景信息。〔E2〕
  3. **IK→执行的精度未闭环验收**：堆积木 575 个动作块的求解器内部位置误差——p95 中位数左 5.73 mm / 右 7.24 mm，p95 超 10 mm 的块占 22.3% / 29.2%，最差块 77.90 / 46.34 mm。这是滤波后目标的内部误差，非物理 TCP 误差、非全控制步全局 p95，且发生在启动限速之前；此前只验收了初始化 FK 与短程平稳。〔E2〕
- **反证与放大因素**：投币有"硬币抬升 8 cm 得 20 分"的里程碑而 28 集均 0 分——低分不能只用"成功须双臂回位太严格"解释；堆积木仅 layout24 得 0.15 原始分；25 Hz 下抓取仅 8 s 预算、启动 2 s 缓升占 50/200 步。未做延长预算对照，不能断言 3600 步就能成功。〔E1/E2/E3〕
- **已排除**：GPU 慢、网络慢、额度耗尽——本地跑仿真、推理在云端，同步请求期间仿真暂停，约 1 s 的 API 等待不自动消耗控制步；容量中断已按基础设施事件处理并断点恢复，不计入模型失败。〔E1/E2〕

## 4 状态与遗留

- **叫停状态**：128 集所属的 600 集批次保持用户叫停，禁止自动恢复或补跑；75 集全字段批次与 YAM 三集是叫停后用户明示发起的新批次，不构成对旧叫停的解除。两个批次的全部进程与 GPU 资源均已核验释放。〔E2〕
- **数据位置**：128 批的完整原始录像（约 1800 段三视角）与 fullfield 的 GPU 侧 exchanges 原始件仍在远程 GPU 节点，本地只镜像了抽样视频、结果与日志（不含全量视频）；YAM 三集的 9 段视频、逐步控制记录与 IO 样例已全量在本地。〔E2〕
- **后续排查顺序（全部未启动，均为计划）**：① 离线核查全部关键视角的裁剪后可见性（用已采 exchanges 逐 tick 帧，不再跑仿真）；② 少量固定轨迹实测"模型原始目标 → 关节命令 → 真实 TCP/夹爪误差"，完成控制链验收；③ 同布局、同控制器下原生预算 vs 延长预算的少量对照——延长组必须标为诊断条件，不得覆盖原生成绩。〔E3〕
- 128 批与 75 批的适配器 SHA 不同（后者为埋点版），两批数字并列引用时须带此口径差。〔E1/E2〕
- 本周公开仓只收录本总结；`DreamLab-repo/` 为实验现场，不建 git、不上传原始数据与录像。

## 5 证据索引（本地工作区相对路径）

| 内容 | 路径 |
| --- | --- |
| 现场总入口（目录导航） | `DreamLab-repo/README.md` |
| 128 集叫停结果表 | `DreamLab-repo/retest-20261021/当前结果-用户叫停.md` |
| 128 集运行/中断/恢复交接（含固定条件表） | `DreamLab-repo/retest-20261021/RUN_HANDOFF.md` |
| 128 集抽样视频索引（成功/失败/部分完成各一 + 另四集） | `DreamLab-repo/results/native_20261021_093945/selected_examples_20261021/视频索引.md`；`.../more_examples_20261021/视频索引.md` |
| 75 集结果概览 | `DreamLab-repo/results/fullfield_20261021_123729/结果概览.md` |
| 75 集交接与事件记录（前三次启动修复、已知偏差） | `DreamLab-repo/fullfield-20261021/RUN_HANDOFF.md` |
| 75 集报告骨架（指标映射，已融入本文 2.4） | `DreamLab-repo/fullfield-20261021/REPORT_SKELETON.md` |
| 75 集失败分析（汇总数字在本文 2.4，未贴原始 json） | `DreamLab-repo/fullfield-20261021/failure_analysis.json`；`.../failure_analysis2.json`（生成脚本 `DreamLab-repo/campaign/analyze_failures*.py`） |
| 75 集单文件离线报告 | `DreamLab-repo/fullfield-20261021/report_fullfield_20261021_123729.html` |
| 五场景 Dream/Astra 对照结果与视频 | `DreamLab-repo/results/multitask_20261020_180709/结果与视频索引.md`（manifest、双模型 verified_summary 同目录） |
| 四瓶 pilot 旧接法/修正版结果与接口约定 | `DreamLab-repo/pilot/README.md`（结果目录 `DreamLab-repo/results/bottles_2026-10-20_15-20-32/`、`.../bottles_corrected_2026-10-20_16-16-05/`） |
| Astra 四瓶三集汇总 | `DreamLab-repo/results/astra_bottles_20261020_three_episode_summary.json` |
| 抖动审计（修正依据与对照数据） | `DreamLab-repo/control-audit/README.md` |
| 第一审计（比例形变发现） | `DreamLab-repo/audit-20261021/官方结果与本地低分差异审计.md` |
| 官方逐次记录对照审计 | `DreamLab-repo/audit-20261021-poststop/Dream官方逐次记录与本地接入对照.md`（原始导出 `official_dream_runs.json` 等同目录） |
| 低分原因与步数口径审计（IK chunk 指标 `chunk_metrics.json`、裁剪重建图同目录） | `DreamLab-repo/audit-20261021-poststop/低分原因与步数口径.md` |
| YAM 四瓶移植交接 | `DreamLab-repo/yam-bottles/RUN_HANDOFF.md` |
| YAM 三集视频索引 | `DreamLab-repo/yam-bottles/results/run_3x3600/视频索引.md` |
| YAM 首个 API 返回与首步控制样例说明 | `DreamLab-repo/yam-bottles/results/run_3x3600/io_example/说明.md` |

视频文件一律按上述索引内的相对路径引用，不在本文内嵌。本报告未复制任何原始数据、凭据或连接方式；远程机器以"远程 GPU 节点"泛称。

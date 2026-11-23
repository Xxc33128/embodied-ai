# Motus LIBERO-Long 续训执行记录：6000 步完整重测至 10000 步收口

> **版本：1.0｜成稿日期：2026-11-23**  
> **记录日期范围：2026-10-26 ~ 2026-10-28（Asia/Singapore，下文时间均为 SGT）**  
> **主题：** Motus 在昇腾 910B3 上的 LIBERO-Long 续训，从 6000 步完整重测到 10000 步收口的执行记录  
> **记录截止：2026-10-28 11:47（源材料最后一次现场核查）**  
> **一句话结论：续训批次按原协议从 6000 步推进到用户设定的 10000 步上限后自动收口；固定 50 集成功率在 9000 步为 38/50＝76%（本协议历史最高），在 10000 步为 30/50＝60%；4000 次新增更新全部确认、未知更新 0。批次于 10-28 07:01:18 完成评测、恢复服务并退出，随后任务按用户指令停止，10000 步复盘未开展。**〔E2，S1、S5〕
>
> **证据口径：** E1＝源码／日志行号实证；E2＝配置、运行汇总与执行记录；E3＝计划、推断或待验证解释；E4＝未复核转述（本篇不使用）。本篇为事后据本地源材料补写，未重读源码、未登录节点，也未核查节点当前状态：E1 项均注明转引来源，节点路径与 SHA 据 S1／S2／S4／S5 摘录，编号见 §7。

## 1. 与 10-27 阶段报告的衔接

[阶段报告](2026-10-27-Motus架构与LIBERO-Long阶段训练报告.md)（v1.0，运行快照截止 2026-10-27 14:59:26）已详述 Motus 架构、数据与因果对齐、训练配方（冻结 VLM 的 FP32 八卡 FSDP SFT、global batch 32、每 1000 步固定 50 集评测）以及 200–8000 步的学习曲线；快照时最新完整 checkpoint 为 8000 步（35/50＝70%），8000→9000 段的八 rank 日志均记录完成到 8423 步。〔E2，R1〕本篇只写其后的增量：8000→9000 段收尾、9000 与 10000 两次完整评测、10000 步收口、服务恢复、用户停止及后续处置。为使增量可读，§2 与 §3.1 用最简篇幅交代续训批次的授权、设计和快照前的经过；架构、数据、loss 与评测协议一律引用阶段报告。更早的八卡全参探针、未适配基线与 128×128 观测审计见 [10-23 执行增量](../2026_1019-1025_Qwen昇腾部署/2026-10-23-Motus联合训练执行增量.md)，预注册见 [1012 周预注册](../2026_1012-1018_RPent多模型性能分析/2026-10-14-Motus与910B3联合端到端RL实验预注册.md)。

## 2. 续训设计与计划

### 2.1 为什么续训、从哪里恢复

周末批次按预设时间边界结束：2026-10-26 08:10 工作截止取消了剩余评测，控制器 08:15:48 以 `stopped / finished / weekend-cutoff / restored` 收口。接受轨迹为 6000 步，含一次 pilot 恢复回放共 6001 次实际更新，训练未知更新 0；但 step 6000 的评测只完成 11/50 集（9 例成功），没有完整 result，不计算正式成功率，18 个已启动 case 留存 6296 次 control attempt、6293 次 return／complete 和 3 次未确认控制。当时最近的完整成绩是 5000 步的 22/50＝44%。〔E2，S4、S5〕

2026-10-26 用户授权新批次：先完整重测 6000 步，再继续训练，并选择“先到 10000 步再复盘”。目标是绝对总步数 10000，即在 6000 之上再接受 4000 次更新，不是再增加 10000 步；正常完成时累计 10001 次实际更新，历史、失败与未知成本不清零。〔E2，S1、S2〕

恢复点是父批次的 step 6000 完整八 rank checkpoint：8 分片共 79,228,547,783 字节，清单哈希 `7446cf95…`（S2 的父证据表把同一值记为 checkpoint6000 complete.json 的 SHA）。父状态须为 stopped／finished／weekend-cutoff／restored，父作业真实退出，6001 次训练更新确认且无未知；模型身份与冻结的 `weekend_train.build_run_identity` 逐值相等，canonical 6000 的路径与 SHA 由父训练收据取得，并在初始评测前验证。〔E2，S2、S4〕

### 2.2 与原训练的差异

| 项目 | 周末批次（原训练） | 续训批次 |
|---|---|---|
| 数值实现 | 冻结的 trainer、eval 等 20 个文件 | 只读复用原文件：不复制改动、不 monkey-patch；原 20 文件与 runtime 逐文件未变 |
| 编排控制器 | `weekend_controller.py` | 新增 `resume_controller.py`、`test_resume_controller.py`、`config.json`，放在新隔离目录 |
| 起点 | 200 步 pilot 迁移 | 父批次 6000 步交接；先完整重跑 6000 的 50 集，16 个子进程 exit0 且 50 份控制审计通过后才允许 6000→7000 |
| 终点 | 时间截止（2026-10-26 09:00） | 绝对步数 10000 为主，兜底硬截止 2026-10-29 09:00 |
| STOP 与截止 | 旧 STOP 文件与旧截止参数 | 新 STOP 文件与新工作截止，只经 CLI 传入 |
| 成本记账 | 6001 次实际更新 | 继承 6001，不清零历史、失败或回放成本 |
| 运行目录 | `weekend-20261023/` | `continuation-20261026/`（首轮，启动失败）与 `continuation-20261026-attempt02/`（实际执行） |
| 6000 步评测 | 截止时只得局部 11 集 | 五个 checkpoint 各一次完整 50 集，不拼接旧 11 集 |

表中的设计意图为〔E3 计划，S2、S3〕；“原 20 文件未变”“不拼接旧 11 集”等执行核对为〔E2，S1〕。

保持不变的部分：八张 910B3、FP32 FSDP、microbatch 1×累积 4×8＝global batch 32；冻结 VLM，只更新 World／Action／Understanding；AdamW、绝对 step 学习率、sampler 与 RNG 的恢复方式；TRAIN 30639／DEV 3444 窗口、划分与 normalizer，不重编码；评测沿用原 seed、H8、K10、20Hz、520 policy controls 加 10 settling controls。本批次仍是 SFT，不是 task-RL。〔E3 计划，S2；各次看护未发现偏离，E2，S1〕

### 2.3 验收与停止边界

- 先 TDD 再独立审查：失败用例覆盖坏父 hash、非 6000 步、父批次未退出或未恢复、6001 成本不符、未知更新、错误模型身份、target 非 1000 倍数、先训练后初始评测、越过 10000、STOP 与绝对截止、缺 gate、初始评测失败时零更新、清理或日志失败不能跳过恢复；通过后再做独立代码审查与精确哈希 gate。
- 单次 claim：控制器验证配置后，以自身真实 PID／start_ticks 独占发布 claim，并绑定 gate、config、controller 三者的 SHA；主 agent 不预创建 claim，也不重复启动。
- 人工停止：`control/STOP.json`；参数更新在 step 边界协作保存，原始部分评测与全部成本保留，无盲目自动重试。兜底硬截止 2026-10-29 09:00，工作截止 08:10，新模型启动截止 08:00，checkpoint grace 600 秒。步数目标预计更早到达，时间截止只作保护。
- 资源：取得原资源租约并校验真实 PID；只允许暂停已授权的 diagnostic、MiniMax-H3、Qwen 三个容器，发现其他 NPU 工作负载即停止，不处理无关进程。
- 看护：每两小时 heartbeat，只在新完整评测、失败、停止或需处理变化时通知；10000 完成后恢复服务并暂停看护。

以上五条均为〔E3 计划，S1、S2、S3〕；实际执行情况见 §3。

## 3. 执行时间线

### 3.1 快照前（10-26 ~ 10-27 14:59）

| 时间 | 事件 | 证据 |
|---|---|---|
| 10-26 08:15:48 | 周末批次按截止收口，6000 评测停在 11/50 | S4；`weekend-20261023/evidence/monitor-20261026T0112/final-closure.json` |
| 10-26 11:50 | 新编排独立验收 PASS（43 tests／OK，另 3 项 CPU 探针通过）；资源只读复核：八卡健康，仅授权 Qwen 服务的两个 TP worker 占 3／7 卡 | S1；`continuation-20261026/evidence/task-review-v1.json`、`root-resource-snapshot-20261026T0350/` |
| 10-26 11:56–11:58 | 最终集成 review PASS／0 findings；发布首轮 gate，`--validate-only` exit0；现场复核后 Popen 一次首轮控制器 | S1；`root-gate-validation.json`、`root-launch.json` |
| 10-26 12:07 | 首轮 failed／finished／restored：CPU wrapper 导入失败，新增训练更新 0 | `continuation-20261026/evidence/monitor-failure-20261026T0616/failure-closure.json` |
| 10-26 14:17 | 核实旧进程全部退出、原 3 服务恢复；修复放入隔离的 `continuation-20261026-attempt02` | 同上 |
| 10-26 14:25 | attempt02 单次启动（此前 45 项回归、定向 review、只到 ready 的真实桥接 smoke 均通过） | S1 |
| 10-26 14:31 | 8 个 CPU bridge ready，8 个 NPU actor 分别在 npu:0–7，16 个注册作业存活；6000 步模型加载中 | `continuation-20261026-attempt02/evidence/root-start-verification.json` |
| 10-26 15:51:38／15:51:39 | 6000 完整 50 集结算 26/50＝52%；随即启动 6000→7000 | S1 |
| 10-27 01:30:50／01:30:52 | 7000 结算 35/50＝70%；启动 7000→8000 | S1 |
| 10-27 11:17:11／11:17:13 | 8000 结算 35/50＝70%；启动 8000→9000 | S1 |
| 10-27 14:59:26 | 阶段报告快照：八 rank 日志均记录完成到 8423 步 | R1 |

本表是阶段报告已覆盖部分的索引式回顾，时间与状态为〔E2，S1、S4〕，首轮失败根因为〔E1，转引 S1〕。未带目录前缀的路径按 S1 原文摘录，推定在 `continuation-20261026/evidence/` 下，其余相对于 `reproductions/libero-long-20261023/`。

### 3.2 快照后增量（10-27 下午 ~ 10-28）

| 时间 | 事件与数字 | 证据 |
|---|---|---|
| 10-27 20:50 | 9000 训练与完整 checkpoint 完成（`accepted_step 9000`）：8000→9000 作业 exit0，八 rank receipt passed，新段 1000 次 intent／complete 一致、unknown 0；published cost charged＝confirmed＝9001、retries 0。checkpoint 清单 SHA `cb3e218e…`，canonical 32,090,317,686 字节。9000 评测当时 46/50，6 个注册进程存活、10 个已有 complete 收据；完整 result 未发布，不计分 | `monitor-20261027T1248/verification.json` |
| 10-27 21:14:21／21:14:23 | 控制器结算 9000，并启动最后一段 9000→10000 | S1 |
| 10-28 01:48 | 9000 完整 50 集：38/50＝76%，比 8000 的 70% 高 6 个百分点；同一 50 例 8 例改善、5 例退化，净增 3 例。16 个评测作业 exit0，infra 0、评测 optimizer_updates 0，50 份原始控制审计与 journal SHA 复核通过，共 17,205 次控制；result SHA `4782e5f4…`。训练侧八 rank 各 539 次 intent／complete、unknown 0，trajectory 9539，含历史回放总确认更新 9540；完整 checkpoint 仍为 9000 | `monitor-20261027T1747/verification.json` |
| 10-28 06:44 | 训练预算达到（`accepted_step 10000`）：9000→10000 作业 exit0，八 rank receipt passed，1000 次 intent／complete 一致、unknown 0；四个续训段均不超过 10000，总新增更新 4000。published cost charged＝confirmed＝10001、unknown_rank_updates 0、retries 0，含历史 6001，未重置失败或回放成本。10000 checkpoint 清单 SHA `59a422e6…`，canonical 32,090,317,686 字节。最终 50 集当时 47/50，6 个作业存活、10 个已有 complete 结果；服务恢复 pending | `monitor-20261027T2243/verification.json` |
| 10-28 07:01:18 | 控制器 complete／finished／accepted_step 10000／restored，`restoration_errors=[]`。最终 result complete，50/50 完成，30 成功（60%） | `user-stop-20261028T0336/verification.json` |
| 10-28 11:36:52 | 用户要求停止本任务。前两次远程连接在 banner 阶段超时，其间已用原生工具删除 motus 自动看护（deleteStatus=deleted）；第三次连接成功后独占写入 `control/STOP.json`（reason=human-stop，source=explicit-user-request） | S1 |
| 10-28 11:38 | 核实：controller、注册作业与 CPU bridge 的原 PID＋start_ticks 均无存活身份，八卡训练容器 exited，原 diagnostic、MiniMax、Qwen 容器 running。只读取现有结果，不补评、不继续分析 | `user-stop-20261028T0336/verification.json` |
| 10-28 11:43–11:47 | 用户追加要求关闭 3、7 卡 Qwen 服务：确认占卡者只有 Qwen3.8-27B 双卡容器的两个 VLLMWorker（各约 56,309／56,289 MB），停止后容器 exited；11:47 复核两卡无运行进程，HBM 降至 3,426／3,423 MB，其他空闲卡约 3,405–3,411 MB。容器、模型、数据与 checkpoint 均未删除 | `evidence/user-stop-qwen38-npu3-7-20261028T034519Z/` |

本表均为〔E2，S1〕。`monitor-*` 与 `user-stop-20261028T0336/` 在 `continuation-20261026-attempt02/evidence/` 下，`user-stop-qwen38-*` 在实验根的 `evidence/` 下。各次看护均未改活动源码、数据、依赖，也未重启作业。阶段报告快照的 8423 步与 10-27 15:47 看护记录的 trajectory 8521 相隔约 48 分钟，98 步约合 29 秒每步，与稳态节奏相符，两者不矛盾。〔E2；节奏为推算，E3〕

## 4. 结果汇总

### 4.1 固定 50 集结果

| checkpoint | 成功数 | 成功率 | 评测集 | 结算时刻（SGT） | 环境控制次数 | 对上一 checkpoint 的逐例变化 |
|---|---:|---:|---|---|---:|---|
| 6000 | 26/50 | 52% | 10 任务×5 官方初态＝50 集 | 10-26 15:51:38 | 20,393 | 对 5000（22/50）：14 例改善、10 例退化，净 +4 |
| 7000 | 35/50 | 70% | 同一 50 集 | 10-27 01:30:50 | 18,137 | 11 例改善、2 例退化，净 +9 |
| 8000 | 35/50 | 70% | 同一 50 集 | 10-27 11:17:11 | 18,274 | 9 例改善、9 例退化，净 0 |
| 9000 | 38/50 | 76% | 同一 50 集 | 10-27 21:14:21 | 17,205 | 8 例改善、5 例退化，净 +3 |
| 10000 | 30/50 | 60% | 同一 50 集 | 10-28 07:01:18（批次完成） | 源材料未记录 | 净 −8（两数相减）；逐例拆分未记录 |

五次评测均为完整 50 集：6000–9000 四次均记录 16 个评测作业 exit0、infra 0、评测 optimizer_updates 0，并有 50 份原始控制审计与 journal SHA 复核；10000 步记录为 result complete、50/50 完成、基础设施异常 0。旧周末批次中断的 11 集与 3 次未确认控制单独保留，未拼接。200–5000 步的完整结果与曲线见阶段报告 §7.1。10000 步的环境控制次数、逐例变化和 result.json 的 SHA 不在本地源材料中。〔E2，S1、S5〕

### 4.2 逐任务成功数

| task ID | 6000 | 7000 | 8000 | 9000 |
|---|---:|---:|---:|---:|
| 0 | 3/5 | 3/5 | 5/5 | 3/5 |
| 1 | 4/5 | 5/5 | 5/5 | 5/5 |
| 2 | 4/5 | 4/5 | 3/5 | 5/5 |
| 3 | 4/5 | 5/5 | 1/5 | 2/5 |
| 4 | 3/5 | 3/5 | 5/5 | 5/5 |
| 5 | 4/5 | 4/5 | 5/5 | 5/5 |
| 6 | 1/5 | 4/5 | 5/5 | 4/5 |
| 7 | 2/5 | 5/5 | 2/5 | 5/5 |
| 8 | 0/5 | 0/5 | 0/5 | 1/5 |
| 9 | 1/5 | 2/5 | 4/5 | 3/5 |
| **合计** | **26/50** | **35/50** | **35/50** | **38/50** |

每格为固定 5 集中的成功数；任务中文名见阶段报告 §7.2，7000 与 8000 两列与该表相同，为便于逐行对照一并列出。10000 步的逐任务成功数不在本地源材料中，只有合计 30/50。〔E2，S1、S5〕

- 9000 步相对 8000 步的净变化：task 7 由 2/5 升到 5/5，task 2 由 3/5 升到 5/5，task 3、task 8 各多 1 例，其中 task 8 是 5000 步之后的第一个成功；task 0 由 5/5 回落到 3/5，task 6、task 9 各少 1 例。〔计数 E2〕
- 相邻 checkpoint 间摆动达到 2 例的任务有 0、2、3、4、6、7、9 七个：task 3 最大一次摆动 4 例（5/5→1/5），task 7 三次相邻变化都是 3 例，task 6 由 1/5 升到 4/5；只有 task 1、5、8 的相邻变化不超过 1 例。平均分相同时逐例并不相同（7000→8000 为 9 例改善、9 例退化），平均分上升时也有任务回落。〔计数 E2；含义见 R1 §7.2〕
- 10000 步比 9000 步净少 8 例，相当于 16 个百分点；回落落在哪些任务、是否集中于长程任务，本地材料不足以判断。〔差值由两数相减；归因未定，E3〕

### 4.3 训练推进与节奏

| 训练段 | 启动（SGT） | 看护快照（SGT） | 八 rank 各自完成 | trajectory 步 | 含历史回放的总确认更新 | 段末 published cost（charged＝confirmed） |
|---|---|---|---:|---:|---:|---:|
| 6000→7000 | 10-26 15:51:39 | 10-26 19:39 | 445 | 6445 | 6446 | 7001 |
| 7000→8000 | 10-27 01:30:52 | 10-27 05:42 | 485 | 7485 | 7486 | 8001 |
| 8000→9000 | 10-27 11:17:13 | 10-27 15:47 | 521 | 8521 | 8522 | 9001 |
| 9000→10000 | 10-27 21:14:23 | 10-28 01:48 | 539 | 9539 | 9540 | 10001 |

每次快照 unknown 均为 0，published cost 的 retries 均为 0。〔E2，S1〕

- 下一段训练在前一段产出 checkpoint 的评测结算后 1–2 秒即启动，因此相邻启动时刻之差就是一个完整周期（训练、保存导出、50 集评测与审计），依次为 9 小时 39 分、9 小时 46 分、9 小时 57 分；最后一段从启动到批次完成为 9 小时 47 分。〔E2 时间戳，间隔为相减所得〕
- attempt02 控制器 14:25 启动到 07:01:18 完成，共约 40 小时 36 分；启动前的估计是“约 40 小时”。〔E2；估计为 E3 计划，S5〕
- 从段启动到看护快照的段内平均节奏为 30.5–31.1 秒每步（含模型构建与精确恢复），与阶段报告记录的段末 100 步 28.2–28.8 秒稳态不矛盾；该值混合了设备计算、通信与主机等待，不能单独归因于某个环节。〔E3，由 E2 时间戳与步数相除；归因边界见 R1 §6.4〕
- 6000 步评测从确认 8 个 actor 进入模型加载阶段（14:31）到结算（15:51:38）约 80 分钟；其余各次评测的启动时刻源材料未逐字记录，只能给出结算时刻。〔E2 时间戳；缺口说明 E3〕

## 5. 问题与处置

| 问题 | 影响与处置 | 等级 |
|---|---|---|
| 首轮启动失败（10-26 12:07） | CPU wrapper 用 `python -c` 经 runpy 启动时缺少 target.parent 的 sys.path，`long_eval_sim` 第 5 行 `import long_eval_protocol` 失败；原 CPU 测试覆盖取消与进程所有权，漏了真实工作入口的同目录导入。actor、控制 journal、新增训练更新、未知更新均为 0，旧进程全部退出，原 3 服务恢复。只修 wrapper 导入路径，补 actual subprocess 的 RED→GREEN 与回归（45 项，49.434 s／OK／exit0），主 agent 做只到 ready 的真实桥接 smoke，定向独立 review PASS／0 findings。修复放隔离的 attempt02，新 gate 与 config 绑定首轮失败收口 SHA，首轮 claim、gate、源码、日志原样保留。看护规则同步调整：核实 claim 与 PID 后即恢复两小时看护，不再等 8 个 actor 就位 | 〔E1 源码行号与测试日志，转引 S1；影响 E2〕 |
| 评测收尾期 state 的 returncode 为 null（10-27 20:50） | 控制器不把进行中的 wait 快照落盘，进程已退出不等于异常；已有 complete 收据的 actor 为 optimizer_updates 0、all_model_parameters_frozen=true。只读核查，不判故障、不重启 | 〔E2〕 |
| 评测进行中的局部进度（9/50、18/50、46/50、47/50） | 一律不计算局部成功率，不提前启动下一段，不重复启动任何作业 | 〔E2〕 |
| 停止指令发布受阻（10-28） | 两次远程连接在 banner 阶段超时；先删除自动看护，第三次连接成功后才写入 STOP。远端批次此时已完成约 4 个半小时，STOP 没有打断任何训练或评测 | 〔E2〕 |
| 10000 步回落 | 固定集从 9000 步的 76% 降到 60%，净少 8 例；逐任务与逐例拆分不在本地源材料。按用户指令停止后未做复盘，原因未定 | 〔计数 E2；原因 E3〕 |
| 无 OOM／hang／数据／精度类故障记录 | 五次完整评测 infra 0；四段训练 unknown 0、retries 0；各次看护均未重启，未改活动源码、数据、依赖 | 〔E2，据 S1 各次记录〕 |
| 资源与存储 | 八卡作业期间按授权暂停三个既有服务容器，07:01:18 恢复且 restoration_errors 为空。10-27 15:47 共享盘 stat 快照空闲 274,877,906,944,000 字节，不代替配额证明 | 〔E2〕 |

## 6. 截至 10-28 的状态与未完成项

### 6.1 状态（2026-10-28 11:47 核实）

| 对象 | 状态 |
|---|---|
| 训练 | 停在用户上限 10000 步。最终完整 checkpoint 为 `checkpoint-010000`（清单 SHA `59a422e6…`），canonical 已导出；总实际训练更新 10001（含历史一次恢复回放），unknown 0。〔E2〕 |
| 评测 | 最终 10000 步 30/50＝60%；本协议固定 50 集的历史最高为 9000 步 38/50＝76%。〔E2〕 |
| 进程与资源 | 本实验的 controller、注册作业与 CPU bridge 全部退出，八卡训练容器 exited；diagnostic 与 MiniMax 容器 running；Qwen3.8-27B 双卡容器按用户追加指令保持停止，不得自动恢复。〔E2〕 |
| 自动化与记录 | motus 两小时看护已删除，`control/STOP.json` 已发布；本地 README 与 AGENTS 已改为停止态；checkpoint 与全部历史证据保留。〔E2〕 |
| 结论边界 | 以上是冻结 VLM 的 FP32 SFT 在同一固定 50 集上的观察。每任务只有 5 集，且这 50 集被反复用于开发观察；若选 9000 步作为交付 checkpoint，应另用独立初态与 seed 组确认。它也不是论文 97.6% 的同条件复现。〔样本结构 E2；解释 E3〕 |

### 6.2 未完成项

| 项 | 状态 | 说明 |
|---|---|---|
| 10000 步复盘 | 未开展，随停止指令终止 | 原定要回答“更高分有没有、哪些任务变差、收益是否值得计算成本、是否具备进入下一类对照的条件”。〔E3 计划，R1 §8〕 |
| 10000 步的逐任务与逐例拆分、控制次数、result SHA | 本地源材料未记录 | 需要读取节点上的 result.json；恢复须用户新授权。〔E2／E3〕 |
| 执行计划 Task 3 首项 | 计划文件中仍未勾选 | 属历史记录，不据此继续推进。〔E2，S3〕 |
| 新初态／多 seed 独立验证、VLM 解冻对照、World 分支消融、task-RL | 均未开始 | 阶段报告 §8 列出的后续建议，不是已完成结果。〔E3〕 |

## 7. 证据索引

### 7.1 本地现场（路径相对于工作区根）

| 编号 | 路径 | 内容 |
|---|---|---|
| S1 | `Motus-joint-rl/docs/superpowers/plans/2026-10-26-libero-long-resume-progress.md` | SDD ledger，本篇主材料；最后更新 2026-10-28 |
| S2 | `Motus-joint-rl/docs/superpowers/specs/2026-10-26-libero-long-resume-design.md` | 续训设计、边界与父批次证据 |
| S3 | `Motus-joint-rl/docs/superpowers/plans/2026-10-26-libero-long-resume.md` | 续训执行计划 |
| S4 | `Motus-joint-rl/docs/superpowers/plans/2026-10-23-weekend-long-train.md` | 周末长训计划与 10-26 收口核查 |
| S5 | `Motus-joint-rl/README.md` | 进度快照，更新于 2026-10-28 |
| S6 | `Motus-joint-rl/AGENTS.md` | 操作边界与停止指令 |
| R1 | `embodied-ai/weeks/2026_1026-1101_具身蒸馏调研/2026-10-27-Motus架构与LIBERO-Long阶段训练报告.md` | 阶段报告 v1.0 |

### 7.2 节点产物（npunew，路径相对于实验根 `/data/motus-joint-rl-20261015/`）

- `reproductions/libero-long-20261023/continuation-20261026/`（首轮，启动失败）：`evidence/task1-report.md`、`evidence/task1-green.log`、`evidence/task-review-v1.json`、`evidence/root-preflight-20261026/result.json`、`evidence/root-resource-snapshot-20261026T0350/`、`evidence/root-gate-validation.json`、`evidence/root-immediate-prelaunch.json`、`evidence/root-launch.json`、`evidence/monitor-failure-20261026T0616/failure-closure.json`。
- `reproductions/libero-long-20261023/continuation-20261026-attempt02/`（实际执行）：`evidence/root-start-verification.json`、`evidence/heartbeat-update.json`、`state.json`、`control/STOP.json`；`evidence/controller-01/eval-step00NNNNN/result.json`（6000–8000 的路径见 R1 附录 A.2，9000 与 10000 按同一命名规则推断，E3）；`evidence/monitor-20261026T1135/`、`monitor-20261026T1639/`、`monitor-20261026T2141/`、`monitor-20261027T0237/`、`monitor-20261027T0745/`、`monitor-20261027T1248/`、`monitor-20261027T1747/`、`monitor-20261027T2243/`（各次看护的 verification.json 与 npu-smi.txt）；`evidence/user-stop-20261028T0336/verification.json`。
- `reproductions/libero-long-20261023/weekend-20261023/evidence/monitor-20261026T0112/final-closure.json`（周末批次收口）。
- `evidence/user-stop-qwen38-npu3-7-20261028T034519Z/`（request.json、result.json、release-check.json 与前后 npu-smi 快照）。

### 7.3 关键 SHA256

| 对象 | SHA256 |
|---|---|
| 9000 步 result.json | `4782e5f46331d9a001fafe549919882d73e577c24071e3bcf5824814ded6ef9e` |
| 9000 步 checkpoint 清单 | `cb3e218e0a0650e03b335c4ed846d93bb5523fbbaea9b7d64f36f11ab8e6ae15` |
| 9000 步 canonical（receipt／plan 绑定） | `fd23ff32b3365636a20055976f8e8f27e151dcb38f3904301973669969fd25ce` |
| 10000 步 checkpoint 清单 | `59a422e699ec0cb6a6e8f5d8d32c147204ff12b0534cb05625543cd4948f60c1` |
| 10000 步 canonical（receipt／plan 绑定） | `803957767a6638a41253940fc5c9db5f297cd0ec5714b90f2a638d596be586ca` |
| attempt02 gate | `221d08b2887b460dddd3199c85ffc18a3fdaa5f6855663e9c53b13f393dc255e` |
| attempt02 config | `a0fe1fb9b3cb82addcfbf0fdaecf42ebb5e1e56f5d16fc50fe1ae918ef89e3c7` |
| 父批次 checkpoint6000 complete 清单 | `7446cf95cbdb5d38de0914e707d6ef2929e4d5353442b86897cab84c2adb8a98` |

6000／7000／8000 的 result.json SHA 与 7000／8000 的 checkpoint 清单 SHA 已列于 R1 附录 A.2，本篇不重复。各次看护对大 payload 只核对清单、metadata 与分片 stat，未重新 hash 约 79 GB 的分片与约 32 GB 的 canonical。〔E2，S1〕

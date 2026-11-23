# HomeBody VLM×SONIC 扩展与 ARDY 诊断执行总结（2026-11-03 ~ 11-06）

> **版本：1.0｜补录日期：2026-11-23**  
> **记录日期范围：2026-11-03 ~ 2026-11-06（Asia/Singapore）**。主线复现（10-27 ~ 10-28）见 [复现报告](../2026_1026-1101_具身蒸馏调研/2026-10-28-HomeBody复现报告.md)；11-07 以后的进展见 [分支报告摘要（截至 11-13）](../2026_1109-1115_Qwen换脑与HALO调研/2026-11-13-HomeBody-VLMxSONIC分支报告摘要.md)，本篇不写。  
> **一句话结论：** 11-04 先把主线复现冻结在 `6f7bbb8`（tag `freeze/homebody-20261104`），随后在实验分支 `codex/sonic-interface` 上分两步走。第一步让 VLM 经新技能 `perform_motion` 编排 SONIC 动作库，15 段连续验收全过。第二步转入「自然语言 → 生成动作 → SONIC 跟踪」的开放生成线；截至 11-06，六个正式套件只有 pose 通过（60/60，门槛 57，且 60 例都没有调用 VLM），generate、edit、program、soak、faults 五套的正式运行尚未执行。ARDY 链路卡在接触、供给、跟踪三处，真实 VLM 时延仍远超预算。〔E2，S7、S8；E3，S1:L123〕  
> **证据口径：** E1＝源码／日志行号实证；E2＝配置／数据文件实证（含 git 提交记录，以及 run 目录里落盘的 summary、score、measurements 文件）；E3＝文档解释或推断（须标注）；E4＝未复核转述（本篇未使用）。本篇为事后补写：E1 是对分支提交的只读源码核对，E2 是对本机镜像小文件的重算，没有重跑实验，也没有登录远程节点。进度日志（S1）里有叙述、但没有回查对应数据文件的数字一律标 E3；计划类内容标〔E3 计划〕。进度日志自带的 E 前缀是另一套图例，本篇没有照抄，按上述定义重新判定。本篇只取 S1 中 11-06 及以前的条目（L1–L160）。  
> **编号与媒体：** P01–P20、G01–G10、A01–A10、W01、R06 是预注册表与覆盖矩阵里的试次编号，不是证据等级。回放 mp4 与截图都在实验现场 `runs/` 下，未随仓库入库，下文只给现场相对路径。

## 1. 背景与目标

[复现报告](../2026_1026-1101_具身蒸馏调研/2026-10-28-HomeBody复现报告.md)的结论是主线复现成立，且决策接口不挑决策源：GLM 子 agent 与 DeepSeek 接入时都没改框架代码。〔E3，S2 §四〕此后的工作分三段：10-27 ~ 10-28 的主线复现见 S2；11-03 ~ 11-04 把 17 任务扩成 17×3 波次并冻结基线（§2）；11-04 起在冻结点旁接入 SONIC，先做动作库，再做开放动作生成与在线编辑（§3–§6）。〔E2，S6；E3，S2〕

开放生成线要达成的是：用自然语言临时指定全身动作和左右手臂姿态，能持续执行、组合、中途修改。复杂动作不要求提前录制，也不要求 VLM 为每个小动作逐次决策；简单行为允许用确定性的参数化实现。〔E3 计划，S3-c §1〕

已选定的 A 路线：冻结通用 VLM、预训练动作生成器和 SONIC 权重；VLM 低频产出类型化的持续意图或局部 patch；本地 grounding／IK／编排器保存并推进目标；ARDY-G1 持续补充未来运动；SONIC 是唯一的 body motor writer。不转为训练 VLA，不做预录片段检索或固定舞蹈循环。〔E3 计划，S3-c §3、S4-b〕动作库播放和 15 段跟踪只算底层基线，不算这条线的完成条件。〔E3 计划，S3-c §1〕

## 2. 基线冻结与 51 场波次

### 2.1 冻结点

| 项 | 内容 | 证据 |
|---|---|---|
| 冻结提交 | `6f7bbb8`，2026-11-04 10:29，提交说明 "chore: freeze HomeBody reproduction baseline before SONIC experiment" | 〔E2，S6〕 |
| 冻结标签 | `freeze/homebody-20261104` 指向 `6f7bbb8`；补录日复核，`main` 与该标签仍指向它 | 〔E2，S6〕 |
| 冻结内容 | 14 个文件：波次配置 `configs/waves/all17.json`（17 任务）与 `coverage13.json`（13 任务，带 targets）；`vlm/{catalog,compat,relay}.py`、`session/config.py`、`tools/run_wave.py`、`tools/smooth_video.py`；三份 HTML 报告与一份 Markdown 报告；两份本地运行说明 | 〔E2，S6、S13〕 |
| 实验分支 | `codex/sonic-interface`；冻结后首个提交 `fe088eb`（11-04 10:45） | 〔E2，S6〕 |

### 2.2 51 场波次

| 指标 | 数值 | 证据 |
|---|---|---|
| 设计 | 官方 17 任务，每任务先跑 1 次覆盖，再以 3 次重复统计；51 场，6 路并发；决策模型 DeepSeek V4.1 Flash | 〔E3，S2 §二；任务名单与冻结配置逐项一致，E2，S13〕 |
| 结果 | 首轮覆盖 10/17；3 次重复 32/51＝63%（95% CI 49–75%） | 〔E3，S2 §二；逐场评分原件在远程节点，本篇未重算〕 |
| 成本与时间 | 含过程实验共 76 场有效会话、约 800 次决策调用；模型等待占墙钟 56%（均值 51 s/步）；波次墙钟 12.8 h 经 6 路并发压到 4.3 h；最大健康并发 6（内存约束 3.5 GB/进程） | 〔E3，S2 §二〕 |
| 本机镜像 | 17 条任务跟拍 `runs/tasks-smooth/01-default.mp4` ~ `17-moved_juice_to_bin.mp4` | 〔E2，主检出目录清单〕 |
| 完成时间 | 11-03 19:46 保存的报告修订备份已含 32/51 表，波次在 11-03 晚之前结束 | 〔E2，工作区 `.revision-backups/homebody-control-20261103/复现报告.md`〕 |

读数时注意分母：复现报告 §二已是 51 场口径，§三「详细分析」与 §四「结论」仍是早期 9 场口径（物体级 21/30、default wave 1/4 等）。〔E3，S2〕

## 3. SONIC 动作库扩展（11-04）

| 提交 | 内容 | 证据 |
|---|---|---|
| `fe088eb` 10:45 | 离线 Protocol v1 探针：`src/homebody/sonic.py` 的 `pack_reference()` 与 `tools/sonic_probe.py`。只封包和校验，不开 socket、不加载模型、不启动物理 | 〔E2，S6；E3，S5-b〕 |
| `81186c8` 11:44 | SONIC 全身技能与 VLM 反馈闭环：`skills/perform_motion.py`、`primitives/whole_body.py`、`backends/sonic.py`、`sonic_policy.py`、`sonic_motion.py`、`sonic_evaluation.py`、`tools/sonic_run.py`、`tools/sonic_relay.py`、`configs/sonic-assets.json`，以及 10 个测试文件 | 〔E2，S6 文件清单〕 |

探针回执：全量 421 passed / 11 skipped / 1 failed，失败项与冻结基线相同（396 passed），是 relay provider 的一条既有断言；资产完整性 802 个文件、0 issue。〔E2，S13 日志尾行〕新增 25 项测试通过，并与官方封包函数逐字节一致（1796 字节）。〔E3，S5-b〕

技能契约：`perform_motion` 只收目录里的 `motion_id`，不接受文件路径；声明失败码 `TRACKING_LOST` 与 `TIMEOUT`；执行走 `preparing → executing → settling → completed`。〔E1，`src/homebody/skills/perform_motion.py:12、48-55` @`81186c8`；E3，S5-a〕

控制边界：29 个身体关节由 SONIC 独占（encoder `[1,1762]→[1,64]`、decoder `[1,994]→[1,29]`，50 Hz，每个 tick 走 4 个 5 ms 物理子步），14 个手指关节保持张开；会话注册表只有 `perform_motion` 与 `done`，默认的 AMO 和 navigate／pick／place 不加载；只在 `reset()` 设置 qpos；96 个资产文件按 SHA256 清单在启动物理前逐文件校验。〔E3，S5-a、S5-b、S5-c（文档引源码行号，本篇未逐行回查）〕

验收结果如下。预注册阈值是完成率 ≥95%、无非预期摔倒、joint RMSE <0.35 rad、root 高度误差 P95 <0.20 m。〔E3 计划，S3-b〕

| 运行 | 结果 | 整条路径 joint RMSE | root 高度误差 P95 | 证据 |
|---|---|---|---|---|
| 15 段连续验收（5 动作 × 3 次，只初始化一次） | 15/15 通过 | 0.12058 rad | 0.04769 m | 〔E2，S8 `acceptance-final`〕 |
| 真实 VLM 五段闭环（GPT-6 Astra） | 5/5，`task_verified=true` | — | — | 〔E2，S8 `vlm-closed-loop`〕 |
| 真实 VLM 三段闭环（GPT-6 Astra） | 3/3，`task_verified=true` | 0.11263 rad | 0.03545 m | 〔E2，S8 `vlm-final`〕 |
| 5 类动作中点取消 | 均回到站姿 | 站姿 RMSE 0.0405–0.0777 rad | root 高度 0.7619–0.7635 m | 〔E2，S8 `cancellation`（5 条结果）；数值 E3，S5-a〕 |

逐动作（每动作 3 次）：dance_party 497 帧、0.16995–0.17035 rad；squat 424 帧、0.12854–0.12926 rad；kick 165 帧、0.12650–0.12709 rad；macarena 1375 帧、0.12967–0.13002 rad；turn 455 帧、0.11165–0.11174 rad。〔E2，S8 `acceptance-final` 重算〕

边界：没有跑原版 C++／DDS 部署链；没有做 NPU 适配；这一步不生成新动作，没有音乐节拍输入，也没有自由文本到动作的生成。〔E3，S2 §5.6、S5-a〕

## 4. VLM×SONIC 设计与计划要点（11-04）

### 4.1 三份设计

| 文档 | 要点 | 证据 |
|---|---|---|
| S3-a 接口探针设计 | 冻结 `6f7bbb8` 与标签；只验证「上游运动参考 → SONIC 部署端」的数据边界：Protocol v1 数据包（1280 字节 header、小端 float32／int64、显式 29 关节顺序）与离线探针；不改 navigate／pick／place、AMO、Session；不启动远程服务、不下载模型 | 〔E3 计划，S3-a〕 |
| S3-b 复杂动作闭环接入 | 五条完成要求：真实 encoder／decoder 与 MuJoCo 接触动力学，身体关节只有一个写入方；连续参考与开始、状态、结果、取消；VLM 依据任务、观测和动作目录选择与组合动作；至少覆盖舞蹈、深蹲及另一类动态动作，正式验收每类至少三次；冻结基线可恢复。给出预注册指标与模块边界 | 〔E3 计划，S3-b〕 |
| S3-c 开放动作生成与在线编辑设计 v1.0 | 五类典型交互；「已有事实与未验证事项」表；架构为「慢语义决策、持续本地动作、快本体控制」七个模块；意图带 schema version、revision、`pose/generate/program` 路径、具名约束、保留目标；局部修改用 compare-and-swap；本地 program 只支持白名单的阶段与条件；所有 reference 以 50 Hz 整数帧标记，ARDY 原生 25 Hz 须重采样；生成 worker 不访问 MuJoCo data、不推进物理 | 〔E3 计划，S3-c〕 |

### 4.2 三份计划

| 文档 | 要点 | 证据 |
|---|---|---|
| S4-a 接口探针计划 | 一个 Task（参考数据包与离线探针），按 TDD 顺序写定；末尾附 11-04 执行回执 | 〔E3 计划，S4-a〕 |
| S4-b goal 模式提示词 | 可直接复制到执行会话的目标说明：A 路线、分支与冻结点约束、完成时必须有的六类证据（未预录的姿态、真实生成、两次以上在线修改、本地编排、真实 VLM 闭环、全部正式套件与阈值） | 〔E3 计划，S4-b〕 |
| S4-c 详细执行计划（829 行） | §0 接手与恢复；§1 验收合同；§2 模块与文件地图（新命名空间 `src/homebody/intent/`）；§3 接口合同；Task 0–10 对应阶段 P0–P6；§15 常见失败定位；§16 续跑规则；附录 A 正式任务清单 | 〔E3 计划，S4-c〕 |

### 4.3 验收合同（预注册）

下表与跨套件门槛（无非预期摔倒与 solver warning；joint RMSE <0.35 rad、root 高度误差 P95 <0.20 m；供给 P95 不超过新增可用参考时长的 80%；常见命令时延 P95 ≤3 s、更新时延 ≤2 s、复杂指令 ≤15 s）都是预注册内容。〔E3 计划，S4-c §1、附录 A〕

| 套件 | 规模 | 通过门槛 |
|---|---|---|
| pose | 20 条中文指令 × 3 种实际初态＝60 trial | ≥57/60；位置误差 ≤0.05 m，朝向误差 ≤15°，保持 ≥0.5 s |
| generate | 10 条复杂动作指令 × 3 seed＝30 trial | ≥27/30，语义与物理同时通过 |
| edit | 6 个持续会话 × 3 seed＝18 episode（至少 36 patch） | ≥17/18，两次编辑均成功 |
| program | 6 个本地编排任务 × 3 初态＝18 trial | ≥17/18 |
| soak | 3 次至少 120 s 的连续生成 | 无 underflow、无自动 reset |
| faults | 15 项故障清单 | 拒绝与故障路径正确，证据完整 |

到 11-06 末，`src/homebody/intent/` 实有 14 个文件，比计划多出 `contact.py`、`fastpath.py`、`registration.py`、`textop.py`；计划里的 `primitives/motion_intent.py` 与 `skills/{start,update,stop}_intent.py` 在树里不存在。〔E2，S6 @`0b109e2` 文件清单〕

## 5. 正式套件结果

| 套件 | 通过／总数 | 日期 | 证据等级 |
|---|---|---|---|
| pose | 60／60（门槛 57） | 2026-11-06 | E2（S7，独立评分文件重算） |
| generate | 未验收／30 | 截至 11-06 未正式运行 | E3（S1:L123） |
| edit | 未验收／18 | 截至 11-06 未正式运行 | E3（S1:L123） |
| program | 未验收／18 | 截至 11-06 未正式运行 | E3（S1:L123） |
| soak | 未验收／3 | 截至 11-06 未正式运行 | E3（S1:L123） |
| faults | 未验收／15 | 截至 11-06 未正式运行 | E3（S1:L123） |

五套未验收的套件是「正式运行尚未执行」，不是「跑了没过」；当天的生成、编辑、演示结果都是开发诊断，不计正式分母。〔E3，S1:L69、L123、L150〕pose 套件的经过见下表。

| 轮次 | 结果 | 说明 | 证据 |
|---|---|---|---|
| 预检 05 | 2／3 | standing 例首个活跃帧间隔 104.65 ms，超时钟阈值 | 〔E2，S7；数值 E3，S1:L78〕 |
| 预检 06 | 3／3，seal | — | 〔E2，S7 `sealed/`〕 |
| formal-pose-01 | 45／60 | 15 例失败：P15–P17 共 8 例经 VLM 的目标窗口没有形成；P20 两例 user→response 29.116／27.058 s；另 5 例时钟断点（P03、P12、P15、P18、P20 各一） | 〔E2，S7 重算；归因 E3，S1:L94、L96〕 |
| fastpath 定向复测 | 7／7 | 针对上述失败的初态，0 次 VLM，最长 user→actual 1.850 s | 〔E3，S1:L106〕 |
| 预检 07 | 3／3，seal | — | 〔E2，S7 `sealed/`〕 |
| formal-pose-02 | 60／60 | `all_trials_physics_passed` 与 `complete_evidence` 均为真；user→response 最大 1.850 s、中位 1.166 s（日志记 1.167 s）；joint RMSE 最大 0.0489 rad；root 高度 P95 最大 27.56 mm；重验 2011 个文件 | 〔E2，S7 重算〕 |

两点限定。第一，formal-pose-02 的 60 例 `normal_vlm_calls` 全为 0：指令由本地 fastpath 解析，语义目标由 evaluator 独立核验，所以 60/60 证明的是几何、跟踪与时钟，不证明真实 VLM 的语言理解。〔E2，S7；解释 E3〕第二，评分文件的 `scope` 字段是 `normalized_core_score_not_all_required_project_gates`，pose 通过不等于总 goal 完成。〔E2，S7〕

## 6. ARDY 诊断与备选线

### 6.1 11-06 当日脉络

当日提交锚点：00:05 `323dc09`（注册 pose 套件与可验证原始执行证据）；09:28 `6aad114`（TextOp 集成）；12:19 `5551c0d`（formal-pose-02 独立评分 60/60，该运行 12:06 结束）；12:25 `9accbf4`（朝向约束）、12:55 `0783d77`（思考门控）、14:14 `a95d14a`（DeepSeek 姿态会话）；16:09 `a97019d`（演示门控迁到 settings）、16:44 `bd78139`（演示修复）；17:49 `929bf70`（挥臂措辞校准）；18:12 `0b109e2` 为当日末提交。〔E2，S6、S7；E3，S1:L122〕11:46 起 Codex 客户端用量限额暂停，由 ZCode 会话代查。〔E3，S1:L120〕

### 6.2 11-05 的铺垫：姿态线诊断与 VLM 时延

- 姿态 smoke：左右各 +0.05／+0.10 m 共 4/4 达到诊断阈值；joint RMSE 0.02621–0.03740 rad。〔E3，S1:L37〕
- 连续姿态 06：两个 patch 连续完成，revision 1→2→3，无 reset／warning；user→response 1.791／1.850 s。〔E3，S1:L58〕
- 两阶段 program 01：按序消费 `reference_phase_id`，`program_complete`；VLM 0 次。〔E3，S1:L59〕
- 首次真实 VLM 在线姿态会话：Astra 1 次调用，远程等待 40.811 s；user→accepted 41.250 s，时延 FAIL。〔E3，S1:L60〕
- 其他 VLM 探针：Astra 30.474 s、Luna 20.202 s；复杂动作文本 20.204 s；Luna 带新图 25.298 s。〔E3，S1:L61、L77〕

这些都是开发诊断，不计正式分母。〔E3，S1:L58、L69〕VLM 一侧的时延问题贯穿到 11-06，用户因此指定探测 DeepSeek 替代 Astra（§6.8）。〔E3，S1:L125〕

### 6.3 文本编码器解锁（11-05 ~ 11-06）

ARDY 的文本编码依赖 gated 的 Meta-Llama-3-8B-Instruct，正常 HF 客户端访问返回 401，官方生成的首次尝试因此失败。〔E3，S1:L27、L36、L72〕用户先后授权了 NPU 节点和新增的第二台 GPU 节点。新 GPU 上已有他人任务常驻约 9.5 GB、利用率 96% 以上，可用 RAM／VRAM 放不下 8B 编码器，于是改用 NPU 节点的空闲 CPU，八张 NPU 上的既有任务不动。〔E3，S1:L71、L84〕

| 步骤 | 结果 | 证据 |
|---|---|---|
| 权重恢复与 CPU 加载 | 四片权重共 16.06 GB，逐片 SHA256 与 canonical revision 一致，两套 LLM2Vec adapter 校验完整；FP32 冻结参数 7,546,867,712；两种文本都得到有限非零的 `[1,4096]`，独立探针 30.26–31.05 s | 〔E3，S1:L72、L84–L85〕 |
| 回环 API | 热调用 3.885–4.296 s，输出 `[1,1,4096]`，与独立输出的最大差为 0 | 〔E2，S10 `encoder-api-probe-01.json`〕 |
| 官方生成 03 | CSV `[104,36]` 转成 207 帧 50 Hz reference（4.12 s），最大关节速度 0.445 rad/s、加速度 16.76 rad/s²，动力学预检无拒绝 | 〔E2，S10 `reference-validation-01.json`；10 步、seed 0 与 4.16 s 请求为 E3，S1:L86、S5-d〕 |
| 坐标往返 | 16 帧特征 `[1,16,414]`，root 位置误差 ≤2.99e-8 m、转角 ≤5.91e-9 rad、hinge ≤1.23e-7 rad | 〔E2，S10 `ardy-history-roundtrip-01.json`〕 |

### 6.4 ARDY → SONIC 的三道关

| 关 | 现象 | 当日进展 | 证据 |
|---|---|---|---|
| 接触 | 原始 22 窗用 SONIC 几何做接触预检 0/22 通过，最深 55.78 mm（左拇指末节与左 hip roll 自碰撞） | 逐帧 ±0.15 rad 投影在 25 Hz 认证 14/22，转 50 Hz 后只剩 3/22；常量关节修正 19/22、全帧线性修正 18/22，两者并集 22/22 | 〔E3，S1:L98–L100〕 |
| 供给 | 投影 wall 中位 2.360 s、最大 5.332 s，超过单窗可用 tail；CPU 采样加转换 P95 1.584 s，CUDA 为 0.132 s | 加入分项计时与调度；仍无正式供给判定 | 〔E3，S1:L97、L99、L112〕 |
| 跟踪 | live08 连续 915 帧（18.30 s）后，左肩 yaw 误差 0.5767 rad 超过 0.50 门槛，触发受控停止 | RNG 隔离后 live09 连续 30.00 s／15 次续接，但有一次 281.87 ms 的活跃帧间隔，未过时钟；冷桥接触修正后 live11 连续 42.00 s／21 次续接，最大活跃帧间隔 42.92 ms | 〔E2，S10 `ardy-rng-stream-compare-01.json`、`ardy-owner-clock-01.json`〕 |

这三条都是结构化输入、冻结模型、真实 SONIC 力矩仿真，没有自然语言 VLM 闭环，不计正式结果。〔E3，S1:L102、L113〕用冻结 SONIC 重放 live08 的 905 个实际帧，重放与实测的逐帧最大 action 误差中位 0.000770、最差 0.01362；日志的结论是这只是同状态离线对照，没有证明物理上可修复。〔E3，S1:L107〕

### 6.5 真实中文入口

- `ardy-language-live-01`：真实 Astra 决策 1 次，24.948 s 后准入；模型到首个生成 worker 才装载，触发 `worker_timeout`，0 在线帧。〔E2，S10 `ardy-language-analysis-01.json`〕
- `ardy-language-live-02`：显式预热 8.236 s；user→accepted 20.809 s、user→online 29.134 s、accepted→online 8.326 s（首条文本编码占 7.083 s）；连续 818 帧（16.36 s）、8 次续接；frame 2324 右肩 yaw 误差 0.5069 rad 触发停止。〔E2，S10 同上；7.083 s 为 E3，S1:L109〕

两轮都没过正式的时长、响应或语义验收。〔E3，S1:L109〕

### 6.6 朝向稀疏约束（11-06 午间）

3 个 seed × base／相对 ±45°，冻结 ARDY API 编码器，无物理。base 组偏航自由漂移（−0.33 ~ +0.75 rad），说明收敛来自约束而不是 prompt；右转 3/3 收敛（末端误差 0.132／0.139／0.134 rad），左转 3/3 收敛（0.290／0.066／0.209 rad）；mask 28 格，CPU 单窗采样约 5.2 s。〔E2，S10 `ardy-heading-core-01/report.json`；「收敛来自约束」的解读 E3，S1:L124〕大转角的末端误差偏大，日志判断接入实测时可能需要 dwell 内重采样或更长时间窗。〔E3，S1:L124〕

### 6.7 TextOp 备选线（11-05 ~ 11-06）

用户在 11-05 回复「那就先算了」，暂停 Llama 授权请求，继续已获准的 TextOp 备选；TextOp 只做隔离调研，没有切成正式 generator。〔E3，S1:L40、L46〕

| 项 | 结果 | 证据 |
|---|---|---|
| 供给 | 固定源码与公开权重完整下载核验；22 个窗口（2 预热＋20 测量），50 Hz、23 关节，独立生成 P95 0.0512 s／0.16 s | 〔E3，S1:L47〕 |
| 适配边界 | 6 个锁定手腕参考取自本体感知与已调度历史；硬空间约束和 tempo 不支持，显式拒绝 | 〔E3，S1:L48；E2，S12〕 |
| smoke 01–02 | 源码哈希被 AppleDouble sidecar 污染，预热失败；缺 vendor import，物理未启动 | 〔E3，S1:L53〕 |
| smoke 03 | 预热 11.57 s、物理 3.86 s；首条参考被 8 rad/s 速度校验拒绝，0 在线帧 | 〔E2，S12；193 个 tick E3，S1:L54〕 |
| 速度与接触 | 原生输出最大割线速度 9.5538 rad/s、加速度 249.238 rad/s²，不是单位或冷入场问题；±0.15 rad 有界投影在第 6 窗被 LP 证明不可行 | 〔E3，S1:L62〕 |
| 续接与编译版 | 参考足底穿地约 20–30 mm，加入有界接触投影后 contact30-21 实际 10.20 s 后低水位；三次续接成本 1.932–2.026 s，有效新增 2.28 s，超 0.8 供给阈值；compiled30-24 实际 588 在线帧（11.76 s），供给 P95 比例 1.2949，仍失败 | 〔E3，S1:L74、L88〕 |

### 6.8 DeepSeek 关思考与时延探针（11-06 午间）

动机：Astra 的姿态会话 user→accepted 41.250 s，远超 3 s 响应上限；用户指定用既有的云端 OpenAI 兼容端点探测 DeepSeek 能否替代。〔E3，S1:L60、L125〕

| 探针 | 结果 | 证据 |
|---|---|---|
| 思考开启 | 文本 14.57–24.56 s（3/3 解析通过）；带 ego 图 7.87–21.17 s（2/2） | 〔E2，S9 `report.json`〕 |
| 同端点模型全扫 | 只有 kimi-k2.6 另外带视觉，耗时 106.8 s；glm、qwen、openpangu 系都拒收图片 | 〔E2，S9 `sweep.json`；模型归类 E3，S1:L125〕 |
| 关思考方言 | `thinking:{type:disabled}` 1.96 s，无 reasoning；`chat_template_kwargs` 2.38 s，无 reasoning；`thinking:false` 返回 HTTP 400；`reasoning_effort` low／minimal 20.4／15.7 s，仍带 reasoning；`budget_tokens: 512` 36.1 s | 〔E2，S9 `thinking.json`〕 |
| 稳定性 | 5/5 通过，2.04–3.17 s，中位 2.53 s | 〔E2，S9 `stability.json`〕 |
| 路由规律 | 量化措辞（0.15 m、15 cm）路由 pose；「举起右手并在肩上方保持不动」路由 generate | 〔E2，S9 `route.json`〕 |

provider 侧的门控：环境变量 `OPENAI_COMPAT_THINKING` 设为 `disabled` 时，请求体加 `thinking:{type:disabled}`；未设或未知值严格拒绝。〔E1，`src/homebody/vlm/compat.py:65-70` @`0b109e2`〕提交 `0783d77`。〔E2，S6〕非思考模式的语义准确性只在单一姿态 prompt 上验证，复杂的自然语言编辑指令还没测。〔E3，S1:L125〕端到端姿态会话三连（DeepSeek 关思考、真实 SONIC 力矩、`--realtime`）：

- live-01：模糊指令「请把右手举到右肩上方」被路由到 generate；没有生成器，按设计拒绝（`real_motion_generator_required`）。〔E3，S1:L127〕
- live-02：数值加掌向的指令，pose 路由，执行 351 帧；高度误差 6.5 mm 达标，`palm_normal` 误差 0.6045 rad 超容差，触发 `measured_reference_divergence` 保护停止。〔E2，S9 `deepseek-pose-live-02/summary.json`〕
- live-03：纯高度指令，649 帧完整执行，末段高度误差 5.0 mm；user→accepted 2.37 s，VLM 1 次调用。〔E2，S9 `deepseek-pose-live-03/summary.json`；时延 E3，S1:L127〕

live-03 的 summary 里 `runtime_succeeded=true`，但 `passed=false`、`formal_gate=not_evaluated`、`mode=diagnose`：它是开发诊断，不是正式通过。〔E2，S9〕日志还记下非思考模式的路由对模糊指令偏保守，pose／generate 边界的提示词要在正式套件前强化。〔E3，S1:L127〕

### 6.9 全路由交互演示（11-06 下午）

主树同步到分支 HEAD 后，ARDY 第一次经 `intent_run` CLI 全链路打通：`demo-ardy-03` 20 s／1049 帧、5 个生成窗全部拼接准入、user→accepted 2.01 s；`demo-edit-01` 的运行中编辑「右手再抬高 5 厘米」被接纳为 revision 2。〔E3，S1:L131–L132〕

**一个已知分支缺陷。** generate 路由下，阶段切换以 `immediate=True` 重新 staging，起点落在已准入续接窗的起点之前，被 buffer 以 `out_of_order` 拒绝，导致 `phase_failed`。〔E1，`src/homebody/intent/runtime.py:289` @`a95d14a`；E3，S1:L133〕当日没有正式修，演示 overlay 用 `goal_dwell_seconds=600`、`phase_timeout_seconds=3600` 抑制阶段切换来绕过；到 11-06 末提交，这处调用仍在。〔E1，`runtime.py:295`、`configs/intent-ardy-demo.toml:15-16` @`0b109e2`〕

**演示门控。** 用户要求错误也可见、不提前停止，随后又要求关闭踏步类安全拒绝。四项放宽先做成环境变量，因拼写分裂、且经多层远程执行时环境变量偶发丢失，整体迁到 settings：`intent.demo_unconditional` 默认 `False`，只在演示 overlay 里置真。〔E1，`src/homebody/session/config.py:95`、`configs/intent-ardy-demo.toml:17`；E3，S1:L134–L137〕overlay 首行写明 "never use this overlay for formal scoring"。〔E1，`configs/intent-ardy-demo.toml:1`〕它与正式默认的差距（正式默认 → 演示 overlay）：`max_joint_velocity` 8.0 → 16.0，`max_joint_acceleration` 80.0 → 300.0，`max_measured_joint_deviation` 0.50 → 3.0，`max_measured_root_deviation` 0.20 → 1.2，`orientation_tolerance` 0.2618 → 3.14 rad，`max_contact_penetration` 0.002 → 0.006，`max_splice_position_jump` 0.05 → 0.15，`goal_dwell_seconds` 0.50 → 600，`phase_timeout_seconds` 30 → 3600。〔E1，`src/homebody/session/config.py:67-68、75、78-82、90` 与 `configs/intent-ardy-demo.toml:2-17` @`0b109e2`〕演示修复链：

- `bd78139`：新动作不再让已连接的 SONIC 读到空参考；抬手走本地掌高不等式和 IK；撤销演示用的无界冷桥抬升（复现时曾到 3.54 m）。`demo-fix-e2e-06` 右掌起点在肩下 34.05 cm。〔E3，S1:L139；起点值 E2，S12 `summary-latest.json`〕
- 「放下」被 VLM 误判为 `stop_intent`（`demo-live-1106-164627`），改为同一会话内的局部姿态修订；`demo-fix-e2e-07` 左掌从肩上 6.29 cm 降到肩下 33.31 cm。〔E3，S1:L140；E2，S12 `raise-lower-recovery.json`（+0.0629 m → −0.3331 m）〕
- 踏步的推进：live10 0 帧（冷桥中段自碰撞）→ live12 7.16 s → live13 28.24 s（根位移 3.07 m，不满足「原地」）→ live19 5.56 s、7 个交替离地片段、漂 0.68 m → live21 7.50 s、8 个片段、漂 0.663 m，终态都是低水位。〔E2，S10 `ardy-stepping-live-13.json`、S12 `ardy-early-refill-physics-01.json`、`ardy-join-physics-final.json`；E3，S1:L114–L116、L141–L143〕
- 反例：live16 单手挥动 30 s，后验发现首窗右腕／髋碰撞约 71 mm；另一条宽松首窗的踏步在回待命期间实测跌倒。修复后演示冷启动也拒绝最终接触不合格的候选。〔E3，S1:L142〕

### 6.10 挥臂措辞校准（11-06）

固定 ARDY 权重、seed 0、CFG 2 的结构化对照，共 7 条。日志的结论是小动作在生成文本和原生样本阶段就已经出现。〔E3，S1:L147〕下表的在线时长取 `online_rows`／50。

| 提示 | 在线时长 | 左右掌相对躯干前后范围 | 终态 | 证据 |
|---|---|---|---|---|
| 原句（含 gently） | 30.00 s | 4.48／3.57 cm | 正常停止 | 〔E2，S11 `original/summary.json`；幅度 E3〕 |
| 删掉 gently | 30.00 s | 2.24／1.66 cm | 正常停止 | 〔E2，S11 `without_gently`；幅度 E3〕 |
| 站立对照 | 30.00 s | 1.07／0.74 cm | 正常停止 | 〔E2，S11 `no_motion`；幅度 E3〕 |
| 要求清楚可见、不给高度 | 30.00 s | 16.87／12.88 cm，双侧各 6 个峰 | 正常停止 | 〔E2，S11 `wide_without_endpoint`；幅度 E3〕 |
| 显式摆到肩高（两条） | 14.40 s、14.42 s | 63.86／70.29 cm、53.74／53.98 cm | 被 immutable splice head 预算拒绝 | 〔E2，S11 `wide_swing`、`gently_with_endpoint`；幅度 E3〕 |
| 过顶（三次重采样） | 0 在线帧 | — | hinge 跳变或肩 pitch 投影不可行 | 〔E2，S11 `overhead*`〕 |

语言侧：演示语言提示单独引导已实测的可见幅度用语。最终校准快照上从站立待命输入同一句中文，SONIC 消费 60.16 s、32 个 ARDY 参考窗，正常停止；去掉入场后左右掌前后范围 21.79／22.10 cm、显著峰 6／9、根水平最大偏移 7.21 cm。〔E2，S11 `measurements-calibrated.json`、`language-calibrated/summary.json`（3966 帧，`vlm_calls=1`，`formal_gate=not_evaluated`）〕全程直立无警告，没有验证精确节拍或同步。〔E3，S1:L150〕

### 6.11 官方动作与中文组合覆盖（11-06）

矩阵：上游交互 UI 的 10 条 Prompt List 原英文直接送进 ARDY（A01–A10），加计划 G01–G10 的中文原句经真实 VLM 进入同一 SONIC 力矩仿真；每条目标约 30 s，不计正式 generate30。〔E3，S1:L154；E2，S11 条目名〕

| 批次 | 结果 | 证据 |
|---|---|---|
| 基线 20 条 | 5 条正常停止到约 30 s：A06 站立对照、G01、G03、G04、G05；其余 15 条失败 | 〔E2，S11 `ardy-coverage-20261106-01/measurements.json` 重算〕 |
| 失败构成 | A01／A03／A04／G08 接触投影越界、A05 关节限位，均 0 s；A09／G10 入场或拼接违反接触几何，0 s；A08／A10 `worker_timeout`；A07 2.70 s 后拼接预算；A02 19.60 s 后 `measured_reference_divergence`；G07 5.56 s 后低水位；G02／G06／G09 采样前被「requires generation route at native tempo」拒绝 | 〔E2，同上；逐条细分 E3，S1:L155–L157〕 |
| 复测 8 条（entry-02） | 只有 W01 挥臂回归（30.20 s）正常停止；右脚踢从 0 s 推进到 9.44 s；倒走 1.22 s；前走、侧步、G01 原句复测仍 0 s | 〔E2，S11 `ardy-coverage-20261106-entry-02/measurements.json` 重算〕 |
| 两层问题 | 右转不指定慢速时，真实 VLM 输出 `until=goals_satisfied` 却没有目标，合同在采样前拒绝（R06，`language_failed`）；直送 ARDY 的右转加摆臂三个原生候选净偏航 −83.7° ~ −91.2°，全部被接触几何拒绝（A11） | 〔E2，同上；E3，S1:L160〕 |

基线里两条值得记的现象：A02 后跳的准入参考有 3 次双脚同时离地、实测 0 次，不能把参考离地算实际跳跃；G03 实测向前 0.783 m，左右脚显著抬脚段从准入的 21／27 降到实测 0／1。〔E3，S1:L155–L156〕冷入场失败的原因定位到入场五次曲线把起末关节速度扩散到整段 2.6 s；演示候选只对超限轴局部化端点速度后，三条归档复算都消除了越界。〔E3，S1:L155〕

## 7. 截至 11-06 的状态与遗留

### 7.1 状态

| 线 | 状态 | 证据 |
|---|---|---|
| 主线 | `main` 与标签仍为 `6f7bbb8`；实验分支没有配置 upstream，也没有远端跟踪引用；全程未 push、未 merge | 〔E2，S6；过程陈述 E3，S1:L89〕 |
| 姿态与其余五套 | 正式 pose 60/60（第二轮，首轮 45/60）；其余五套正式运行未执行 | 〔E2，S7；E3，S1:L123〕 |
| ARDY 连续生成 | 结构化输入最长 42.00 s；真实中文入口 16.36 s 后保护停止 | 〔E2，S10〕 |
| 决策 VLM | Astra 姿态会话 41.250 s；DeepSeek 关思考单例 2.37 s | 〔E3，S1:L60、L127；E2，S9〕 |
| 全仓测试 | 935 passed / 11 skipped（11-06 上午，L89）；其后是定向测试，如 `tests/intent` 464 passed（L159） | 〔E3，S1；本篇未复跑〕 |
| 提交量 | 冻结后到 11-06 末共 56 个本地提交：11-04 8 个、11-05 7 个、11-06 41 个 | 〔E2，S6〕 |

进度日志顶部的阶段表停在 11-06 上午，写的是「正式 pose60 运行中，未出最终分数」；以当日午间条目为准，P5 的 pose 套件已通过，P6 总体验收仍未验收。〔E3，S1:L3、L7–L15、L123〕

### 7.2 遗留（截至 11-06）

1. 五套正式验收未运行，总 goal 仍为 active。〔E3，S1:L123〕
2. pose 套件 60 例都没调用 VLM，真实 VLM 的语义验收不由它覆盖。〔E2，S7〕
3. ARDY 三道关都没解：接触投影耗时超单窗 tail；肩 yaw 的可跟踪范围没确认；冷桥中段自碰撞和腕部接缝只有部分修复。〔E3，S1:L99、L109–L110、L115〕
4. 语义与「原地」约束没解决：踏步根位移 0.66–3.07 m；官方的动态原句没有一条稳定跑满 30 s。〔E2，S10、S12；E3，S1:L116、L158〕
5. generate 路由的阶段切换缺陷只有演示绕过。〔E1，`runtime.py:295` @`0b109e2`〕
6. 首轮 pose 的 5 例时钟断点（约 0.2 s）根因未确认，复测没有复现。〔E2，S7；E3，S1:L96、L106〕
7. 真实 VLM 时延：Astra 仍远超预算；DeepSeek 非思考模式的语义只在单一姿态 prompt 上验证。〔E3，S1:L60、L125〕
8. 演示 overlay 放宽了九项参数并开启 `demo_unconditional`，其结果不能用于正式评分。〔E1，`configs/intent-ardy-demo.toml:2-17` @`0b109e2`〕

### 7.3 「复现报告 §09–§26」指什么

进度日志从 L150 起有 20 处写作「本机复现报告 §09」「原工作目录复现报告 §26」这样的引用，覆盖 §09–§26 共 18 个章节号。〔E2，S1 引用行清点〕核实结论有四条。

- **不是详细执行计划。** S4-c 的章节号是 §0–§16 加附录 A；它的 §9、§10 是 Task 5（滚动 reference buffer）和 Task 6（运行层），与日志引用处的内容（挥臂对照结果、动作覆盖结果）对不上，§17–§26 在计划里不存在。〔E2，S4-c 标题清单〕
- **是原工作目录里的分支复现报告。** 日志的用词是「本机复现报告」「原工作目录复现报告」。章节号与日志自 11-06「ARDY 挥臂措辞接入 SONIC」起的每个日期小节一一对应、顺序递增：11-06 两节对应 §09、§10；11-09 九节对应 §11–§19；11-10 七节对应 §20–§26。〔E2，S1 标题与引用行对照〕
- **引用的报告章节在本地未见对应文件。** 日志 11-11 的「定时编排复测与报告精简」一节写明「对外报告保留最新结果、代表性回放与当前验收差距，早期过程仅保留原始归档」。〔E3，S1:L289–L293〕主检出里现存的分支报告只有 HTML 一份，分四节（01 分支架构、02 三个样例、03 ARDY 还做了哪些动作、04 正式验收与当前卡点），没有 § 编号。〔E2，主检出文件清单与标题〕
- **本篇涉及的 §09、§10，底层证据仍在。** 分别对应 `runs/ardy-sonic-amplitude-20261106-01/` 与 `runs/ardy-coverage-20261106-01/`。〔E2，S11；对应关系 E3，S1:L150、L157、L160〕

## 8. 证据索引（现场相对路径）

分支文档在实验分支 `codex/sonic-interface`（worktree）上。`runs/` 路径里，标「主检出」的在主检出的 `runs/` 下，标「worktree」的在分支 worktree 的 `runs/` 下；`live01`–`live08`、`formal-pose-02` 原始 run、`pose-suite-preflight-*`、`demo-*` 只在远程实验目录，本篇经 S1 转引，没有直接读到。`runs/` 下的 jsonl、npz 与大体积 json 没有打开。源码行号对应的提交号写在引用处。

| 编号 | 类型 | 路径 | 说明 |
|---|---|---|---|
| S1 | 文档 | `docs/superpowers/plans/2026-11-04-vlm-sonic-generative-progress.md` | 进度日志，486 行、按日期分节；本篇只用 L1–L160 |
| S2 | 文档 | [复现报告公开版](../2026_1026-1101_具身蒸馏调研/2026-10-28-HomeBody复现报告.md) | §二 波次；§四 结论；§五 SONIC 扩展 |
| S3-a/b/c | 设计 | `docs/superpowers/specs/2026-11-04-{sonic-interface,sonic-runtime,vlm-sonic-generative}-design.md` | 依次 56、34、107 行 |
| S4-a/b/c | 计划 | `docs/superpowers/plans/2026-11-04-{sonic-interface-plan,vlm-sonic-goal-prompt,vlm-sonic-generative-execution}.md` | 依次 46、35、829 行 |
| S5-a/b/c/d | 说明 | `guides/sonic-runtime.md`、`guides/sonic-interface.md`、`guides/sonic-policy-evidence.md`、`guides/ardy-api-evidence.md` | 依次 123、106、82、66 行 |
| S6 | git | 分支提交记录 `6f7bbb8..0b109e2`；源码 `src/homebody/vlm/compat.py`、`src/homebody/session/config.py`、`src/homebody/intent/runtime.py`、`src/homebody/skills/perform_motion.py`；`configs/intent-ardy-demo.toml` | 只读核对 |
| S7 | 数据 | 主检出 `runs/intent-pose-suite-prototype-20261106-07/formal-pose-02-independent-score/score.json`；`runs/intent-pose-suite-prototype-20261106-06/formal-pose-01-independent-score/score.json`；`runs/intent-pose-suite-prototype-20261105-05/independent-score/score.json`；前两个目录下的 `sealed/` | pose 评分 |
| S8 | 数据 | worktree `runs/sonic-runtime/` 下的 `acceptance-final/`、`vlm-closed-loop/`、`vlm-final/`、`cancellation/` | SONIC 动作库验收 |
| S9 | 数据 | 主检出 `runs/deepseek-latency-probe-01/` 下的 `report.json`、`thinking.json`、`sweep.json`、`stability.json`、`route.json`；`runs/deepseek-pose-live-02/summary.json`、`runs/deepseek-pose-live-03/summary.json` | DeepSeek 探针 |
| S10 | 数据 | 主检出 `runs/intent-llama-recovery-20261105/` 下的 `encoder-api-probe-01.json`、`official-generation-03/reference-validation-01.json`、`ardy-history-roundtrip-01.json`、`ardy-language-analysis-01.json`、`ardy-rng-stream-compare-01.json`、`ardy-owner-clock-01.json`、`ardy-stepping-live-13.json`、`ardy-heading-core-01/report.json` | ARDY 恢复与诊断 |
| S11 | 数据 | 主检出 `runs/ardy-sonic-amplitude-20261106-01/`（各子目录的 `summary.json`、`measurements-calibrated.json`）；`runs/ardy-coverage-20261106-01/measurements.json`；`runs/ardy-coverage-20261106-entry-02/measurements.json` | 挥臂与覆盖矩阵 |
| S12 | 数据 | 主检出 `runs/demo-fix-evidence/` 下的 `summary-latest.json`、`raise-lower-recovery.json`、`ardy-early-refill-physics-01.json`、`ardy-join-physics-final.json`；`runs/intent-textop-sonic-smoke-03/diagnostic-summary.json` | 演示修复与 TextOp smoke |
| S13 | 配置与日志 | `configs/waves/all17.json`、`configs/waves/coverage13.json`（@`6f7bbb8`）；主检出 `runs/sonic-interface-probe-20261104/` 下的 `baseline-tests.log`、`experiment-tests-final.log`、`assets-check.log` | 冻结波次配置；11-04 测试与资产检查 |

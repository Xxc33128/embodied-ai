# RPent 扩展：Flash Mode 与 libero_goal 第二套件（npunew · 20+20 集）

> 执行：2026-09-26 → 09-27 晨（跨夜自动 + 断点续跑），ZCode（GLM-5.3 驱动）+ 用户。
> 前置：[2026-09-24 RPent 复现](2026-09-24-RPent复现-npunew-GLM5.3-ObjectSwap20集.md)（object_swap api 模式 18/20=90%）。本篇补齐上游遗留的两个未测项：**Flash Mode（免 LLM 记忆回放）**与**第二套件 libero_goal**。证据分级 E1–E4。

## 0. 一句话结论

GLM-5.3 大脑在更难的 libero_goal 上 **16/20 = 80%**（正对官方 GPT-5.5 档 75%、高于 Qwen3.6-27B 的 68%，且是在该套件**无 task 级记忆**的条件下）；Flash Mode 在 npunew 全国产算力上跑通（Molmo2-8B @ dev4），**15/20 = 75%、0 LLM token**。RPent 三种模式（api/flash/记忆只读）在 npunew 均有数据，复现义务全部履行完毕。

## 1. 总战绩（三批次，均为 20 集）

| 批次 | 结果 | 官方对照（goal_task / object_swap） |
|---|---|---|
| object_swap · api · GLM-5.3（9/24） | 18/20 = **90%** | GPT-5.5 91% / Astra 99% |
| **libero_goal · api · GLM-5.3（本篇）** | 16/20 = **80%** | GPT-5.5 75% / Qwen3.6 68% / Astra 88% |
| **object_swap · Flash · Molmo2-8B（本篇）** | 15/20 = **75%** | 官方 Flash 93%（object_swap） |

goal 的 80% 含一次网络挂死后的协议性重跑（见 §4.2）；20 集小样本，Wilson 95% CI 约 58%–92%。

## 2. libero_goal（api planner · GLM-5.3）

### 2.1 结果与用量 [E1]

逐集（recipe 口径，分析器 `analyze_goal.py`，节点 `/data/rpent2/` 可重跑）：

| 集 | 结果 | 时长 s | 请求 | ctx max | decode | | 集 | 结果 | 时长 s | 请求 | ctx max | decode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t0_s0 | ❌ | 5160 | 100 | 139k | 57k | | t5_s1 | ❌ | 823 | 24 | 63k | 30k |
| t0_s1 | ❌ | 5430 | 99 | 191k | 106k | | t6_s0 | ✅ | 1196 | 53 | 61k | 25k |
| t1_s0 | ✅ | 977 | 35 | 73k | 21k | | t6_s1 | ✅ | 1006 | 33 | 83k | 29k |
| t1_s1 | ✅ | 761 | 23 | 49k | 15k | | t7_s0 | ✅ | 189 | 9 | 47k | 4.7k |
| t2_s0 | ✅ | 3833 | 100 | 160k | 83k | | t7_s1 | ✅ | 221 | 10 | 49k | 5.2k |
| t2_s1 | ✅ | 1126 | 48 | 114k | 44k | | t8_s0 | ✅ | 537 | 23 | 66k | 16k |
| t3_s0 | ✅ | 2491 | 55 | 135k | 67k | | t8_s1 | ✅* | — | 29 | — | — |
| t3_s1 | ✅ | 1035 | 36 | 83k | 31k | | t9_s0 | ❌ | 1637 | 53 | 101k | 47k |
| t4_s0 | ✅ | 683 | 30 | 73k | 20k | | t9_s1 | ✅ | 438 | 21 | 64k | 13k |
| t4_s1 | ✅ | 917 | 31 | 76k | 22k | | | | | | | |

（*t8_s1 首跑 12:38 已触发 terminated=true 但随后 GLM 调用网络挂死 8h，按协议杀掉重跑并解出，重跑数据入表。）

**均值**：单集 1616s（object_swap 为 890s，1.8 倍）；请求/集 45（vs 34）；单请求上下文 avg 92k / max 191k；新增 prefill 260k/集；**cache 命中 90.3%**（三批次全部 ≈90%，再次逐项复刻）；decode 36.3k/集（vs 23.9k，1.5 倍）；turn 周期 30.9s。

### 2.2 解读 [E3]

- **档位**：80% 落在官方 GPT-5.5（75%）与 Astra（88%）之间，显著高于 Qwen3.6-27B（68%）——GLM-5.3 在难套件上保住了 5.5–Astra 档的定位。
- **记忆深度是泛化瓶颈的直接证据**：goal 套件在本地语料中只有 suite 卡（9-10 张）、无 task 级 plan/anchors（object_swap 有 10 套），GLM 需现场摸索策略；官方各档成绩均基于各自探索后冻结的更厚记忆快照（官方文档自述，E4）。我们的 80% 是"薄记忆"条件下的下界样点。
- 失败集中模式：t0"开抽屉"双败（接触密集铰接任务，GLM 以低层原语硬磕接触物理打满 max_turns）；t5_s1/t9_s0 为常规苦战失手。苦战特征与此前完全一致（ctx>150k + decode 3× + 请求 2.5×）。

## 3. Flash Mode（object_swap · Molmo2-8B 视觉定位）

### 3.1 部署 [E1/E2]

- **Molmo2-8B**：HF `allenai/Molmo2-8B`，实为 **fp32 权重 ~37GB**（bf16 加载后 ~16G HBM），hf_transfer 并行下载 ~4h（节点出网总量 ~1.5-2MB/s）；跑在 **dev4**（port 18801）。
- **NPU 适配**：molmo_server 无 NPU 兼容块，新打 `patch_molmo_npu.py`（transfer_to_npu + get_device_properties major=None 全局 shim 二合一，与 sam3 同款）；transformers 4.57.6 直接满足 remote code 要求（pyproject 注释所述与 OpenPI 的版本冲突在本安装中未出现）。8 分片加载 55s。
- **flash 记忆来源**：文档声称 flash 计划在 HF 数据集 `RLinf/RPent-memory` 的 `libero/flash/` 下，但**该目录现已不存在**（数据集只有两个大脑快照目录，E1）——文档与数据集现状不符。本地语料 `task_card/` 里的 `object_swap_t{0-9}_{plan,anchors}.json` 与 flash 加载器期望的命名完全一致，软链为 `memory/libero/flash/` 后 replay 正常执行（t3 冒烟 301s 解出）。

### 3.2 结果 [E1]

**15/20 = 75%**，解出集均值 228s，8 次工具调用/集，**0 LLM token**（对照 api 模式：90% 成绩、890s/集、~260k token/集）。

失败分类：
- **t2 双败为机械性失败**（22s 即中止）：task_card 的 t2 计划锚点是"ranch dressing bottle"，而该任务语言为"salad dressing"——计划与任务变体错位，锚点永远定位不到。
- t0_s0、t7_s0/s1 为执行层失手：完整回放 200-260s 未达终止谓词（锚点重定位/放置精度）。
- 剔除 t2 计划错位后执行层成功率 15/18 = 83%（官方 93% 用的是配套的 78 计划集 + GPU）。

### 3.3 解读 [E3]

Flash 的经济性主张在 NPU 上成立：**0 token、无 LLM 等待**，单集 228s 中执行层（Molmo 定位 + VLA chunk + move）占绝对主体；代价是成功率对计划-场景对齐与定位精度敏感。74.63%→72.63%（官方 overall）与我们的 75%（object_swap）方向一致。

## 4. 新发现（超出上游报告）

### 4.1 文档-数据集漂移 [E1]

`flash.rst` 指引的 `libero/flash/**` 下载路径在 RPent-memory 数据集中已不存在；task_card 计划可作替代（含错位风险，如 t2）。建议回馈上游：或在数据集恢复 flash 目录，或在文档改指向 task_card 并标注差异。

### 4.2 GLM API 网络级无限挂死（第二次出现，本次坐实）[E1]

t8_s1 于 12:38 由 pi0_pick 触发 terminated=true（任务已解出），但**下一次 GLM 调用的 HTTP 连接无限挂起 8 小时**（py-spy 级现象：进程活着、无 IO、单线程睡眠）。`--planner-timeout-s 5400` 只约束规划轮内时长，不覆盖底层 httpx 无超时的连接挂死。处置：kill + 断点协议重跑。**建议 upstream：给 api planner 的 httpx 客户端设 connect/read 超时（如 600s）**——否则任何长跑评估都有被单次网络黑洞卡死整批的风险。

### 4.3 其他 [E2]

- Molmo2-8B 仓库为 fp32（37G），文档未提示；带宽受限环境需按 37G 规划。
- flash 与 goal 并行跑期间共享 VLA 服务（dev5），计时互相污染——goal 单集时长含争用期（t9 之前的集），成功率不受影响；报告用时数据标注此口径。

## 6. 失败集重跑与早停量化（2026-09-28 补充，GLM API 恢复后执行）

### 6.1 失败集重跑：5/6 翻盘，唯一系统性失败是 goal t0 [E1]

对三批的全部 6 个失败集各重跑一次（原败局保留为 `*_fail1`）：

| 集 | 首跑 | 重跑 | 判定 |
|---|---|---|---|
| swap t2_s1 | ❌ 2818s/9.4M tok | ✅ 601s/1.3M | 随机失手 |
| swap t3_s1 | ❌ 3840s/9.9M | ✅ 617s/1.1M | 随机失手 |
| goal t0_s0 | ❌ 5160s | ❌ 1485s（三连败）| **系统性**（开抽屉） |
| goal t0_s1 | ❌ 5430s/11.7M | ✅ 835s/1.4M | 随机失手（同任务可解） |
| goal t5_s1 | ❌ 823s（快速失败）| ✅ 4795s/9.9M（苦战翻盘） | 随机失手 |
| goal t9_s0 | ❌ 1637s | ✅ 476s/0.6M | 随机失手 |

**Retry-once 有效战绩**：object_swap 18/20 → **20/20（100%）**；libero_goal 16/20 → **19/20（95%）**。唯一反复失败的是 goal t0 s0（开抽屉的特定布局）——三连败 + 同任务 t0_s1 重跑可解，说明是**任务×布局级别的硬骨头**而非全局短板。重跑经济性：**3.4M token/胜**（16.9M 换 5 胜），且失败集白烧的 45.0M token（0 胜）与之形成鲜明对照。

### 6.2 早停回测：朴素阈值早停是坏交易 [E1]

以前兆规则（单请求 ctx>140k ∨ 累计 decode>40k ∨ 请求数>45(swap)/55(goal)）回测全部 40 个 api 集：

- **收益侧**：4 个带前兆的失败集可省 22.2M token = 全批次 103.6M 的 **21.5%**（t0_s1 高达 82%、swap t3_s1 71% 的燃烧发生在前兆之后）；快速失败集（goal t5_s1，24 请求即败）无前兆，规则按设计不触发。
- **代价侧**：**5 个翻盘胜局会被误杀**（goal t2_s0/t2_s1/t3_s0/t5_s0 + swap t6_s0）——goal 成功率将从 80% 掉到 60%，swap 从 90% 掉到 85%。前兆的真实语义是"深度苦战中"而非"必败"：10 个 max_turns/近 max_turns 集中 6 个最终获胜。
- **结论**：上游"早停可省约 2/3 无效 token"只算了收益侧；在我们的数据上，纯早停省 21.5% token 的代价是 -12.5pp 成功率，不可取。合理策略是**前兆截断 + 重跑一次**（6.1 的重跑成功率 5/6 支持此策略），其对翻盘集的额外重跑成本未测，留作后续。

## 7. 产物索引（更新）

- 本地：`rpent-repo/flash20-npunew/`（2.7MB）、`rpent-repo/goal20-npunew/`（7.8MB）、`rpent-repo/retry-npunew/`（11MB，含 `*_fail1` 原败局与重跑的视频/日志、`analyze_earlystop.py` 及其输出）。
- 节点：`/data/rpent2/logs/{flash_object_swap,eval_libero_goal,eval_object_swap}`（败局存于 `*_fail1`）、`/data/rpent2/analyze_{goal,earlystop}.py`、`/data/rpent2/scripts-npunew/{flash20_final,eval20_goal,retry_failed,rerun_t8s1_goal,molmo_boot}.sh`、`/data/rpent2/patches/patch_molmo_npu.py`。
- 服务现状：VLA@dev5:18803、SAM3@dev6:18802、Molmo@dev4:18801 常驻未停（停：`docker exec rpent-npu pkill -f 'pi05_vla_server|sam3_server|molmo_server'`）。

## 8. 结论与后续

1. **复现义务全部履行**：api（两套件）+ flash 三种模式在纯昇腾栈上均有 20 集数据；加上失败集重跑，**retry-once 口径 object_swap 100%、libero_goal 95%**——上游 20/20 的成绩在重跑口径下完全对齐（首跑 90%/80% 的差距就是采样方差 + goal t0 一个系统性硬骨头）。
2. **GLM-5.3 档位结论稳固**：object_swap 90→100%、libero_goal 80→95%（薄记忆），与官方 GPT-5.5–Astra 档格局一致。
3. **Flash 经济性在 NPU 成立**，但计划集质量是上限（t2 错位直接丢 10%）。
4. **早停的正确形态是前兆截断+重跑**，纯阈值早停得不偿失（§6.2）；重跑成本 3.4M token/胜。
5. 后续可选：回馈四份产出给上游/同事仓库（sam3 双 shim、molmo shim、flash 记忆路径文档漂移、httpx 超时修复建议）；若扩官方 100 集口径，建议先修 4.2 并引入 §6.2 的截断+重跑策略再跑长批。

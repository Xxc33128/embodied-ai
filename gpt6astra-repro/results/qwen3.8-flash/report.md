# qwen3.8-flash bowl-eval-v1 标准 20 次子 agent 测试报告

批次：`logs-qwen3.8flash-subagent20-20260915-1545/`（模式 A，唯一正式口径）
协议：`robosuite-bowl-paired20-v1` v1.0（SOP 见 `2026-09-15-bowl标准测试流程.md`）
日期：2026-09-15 15:45 立项 → 21:53 全部收口，盲评 22:05 完成
证据等级：**E1** 源码/日志行级实证 · **E2** 配置与哈希实证 · **E3** 文档/推断（须标注）

## 1. 结果

**主结论：20 个场景里复核 stage=4（松爪后方块静止落在碗内）4 次 = 4/20 = 20%。**

| 指标 | 数值 |
|---|---|
| 有效 attempt | 20/20（`error.json` 0 个；`recording_checks` 20/20 PASS） |
| 复核 stage=4 | **4**（seed 2003、2012、2016、2018） |
| 环境自动判据 `success_at_end` | 4（与复核完全同行，无分歧） |
| 平均最高阶段 | 1.45 |
| 阶段分布（0/1/2/3/4） | 6 / 8 / 1 / 1 / 4 |
| 终止原因 | give_up 15 · success 4 · max_steps 1 |
| 决策数 | 15–20 次（均值 18.9），无一行超预算 |
| 控制步 | 163–900（均值 675），无一行超 900 |

逐行表见 `summary.csv`（含 `reviewed_stage` / `reviewed_success`），跨模型同 seed 对照见 `paired-by-seed.csv`，逐行盲评依据见各 `trial-NN/review.json`，可播放录像与审核阶段汇总见 `index.html`。

### 四次成功的关键证据（复核者引用帧步）
| trial | seed | 决策 | 控制步 | 释放步 | 末帧方块底 z | 距碗轴 | 末帧速度 |
|---|---|---|---|---|---|---|---|
| 04 | 2003 | 19 | 713 | 710 | 0.7932 | 0.0102 m | 0.050 m/s |
| 13 | 2012 | 17 | 163 | 160 | 0.7898 | 0.0091 m | 0.087 m/s |
| 17 | 2016 | 18 | 835 | 829 | 0.7956 | 0.0100 m | 0.050 m/s |
| 19 | 2018 | 17 | 707 | 705 | 0.7933 | 0.0095 m | 0.040 m/s |
桌面顶 z=0.80 m，碗内静止必然低于该值 —— 四例都在碗轴 1 cm 内且末帧速度 ≤0.09 m/s（E1：`physics.jsonl` 私有轨迹）。

**trial-19 边界样本二次复核分歧**：第二轮复核只看录像给 3（成功即终止，松爪后只剩 2 帧），但包内去真值轨迹显示它已落到桌面以下、距碗轴 9.5 mm、末帧 0.040 m/s（与另三例同一静止特征），故记 stage=4 并在 `review.json` 留 `strict_frame_only_stage=3` 的保守读法。**若一律只认画面证据，本报告口径为 3/20。**

### 最有信息量的失败
- **trial-03（seed 2002）**：单 worker 全程干净，真抓真抬（方块底 z 0.925，持续 160 步），但到碗最近处仍在碗体外侧，松爪掉在碗旁桌面上 → 两位复核者都给 **stage 2**（自动轨迹判据当时给 3，被盲评纠正：这是自动距离近似不可当成绩的证据，SOP §6）。
- **trial-20（seed 2019）**：抓到、抬到 0.866、移到碗口上方并下降中，但 20 次决策用尽仍未松爪 → stage 3。
- 6 行 stage 0 = 全程没碰到方块（07、09、10、11、14、18），其中 trial-09 把 900 控制步跑满仍没抓起来；8 行只到 stage 1（碰到/拨动但没离桌）。

## 2. 接入方式与可比性（必须一起读）

- 模式：**每个 trial 一个 `fork_turns=none` 无历史子 agent 连续做完整个 trial**（`bowl-policy` 定义，模型 `qwen-token-plan-cn/qwen3.8-flash`，工具只 `read`+`bash`，`thinking=medium`），通过文件信箱转接未改动的 `LLMAgentPolicy` 请求；worker 只读当次规范请求与其引用的图像，禁读项目文件/源码/私有日志/其他 trial。
- 规范面与 API 条件一致：统一 system+embodiment 提示（SHA256 校验）、任务句、20 次决策预算、900 控制步、`max_speed_frac=0.25`、`effort=medium`、wire 只带最近 2 组图像、私有真值（seed/方块位姿/接触/评分）绝不回流。
- **差异披露（E3）**：wire 是 `chat`（原 Astra 样本为 `responses`）；持久 worker 保留本 trial 记忆（API 条件下模型只有请求里的显式历史）；`thinking=medium` 是父 harness 前端旋钮，不等于供应商同名档位；墙钟含排队，不作串行比较。
- **分组**：provider 429 暂停期间被换新 worker 的行不并入主口径 ——

| 接入子组 | 行数 | stage=4 | 成功率 | 平均阶段 | 阶段分布 0/1/2/3/4 | 行号 |
|---|---|---|---|---|---|---|
| 单 worker 全程（persistent-worker） | 12 | 3 | **25%** | 1.75 | 3/4/1/1/3 | 03、05、06、08、09、12、13、14、17、18、19、20 |
| 暂停后换 worker（worker-restart） | 8 | 1 | 12.5% | 1.00 | 3/4/0/0/1 | 01、02、04、07、10、11、15、16 |

判定依据 = workflow journal 的 agent 终态 + `call-NN-started.json` 时间跳越恰好跨过暂停时刻（16:37、17:11、18:35、20:36）。换 worker 时所在决策序号已逐行写进 `review.json`。

## 3. 与同场景其他模型的逐 seed 对照

同一份冻结初始化库（seed 2000–2019，聚合逐文件 `diff -r` 一致），同一提示词/预算/物理。

| 模型 | 复核 stage=4 | 平均最高阶段 | 阶段分布 0/1/2/3/4 | 备注 |
|---|---|---|---|---|
| **qwen3.8-flash（本批次）** | **4/20** | 1.45 | 6/8/1/1/4 | 本报告 |
| glm-5.3-flash | 0/20 | 0.65 | 11/6/2/1/0 | 已盲评；`env_success` 0/20 |
| deepseek-v4.1-flash | 1/20 | 0.60 | 14/3/1/1/1 | `env_success` 1/20（22 个 attempt 里 1 个） |
| gpt-6-astra | 未完成整组 | — | — | 该批次只留下 5 个 attempt（含 trial-02 冻结现场），不纳入对照结论 |

逐 seed 对照见 `paired-by-seed.csv`。**限定**：只有 20 个固定场景，结论不外推；且这是仿真 embodiment（robosuite Panda + OSC_POSE）与原真机 I2RT YAM 双臂不同，成功率**不可**与 OpenAI×RoboCurve 报告的 19/20=95% 直接比较（复现的是协议，不是数字）。

## 4. 成本与限流（为什么这一批跑了 6 小时）

| 窗口 | 并发 | tokens（含 cache） | cache 占比 | 结果 |
|---|---|---|---|---|
| R1（16:03–17:31，5 槽） | 5 | 73.6M | 94% | 7 行完成，2 次 429 暂停 |
| R2（17:33–19:16，2 槽） | 2 | 36.2M | 94% | 5 行完成，1 次 429 暂停 |
| R3（19:32–21:53，2 槽） | 2 | 27.9M+3.6M | 94% | 8 行完成，1 次 429 暂停后 auto-resume 成功 |

- 一个**持久 worker 跑完单 trial ≈ 7–8.5M tokens，94% 是把自身累计上下文重发**（cache read）。20 个持久 trial ≈ 140M tokens，provider（阿里云百炼 token plan `qwen-token-plan-cn`）的额度窗口装不下：共 **4 次 429 `insufficient_quota`**（16:37、17:11、18:35、20:36），并发从 5 降到 2 只能拉长间隔（~35 min → ~62 min），不能消除。
- 每次 429 暂停都会掐死当时在跑的 worker（auto-resume 换新 worker 接入）→ 这就是 8 行落入 worker-restart 子组的唯一原因。**设置一次都没为提速而改**（用户指令：不改设置、只调并行、额度没了就等）。
- 盲评成本：第一轮 5 个复核 agent 530k tokens、边界第二轮 3 个 207k tokens（复核模型 `qwen3.8-max`，与被测模型不同）。
- API usage/cost 一律 **null**（子 agent 账号用量不能冒充 API usage，SOP §7）。

## 5. 过程事故与留痕（不计入成绩，全部保全）

| 目录 | 起因 |
|---|---|
| `attempts-aborted-wave1-1552/` | 用户要求从固定波次改为动态滚动池，5 个半成品（call≈2）中止 |
| `attempts-aborted-duplicate-launch-1603/` | 我的操作失误：重复起了一个索引重叠的池，5 个 runner 建了目录但 0 次模型决策 |
| `attempts-aborted-quota429-r2-1720/` | 17:11 暂停的残留无主 runner（含 18 次决策的 trial-06 等 6 个）+ 一次抢跑（0 决策） |
| `../logs-qwen3.8flash-perdecision8-20260915-1917/` | 我曾按 SOP §3.4 改起 per-decision 严格等价模式（4 个半成品，0 终态），19:29 按用户指令停止，目录内贴 `OFF_SPEC_NOTE.md`，不计入成绩 |
| seed 2012/2013/2014 首启失败 | 另一会话 19:0x 把 `embodied-ai/sonic-npu/weeks/2026_0914-0920_GPT6Astra评测复现/` 移到仓库根 → runner 写死的 `bowl-eval-v1` 路径 FileNotFoundError（0 次决策、无残留目录） |

基础设施改动仅两处、零协议字节变化，runner 加载时仍校验 prompt/system/环境源码 SHA256：
`run_subagent_trials_qwen.py` 的 standard 目录解析回退（`edff5d10…` → `247b8498…`）；`launch_trial_r2.sh` 导出 `BOWL_EVAL_STANDARD` 指向批次本地 `standard/`（`9dd76d73…`）。协议文件与被测环境源码哈希与 `protocol.json` 记录一致（E2）。

## 6. 留痕产物

每 trial：`video.mp4`（每控制步一帧、20fps 仿真时钟、front|side|wrist 三视图、含首末帧）、`initial-/final-*.png`（三路）、`outgoing-/incoming-NN.json`（逐调用请求与回复）、`frames/`（模型所见原图无损导出 + SHA256 索引）、`actions` 展开在 `physics.jsonl`、`initialization.json` + `initialization-check.json`（与冻结库逐字段比对）、`harness/*.json`（原始 log）、`result.json`、`review.json`。批次级：`experiment.json`、`dispatch.jsonl`（含中止/更正事件）、`standard/`（协议与提示词冻结件）、`initializations/`（20 场景库）、`blind-review/`（20 份匿名包 + `blind-review-map.json`）、`summary.csv`、`recording-audit.json`、`paired-by-seed.csv`、`index.html`。

## 7. 遗留

1. 8 行 worker-restart 若要并入唯一口径，需在额度充足时整 trial 重跑（原 attempt 保留）。
2. `launch_trial_r2.sh` 的 READY 判据仍靠轮询 `public/status.json`，未按 runner 所有权校验（本次靠"attempt 目录独占"避免双接）；下批应改成 pid 级隔离。
3. 共享生成器 `audit_standard_trials.py` 的 index.html 表头仍写死上一批次的模型名，本批只修正了生成的 HTML 文件本身；应参数化到 `experiment.json`。
4. 自动阶段判据（距离近似）与盲评在 trial-03 上不一致（3 vs 2）——主表只认盲评，自动值仅作候选指标（SOP §6 要求，本批已实证这条纪律有必要）。

## 8. token 与时间记录（22:40 补充，逐 trial 表 = `tokens-and-timing.csv`）

**时间：每个决策都单独记了时。** 来源 = runner 写的 `call-NN-started.json` / `call-ended.json`（含 `wall_time`、`wait_seconds`、当时刻的物理步号）与 `result.json`。

| 项 | 数值 |
|---|---|
| 墙钟合计（20 trial 相加） | **11.5 h** |
| 其中等模型决策 | **11.4 h = 99.1%**（逐行 98.6–99.6%） |
| 物理推进耗时 | 388 s（13,506 控制步 × ≈20–24 ms，含每步录像写帧） |
| 仿真时间合计 | 675 s（13,506 步 / 20Hz；单 trial 8.15–45.0 s） |
| 单次决策等待 | 跨 trial 均值 72–140 s，全程最大 583 s |
| runner 每决策硬超时 | 1800 s，**20 个 trial 一次都没触发** |
| 推理期间仿真 | 暂停（审计断言 `call-started` 与 `call-ended` 的 `physics_steps` 相等，20/20 通过） |

**token：有记账，但不是 API 账单。** 权威来源 = pi workflow journal 的 `tokenUsage`（父 harness 侧统计，含 cached 读取）。

| 窗口/用途 | 并发 | tokens 合计 | 其中 cache 重发 | 新鲜 input+output |
|---|---|---|---|---|
| R1 16:03–17:31 | 5 槽 | 73.62M | 69.04M (93.8%) | 4.58M |
| R2 17:33–19:16 | 2 槽 | 36.16M | 33.79M (93.4%) | 2.37M |
| R3 19:32–21:53 | 2 槽 | 60.65M | 57.05M (94.1%) | 3.60M |
| 盲评 pass1 + 边界 pass2 | 5 / 3 复核 agent | 12.00M | 11.26M | 0.73M |
| 作废的 per-decision 尝试（不计成绩） | 4 槽 | 2.94M | 2.49M | 0.45M |
| wave1 中止（改滚动池前） | 5 | 2.78M | 2.38M | 0.40M |
| **合计** | — | **188.2M** | **≈94%** | **≈12.1M** |

- 逐 trial 列在 `tokens-and-timing.csv`：13/20 行的 journal 条目非零（0.05M–9.85M，中位 1.2M，cache 占 93.1%）。**注意这些是下限**：凡中途换过 worker 的行，旧段条目在暂停-重放时被压成 0，只剩接续段的量；所以单 trial 成本以 run 级总量除以完成行数为准（**R3 实测 60.65M / 8 行 ≈ 7.6M tokens/行**）。
- `api_usage` 与 `cost` 一律 **null**（SOP §7：子 agent 账号用量不得冒充 API usage，供应商账单侧无法按 trial 归因）；上表 token 是本机 harness 记账，不是发票数字。
- 录像/图像侧的量：20/20 `video_frames == physics_steps+1`；模型可见图像引用合计 **2,208** 张，`frames/index.json` 逐文件 SHA256 可追到对应 wire 请求。
### 8.1 每 trial 的 token 可用性说明 + 协议侧计量（`tokens-and-timing.csv` 新增列）

| 层 | 每 trial 是否都有 | 量级 | 说明 |
|---|---|---|---|
| 时间（墙钟/等模型/仿真/步号） | **20/20 全有，且到每个决策** | 22.0–47.2 min；等模型占 98.6–99.6%；单次决策均值 72.5–140.2 s，最大 582.8 s | `call-NN-started/ended.json` 原始件 |
| 协议侧计量（真实发给模型的内容量） | **20/20 全有** | 15–20 个请求、87–117 张图、5.7–8.2 MB 请求体/行；合计 149 MB、2,208 张图 | 由 `outgoing-*.json` 直接数出来（E1） |
| 协议侧 token **估算** | 20/20 | ≈3.6 万–5.6 万 tokens/行 | 224² 图按 64 token、文本按 chars/4（E3，仅供数量级参考） |
| harness 记账 token | 13 实测 / 2 下限 / 5 缺失 | 0.05M–9.85M/行 | journal 只保留最后一段：R1 窗口 5 行（trial-01..05）被暂停重放清零，本机不可恢复；trial-06/09 只剩接续段 → 标下限 |

**两层 token 差 100–200 倍，这就是 429 的真因**：协议本身每行只要约 5 万 token，但持久 worker 的 agent 会话每轮要重发自己的全部历史（读过的图 + 每次请求 JSON），且一个机器人决策内部含 3–4 次模型往返 → 实测 **≈7.6M tokens/行**（R3 60.65M / 8 行）。所以"省 token"只能靠减少重发（换 per-decision 接入，属另一口径需你点头），不是靠调并发。

## 9. 时间到底花在哪（逐项拆开，全部来自留痕可复算）

### 9.1 墙钟台账（15:45 立项 → 22:50 归档，共 7.1 h）
| 段 | 时长 | 产出 / 性质 |
|---|---|---|
| 读规程 + 建批次 + 冒烟预检 | 18 min | 必要 |
| R1（5 槽） | 88 min | 7 行 |
| R1→R2 换池（停池、清孤儿、起池） | 2 min | 我的调度变更 |
| R2（2 槽） | 103 min | 5 行（含 trial-06 重跑） |
| **我误起的 per-decision 支线，发现后停回** | **16 min** | **纯浪费**（2.9M tokens，4 个半成品作废） |
| R3（2 槽） | 141 min | 8 行 |
| 审计 + 两轮盲评 + 归档 + 提交 | 57 min | 报告/复核/镜像 |
| 合计跑测时间 | 332 min | 20 行 + 3 次换池开销 |

### 9.2 一次机器人决策内部（这才是主成本）
378 次决策、等待中位 **81 s**、p10 21 s、p90 201 s、均值 109 s、最大 583 s。拆开：

| 环节 | 量级 | 证据 |
|---|---|---|
| worker 内部 LLM 往返（`next` → 读 3 张图 → `submit` → 收尾，约 4–6 次调用） | **≈ 全部 80–90 s** | 中位每次规范请求 harness 记账 **47k tokens**，而协议侧真实内容只有 ≈2.5–3k tokens/请求 → **94% 是把 worker 自己的历史重发**；单次推理被推大 |
| 环境步进 + 接触求解 + 写帧 | 决策内 ≈0.6–0.7 s；整 trial 平均 **19 s**（20–24 ms/步） | `physics.jsonl` 行数与 `video_frames == steps+1`；全批物理累计仅 388 s |
| 信箱文件轮询 | 0.05–0.1 s | `subagent_mailbox.py` sleep(0.1) |
| ffmpeg 编码（672×224） | 可忽略，不阻塞主循环（管道写） | runner 用 stdin 流式喂帧 |
| CPU / 显存争抢 | **无**（82.8% idle，load 2.9/10 核，DeepSeek 12 个空转 runner 合计 0% CPU） | 见 `RUN_STATUS.md` 瓶颈实测条 |

**同一模型的对照实验（作废的 mode-B 支线正好当了消融）**：fresh-context worker（只看当次请求）中位 **42 s**/决策（p10 21 s）vs 持久 worker 中位 **86 s** → 持久模式单决策慢约 **2 倍**，单 trial token 多约 **8 倍**（7.6M vs ≈1M）。

反直觉的一条：持久 worker 的决策延迟在 trial 内是**递减**的（决策 1–5 中位 93 s → 16–20 为 65 s），因为上下文主要是 cache 读、越往后命中越好；但这也意味着**窗口后期 token 吞吐最高 → 429 恰恰在 trial 收尾期砸下来**，一砸就掐死在跑的行。

### 9.3 429 空档与我的调度损耗
- 4 次暂停造成的空档（各行超出自身中位数的部分相加）= **6,952 s ≈ 1.93 h**（聚合量；并行 2–5 路，折到墙钟约 0.5–1 h）。
- 我自己的调度损耗：wave1 中止 + 改滚动池 ≈ 16 min、重复起池清理 ≈ 2 min、mode-B 支线 ≈ 16 min → **约 34 min**（且换掉了 8 行的接入纯度）。

### 9.4 提 speed 的杆（按性价比，都不动协议字节）
1. **提额度 / 换按量计费 key**（唯一触及根因的）：可同时消掉 ~1.9 h 空档并允许 4–6 槽 → 本批 332 min 的跑测段理论可压到 ≈120–150 min。
2. **per-decision 接入**：单决策 86→42 s、单 trial 7.6M→≈1M tokens，2 槽也能 ≈2× 提速且**不再被暂停污染 trial**；代价是换接入组（SOP §3.4 允许但必须分组呈现）——需你点头，属口径决定。
3. **减少 worker 内部往返**（一次回复里批量读图 + 直接提交）：属于 dispatch 提示词改动 → 要开新 protocol 版本并对所有模型重测，本批不允许中途改。
4. 不用碰的：并发已经贴着额度、决策数贴着预算（均值 18.9/20）、900 步只撞满 1 行 —— 说明慢不是"模型犹豫"或"仿真太贵"，是**重发上下文 + 等额度**。

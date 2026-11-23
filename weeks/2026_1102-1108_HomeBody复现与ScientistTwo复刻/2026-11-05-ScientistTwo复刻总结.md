# ScientistTwo 复刻总结（2026-11-05）

> 记录对象：工作目录 `scientisttwo-repro/`（非 git 仓库，4.3 MB，737 个文件，不含 `__pycache__`）。全部产出集中在 2026-11-05 一天，现场文件的修改时间跨度为 14:49–18:19〔E2〕。此前这项工作在记录仓库里没有入库文档。

- **日期**：2026-11-05（单日）；本报告撰写于 2026-11-23。
- **主题**：Google ScientistTwo（arXiv 2609.19644）六阶段自主科研 agent 框架的复刻——沙箱执行、种子派生、静态门禁，以及当日的五轮"审查→修复"。
- **一句话总结论**：一天内做出了端到端可跑的复刻（纯标准库，26 个角色，63 个测试全绿〔E1〕），五轮对抗性审查把可信度评分从 4/10 推到 7/10 并判 GO；但 8 次试跑全部是脚本化 fake 后端、没有接过真实模型，防作弊栈在同用户权限下挡不住蓄意爬取，**接真实模型前必须先落实 OS 级隔离与网络封堵**〔E3，综合结论，依据见 §4–§6〕。

证据口径：E1＝源码／日志行号实证，E2＝配置／数据文件实证，E3＝文档解释或推断，E4＝未复核的转述。论文内容取自 `docs/PAPER_NOTES.md`，本报告没有对照论文原文，一律记 E4。五轮审查留档里的攻击复现数字本报告没有复现，记 E4；修复是否落地以读码为准，记 E1 或 E3。标注"11-23 复核"的条目是撰写本报告时新取的证据，不属于 11-05 的现场记录。

## 1. 背景与目标

**ScientistTwo 是什么**（以下为论文与公开资料的转述）：

- Google Cloud AI Research 的自主科研系统。论文 arXiv 2609.19644 只有 HTML v1，附录 A.2 截断不可见；**官方未开源**，GitHub 上只有项目网页的源码〔E4〕。
- 六阶段流程（论文 §3.1–3.6）：找局限 → 生成种子想法 → 子集／全量评测 → 想法进化 → 消融 → 写稿＋模拟评审＋元审稿。Table 1 列出全部 agent 角色与职责〔E4〕。
- 每个阶段内部是通用的 refine 循环（Listing 4）：candidate → critic 判 accept／refine／discard，超过 `max_rounds` 丢弃〔E4〕。
- 唯一明示的数值是 peer review 过线分 "e.g., 8"（1–10 制）。循环变量 N_seed、N_0、N_eng、K、S、N_e、N_p、N_abl、N_peer、N_meta 的取值称在附录 A.2，实际不可见〔E4〕。
- 论文报告的结果：107 题中 86 题（80.4%）超过原论文 SOTA，平均提升 25.2%（中位 7.7%）；成本约 3,765 美元／题，2–3 天／题〔E4〕。
- 模型分工：轻模型承担分析与评审，重模型（Claude Code）承担研究性编码〔E4〕。

**论文没有公开的**：prompt 模板（一份都没有）、沙箱与调度细节（只知用 VM）、硬件配置、附录 A.2 的超参数取值〔E4〕。

**为什么复刻**：论文只给了流程骨架，复刻是把骨架落成可运行、可审计的实现——输入一篇已录用论文＋可跑代码＋只读基准，输出改进方法、消融、论文稿、模拟评审、完整性审计与独立复核报告（`README.md:3-5`）。论文没给的部分一律自定并显式标注，不冒充论文原有设计〔E3，现场文档自述〕。

## 2. 框架设计

### 2.1 六阶段与角色

流水线是一个 11 段的状态机（`orchestrator.py:22-24`〔E1〕）。每个角色是一次无状态调用，critic 每次拿到的是全新请求，不带被审内容的对话历史（`agent.py:1-4`〔E1〕）。下表职责列的出处是 `DESIGN.md` §1 与 `README.md`〔E3〕，实现列的函数名以读码为准〔E1〕。

| 论文章节 | 职责 | 实现（`orchestrator.py`） |
|---|---|---|
| §3.1 找局限、出想法 | Limitation Extractor ⇄ Verifier 循环，verifier 的 missing 清单回灌下一轮；Idea Generator 出不超过 n_seed 个想法，Novelty Checker 打分排序，top-n0 进第 0 轮，进化想法同样过 novelty 检查 | `stage_limitations`、`stage_ideas_seed` |
| §3.2 子集与全量评测 | Baseline Coding 子集复现；Subset Coder／Critic／Engineer 判 Good／Engineer／Bad，全量同构；success＝全量 Good 且数值严格更优 | `stage_baseline`、`stage_evaluate` |
| §3.3 进化与选优 | Evolver 聚合成功与失败的 trace，迭代至 s_target 个成功，每轮混入 n_e 个新种子；Selector 选最优 | `_evolve`、`stage_select` |
| §3.4 消融 | Planner 出 n_p 个计划，Coder 写变体，轮级 Critic；判 Refine 时由 engineer 修订方法、全量重评，严格更优才替换 best 并重新规划，上限 n_abl | `stage_ablation` |
| §3.5 写稿与模拟评审 | Drafter → Peer-Reviewer（过线分 8）；不过线则 Rebuttal Planner／Coder 真跑补充实验，再由 Enhancer 改稿，上限 n_peer | `stage_paper` |
| §3.6 元审稿 | Accept，或 Refine 并按 ideas／ablation／paper 三档打回、抬高成功数门槛，上限 n_meta | `stage_meta`、`META_RESET`（`:26-30`） |
| §4 完整性审计（另加独立复核） | 分数复现、spec 合规、引用 grounding、方法-代码对齐；独立复核用未见种子做配对检验 | `integrity.py`、`verify.py` |

阶段划分有两套口径：`README.md:13-23` 把 §3.1 拆成阶段 1、2，并把元审稿列在六阶段之外；`PAPER_NOTES.md:14` 把 §3.1–3.6 记为六阶段，元审稿在其中〔E3〕。

与论文逐条对照记在 `DESIGN.md` §1。§4 列了六项刻意的差异，其中独立复核是论文之外新增的；其余是加固，例如 critic 与 selector 的判定以基准数值兜底、任务包整树哈希锁、断点续跑与预算〔E3〕。超参数默认值全部是自定的猜测，只有 `peer_threshold` 8 来自论文（n_seed 6、n0 3、n_eng 3、k_max 3、s_target 2、n_e 1、n_p 3、n_abl 2、n_peer 2、n_meta 2；`config/default.json`）〔E2〕。

### 2.2 沙箱执行（两进程隔离）

- **真值不出框架进程**：数据由框架进程内的 `datagen` 生成，测试集真值与真系数不落盘、不进候选进程。候选只收到 `{X_train, y_train, X_test}`，经 stdout 交回预测，指标由框架对真值计算（`bench.py:175-225`，`tasks/toy-hfd/bench/run_candidate.py:13-19`〔E1〕）。
- **复制执行**：runner 与 `method.py` 先复制进一个空的临时目录再运行，argv、cwd 与 `sys.path[0]` 都不含仓库路径；子进程环境变量是白名单，只留 `PATH`、`LANG`（`bench.py:137-173`〔E1〕）。
- **资源与数值防线**：stdin／stdout 走沙箱内的文件，stdout 上限 10 MB；超时杀整个进程组；预测长度、coef 维度不符或出现 NaN／Inf 一律判 `CandidateFailure`（`bench.py:38, 159-172, 190-215`〔E1〕）。
- **异常分流**：`SpecViolation` 表示任务面被篡改，run 终止；`CandidateFailure` 表示候选自身失败，进 engineer 修复循环，预算耗尽判弃（`bench.py:42-47`，`orchestrator.py:91-113`〔E1〕）。

这是进程级隔离，不是 OS 级隔离：候选仍以同一用户身份运行〔E1〕。

### 2.3 种子派生

- 实际随机种子＝HMAC-SHA256(run 级密钥, 名义种子) 的前 4 字节（`bench.py:118-123`〔E1〕）。密钥存在 run 目录旁的 0600 独立文件里，不进 `state.json`（`bench.py:69-84`〔E1〕）。
- 注入 prompt 的协议文本 `BENCHMARK.md` v0.2 只保留定性结构，不含生成公式与种子表（`tasks/toy-hfd/BENCHMARK.md:3-16`〔E2〕）。
- 跨 run 的数值因此不可比，这是设计使然。`seed_derivation: nominal` 仅供框架侧测试取确定性断言（`DESIGN.md` §3〔E3〕）。

### 2.4 静态门禁

- 禁令表在配置里，共 17 条：评测内部符号、自省逃逸类（`sys.modules`、`__main__`、`importlib`、`subprocess`、`eval`、`exec`、`open` 等）、种子特判等（`config/default.json`〔E2〕）。
- 候选写盘后、评测前扫描，命中即 `CandidateFailure`，违规清单作为修复指令交给 engineer（`orchestrator.py:78-89`〔E1〕）。
- 终审对最终 best 的 `method.py` 再做一次静态扫描加 validation_filter（`integrity.py:26-52, 120`〔E1〕）。审计不过则 run 标 `integrity_failed`、退出码 5（`orchestrator.py:665-672, 769`〔E1〕）。

### 2.5 接口契约

`docs/CONTRACTS.md` 是 26 个角色输入／输出 JSON 契约的单一权威出处，改契约先改它（`AGENTS.md:7`〔E2〕）。要点：

- 分析类角色的回复必须以最后一个 ```json 代码块结尾。解析是严格模式：最后一个块解析失败立即报错，不回退更早的对象（`util.py:55-74`〔E1〕）。
- 评审判定三元组：`Good`＝accept，`Engineer`／`Refine`＝refine，`Bad`＝discard；未知取值一律按丢弃处理（`refine.py:8-15`〔E1〕）。
- 角色分两档：6 个研究性编码角色走 heavy，其余 20 个走 light（`roles.py:4-19`〔E1〕）。
- 基准侧：`method.py` 暴露 `fit_predict(X_train, y_train, X_test) -> list[float]`，可选 `coef()`。`strictly_better`＝主指标改善超过 eps，且次要指标不恶化超过 max(eps, 0.1%·|base|)（`bench.py:241-258`〔E1〕）。
- 状态：`state.json` 记录配置与配置指纹，续跑时配置不一致拒绝启动；想法终态一经写入不可回退（`state.py:130-142`，`orchestrator.py:56-59, 745-746`〔E1〕）。

## 3. 实现规模

| 项 | 数量 | 出处 |
|---|---|---|
| 框架包 `scientisttwo/` | 12 个模块，2,330 行 Python，零第三方依赖 | `wc -l`〔E1〕 |
| 其中 orchestrator／fakebank／bench | 774／551／264 行 | 同上 |
| 角色提示词 `prompts/` | 26 个角色文件＋`_common.md`，共 561 行 | 同上 |
| 任务包 | 1 个（`toy-hfd`）：虚构论文、基线源码、`datagen`、`run_candidate`、一次性测量工具 | `tasks/toy-hfd/`〔E2〕 |
| 测试 | 38 个 `unittest.TestCase` 类、63 个测试方法，单文件 1,304 行 | `tests/test_pipeline.py`〔E1〕 |
| 文档 | DESIGN 108 行、CONTRACTS 75 行、PAPER_NOTES 43 行、五份审查留档共 240 行 | `docs/`〔E2〕 |
| 运行留痕 | 8 个 run 目录，每个 43–45 次 agent 调用，带 `log.jsonl` 与 `transcripts/` | `runs/`〔E2〕 |

不含 run 产物的手写 Python 合计 3,798 行〔E1〕。

测试数有一处对不上：R5 留档写"60/60 测试"，当前文件静态计数是 63。多出的 4 个在文件末尾两个类里（`TestHTTPBackend` ×3、`TestMetaRefineLoopback` ×1），测试文件的修改时间 18:12 晚于 R5 留档的 17:32〔E2〕。但 R5 之前的 36 个类逐类求和只有 59 个，比留档少 1，这 1 个的出入现有材料无法判定〔E3〕。

11-23 复核〔E1〕：把源目录（不含 `runs/`）复制到工作区外的临时目录，`python3 -m unittest discover -s tests` 结果为 `Ran 63 tests … OK`，耗时 66.7 秒。HTTP shim 测试有 2 条 socket 未关闭的 `ResourceWarning`，不影响结果。源目录未改动。

## 4. 五轮审查与修复

每轮由一名新的子 agent 审查员对当时的代码做攻击性审查，再逐条修复并补回归测试；五名审查员分别用了 46、53、46、80、55 次工具调用〔E4〕。日期时间取审查留档文件的修改时间，即该轮修复落点的记录时刻〔E2〕。"主要发现"与"评分"两列是留档原文的摘录（E4），"修复内容"列按读码结果标注。

| 轮次 | 日期时间 | 主要发现 | 评分 | 修复内容 |
|---|---|---|---|---|
| R1 | 11-05 15:53 | P0×2：防作弊核被击穿，候选经 `importlib` 加载进评测进程，内省即可拿到真值（mse=0 且全审计通过）；消融"更优变体替换 best 并重规划"整块缺失。另 P1×9、P2 若干〔E4〕 | 4/10〔E4〕 | 两进程隔离，禁令扩充〔E1〕；补齐消融修订回路〔E1〕；rebuttal 变体隔离到独立目录〔E1〕；续跑一致性与配置入档〔E1〕；精确二项符号检验〔E1〕；严格 JSON 解析〔E1〕；整树哈希〔E1〕。审查员当时看到的是 20 个测试〔E4〕 |
| R2 | 11-05 16:22 | 第一轮修复逐条复核通过，但指出 R1 留档声称的"消融双路径测试"实际只有替换路径一条。两条新 P0：NaN／Inf 穿透全部数值防线；协议文本含生成公式与种子表，真值可重构。另 P1×4、P2 若干〔E4〕 | 5/10〔E4〕 | 三层有限性断言〔E1〕；种子 HMAC 派生，协议文本去掉公式〔E1〕；`CandidateFailure` 分流〔E1〕；coef 长度校验〔E1〕；审计范围收窄到最终 `method.py`〔E1〕；run 目录锁，prompt 走 stdin〔E1〕。修复后 40/40 测试〔E4〕 |
| R3 | 11-05 16:51 | 新 P0：子进程的 argv、`sys.path[0]`、`__file__` 泄露仓库路径，顺藤摸瓜读到 `datagen` 与状态文件，绕过种子派生。另 P1×2、P2 一组〔E4〕 | 6/10〔E4〕 | 沙箱复制执行，环境变量白名单，密钥外置〔E1〕；stdout 上限与进程组清理〔E1〕；指标白名单〔E1〕；工程预算统一池〔E1〕；CLI shim 测试抓到 `{model}` 占位符从未被替换的真 bug 并修复〔E1〕。修复后 49/49 测试〔E4〕 |
| R4 | 11-05 17:14 | 三轮修复复核全部通过。新 P0：无路径先验的全盘爬取，据留档 0.7 秒定位密钥目录、1.6 秒走完整条攻击链。另 P1×1、P2×10〔E4〕 | 5.5/10〔E4〕 | 评估期静态门禁〔E1〕；审计不过的硬门〔E1〕；evolver 的 id 复用守卫〔E1〕；数值强转、id 清洗、信封解析从严〔E1〕；其余各项按留档已修〔E3〕。修复后 56/56 测试〔E4〕 |
| R5 终审 | 11-05 17:32 | 抽验前两轮 6 项修复全部通过，无 P0；P1×2（变体进评测前无静态门禁、两类裸崩），均为边角级〔E4〕 | **GO，7/10**〔E4〕 | 三处变体接入静态门禁〔E1〕；baseline 循环与解析失败统一为可续跑的 `AgentError`〔E1〕；finalize 不再覆盖 `meta_not_accepted`〔E1〕。留档记 60/60 测试〔E4〕 |

评分轨迹 4 → 5 → 6 → 5.5 → 7。R5 留档的分项是管道工程 8、防作弊核 6、真实就绪度 6.5〔E4〕。R4 的回落对应的是全盘爬取攻击，而不是修复退步〔E3〕。

评分评的是哪一版代码需要说清：每轮分数都给在**该轮修复之前**，所以 7/10 对应的是 R4 修复后的状态。R5 自己提出的 2 个 P1 修复之后，以及 18:11 起新增的 HTTP 后端，都没有再经过审查〔E3〕。

## 5. 试跑结果

### 5.1 toy-01 的执行方式

toy-01 创建于 15:13:21，日志首尾相隔约 5 秒，共 44 次 agent 调用、36 条基准记录〔E1〕。它出自 R1 修复之前的代码，`state.json` 上有三个特征可以互证：`config` 与 `config_sha256` 为 null；任务包哈希只锁了 4 个文件，不含 `BENCHMARK.md`〔E2〕；独立复核的 p 值是 0.001565，与修正前"无连续性校正的正态近似"吻合（修复后同样的 10 胜 0 负记为 0.001953）〔E3〕。

执行脚本是当时的 `tasks/toy-hfd/bench/run_trial.py`（81 行）〔E1〕：

- 命令行接 `--method`、`--seed`、`--out`、`--task`；`--task` 是死参数，R1 记为 P2〔E1〕。
- `gen_data(seed)` 以 `random.Random(seed * 9301 + 49297)` 生成 AR(1) ρ=0.6 的 10 维相关特征线性回归：5 个强系数、5 个弱系数，训练 40 行、测试 200 行，噪声 σ=1.2（`run_trial.py:11-31`〔E1〕）。
- `load_method` 用 `importlib` 把候选 `method.py` **加载进评测脚本自己的进程**，在同一进程里算 mse 与 |coef| 对真系数的 Kendall τ，写出一个含 `seed`、`metrics`、`config.method` 的 JSON（`run_trial.py:47-77`〔E1〕）。

也就是说，toy-01 是"框架进程 → 子进程跑 `run_trial.py`"的两层结构，但真值与候选代码同住一个进程，这正是 R1 的 P0-1〔E1〕。每次试验的绝对方法路径被原样写进了结果文件的 `config.method`〔E2〕。

**模型后端：fake，没有接真实模型**〔E1〕。toy-01 的 transcript 回复段带 `(fake)` 前缀；44 次调用全部落在 5 秒内；日志里每次调用带固定的 `cost_usd`（heavy 档 0.5、light 档 0.05，`fakebank.py:8`），13×0.5＋31×0.05＝8.05，正是 `state.json` 记的 `usd: 8.05`。这是模拟计价，不是真实花费。

**任务**：`toy-hfd` 是自带的试点任务，参考论文《CorrScreen-OLS》是为验证流水线而虚构的（`tasks/toy-hfd/paper.md:3`〔E2〕）。基线方法是"按边际相关筛 top-5 再做 OLS"，在相关特征上会系统性丢信号（`tasks/toy-hfd/src/baseline.py:1-4`〔E1〕）。

**结果**〔E2〕：status `complete`。5 个想法里 S1（Ridge on all features）与进化想法 E1 成功，S2–S4 被弃。best 是 S1，全量 mse 1.852、τ 0.613；基线全量 mse 3.136、τ 0.253。3 个消融计划；评审两轮 7 分、8 分（首轮未过线，rebuttal 后过线）；审计 PASS；独立复核在 10 个未见种子上 10 胜 0 负。

这些数字**不能当研究结论**。S1–S4、E1 以及消融变体 A1、A2 的代码都是 fakebank 里的内置常量（`fakebank.py:32-193`〔E1〕），novelty 分数是一张写死的表（`fakebank.py:299`〔E1〕），"ridge 优于边际筛选"是脚本预设的结局。fake 模式下 critic 的判定就是框架自己的数值比较（`fakebank.py:343-356`〔E1〕），端到端测试验证的是管道行为，不是研究判断的质量（`DESIGN.md` §4〔E3〕）。

### 5.2 八次 run 一览

8 个 run 全部跑 `toy-hfd`，全部 fake 后端（toy-01 见 5.1，其余 7 个的 `config.tiers` 都指向 fake〔E2〕），全部 `complete`，best 全是 S1，评审分全是 [7, 8]〔E2〕。

| run | 创建时间 | 代码状态（按时序与状态字段推断，E3） | 调用 | best 全量 mse / τ | 基线全量 mse / τ | 复核 胜/负，p |
|---|---|---|---|---|---|---|
| toy-01 | 15:13:21 | R1 修复前 | 44 | 1.852 / 0.613 | 3.136 / 0.253 | 10/0，0.001565 |
| toy-02 | 15:52:14 | R1 修复后 | 43 | 1.852 / 0.613 | 3.136 / 0.253 | 10/0，0.001953 |
| toy-03 | 15:53:55 | R1 修复后 | 43 | 1.852 / 0.613 | 3.136 / 0.253 | 10/0，0.001953 |
| toy-04 | 16:22:35 | R2 修复后（种子已 HMAC 派生） | 45 | 1.953 / 0.620 | 3.493 / 0.316 | 9/1，0.021484 |
| toy-05 | 16:51:02 | R3 修复后 | 43 | 1.888 / 0.644 | 3.355 / 0.238 | 9/1，0.021484 |
| toy-06 | 17:16:19 | R4 修复后 | 43 | 1.895 / 0.609 | 3.425 / 0.267 | 10/0，0.001953 |
| toy-audit | 17:18:59 | R5 终审员自己的实测（R5 留档自述） | 43 | 1.802 / 0.687 | 3.365 / 0.320 | 9/1，0.021484 |
| toy-07 | 17:32:12 | R5 修复后 | 43 | 1.975 / 0.687 | 3.189 / 0.318 | 10/0，0.001953 |

表内数值取自各 run 的 `state.json`〔E2〕。

- toy-04 起数值逐 run 不同，原因是种子改为 HMAC 派生〔E3〕。
- 基线子集复现的容差放在任务包里：相对 0.35，τ 另有 0.25 的绝对下限，比代码默认的相对 0.25 宽（`orchestrator.py:182-185`，`tasks/toy-hfd/task.json:19-20`〔E1〕）。代码注释给的理由是 HMAC 种子下有界指标的采样漂移可达 ±0.15。toy-06 与 toy-audit 的基线子集 τ 相对论文报告值偏差约 55% 与 59%，是靠这个绝对下限通过的〔E2〕。
- 只读任务包的锁定哈希在当日变过〔E2〕：`BENCHMARK.md` 一次（toy-03 → toy-04 之间，对应 R2 去公式与种子表的修复〔E4〕）；`task.json` 两次（toy-01 → toy-02、toy-05 → toy-06 之间），后一次改了什么，审查留档没有记载〔E3〕。
- toy-audit 在 `runs/` 下留了 84 个文件；R5 留档写明用了 `runs/toy-audit` 实测，同时自述"仓库未改"〔E2〕。
- R5 之后没有新的 run 目录〔E2〕。

**真实后端的准备情况**：当日最后一个改动是 18:19 的 `config/smoke-glm.json`，一个 OpenAI 兼容 HTTP chat/completions 后端的冒烟配置。凭据从环境变量读，不写进配置；循环参数与调用上限都调小（上限 150 次）〔E2〕。配套的 HTTP 调用在 `agent.py:116-163`，有 3 个测试覆盖，其中 2 个起本地回环 shim 服务器〔E1〕。但 `runs/` 下没有对应的 run 目录，**没有任何一次接真实模型的流水线运行留痕**〔E2〕。配置注释写的是"模型 id 探测后按结果改"，说明当时尚未定型〔E3〕。

## 6. 终审结论与未完成项

**结论**：作为工程壳可以判 GO——流程齐全、留痕完整、断点续跑与预算闭环可用、63 个测试全绿。作为"能对真实模型做可信科研评测"的系统还不成立〔E3，综合结论〕。

接真实模型的前置条件，按阻塞程度排列：

1. **OS 级隔离（未落实，硬阻塞）**。`DESIGN.md` §3 的部署硬要求是候选进程换 uid、进容器或只读白名单沙箱，网络出口默认封堵〔E3〕。现状是复制执行加环境变量白名单，候选仍是同用户进程（`bench.py:148-156`〔E1〕）；留档记载同用户下 0.7 秒即可定位密钥目录〔E4〕。DESIGN 的原话是：无隔离时防作弊栈只对"诚实但出 bug 的模型"有效，对蓄意作弊的模型无效——这是能力边界，不是配置项。R5 也把"OS 隔离、断网"写成 GO 的前提〔E3〕。
2. **`_common.md` 没有接线**〔E1〕。README 称它是全角色共享的"红线与共用背景"（`README.md:60`），但运行时只读 `prompts/<role>.md`（`agent.py:28`，`roles.py:26-28`），代码、测试和任何一份 transcript 都不引用它。"files 映射交付"与"静态禁令提示"两条规则只写在这个文件里，26 个角色 prompt 都没有，R2 修复表里的"`_common` 禁令提示"因此没有送达任何模型。禁令与接口另经 `BENCHMARK.md` 送达（`orchestrator.py:162-168`〔E1〕），所以对有文件访问权的 CLI 后端影响有限〔E3〕。对没有文件访问权的 HTTP 后端，编码角色并不知道要在 JSON 里带 `files`（`agent.py:58-69`〔E1〕），首个 coder 很可能因为没写出 `method.py` 而直接失败〔E3〕。
3. **一处评测路径没有静态门禁**〔E1，读码观察，未做攻击验证〕。critic 判 Engineer 后由 engineer 改写 `method.py`，随后的重跑（`orchestrator.py:354-362`）前面没有 `_static_gate`；其余候选路径都有（`orchestrator.py:97, 171, 176, 468, 506, 598`）。终审对 best 的静态扫描仍在，风险限于不进 best 的想法的数字。
4. **费用上限在 HTTP 后端下不生效**〔E2〕。冒烟配置的单价为 0，USD 记账恒为 0，只有调用次数上限是硬的。
5. **引用核验是降级模式**〔E2〕。`tools.search_cmd` 为 null，审计报告标注 `memory-only(degraded)`，即模型凭记忆核验。
6. **只能评这一个任务**〔E1〕。指标计算内联在框架，白名单只有 `mse`、`rank_tau`（`bench.py:39`），换第二个任务要先把这些移进任务包。

其他遗留：

- **`run_trial.py` 在待删除区**。toy-01 实际执行的脚本在 15:47 前后被移入工作区的 `_to_delete/`，现行目录里已经没有它，源码与 `__pycache__` 都没有〔E2〕。它的 SHA256 与 `runs/toy-01/state.json` 的记录逐字符一致〔E1〕，是 toy-01 执行方式唯一可核的证据。**删除前需确认是否捞回**：删了之后 toy-01 的哈希记录无法再核。
- **文档滞后于代码**〔E2〕。README 与 `AGENTS.md` 只记了第一轮审查，README 还写着"DESIGN.md v0.2"。DESIGN 的版本头停在"v0.3（二轮审查后）"，正文却已含三、四轮的内容；CONTRACTS 的版本头同为 v0.3。HTTP 后端在 `docs/` 里没有任何记载。CONTRACTS 的 `state.json` 字段表少了实际存在的 `limitations`、`baseline`、`ablations`、`ablation_progress`、`paper` 五个字段。`bench.py:21-22` 的注释仍写残余爬取"慢"，与 DESIGN §3 的 0.7 秒／1.6 秒口径不符。
- **`runs/` 若日后入库需先清理**〔E2〕：toy-04 的 `state.json` 里有 1 个明文种子密钥（R3 修复前的旧形态，本报告未取值）；`runs/` 下另有 4 个 0600 密钥文件（本报告未读取）；8 个 run 的 `state.json`、`log.jsonl`、结果文件与部分 transcripts 都含本机绝对路径。
- **目录尚未 git 初始化**（`AGENTS.md:20`：用户确认后再 `git init`）〔E2〕。

## 7. 证据索引

路径相对工作区根，前缀 `scientisttwo-repro/` 的为现行现场。

| 内容 | 路径 |
|---|---|
| 框架概览与声明 | `scientisttwo-repro/README.md`、`scientisttwo-repro/AGENTS.md` |
| 论文公开信息盘点 | `scientisttwo-repro/docs/PAPER_NOTES.md` |
| 论文↔实现对照、威胁模型 §3、超参数 §5 | `scientisttwo-repro/docs/DESIGN.md` |
| 26 角色契约、基准侧与状态契约 | `scientisttwo-repro/docs/CONTRACTS.md` |
| 五轮审查留档 | `scientisttwo-repro/docs/REVIEW-2026-11-05.md`、`…-r2.md`、`…-r3.md`、`…-r4.md`、`…-r5.md` |
| 配置 | `scientisttwo-repro/config/default.json`、`real.example.json`、`smoke-glm.json` |
| 后端调用、角色注册、fake 后端 | `scientisttwo-repro/scientisttwo/agent.py`、`roles.py`、`fakebank.py` |
| 沙箱、种子派生、数值比较 | `scientisttwo-repro/scientisttwo/bench.py` |
| 流水线状态机、静态门禁、硬门 | `scientisttwo-repro/scientisttwo/orchestrator.py` |
| 状态、续跑、锁、预算 | `scientisttwo-repro/scientisttwo/state.py` |
| 审计四件套、独立复核、refine 循环 | `scientisttwo-repro/scientisttwo/integrity.py`、`verify.py`、`refine.py` |
| 共用 prompt（未接线） | `scientisttwo-repro/prompts/_common.md` |
| 任务包 | `scientisttwo-repro/tasks/toy-hfd/`（`task.json`、`BENCHMARK.md`、`paper.md`、`src/baseline.py`、`bench/datagen.py`、`bench/run_candidate.py`、`tools_measure.py`） |
| 测试 | `scientisttwo-repro/tests/test_pipeline.py` |
| toy-01 留痕 | `scientisttwo-repro/runs/toy-01/`（`state.json`、`log.jsonl`、`FINAL.md`、`audit_report.md`、`paper.md`、`verify/`、`transcripts/`） |
| 其余 7 个 run | `scientisttwo-repro/runs/toy-02` 至 `toy-07`、`scientisttwo-repro/runs/toy-audit` |
| **toy-01 的实际执行脚本（当前位于待删除区，删前需确认是否捞回）** | `_to_delete/scientisttwo-repro/tasks/toy-hfd/bench/run_trial.py`（2,800 字节；SHA256 `901237f6a8e42ced38442b850ec547075332895e3027902ee9b3979c9a5d8c2d`，与 `runs/toy-01/state.json` 的记录一致） |

未读取、未引用的敏感件：`scientisttwo-repro/runs/` 下的 4 个密钥文件，以及 toy-04 `state.json` 中的种子密钥值。

# LIBERO-PRO 单线执行计划

> v2.0 · 2026-09-19 · 用户范围变更：仅继续 LIBERO-PRO，取消 RoboDojo / RoboLab 后续工作。
> 状态：执行前计划，未启动正式采集。本版替代本文件 v1.0 的 T0–T12 工作队列。
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 LIBERO 仿真与昇腾推理环境上，完成 Student-only、GPT Direct、GPT Hybrid 的四套件六条件评测，交付可复核的逐格结果、配对比较与失败归因。

**Architecture:** 使用现有 LIBERO-PRO 原生环境、LiberoSession 和 pi05_libero 转换权重；补齐条件选择、LIBERO 动作契约、GPT 工具循环及正式 runner。复用已修正的结果汇总与请求账本，通过集成测试确认其适用于 LIBERO。Mac 负责代码、编排和轻量测试，Linux 容器负责仿真/渲染和 NPU 推理。

**Tech Stack:** Python / pytest / LIBERO-PRO / robosuite / MuJoCo / OSMesa / JAX / PyTorch / torch_npu。

**Spec:** 本版用户范围决定优先；`2026-09-17-LIBERO-PRO补充面板初步计划.md` v0.2 中的实验契约作为技术参考；`configs/libero_pro.lock.json`、`configs/validation.json`、`configs/lp_a2_validation.json` 作为已有锁定与验收依据。旧总计划中仅复用被本版明确列出的日志、重试、统计规则。

## Global Constraints

- 当前唯一交付对象是 LIBERO-PRO。RoboDojo / RoboLab 已退出范围，既不是前置条件，也不是未来默认队列；保留历史代码和证据，不删除、不回滚。
- 取消旧 T3–T6 中 RoboDojo reset、USD转换、双臂场景、三相机、十任务、支持臂、衣物、RoboDojo 原RNG验收，以及旧 T12 RoboLab审计/迁移。已做成果可留存，但不再为它们投入后续实现或验收。
- 4 suites × 6 conditions × 3 methods = 72 个方法结果格；24个条件格指尚未乘方法数。suite为 goal / spatial / object / 10；condition为 Ori / Obj / Pos / Sem / Task / Env。
- Student-only / Direct / Hybrid 是方法；app-server / API网关是接入方式，二者不得混称。保留已有两接入适配方向，真实运行的接入集合写入冻结配置，跨接入结果分层展示。
- 沿用官方客户端约束：wait10、学生 replan5；max_steps：spatial=220、object=280、goal=300、libero_10=520。Direct/Hybrid控制语义必须按LIBERO控制器单独定义和验证。
- LIBERO使用自己的图像、本体及7D学生动作协议；不得将RoboDojo的双臂14维schema或DLS实现直接当作LIBERO接口。
- checkpoint使用锁定的 pi05_libero / pi05_libero_pt；原RoboDojo 8642服务health正常不能替代LIBERO权重服务验收。
- 不改变原物理/评分语义换取高成功率；Task的指令、对象/BDDL、初态、成功判据同步切换；Sem只改变指令。
- 开发、数值验收、正式统计样本分离；正式样本前冻结代码/输入/模型/每格n/配对/预算，结果出来后不放宽门槛。
- 从未进入策略阶段的基础设施初始化最多额外重试2次；策略阶段后不整集重跑挑结果。未知执行状态标记incomplete，全部attempt保留。
- 凭据与敏感原始日志不入公开文档；不修改同事原始交付目录。跨机部署按文件清单和hash核对。
- E1=源码，E2=配置与运行记录，E3=工程判断；本计划任务与排期均为E3，不能当已完成证据。

## 1. 已有基础与剩余问题

### 可以复用的已有结果

- LP0来源锁定、LP1老栈环境、LP2权重转换、LP-A0数值对比已有证据，保留各自原验收范围。
- LP-A1/A2证明 goal/task0 的会话/指纹和设备替换链路可跑；此前本轮目标机session测试9 passed，不等于24个条件格已覆盖。
- 修订时发现仓库已有新的T0–T4提交，不沿用旧审计HEAD bf60127的缺陷状态。共享结果/ACK修复已存在；本次实跑 `test_results_accounting.py + test_transport_recovery.py + test_agent_contract.py` 为 **58 passed**。
- 上述58项是组件复验，不证明共享模块已接入LIBERO正式runner；agent契约测试通过也不证明真实传输/7D控制已实现。
- LP-A0所列中间量/未解码动作证据缺项需核对并关闭，但不重新启动已退出范围的RoboDojo W6/W7。

### 仅保留这些当前缺口

1. 四套件六条件的完整case/变体解析与环境接线。
2. LIBERO学生服务协议、正式随机流/预处理身份及开发覆盖。
3. GPT Direct/Hybrid在LIBERO中的观测、动作和工具契约。
4. 真实GPT传输、工具循环、错误恢复及用量记录。
5. 正式批量runner、attempt账本、配对指纹和统计集成。
6. LP5精度、任务权重、配对样本池与吞吐重算。
7. 冻结、正式采集和报告。

历史审计中F03–F07的RoboDojo专项问题不再阻塞本线；F10只复查LIBERO自己的证据要求。F01/F02不重复盲修，先复用已提交修复并做集成验收。

## 2. 阶段、依赖和完成门

| 顺序 | 工作包 | 产物 | 依赖真实GPT |
|---|---|---|---|
| L0 | LIBERO运行基线与共享组件复验 | 部署/版本/权重/测试清单 | 否 |
| L1 | 24条件格与case身份 | 四套件六条件任务清单、条件测试 | 否 |
| L2 | 学生policy与LIBERO动作契约 | 参数化学生链路、Direct/Hybrid动作适配 | 否 |
| L3 | 批量runner与统计集成 | 可执行Student-only开发campaign、故障日志 | 否 |
| L4 | GPT接入与工具循环 | Direct/Hybrid离线及真实开发闭环 | 后半段是 |
| L5 | 样本与成本冻结 | 每格n/精度/配对/部署hash、正式manifest | GPT成本部分是 |
| L6 | 正式采集与报告 | 72格结果或如实标注的覆盖子集 | GPT行是 |

依赖：L0 → L1 → L2 → L3；L4假传输可提前，真实闭环依赖L2/L3；L5在L1后可开始计算，正式冻结须满足对应方法的验收门。学生可先冻结和采集，但须预先冻结共享case身份及后续GPT配对规则，禁止看到学生结果再挑GPT案例。

## L0 · 运行基线与共享组件复验

**文件：** `README.md`、`docs/acceptance/runtime.md`、`docs/acceptance/status-matrix.json`、`scripts/check_runtime.py`、`tests/test_results_accounting.py`、`tests/test_transport_recovery.py`。

- [ ] 记录gap-sim的Python/mujoco/robosuite/torch版本、PRO commit、资产路径与hash，确认与现有lock一致。
- [ ] 单独确认LIBERO权重服务入口、checkpoint身份、设备/容器和实际调用协议。复用LP-A2服务前检查是否仍在运行，不把其他权重服务冒烟算成LIBERO通过。
- [ ] 确定agent编排在Mac还是目标机，按实际角色同步文件；不因远端缺Mac专用client而无条件拷贝整个仓库。
- [ ] 复验已修共享统计/ACK模块；检查缺日志、身份冲突、重复请求、poisoned、init_fingerprint差异等断言，不以单测通过替代L3故障集成。
- [ ] 更新状态表的active_scope，非LIBERO专有工作包列为out_of_scope；共享模块仍保留实现/单测/目标机集成/正式验收四态。

已验证本地命令：

```bash
./.venv/bin/python -m pytest tests/test_results_accounting.py tests/test_transport_recovery.py tests/test_agent_contract.py -q --tb=short
```

目标机在现有libero_pro环境显式设置OSMesa和LIBERO_CONFIG_PATH后运行 `tests/test_libero_session.py`；Mac的LIBERO跳过不能算目标机通过。

**完成门：** LIBERO环境/模型/入口身份明确；组件复验通过；已有LP-A0/A1/A2证据的范围与缺项写清楚。

## L1 · 四套件六条件与共享case身份

**文件：** 修改 `src/gap_repro/sim/libero_session.py`；新增 `src/gap_repro/libero/cases.py`、`tests/test_libero_conditions.py`、`configs/libero_cases.dev.json`。

**接口：** CaseSpec包含suite/task/condition/variant/init_index/seed、BDDL/初态/指令/成功谓词身份、max_steps、wait和预处理配置。三个方法消费同一个CaseSpec。

- [ ] 对照钉定PRO源码接入Ori/Obj/Pos/Sem/Task/Env选择与资产，不得只将condition写进日志。
- [ ] 建立4×6×10任务级覆盖表，展开实际变体/初态索引并核对可用数量；Sem三改写明确编号。
- [ ] 验证Sem前后只有语言变化；Task的四件套同步变化；其余扰动使用源协议允许的变化，不任意组合。
- [ ] 将全量case池和变体身份hash写入开发manifest，禁止正式结果出来后选择有利案例。
- [ ] 明确reset序列、RNG以及fixture抖动的配对口径；保存三层指纹，区别纸面可比路径和工程钉扎消融。
- [ ] 每个任务级配置完成reset、观测、原成功判据读取和少量dummy step；记录缺失与异常，不能抽掉失败case补新case。

**验收：** 24条件格均有实际生效证据；三方法的指令/布局/预算来源一致；无未解释的BDDL/init/变体缺失。

> 2026-09-19 论文交叉验证修订（证据档 §6b）：Sem 格在论文管线下 = Ori 等价重跑
> （evaluate.py 指令取 benchmark 属性 = 文件名转写原句，_lan 的 BDDL 只改 :language
> 且文件名与基线相同）→ 其"实际生效证据"按管线语义验收：证明四要素与基线全同
> + 按论文协议重跑，不要求改写指令送达模型；是否改为送达改写句是 L5 冻结前的
> 用户决策点，默认保持论文口径保证数字可比。

## L2 · 学生链路与LIBERO方法动作契约

**文件：** 新增 `src/gap_repro/libero/policy.py`、`libero/actions.py`、`libero/agent_contract.py`、`tests/test_libero_actions.py`、`tests/test_libero_policy.py`；复用 `scripts/lp_a2_policy_server.py`，保留原LP-A2验证脚本与证据。

**接口：** policy接收CaseSpec及原生observation，输出7D动作块及candidate/noise/checkpoint身份；actions适配层将Direct/Hybrid目标转换为原LIBERO控制器输入，显式记录单位、坐标系与夹爪符号。

- [ ] 参数化suite/task/condition及步数，沿用锁定图像翻转/resize、本体编码、学生replan5；不用固定goal/task0脚本直接生成全矩阵成绩。
- [ ] 审计正式学生噪声/RNG契约。LP-A2验证用NumPy噪声与正式协议分开，明确正式采用的源序列、重复候选重取规则和hash；修改数值路径时做针对性三腿复验。
- [ ] 核对LP-A0未解码动作/中间量证据要求与实际产物，补齐缺失项或明确未完成范围，不用解码输出PASS替代全部声明。
- [ ] 设计单臂Direct、Hybrid候选/修正/接管/交还的LIBERO契约，对照控制器说明和钉定源码验证。旧双臂schema仅为逻辑参考，不能直接下发。
- [ ] 实测平移、旋转、夹爪正负、步数上限、饱和/越界、NaN/Inf、工具重复调用；确认物理运动方向与观测一致。
- [ ] 24条件格各选预定开发case做学生闭环，输出成功/失败均完整的记录；这批是管线覆盖，不计入正式n。

**验收：** 四套件步数和7D动作正确；全部模式有动作执行证据；学生服务与LIBERO环境可持续往返；失败如实记录，不以成绩高低判管线通过。

## L3 · LIBERO批量runner、故障恢复与结果集成

**文件：** 新增 `scripts/run_lp5_campaign.py`、`src/gap_repro/libero/runner.py`、`tests/test_lp5_runner.py`；接入现有 `results.py`、`transport.py`。

**接口：** `run_episode(case, method, policy, agent, audit_sink) -> EpisodeResult`；CLI区分 `--mode development|formal`，formal验证冻结manifest和所有输入hash。不扩展RoboDojo的run_w11_campaign.py。

- [ ] 接CaseSpec加载、独立reset、观测/候选/动作、原success判据、原步数预算、视频、终局与自动汇总。
- [ ] 持久记录request/dispatch/ACK阶段与candidate/noise身份；网络等待不额外推进物理时间；执行结果未知不得盲重发。
- [ ] 集成故障反例：初始化失败后成功；策略开始后超时；ACK丢失/重取；旧epoch；重复日志；进程重启；缺失case；两个方法指纹不一致。
- [ ] 结果聚合以冻结case清单为分母；使用LIBERO原生success，不要求虚造缺失native_score。缺失分数保留null。
- [ ] worker物理状态/RNG/模型上下文隔离；先单worker开发campaign，再测小规模并发，量化CPU渲染、NPU队列、内存和日志吞吐。

**验收：** 一个CLI真实执行多case Student-only开发campaign；全部attempt可追溯；异常不变成成功、配对不串状态；变更冻结输入会拒绝formal启动。

## L4 · GPT传输与Direct/Hybrid工具循环

**文件：** `agent/client.py`；新增 `agent/app_server.py`、`agent/api_runner.py`、`libero/agent_loop.py`、`tests/test_libero_agent_loop.py`。

- [ ] 先用假传输走通observe→candidate→execute→feedback→stop；验证Direct/Hybrid模式、接管gate与交还。此阶段无需GPT凭据。
- [ ] 落实现有connect/request/next_message/close；两接入统一工具事件，保留请求关联、限流等待、超时和工具错误反馈。
- [ ] 模型只见规定图像、本体、指令与同集历史；BDDL答案、真值评分、未来轨迹不可出现在模型可见输入。
- [ ] 工具参数必须走L2单臂动作校验；格式错误返回同会话修正，不触发未经校验的动作。
- [ ] 真实接入信息登记模型/provider/effort/endpoint或启动命令、凭据环境变量名及限流策略；凭据不进版本库。
- [ ] 使用预定开发case完成Direct/Hybrid真实episode和至少一次接管/交还路径验证，保存对话、动作、用量及耗时。若自然episode未触发接管，另用开发fixture验证控制流，不能伪造正式结果。
- [ ] 两接入能力分别验收；正式采用哪些接入由L5 manifest明确，方法与接入维度分开统计。

**验收：** 真实模型能获取LIBERO图像、发工具、驱动环境并结束episode；可核查token/缓存用量；费用未知记null，不写0。

## L5 · LP5样本、配对与成本冻结

**文件：** 新增 `scripts/plan_lp5.py`、`tests/test_lp5_design.py`；修订 `2026-09-19-LP5冻结提案.md`，满足冻结条件后生成 `configs/lp5_freeze.json`。

- [ ] 用Wilson实际半宽整数搜索重算样本，不把Wald公式标作Wilson：

```python
import math
def wilson_half_width(p, n, z=1.96):
    return z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)

def required_n(p, target):
    n = 1
    while wilson_half_width(p, n) > target:
        n += 1
    return n
```

- [ ] 明确先验规划与最坏情况保证的区别；为Task近零格设定实际精度/检测目标。未知GPT成功率不套用学生p。
- [ ] 冻结suite的任务权重、每任务n和Sem改写分配；如果不等样本，明确微/宏平均及区间口径，不能直接冒充论文等任务权重。
- [ ] 冻结学生/GPT共享case池与配对交集。GPT每格≥30与学生16/20集冲突必须解决；可提高学生对应样本或如实声明较小配对交集。
- [ ] 每格给N、精度、变体/任务分配、方法、接入、配对数和成本来源；分别计算24格学生与72格三方法总量。
- [ ] 使用L3实测并发吞吐和L4实测模型调用量估算墙钟；CPU并行不直接抵消NPU串行耗时，不能以“32worker≈4h”代表完整采集。
- [ ] 统一旧提案的总量、分摊公式与决策表；1,786/1,410仅为旧算术方案，不能直接确认成正式样本。
- [ ] 提交完整可审阅方案，依既定冻结流程确认后生成manifest；学生先行时也须预先锁定case池、协议和后续配对选择规则。

**验收：** 脚本、Markdown表和JSON数字一致；统计假设可追溯；代码/模型/输入/阈值hash齐全；不得看正式成绩后选格、扩样或放宽门槛。

## L6 · 正式采集与报告

**依赖：** 所跑方法的L0–L5完成门通过，冻结生效。

- [ ] 跑前校验LIBERO服务身份、部署hash、资产/权重、运行环境和冻结manifest。
- [ ] 按冻结调度执行Student-only、Direct、Hybrid；记录全部attempt、指纹、对话、候选、动作、成功判据、终止原因和用量。
- [ ] 检查异常、缺日志、配对漂移、接管/交还、视频与评分是否一致；实现/评分修复另建campaign，禁止只重跑失败case拼表。
- [ ] 输出72格的n、成功率、95%区间、配对差与缺失清单；预算或接入限制导致未覆盖的格明确标注，不声称全矩阵完成。
- [ ] 单列学生与论文的协议差异及对照；GPT重点分析Sem语言改写与Task目标变化，两者结论分开。
- [ ] 报告同时列环境/NPU/模型/网络耗时、token、可得费用及版本差异；脱敏结果按周工作流归档。

**完成定义：** 当前项目仅要求LIBERO-PRO所冻结的覆盖范围有完整可追溯结果。无需完成RoboDojo或RoboLab；成功率接近论文不是验收条件。

## 3. 执行节奏

| 批次 | 内容 | 规划（E3） |
|---|---|---|
| 首批 | L0 + L1 + L2契约核查 | 先形成24条件覆盖表与7D动作协议；无需GPT |
| 第二批 | L2学生开发闭环 + L3批量runner | 根据真实吞吐确定worker数；无需GPT |
| 第三批 | L4假传输→真实GPT | 离线部分可与前两批独立推进，真实部分依赖接入信息 |
| 第四批 | L5逐格精度/成本/配对冻结 | 提供具体数字后确认正式规模 |
| 第五批 | L6采集与报告 | 由冻结N与实测吞吐计算耗时 |

不继承旧计划关于场景重建、十任务、衣物及RoboLab的工期，也不简单把已取消工作的天数加到LIBERO上。L1/L2开发覆盖完成后，根据条件适配复杂度与GPT单集耗时更新工时估计。

每包：必要的反例测试→实现/接线→对应环境验收→证据与状态更新→独立审查→关闭阻断项→独立commit。未修改部分不为“全绿”重跑无关benchmark测试。

**立即下一动作：L0核对LIBERO服务与共享组件部署，随后L1建立四套件六条件case清单。** GPT接口、样本规模决策均不阻塞这两项。

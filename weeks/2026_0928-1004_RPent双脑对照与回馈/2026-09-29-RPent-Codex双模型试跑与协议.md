# RPent Codex 双模型试跑与评测协议

> v0.7 · 2026-09-29 · 已审定8集：5成功、3失败；Sol一次成功后收尾中断已单独审查留痕；不是最终成绩报告。
> E1=源码证据；E2=配置、元数据及运行日志；E3=估算与解释。敏感连接信息、认证与原始运行文件留在工作目录，不入公开仓库。

## 1. 已确认范围

用户确认在一天左右预算下，采用 GPT-6 Astra 与 GPT-6 Sol，均为 `medium`；每个模型运行 Spatial Task、Spatial Swap、Long Swap 的全部 10 个 task，每 task 取 seed 0、1，共 120 集。该规模覆盖全部 task，但不是官方每套件 100 集的完整复现。先双线试跑，验证后再决定是否升四线。正常任务失败不重跑刷分；基础设施异常单独留痕。[E2]

suite 名称分别为 `libero_spatial_task`、`libero_spatial_swap`、`libero_10_swap`。所有 30 个 task 的初态数量均已验证为 50；seed0/1 对应初态索引0/1。[E1/E2]

## 2. 无 API key 的接入

保留 RPent 原生 `CodexPlanner` 与 `openai_codex` SDK。SDK stdio 经私有 SSH UNIX socket 转接到 Mac 的 Codex app-server；RPent MCP 的图像与工具结果经独立私有 socket 转发。ChatGPT 登录凭据只在 Mac 使用，不复制到节点。每个 episode 新建 Codex thread 和独立工作目录，模型与 effort 由调用参数控制。[E1/E2]

接入仅映射本机执行 cwd 与 MCP endpoint，不重写实验提示、工具结果或模型输出。原生 shell、web search、apps、plugins 在本批客户端禁用。另显式禁用 Mac 个人与内置 skills；以实际 rollout developer 消息验证未注入技能目录。结果标为 Codex planner，不能当作 API planner 的行为完全等价测量。[E1/E3]

源码基线：RPent `eecf206959008700a315f58dc98146f147006f09`，沿用节点原有 NPU/osmesa 补丁；本次转接脚本放在仓库外，不覆盖原有改动。[E2]

## 3. 资产与冻结记忆

发现现有安装的 Spatial Task t3/t7 初态文件为空。RPent 自带 `robots/libero/guides/pro_hybrid_guide.md` 指示 perturbation 的权威资产应来自 HF 数据集 `zhouxueyang/LIBERO-Pro`。[E1/E2]

本批固定数据集 revision `c86fc3b8293185a6f373677018ff3e37f8391602`，下载三个 suite 的 BDDL+init 共60文件，并通过独立 `LIBERO_CONFIG_PATH` 使用，不修改共享安装包。与旧安装比较：Spatial Task 两份BDDL、十份init不同；另外两套件40文件相同。修复后每个task都有50初态。[E2]

复制既有本地记忆为347文件快照，保存逐文件SHA256，使用 `--memory-profile local --no-auto-merge-memory`、非explore模式。Long Swap t9 缺任务级记忆的现状两模型共享，不临时补解题策略。快照文件权限设为只读；这不是对容器root的强制安全边界，仍需结束后复核hash。[E1/E2]

## 4. 隔离与预算

- 每集独立 env_server、输出目录、Codex会话；两条初始lane各用独立VLA进程，共享带锁的SAM3服务。[E1/E2]
- VLA底层使用进程级随机数；本批两个VLA启动时均设置seed 20260929。不可声称环境seed相同即动作噪声完全相同。[E1/E2]
- 两模型统一：`--planner-timeout-s 5400 --max-episode-steps 10000 --max-turns 100`，外层5700秒watchdog。[E2]
- 当前原生非dashboard Codex planner的 `max-turns` 只是记录，未形成硬截断。时间和环境步数才是本批实际硬预算；不与API planner的100请求上限混称等价。[E1]
- 账户额度、网络及MCP往返也影响吞吐；四线与24小时完成均未验证，不能承诺。[E3]

## 5. 已完成验证

1. Astra、Sol的本机 `medium` 连通性探针均成功。[E2]
2. 远端原生SDK通过本机登录返回READY。[E2]
3. 两模型分别调用远端MCP测试工具，正确读出marker和红色图片；Astra 12.4s、Sol 17.9s，仅为接入探针，不能用于任务速度比较。[E2]
4. 两个独立VLA加载约201s/213s并监听各自端口。[E2]
5. 首对接入试跑曾进入图像定位、移动、VLA抓取阶段；审计实际rollout发现个人skills仍被注入，因此主动中止并将两集永久标记为诊断用途，不计正式成绩，也不作为任务失败。[E2]
6. 修复：遍历并禁用实际Mac个人与内置skill路径，且将配置放在远端SDK参数之后，避免被旧skills.config覆盖。新Astra探针的全部developer块均无Available skills/个人目录，model/effort实录为gpt-6-astra/medium，MCP图像回报正确；Sol同配置探针也已完成。[E2]
7. 正式首对使用 `formal-results/<model>/<suite>/t0_s0/attempt-001`，mkdir拒绝覆盖；VLA重新启动复位随机数，统一时间上限，首对最终结果见第7节。其余118集使用独立调度账本继续，正常失败不重跑。[E2]

## 6. 产物定位

- 本地工作目录：`rpent-repo/codex-eval-20260929/`，含转接脚本、测试、连接日志及隔离客户端运行目录。含认证引用与原始内容，整个目录不得直接上传。
- 节点同名实验目录：冻结资产/记忆manifest、两模型probes、`results/` 保存已排除的诊断试跑；`formal-results/<model>/<suite>/t<task>_s<seed>/<attempt>/` 保存正式结果。
- 可发布版本仅在本周目录整理结论和必要脱敏证据；后续报告更新版本号及git记录。

## 7. 正式首对验收（2026-09-29）

两模型均完成 Spatial Task t0 / seed0 / attempt-001，并主动调用 `finish(status=failure)`；SDK 正常返回 completed，未触及5400秒planner预算或5700秒外层watchdog。进程退出码1对应未解出，不能单独据此认定基础设施故障。环境全部状态均为 `terminated=false`，最终 `truncated=false`。两集计入正式任务失败，保留原始证据，不再以基础设施重试名义重跑。[E2]

| 模型 | 本集结果 | 外层耗时 | SDK耗时 | 工具调用 | 输入token（其中缓存） | 输出token |
|---|---|---:|---:|---:|---:|---:|
| GPT-6 Astra / medium | 失败 | 3374.1s（56.2min） | 3300.1s | 76 | 4,340,980（4,203,264） | 7,587 |
| GPT-6 Sol / medium | 失败 | 3857.9s（64.3min） | 3781.9s | 134 | 6,217,736（6,080,128） | 14,011 |

验收证据：[E2]

- 两份实际 rollout 的 `turn_context` 分别为对应模型与 medium；developer 消息未发现个人 skills 注入。
- 原生 `states.json`：Astra 27条状态、Sol 26条状态，无成功终止；原生 transcript 的 finish 与上述判定一致。无成功 recipe，符合未解出状态。
- 两份 `episode.mp4` 均可打开；Astra 1121帧、Sol1261帧，20fps。全帧解码均通过。此检查不代替环境成功谓词，也不声称已逐帧人工审阅。
- 冻结资产60文件、记忆347文件逐一 SHA256 复核，全部与启动前manifest一致。
- 上游响应流多次断线重连，最终恢复并正常完成。重连事件不会单独将本集排除；耗时包含这些延迟，尚未拆分净模型/网络/仿真时间。

首对检查时只完成每模型1集（2/120）。该 task/seed 两模型均0/1，不能据此公布三套件总体成功率或与官方100集套件成绩直接比较。[E2/E3]

## 8. 后续调度约束与时间判断

首对结束后核验：容器内存约51GiB，实验所在NPU显存约19.2/65.5GB；这些是空闲阶段快照，只支持容量判断，不能证明四线吞吐。维持每模型一条lane，先运行后续相同task/seed的配对，四线仍待独立吞吐验收。[E2/E3]

持久调度采用以下约束：每集重启所属独立VLA并使用固定seed20260929；两集均收尾后再放行下一对；每对启动前检查官方账户额度与冻结文件；额度未知、耗尽、桥接断开、hash不符或未能正常收尾时停止新集。仅保留脱敏额度比例，不复制认证。正常失败继续计分，异常需要独立审查，不自动重试，不覆盖原attempt。[E2，已实现并通过本机与容器测试，见第9节]

按首对外层耗时做线性估算：剩余59对在双线下约63小时，另加VLA复位等开销；按总episode时间除以四并发估算约29.6小时，尚未包括并发争用及加载开销。仅有一对样本，任务难度不同，估计不代表最终耗时；现有实测不足以支持24小时完成承诺。[E3]

账户周额度启动预检读取已用42%；这是账户共享用量，不能全部归因于本评测，也不能由两集精确外推出剩余集数。不自动使用重置券、购买额度或替换模型。[E2/E3]

## 9. 双线持久调度落地

`batch_scheduler.py` 与 `test_batch_scheduler.py` 放在工作目录及节点同名实验目录。15项测试分别在Mac、容器通过；真实 `--check` 导入首对两项正常失败，并确认剩余118集/59对，未触发额外推理。随后以容器detached进程启动调度，从Spatial Task t0/seed1开始加载两套新VLA。[E2]

- 当前每模型一线，VLA用独立新端口，每集结束仅清理调度自身创建的进程组；首对旧VLA已核实退出。[E2]
- 两次额度闸门均调用现有私有转接中的官方 `account/rateLimits/read`，全程25秒截止；合法的单周额度窗口与 `secondary=null` 已覆盖，原始账户响应不落盘。[E2]
- 账本以原子替换落盘，进程锁覆盖异常落盘全过程；重复调度进程不能改写已有状态。测试覆盖已计分失败跳过、attempt拒覆盖、正常失败退出码1、结果与环境不一致、超时、额度未知/耗尽及VLA监听进程归属。[E2]
- `batch/state.json` 是新批次调度状态；`formal-gpt-6-*.status.json` 继续保留首对原始状态，不能将其误读为整批进度。冻结120身份计划保持不变，完成/待运行状态由账本管理。[E2]
- 每对均正常收尾后才启动下一对；needs_review或基础设施异常会阻止新集，需要审查后另行处理。当前实现不自动重试，不复用未验收attempt，不扩大到四线。[E2]

此处证明调度已启动，不代表118集已经完成；逐集最终结果仍须以原生环境判定与产物验收为准。[E2]

## 10. 第二对结果及预算超时审查

Spatial Task t0/seed1：Astra环境最终 `terminated=true`、`truncated=false`，正常finish success，成功recipe写出，原生elapsed 858.1s；录像320/320帧完整解码。Sol触发原生5400秒planner预算，SDK interrupted、finish为空，原生elapsed 5425.3s；全部26条环境状态均未成功，最终 `terminated=false/truncated=false`，无成功recipe，录像1223/1223帧完整解码。[E2]

调度将Sol标为needs_review并阻止下一对。审查确认其按预注册墙钟预算退出，已有环境与结果完整收尾；网络重连多次恢复后仍有工具推进，未发现环境服务崩溃或外层watchdog强杀。该集记为 `task_failure / planner_budget_exhausted`，保留网络等待占用预算这一限制，不扩时、不改判分、不重跑。基础设施等待与模型解题耗时尚未可靠拆分，不能把该耗时全部归因于模型能力。[E2/E3]

审查前后60份资产、347份记忆hash保持一致。原attempt不修改；独立 `batch/review-sol-t0-s1-attempt-001.json` 保存原调度记录、审查理由及日志/状态/录像SHA256，持锁更新调度账本。重新预检后剩余116集/58对，继续保持每模型一线。[E2]

截至本次审查共4/120集：Astra 1成功/2集，Sol 0成功/2集，均只覆盖Spatial Task t0的两个seed；不能据此报告完整套件成功率。账户最新可读周额度已用45%，不使用重置券或额外购买。[E2]

## 11. 第三对验收及自动推进

Spatial Task t1/seed0 两模型均成功：Astra原生elapsed 496.6s、Sol949.1s。两者正常finish success，最终环境 `terminated=true/truncated=false`，各有成功recipe；录像分别261/261帧、293/293帧完整解码。实际rollout模型分别为Astra/Sol且effort=medium，无个人skills注入；冻结60份资产和347份记忆hash无变化。[E2]

调度未经过人工重启，已自动加载下一对t1/seed1的独立VLA。累计6/120集：Astra 2成功/3集，Sol1成功/3集；仅覆盖Spatial Task的部分任务，不能当作完整套件成功率。周额度已用47%。仍保持每模型一线，未验证四线吞吐。[E2]

## 12. 第四对与成功后的收尾异常

Spatial Task t1/seed1：Astra正常finish success，elapsed722.4s，环境成功，成功recipe及293/293帧录像完整。Sol在release后环境已 `terminated=true/truncated=false`，但随后本机rollout在13:45:29 UTC出现 `turn_aborted / reason=interrupted`；对应转接请求仅有initialize、initialized、thread/start、turn/start，没有发出turn/interrupt。中断来源尚未确定，不能称为模型主动结束。[E2]

该本机会话已关闭，远端转接进程仍等待、阻塞SDK收尾。核对进程归属后，仅向这一个转接进程发TERM，没有终止仿真或给模型发送解题信息；原RPent评测进程随后捕获TransportClosedError并自行保存结果。原生elapsed1764.7s包含收尾等待；finish消息为空，但环境成功状态和原生成功recipe均保存，录像266/266帧完整解码。[E2]

调度保守标记 `infrastructure_error / no normal finish` 并停止后续放行。审查依据中断前环境成功及原生recipe，保留本集成功，同时单独标注成功后的基础设施收尾故障；不重新运行、不抹去原始异常。`batch/review-sol-t1-s1-attempt-001.json` 保存原调度记录、干预范围、审查理由和证据hash。资产60文件、记忆347文件SHA256仍一致。[E2]

累计8/120集：Astra3成功/4集，Sol2成功/4集，均仅覆盖Spatial Task t0、t1的两seed。剩余112集重新通过预检后继续双线，未升四线。此次恢复只验证后续连接可用，没有证明此前网络或中断问题的根因已经解决。[E2/E3]

恢复前预检发现私有转接socket拒绝连接；本机SSH控制主进程与原防休眠进程已退出，而桥接服务仍可本地连接。仅在确认没有活动评测且socket无监听后，清除本批遗留socket，按原参数重建SSH控制连接，并为当前桥接进程恢复防休眠。新的官方额度预检通过（周额度51%），队列仍为112集/56对，随后恢复双线调度。SSH退出与会话中断时间相关，但尚不能证明唯一因果；没有修改评测提示、模型或预算。[E2/E3]

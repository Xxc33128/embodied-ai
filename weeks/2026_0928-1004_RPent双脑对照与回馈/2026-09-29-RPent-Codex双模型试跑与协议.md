# RPent Codex 双模型试跑与评测协议

> v0.67 · 2026-09-30 · 已审定108集：93成功、15失败；双线推进至Long Swap t7/seed0；不是最终成绩报告。
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

## 13. 第五对启动异常（不计模型失败）

恢复连接后的Spatial Task t2/seed0未完成有效评测：Sol创建attempt-001后，上游连续报 `workspace routing discovery failed/timed out`，最终will_retry=false、SDK failed；输入、输出和工具调用均为0，原生elapsed225.0s。Astra已加载VLA，但在加载后的额度复检超时，评测进程未启动、formal attempt未创建。两项均保留为基础设施异常，不计入模型失败，也不计入已审定8集。[E2]

调度自动停止放行；SSH控制连接仍存活，账户读取仍显示允许使用，周额度52%，不能将本次错误等同于额度耗尽或再次SSH断线。路由发现失败的上游原因尚未确定；新的独立MCP连通性探针与显式attempt隔离恢复方案正在检查中，不覆盖旧日志/attempt，不盲目放行新集。[E2/E3]

恢复探针实际完成：Astra12.2s、Sol11.0s，均只调用一次inspect_probe，正确返回标记及红色图像，说明本次检查时模型路由与MCP链路恢复。探针不计评测成绩，也不能保证后续网络稳定。[E2]

诊断文件留痕修正：本次临时包装对旧探针固定输出路径的替换未命中，覆盖了 `probes-clean` 中旧探针摘要；随后把本次输出独立复制到带UTC时间的新诊断目录。旧探针本机原始会话rollout仍保留，正式评测attempt、冻结资产和记忆未受影响。后续恢复attempt使用显式输出标识并拒绝覆盖。[E2]

调度增加显式审查后的attempt支持：只有账本中合法且高于旧编号的retry_attempts批准项才可重试基础设施异常；已计分成功/失败仍跳过，同一批准不能反复消费。结果目录、batch日志和记录均含新attempt标识；旧调度脚本保留快照。20项测试在本机与容器通过，包括旧结果拒覆盖、重试日志隔离及已计分失败不重跑。[E2]

`batch/review-t2-s0-retry-002.json` 保存两项原异常记录及10份原产物hash。仅批准这两个身份进入attempt-002：Sol保留attempt-001；Astra虽无正式attempt-001，也使用002以区分已有启动日志。正式预检确认剩余112集/56对、周额度54%，随后重新启动双线调度。不改提示、判分、记忆、模型或预算；若再次异常仍停止，不自行申请下一attempt。[E2]

## 14. 第五对独立attempt验收

Spatial Task t2/seed0的attempt-002两模型均正常finish success，最终环境 `terminated=true/truncated=false`，成功recipe写出。Astra原生elapsed373.5s，录像163/163帧；Sol385.6s，录像162/162帧，均完整解码。实际模型与medium、无个人skills在运行时已复核；资产60文件、记忆347文件hash无变化，审查时保存的10份原异常产物hash也全部一致。[E2]

该结果计入原120身份各一次，不把启动异常额外计入分母，也不覆盖attempt-001。累计10/120集：Astra4成功/5集，Sol3成功/5集；仍只是Spatial Task部分任务。调度已自动进入t2/seed1，周额度55%，继续双线。[E2]

## 15. 后续配对验收记录

以下耗时为原生elapsed，包含运行中的等待；每行对应两模型的同一task/seed。成功均核验正常finish、环境 `terminated=true/truncated=false`、成功recipe与全帧视频解码；不会把视觉观感替代环境判定。[E2]

| suite / task / seed | Astra结果 / 耗时 / 视频帧 | Sol结果 / 耗时 / 视频帧 | 冻结文件 |
|---|---|---|---|
| Spatial Task / t2 / s1 | 成功 / 374.2s / 166 | 成功 / 395.8s / 174 | 60资产+347记忆hash一致 |
| Spatial Task / t3 / s0 | 成功 / 666.4s / 474 | 失败 / 801.2s / 591 | 60资产+347记忆hash一致 |
| Spatial Task / t3 / s1 | 成功 / 739.3s / 463 | 成功 / 749.5s / 466 | 60资产+347记忆hash一致 |
| Spatial Task / t4 / s0 | 成功 / 771.5s / 549 | 成功 / 1171.8s / 918 | 60资产+347记忆hash一致 |
| Spatial Task / t4 / s1 | 成功 / 942.9s / 672 | 成功 / 1629.9s / 1176 | 60资产+347记忆hash一致 |
| Spatial Task / t5 / s1 | 成功 / 370.7s / 208 | 成功 / 580.8s / 309 | 60资产+347记忆hash一致 |
| Spatial Task / t6 / s0 | 成功 / 613.1s / 404 | 成功 / 728.7s / 525 | 60资产+347记忆hash一致 |
| Spatial Task / t8 / s0 | 成功 / 644.3s / 429 | 成功 / 635.1s / 410 | 60资产+347记忆hash一致 |
| Spatial Task / t8 / s1 | 成功 / 520.0s / 331 | 成功 / 586.9s / 430 | 60资产+347记忆hash一致 |
| Spatial Task / t9 / s0 | 成功 / 802.2s / 592 | 成功 / 736.2s / 525 | 60资产+347记忆hash一致 |
| Spatial Task / t9 / s1 | 成功 / 1019.1s / 743 | 成功 / 1055.7s / 802 | 60资产+347记忆hash一致 |

t3/seed0的Sol正常finish failure，11条环境状态均未terminated，最终truncated=false，录像591/591帧完整解码；计为正式任务失败，不重跑。两模型实际medium、无个人skills已在本对运行时复核。[E2]

t3/seed1两模型均正常finish success，环境terminated=true/truncated=false，成功recipe与录像完整；实际模型/medium和无个人skills在运行时复核。[E2]

t4/seed0两模型均正常finish success，环境terminated=true/truncated=false，成功recipe与录像完整；实际模型/medium和无个人skills在运行时复核。[E2]

t4/seed1两模型均正常finish success、环境terminated=true/truncated=false，成功recipe与全帧录像完整。实际模型/medium、无个人skills已复核。[E2]

本次累计20/120集：Astra9成功/10集、Sol7成功/10集，仅覆盖Spatial Task t0–t4；周额度69%，仍允许使用。[E2]

## 16. t5/seed0启动前端口异常

Sol在VLA启动前的端口bind检查报Address already in use；没有创建正式attempt，也没有启动该集模型推理，不计任务失败。随后只读socket检查未见该端口条目，bind探测成功；原占用进程或TIME_WAIT状态未及时捕获，原因仍未确认，不能断言已修复。[E2/E3]

已将原异常记录及调度源码hash保存到独立review-sol-t5-s0-prelaunch-001.json；未杀进程、未覆盖记录、未放行重试。同对Astra继续启动，调度将等待该线结束后按未解决结果停下；届时再审查独立attempt恢复。冻结60资产与347记忆hash仍一致。[E2]

Astra t5/seed0正常finish success，环境terminated=true/truncated=false，成功recipe存在，原生elapsed417.8s，录像206/206帧完整解码，实际Astra/medium及无个人skills已在运行时复核。累计21集：Astra10成功/11集，Sol7成功/10集；冻结文件hash一致。[E2]

调度停下后，持有scheduler.lock复核Sol尚无正式attempt-001，端口再次bind成功；保留原异常与前次审查文件hash，仅批准Sol t5/seed0进入attempt-002，Astra已计分结果不重跑。独立review-sol-t5-s0-retry-002.json保存批准与证据；此次为可用性恢复尝试，不声称已找到或修复瞬时占用根因，再次异常须重新审查。[E2/E3]

正式预检通过：剩余99集/50个调度组（首组仅Sol），周额度70%，允许使用。持久调度以独立scheduler-resume-004.log恢复，仍最高双线，不改模型、预算或实验内容。[E2]

Sol t5/seed0 attempt-002正常finish success，环境terminated=true/truncated=false，成功recipe存在，原生elapsed469.9s，录像252/252帧完整解码。实际Sol/medium、无个人skills已复核；60资产、347记忆及恢复审查保存的原证据hash均一致。只计入该身份一次，不计启动前异常为额外集。[E2]

累计22/120集：Astra10成功/11集、Sol8成功/11集。调度已自动恢复t5/seed1双线VLA加载，额度仍允许使用、周用量70%；原端口占用原因未确认，继续监控。[E2]

后续t5/seed1两模型均正常finish success，环境terminated=true/truncated=false、成功recipe及录像完整；实际模型/medium、无个人skills已复核，冻结文件hash一致。累计24/120集：Astra11成功/12集、Sol9成功/12集，仅覆盖Spatial Task t0–t5。调度自动进入t6/seed0双线，周额度71%，仍允许使用。[E2]

t6/seed0两模型均正常finish success，环境terminated=true/truncated=false、成功recipe及录像完整；实际模型/medium、无个人skills已复核，冻结文件hash一致。累计26/120集：Astra12成功/13集、Sol10成功/13集，仅覆盖Spatial Task部分初态。调度自动进入t6/seed1双线，周额度73%，仍允许使用。[E2]

## 17. t6/seed1上游容量异常

Sol attempt-001在读取记忆和初始view_env_state后收到明确serverOverloaded、will_retry=false；SDK failed且无正常finish。原生elapsed98.3s、SDK73.6s、17次工具调用，只有初始环境状态，terminated=false/truncated=false，未执行机器人动作，未产生成功recipe或视频。计基础设施异常，不计任务失败。[E2]

实际Sol/medium和无个人skills已核验，周额度73%且允许使用，错误指向模型服务容量，不能等同额度耗尽。原异常及8份文件hash已保存到review-sol-t6-s1-overloaded-001.json；未批准新attempt。同对Astra继续运行，配对结束后再审查恢复，不替换模型，不覆盖原结果。[E2]

Astra t6/seed1正常finish success，环境terminated=true/truncated=false、成功recipe存在，原生elapsed828.4s，录像530/530帧完整解码。实际Astra/medium、无个人skills已复核；累计27集：Astra13成功/14集、Sol10成功/13集。[E2]

配对结束并停止后，冻结60资产/347记忆及Sol原attempt的8份文件hash全部一致；端口可用、官方额度复检允许使用（74%）。距容量错误超过12分钟，持锁审查仅批准Sol独立attempt-002，保存review-sol-t6-s1-retry-002.json；Astra已计分成功不重跑。此为一次恢复尝试，上游容量恢复尚未证实，再次异常须新审查。预检剩余93集/47组，持久调度使用scheduler-resume-005.log启动，保持原模型与medium及全部协议。[E2/E3]

Sol t6/seed1 attempt-002正常finish success，环境terminated=true/truncated=false、成功recipe存在；原生elapsed822.0s，录像561/561帧完整解码。实际Sol/medium、无个人skills已复核，冻结60资产/347记忆及原attempt的8份文件hash均一致；保留容量异常attempt-001，该身份仅计一次。[E2]

累计28/120集：Astra13成功/14集、Sol11成功/14集，仅覆盖Spatial Task t0–t6。调度自动恢复t7/seed0双线VLA加载，周额度75%，仍允许使用。[E2]

## 18. t7配对验收及端口预检修正

t7/seed0两模型均正常finish success，环境terminated=true/truncated=false、成功recipe及录像完整。Astra elapsed575.4s、331/331帧；Sol841.7s、538/538帧。实际模型/medium、无个人skills已复核，冻结60资产及347记忆hash一致。累计30/120集：Astra14成功/15集、Sol12成功/15集。[E2]

下一对Sol t7/seed1再次在VLA启动前bind报Address already in use，未创建正式attempt，不计模型失败；Astra继续运行，原异常已保存独立review-sol-t7-s1-prelaunch-001.json，尚未批准重试。[E2]

隔离Linux测试复现旧预检缺陷：主动关闭服务连接留下TIME_WAIT（TCP状态06），旧普通bind返回errno98，设置SO_REUSEADDR后成功；原生HTTP/Socket RPC服务均allow_reuse_address=True。候选修正仅让调度端口预检与原生服务一致，不使用SO_REUSEPORT，仍拒绝真实监听者，并保留后续PID归属验收。两次现场缺少错误瞬间的socket快照，TIME_WAIT作为具体事故原因仍属推断。[E1/E2/E3]

新增TIME_WAIT可复用、真实监听者拒绝两项测试，本机与容器共22项通过；候选文件单独保存，运行中调度尚未替换，待本对结束后持锁部署与审查恢复。周额度76%，仍允许使用。[E2]

Astra t7/seed1正常finish success，环境terminated=true/truncated=false、成功recipe存在；原生elapsed687.3s，录像448/448帧完整解码。实际Astra/medium、无个人skills已复核；累计31集：Astra15成功/16集、Sol12成功/15集。[E2]

配对停止后持有scheduler.lock，确认冻结文件及候选源码hash一致，原调度/测试脚本分别保存pre-portcheck快照，再原子部署已通过22项测试的候选。独立review-sol-t7-s1-retry-002.json记录部署前后源码hash、旧异常与一次性attempt-002批准，仅恢复Sol。预检通过：剩余89集/45组、额度78%且允许使用；scheduler-resume-006.log启动，保持每模型最多一线。此修正消除可复现的TIME_WAIT误判，不将现场原因推断升级为已证实。[E2/E3]

Sol t7/seed1 attempt-002正常finish success，环境terminated=true/truncated=false、成功recipe存在；原生elapsed1435.0s，录像1086/1086帧完整解码。实际Sol/medium、无个人skills已复核；冻结60资产/347记忆hash一致，旧调度快照和新调度hash均与部署审查一致。[E2]

累计32/120集：Astra15成功/16集、Sol13成功/16集，仅覆盖Spatial Task t0–t7。调度自动进入t8/seed0双线VLA加载，本次跨集未再触发端口预检错误；周额度79%，仍允许使用。一次正常转换不足以证明后续无端口异常。[E2/E3]

t8/seed0两模型均正常finish success，环境terminated=true/truncated=false、成功recipe及录像完整；实际模型/medium、无个人skills已复核，冻结文件hash一致。累计34/120集：Astra16成功/17集、Sol14成功/17集，仅覆盖Spatial Task部分初态。调度自动进入t8/seed1双线VLA加载，周额度80%，仍允许使用。[E2]

t8/seed1两模型均正常finish success，环境terminated=true/truncated=false、成功recipe及录像完整；实际模型/medium、无个人skills已复核，冻结文件hash一致。累计36/120集：Astra17成功/18集、Sol15成功/18集，仅覆盖Spatial Task t0–t8。调度自动进入t9/seed0双线，周额度82%，仍允许使用。[E2]

t9/seed0两模型均正常finish success，环境terminated=true/truncated=false、成功recipe及录像完整；实际模型/medium、无个人skills已复核，冻结文件hash一致。累计38/120集：Astra18成功/19集、Sol16成功/19集；Spatial Task仍差最后一对，尚未完成整套验收。调度自动进入t9/seed1双线VLA加载，周额度83%，仍允许使用。[E2]

## 19. Spatial Task整套验收

最后一对t9/seed1两模型均正常finish success，环境terminated=true/truncated=false、成功recipe与全帧录像完整；实际模型/medium、无个人skills已复核，冻结文件hash一致。账本核对每模型20个唯一身份（10任务×seed0/1），旧诊断集与基础设施attempt均不额外计分。[E2]

| 模型（medium） | 成功/正式集 | Spatial Task成功率 | 正式失败身份 |
|---|---|---|---|
| GPT-6 Astra | 19/20 | 95% | t0/seed0 |
| GPT-6 Sol | 17/20 | 85% | t0/seed0、t0/seed1（预算耗尽）、t3/seed0 |

这仅是本批NPU/Codex medium、每任务2个初态的结果，不等同官方low或每任务10个初态的成绩。Sol成功后收尾异常的既有审定保留，详见前文；正常失败不重跑。[E2/E3]

累计40/120集：36成功、4失败；Spatial Swap与Long Swap尚未完成。持久调度已自动进入Spatial Swap t0/seed0双线VLA加载，周额度85%，仍允许使用，保持双线。[E2]

## 20. Spatial Swap配对验收

每行均已核验实际模型/medium、无个人skills、正常finish success、最终环境terminated=true/truncated=false、成功recipe和全帧视频解码。耗时为原生elapsed；冻结60资产与347记忆hash每对复核。[E2]

| task / seed | Astra结果 / 耗时 / 视频帧 | Sol结果 / 耗时 / 视频帧 | 冻结文件 |
|---|---|---|---|
| t0 / s0 | 成功 / 500.0s / 319 | 成功 / 659.0s / 484 | hash一致 |
| t0 / s1 | 成功 / 553.8s / 324 | 成功 / 746.2s / 515 | hash一致 |
| t1 / s0 | 成功 / 793.7s / 483 | 成功 / 918.7s / 611 | hash一致 |
| t1 / s1 | 成功 / 520.6s / 314 | 成功 / 696.3s / 454 | hash一致 |
| t2 / s0 | 成功 / 320.0s / 117 | 成功 / 344.0s / 121 | hash一致 |
| t2 / s1 | 成功 / 335.0s / 153 | 成功 / 762.6s / 193 | hash一致 |
| t3 / s0 | 成功 / 593.6s / 362 | 成功 / 1025.2s / 631 | hash一致 |
| t3 / s1 | 成功 / 558.2s / 352 | 成功 / 971.2s / 715 | hash一致 |
| t4 / s0 | 成功 / 554.6s / 303 | 成功 / 538.0s / 269 | hash一致 |

累计58/120集：54成功、4失败；Spatial Swap目前各9成功/9集，尚非整套成绩。调度自动进入t4/seed1双线VLA加载，周额度98%，仍允许使用。[E2]

## 21. Spatial Swap t4/seed1收尾异常

Astra attempt-001在step10已达环境terminated=true/truncated=false，审计文件写入成功，但finish(success)的MCP传输失败，原生finish为空，调度标记infrastructure_error。原生elapsed为595.4s，录像352/352帧完整解码，recipe存在；60资产与347记忆hash无变化。独立review保存原账本及产物hash，未批准重跑，未覆盖attempt。[E2]

Sol同身份仍在运行；保持配对屏障，待其结束后审定Astra成功后的收尾异常并处理调度。58集统计暂不增加该待审定身份。周额度已用99%，后续由官方额度检查控制放行，不自动重置或购买。[E2]

22:56 UTC复查：官方额度窗口显示已用100%，预计2026-10-05 14:34:52（UTC+8）恢复。当前配对仍仅Sol运行，Astra的infrastructure_error使配对屏障阻止继续放行；本次不重启调度、不干预在途解题。等待额度自然恢复或用户自行处理账户额度，恢复前不启动新集。成功后收尾异常仍保留独立审查记录。[E2]

23:01 UTC收尾验收：Sol t4/seed1正常finish success，环境terminated=true/truncated=false，原生1426.3s，录像1009/1009帧完整解码。Astra按既有“环境成功后收尾异常”标准审定成功：595.4s、352/352帧，保留finish MCP传输异常及原账本hash，不重跑。独立adjudication引用原review hash，在取得调度锁后仅更新账本分类；调度保持stopped，并记录quota_exhausted放行限制。冻结文件hash仍一致。[E2]

累计60/120集，56成功、4失败：Spatial Task为Astra19/20、Sol17/20；Spatial Swap前10集/模型均成功（t0–t4、seed0/1）。剩余60集未运行，尚无最终全套成绩；监控保留，待额度恢复后重新核验再放行。[E2]

## 22. 额度恢复后续跑

2026-09-30用户要求继续后，官方额度查询显示已用0%，远端account/rateLimits/read确认allowed=true；调度--check通过，剩余60集/30对、冻结文件一致。NPU7启动前HBM占用5%，资源检查通过。取得调度锁后保留独立恢复记录、清除旧额度阻塞标记，以独立scheduler-resume-007.log启动原双线；账本确认Spatial Swap t5/seed0两模型均进入独立VLA加载。未重跑已评分身份，未使用重置券、购买额度或替换模型；额度恢复原因未作推断。[E2]

恢复后首对Spatial Swap t5/seed0两模型均正常finish success，环境terminated=true/truncated=false、recipe及全帧录像完整：Astra661.1s、346/346帧，Sol592.6s、310/310帧。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计62/120集（58成功、4失败）；自动进入t5/seed1双线VLA加载，额度已用2%。[E2]

Spatial Swap t5/seed1两模型均正常finish success，环境terminated=true/truncated=false、recipe及全帧录像完整：Astra490.4s、251/251帧，Sol581.3s、334/334帧。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计64/120集（60成功、4失败）；自动进入t6/seed0双线VLA加载，额度已用3%。[E2]

Spatial Swap t6/seed0两模型均正常finish success，环境terminated=true/truncated=false、recipe及全帧录像完整：Astra711.0s、501/501帧，Sol1328.2s、852/852帧。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计66/120集（62成功、4失败）；自动进入t6/seed1双线VLA加载，额度已用5%。[E2]

Spatial Swap t6/seed1两模型均正常finish success，环境terminated=true/truncated=false、recipe及全帧录像完整：Astra771.7s、569/569帧，Sol1313.2s、831/831帧。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计68/120集（64成功、4失败）；自动进入t7/seed0双线VLA加载，额度已用7%。[E2]

Spatial Swap t7/seed0两模型均正常finish success，环境terminated=true/truncated=false、recipe及全帧录像完整：Astra609.4s、379/379帧，Sol975.7s、459/459帧。实际模型/medium及无个人skills已核对。累计70/120集（66成功、4失败）。[E2]

下一对开始前，官方额度RPC返回拒绝，调度按协议停止，未创建下一集attempt。连接复查及再次官方查询恢复allowed=true、额度已用9%；原RPC详细错误未保留，根因未定。取得调度锁后以独立review保留原state及SHA256，批准一次调度恢复；--check验证60资产和347记忆、剩余50集/25对及官方额度通过，以独立scheduler-resume-008.log恢复双线。未重跑已评分身份，也未复用旧infra retry批准。[E2]

Spatial Swap t7/seed1两模型均正常finish success，环境terminated=true/truncated=false、recipe及全帧录像完整：Astra539.4s、274/274帧，Sol1935.2s、1106/1106帧。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计72/120集（68成功、4失败）；自动进入t8/seed0双线运行，额度已用11%。[E2]

Spatial Swap t8/seed0两模型均正常finish success，环境terminated=true/truncated=false、recipe及全帧录像完整：Astra561.9s、348/348帧，Sol895.1s、450/450帧。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计74/120集（70成功、4失败）；自动进入t8/seed1双线VLA加载，额度已用12%。[E2]

Spatial Swap t8/seed1两模型均正常finish success，环境terminated=true/truncated=false、recipe及全帧录像完整：Astra487.0s、291/291帧，Sol751.1s、408/408帧。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计76/120集（72成功、4失败）；自动进入t9/seed0双线运行，额度已用13%。[E2]

Spatial Swap t9/seed0两模型均正常finish success，环境terminated=true/truncated=false、recipe及全帧录像完整：Astra941.2s、498/498帧，Sol752.6s、340/340帧。Astra在环境成功后的收尾阶段出现Broken pipe流中断，SDK自动重连后完成audit写入及finish；未人工干预、未重启或新增attempt，最终原生正常成功。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计78/120集（74成功、4失败），额度已用15%。[E2]

Spatial Swap t9/seed1两模型均正常finish success，环境terminated=true/truncated=false、recipe及全帧录像完整：Astra640.4s、369/369帧，Sol1661.2s、951/951帧。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。[E2]

## 23. Spatial Swap收口与Long Swap启动

Spatial Swap的40个计划身份已全部审定：Astra20/20成功，Sol20/20成功，均为100%。Astra t4/seed1保留环境成功后finish传输异常的既有独立审定，不将其改写为正常finish；其余身份按各配对验收记录计入。[E2]

累计80/120集（76成功、4失败），Spatial Task与Spatial Swap已完成；仍余Long Swap每模型20集。持久调度已自动进入libero_10_swap t0/seed0双线运行，额度已用17%，继续保持每模型一线。本批每任务仅2个初态，不等同官方更大初态样本的最终成绩。[E2/E3]

## 24. Long Swap配对验收

Long Swap t0/seed0两模型均正常finish success，环境terminated=true/truncated=false、recipe及全帧录像完整：Astra1098.2s、953/953帧，Sol1201.1s、850/850帧。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计82/120集（78成功、4失败）；自动进入t0/seed1双线运行，额度已用20%。[E2]

Long Swap t0/seed1两模型均正常finish success，环境terminated=true/truncated=false、recipe及全帧录像完整：Astra940.5s、761/761帧，Sol1072.1s、822/822帧。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。按冻结环境终止谓词计分，Astra审计记载第二罐仍被夹持时已触发成功，不另加释放条件。累计84/120集（80成功、4失败）；自动进入t1/seed0双线VLA加载，额度已用22%。[E2]

Long Swap t1/seed0两模型均正常finish success，环境terminated=true/truncated=false、recipe及全帧录像完整：Astra746.4s、468/468帧，Sol986.7s、703/703帧。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计86/120集（82成功、4失败）；自动进入t1/seed1双线VLA加载，额度已用24%。[E2]

## 25. Long Swap t1/seed1正常stuck分类审查

Astra attempt-001正常调用finish(stuck)，SDK完成、进程退出码0，环境terminated=false/truncated=false；模型认为目标被遮挡而未满足自身感知检查，主动结束。原生168.9s，2条环境状态、录像35/35帧完整解码，失败集未写成功recipe。该结果属于任务失败，不是可重跑的基础设施故障；分类器因未接受stuck状态而记录infrastructure_error。[E2]

独立review保存原账本记录及10个产物SHA256，60资产与347记忆hash一致，实际Astra/medium及无个人skills已核验。Sol同配对仍在运行；保留配对屏障，待其结束并取得调度锁后再将Astra归入task_failure，禁止为此创建新attempt。此处暂保留86集既有入账统计，不提前声称账本已修正。[E2]

Sol t1/seed1已正常finish success：1182.6s，环境terminated=true/truncated=false，recipe存在，录像749/749帧完整解码。配对屏障随后停止调度。取得调度锁并复核Astra原10个产物hash未变后，独立adjudication引用原review hash和原账本，将其归入task_failure；168.9s、35帧、finish(stuck)原记录保留，不重跑。[E2]

累计88/120集（83成功、5失败）。--check复核60资产、347记忆、剩余32集/16对及官方额度通过（已用26%）；以独立scheduler-resume-009.log恢复双线。此为一次审查后的调度恢复，未新增retry批准，也未改动提示、判分、记忆或分类器代码。[E2]

Long Swap t2/seed0配对验收：Astra正常finish failure，环境terminated=false/truncated=false，647.8s、录像375/375帧；夹持物在搬运及原地恢复后的短程搬运中滑落，保留正常失败，不重跑。Sol正常finish success，环境terminated=true/truncated=false、recipe存在，1464.4s、录像940/940帧。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计90/120集（84成功、6失败）；自动进入t2/seed1双线VLA加载，额度已用28%。[E2]

Long Swap t2/seed1配对验收：Astra正常finish success，环境terminated=true/truncated=false、recipe存在，856.8s、录像576/576帧；Sol正常finish failure，环境terminated=false/truncated=false，1762.4s、录像1094/1094帧。Sol审计记载炉灶已开启，但多次抓取未能搬起摩卡壶，保留正常失败，不重跑。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计92/120集（85成功、7失败）；自动进入t3/seed0双线VLA加载，额度已用31%。[E2]

## 26. Long Swap t3/seed0失败验收与stuck分类审查

Astra正常finish failure，1061.2s、录像1014/1014帧；Sol正常finish stuck，2419.4s、43条状态、录像2025/2025帧。两者环境均terminated=false/truncated=false，未写成功recipe。Astra未能将碗放入抽屉；Sol已将碗放入，但抽屉未完全关闭，均保留任务失败，不重跑。实际模型/medium及无个人skills已核验，60资产与347记忆hash一致。[E2]

Sol进程退出码0、SDK completed且finish工具正常完成；分类器未接受stuck状态，将其误记为infrastructure_error并停止配对调度。取得调度锁后，独立review保存本attempt全部文件SHA256，独立adjudication保存原完整账本及其hash并引用review hash，将该记录归入task_failure；原attempt与历史审定保持不变，未修改分类器或解题协议。[E2]

累计94/120集（85成功、9失败）。--check复核冻结文件、剩余26集/13对与官方额度通过（已用34%）；按本次独立审查的一次性恢复授权，以新scheduler-resume-010.log恢复双线，账本确认running。下一配对为t3/seed1；未创建任何重试attempt。[E2]

Long Swap t3/seed1配对验收：两模型均正常finish failure，环境terminated=false/truncated=false，未写成功recipe。Astra979.6s、录像969/969帧，碗在抽屉边缘未能落稳；Sol2376.3s、录像1902/1902帧，碗留在近乎关闭的抽屉外桌面。正常失败保留，不重跑。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计96/120集（85成功、11失败）；自动进入t4/seed0双线VLA加载，额度已用37%。[E2]

Long Swap t4/seed0配对验收：Astra正常finish success，环境terminated=true/truncated=false、recipe存在，1383.7s、录像1936/1936帧。Sol正常finish failure，环境terminated=false/truncated=false，2065.4s、录像2291/2291帧，未写成功recipe；模型认为目标杯已摆上盘，但环境谓词未触发，后续白杯重抓未成功，按冻结谓词保留失败，不重跑。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计98/120集（86成功、12失败）；自动进入t4/seed1双线VLA加载，额度已用41%。[E2]

## 27. Long Swap t4/seed1传输中断审查与独立重试

两模型attempt-001日志同时停止更新，本地会话均记录turn_aborted，SSH控制连接消失；残留转发socket连接被拒绝。连接消失的上游原因尚未确定，不将其归因于模型。先保存事故快照与产物hash，再仅终止两条已断连的remote_codex中继，使原harness正常收尾。两集finish均为空、退出码1，所有环境状态均无成功：Astra943.4s、录像740/740帧；Sol942.2s、录像541/541帧。均归为基础设施中断，不计任务失败。[E2]

失效socket重命名保留，重建SSH转发并确认master存活。取得调度锁后，独立review保存原完整账本及hash、事故记录hash和两套最终产物hash；60资产与347记忆hash一致，实际模型/medium及无个人skills沿用本配对已核验证据。仅批准每模型一次attempt-002，原attempt-001与已有98集审定不变；新attempt若再异常须重新审查。[E2]

--check通过：剩余22集/11对，官方额度已用43%，允许放行。以独立scheduler-resume-011.log恢复双线，账本确认running。本次仅修复连接并另建attempt，未改提示、物理、判分或记忆。[E2]

Long Swap t4/seed1的attempt-002配对验收：Sol正常finish success，环境terminated=true/truncated=false、recipe存在，1567.6s、录像1867/1867帧；Astra正常finish failure，环境terminated=false/truncated=false，1829.3s、录像2318/2318帧，未写成功recipe。Astra认为两杯已摆在其识别的盘上，但谓词未触发，白杯重抓恢复未成功；保留正常失败，不重跑。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。原attempt-001基础设施中断及其独立review保留，重试批准已消费。累计100/120集（87成功、13失败）；自动进入t5/seed0双线，额度已用47%。[E2]

Long Swap t5/seed0配对验收：Astra正常finish success，环境terminated=true/truncated=false、recipe存在，570.9s、录像408/408帧；Sol正常finish failure，环境terminated=false/truncated=false，865.5s、录像581/581帧，未写成功recipe。Sol释放后书掉至收纳盒后方，两次原地重抓未成功，保留正常失败，不重跑。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计102/120集（88成功、14失败）；自动进入t5/seed1双线VLA加载，额度已用49%。[E2]

Long Swap t5/seed1两模型均正常finish success，环境terminated=true/truncated=false、recipe及全帧录像完整：Astra454.5s、409/409帧，Sol584.3s、404/404帧。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计104/120集（90成功、14失败）；自动进入t6/seed0双线VLA加载，额度已用50%。[E2]

Long Swap t6/seed0配对验收：Sol正常finish success，环境terminated=true/truncated=false、recipe存在，887.1s、录像746/746帧；Astra正常finish failure，环境terminated=false/truncated=false，898.4s、录像964/964帧，未写成功recipe。Astra白杯接触放置后在退离时倾倒，保留正常失败，不重跑。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计106/120集（91成功、15失败）；自动进入t6/seed1双线VLA加载，额度已用52%。[E2]

Long Swap t6/seed1两模型均正常finish success，环境terminated=true/truncated=false、recipe及全帧录像完整：Astra980.9s、977/977帧，Sol835.7s、671/671帧。实际模型/medium及无个人skills已核对，60资产与347记忆hash一致。累计108/120集（93成功、15失败）；自动进入t7/seed0双线VLA加载，额度已用55%。[E2]

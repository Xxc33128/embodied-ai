# RPent Codex 双模型试跑与评测协议

> v0.2 · 2026-09-29 · 接入验证完成，修复技能隔离后重新启动首对；不是最终成绩报告。
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
7. 正式首对使用 `formal-results/<model>/<suite>/t0_s0/attempt-001`，mkdir拒绝覆盖；VLA重新启动复位随机数，统一时间上限，尚无最终任务判定。当前只放行这两集，其余118集待首对验收后再调度。[E2]

## 6. 产物定位

- 本地工作目录：`rpent-repo/codex-eval-20260929/`，含转接脚本、测试、连接日志及隔离客户端运行目录。含认证引用与原始内容，整个目录不得直接上传。
- 节点同名实验目录：冻结资产/记忆manifest、两模型probes、`results/` 保存已排除的诊断试跑；`formal-results/<model>/<suite>/t<task>_s<seed>/<attempt>/` 保存正式结果。
- 可发布版本仅在本周目录整理结论和必要脱敏证据；后续报告更新版本号及git记录。

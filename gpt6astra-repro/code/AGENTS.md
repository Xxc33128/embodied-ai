# AGENTS.md — GPT-6 Astra 机器人评测复现（gpt6astra-repro）

> **20次测试恢复执行**：trial-01有效完成并保留；trial-02首次因账户/子agent中断，等待第14决策超时退出（13个已执行动作），原目录原样保留且不计成绩。当前从相同scene seed 2001重跑至 `logs-persistent-subagent20-20260915-113640/trial-02-attempt-02/`，之后继续trial03–20。一轮一个全新子agent，标准prompt只替换mailbox路径。

> **2026-09-15 15:50 GLM 批次完成（与 GPT 批次完全隔离）**：应用户”跑你的、结果存 glm 文件夹”要求，新增 `logs-glm-subagent20-20260915-120407/`：bowl-eval-v1 标准 20 trial 全部完成，20/20 记录检查 PASS，初态与共享冻结库逐 trial 精确匹配（与 GPT 批次同 bank，diff 一致）。结果：盲评 stage=4 成功率 0/20，env_success 0/20，阶段分布 0×11/1×6/2×2/3×1（trial-03 stage3 双盲评+物理交叉核验）。runner 仅改模型标签（`run_subagent_trials_glm.py`，环境/提示词哈希照常校验）；最多 6 并发已披露，墙钟不作串行比较。GPT 批次 trial-02 中断现场冻结于 `gpt-trial02-freeze-20260915-1153/`（FREEZE_NOTE.md 含续跑路径），`logs-persistent-subagent20-20260915-113640/` 未改动，其 20 有效 trial 未凑齐、两模型对照暂不可下。详见周目录 `2026-09-15-GLM标准20次子agent测试结果.md`。

> **2026-09-15 11:36 新批次执行中**：用户再次明确”测试二十次”。当前目录 `logs-persistent-subagent20-20260915-113640/`，20个共享初态复用并逐trial精确复核。每trial一个 GPT-6 Astra/medium、fork_turns=none 子agent连续完成；20次policy决策/900控制步，内部读图工具不计机器人决策。wire仍保留最近2组图，但子agent保留本trial此前记忆，已在manifest披露接入方式差异，不宣称独立API完全等价。不得改物理/提示词、读取.env或把私有评分反馈给模型。

> **2026-09-15 用户澄清（优先于下文）**：每个trial仅创建一个无过往聊天上下文的子agent，由它连续完成整个任务并保留本trial记忆，不能每次决策重建agent；下一个trial再创建新agent。此前每决策新agent的批次已中断：0/20完整trial，trial-01执行6个动作后停止并保留视频，不能计入正式成绩。账户曾返回额度错误，但随后fresh usage查询显示允许使用，不认定为持续额度阻塞。

> **2026-09-15 标准实验启动**：用户已要求按标准开子 agent 执行20次测试，覆盖下文“暂不运行”的旧状态。使用 `run_subagent_trials.py`、`.venv-robosuite/` 和 `logs-standard-subagent20-20260915-112007/`；标准提示词薄适配与900步已接入，桥接3项/环境13项回归通过，20个初态已冻结。每次决策均为 fork_turns=none 的新 GPT-6 Astra/medium 子agent，只读当前公开请求与图像。正在运行，尚无整组正式成绩。不得把协调器或过往聊天信息交给决策模型。

> **2026-09-15 标准流程**：用户最新要求先制定统一测试规程，暂不运行此前提出的10次子agent实验。正式规范见周目录 `2026-09-15-bowl标准测试流程.md` 及 `bowl-eval-v1/`：每模型20次，共享scene seeds 2000–2019，每trial新上下文、20调用/900步、统一精简提示词、视频与请求/动作/评分完整记录。现有eval_bowl.py与子agent桥接尚未完全适配该标准；不可把制定规范写成正式实验已完成。1000系列与此前交互演示均属开发记录。

> **2026-09-15 最新独立复核**：robosuite 底层脚本入碗10/10、现有契约8/8均复测通过，但尚未打通模型闭环。适配器夹爪公式反向、yaw把xyzw误读为wxyz、碗壁group0被默认渲染隐藏，均已独立实测；本轮未修生产代码。后续先处理这些接口问题，详情见周目录 `2026-09-15-robosuite适配器独立复核.md`。下文“闭环打通”等状态须依此限定。

> **2026-09-14 原始来源更正**：后续复现范围以 `../embodied-ai/2026_0914-0920_GPT6Astra评测复现/2026-09-14-原始评测复核与复现范围.md` 为准。原报告概要写 0.58.0，但抽查 5 次 Astra bowl transcript 均写 0.57.1、wire=responses、max_steps=900，且观测含 joint_eff。不能再声称协议/版本/观测已 100% 对齐。Panda 与自研 IK/夹爪控制器是本地实现选择，不是原评测必要前置；用户明确无截止时间，旧 Day3 规则不再有效。保留现有版本及历史实验，不自动升级或降级。

复现 **OpenAI × RoboCurve「GPT-6 Astra on robot arms」** 评测（<https://openai.robocurve.org/gpt-6-astra/>）的**技术协议**：用开源 harness `inspect-robots` + LLM agent policy（`move_*` 工具调用 / 回合制 ≤20 次 LLM 调用 / 限速护栏 / transcript+wire 留痕），在仿真 embodiment 上跑 pick-place 任务 20 trials。

**复现的是协议（E1/E2），定性对照的才是数字（E3）**：真机是 I2RT YAM 双臂，我们只有 mock/Isaac；成功率数字**不可**与 Astra 的 19/20=95% 直接比。任何写进结论的对比都必须带这条声明。

- 权威结论文档在本工作区的周报（见下「与周报的分工」），本目录只放脚本 + 日志 + venv。
- 本目录**不是 git 仓库**（无 remote、含 `.env` 明文凭据）；`git init` 前必须先建 `.gitignore` 排除 `.venv/`、`logs*/`、`.env`。

## 一分钟上手

```bash
cd /Users/xerxes3/Documents/huawei实习/gpt6astra-repro
source .venv/bin/activate                 # py3.11.15（uv 创建，勿用系统 python）
set -a; source .env; set +a               # 导出 ZHIPUAI_API_KEY（脚本不会自动读 .env）

.venv/bin/python run_mock_agent.py        # 零 API 费全链路自检（改动后先跑这个）
.venv/bin/python eval_bowl.py --model glm-5.3-flash \
    --base-url https://open.bigmodel.cn/api/coding/paas/v4 \
    --key-env ZHIPUAI_API_KEY --embodiment cubepick --trials 3 \
    --log-dir logs-glm-smoke              # 真实 LLM 冒烟
```

Isaac / `--embodiment isaacsim-liftcube` **不在这台 Mac 上跑**（本机无 CUDA/Vulkan，插件是惰性导入因此连 import 都会失败）→ 见 `kaggle_recon_and_setup.py`。

## 文件清单

| 文件 | 作用 | 状态 |
|---|---|---|
| `quickstart.py` | 上游 `examples/quickstart.py` 拷贝：`ScriptedPolicy` + CubePick mock 冒烟，验证 harness 本身 | 可跑 |
| `run_mock_agent.py` | ★内置 stdlib 手写 OpenAI 兼容 mock LLM server 驱动 `LLMAgentPolicy` 闭环（chat wire / tool call / 限速 / 预算 / 三层日志），零成本 | 可跑 |
| `test_glm_vision.py` | 探测智谱「端点 × 模型」的 文本/图像/tool_calls 三能力矩阵 | 可跑（需 key）|
| `eval_bowl.py` | Phase 3 正式评测 CLI：model / embodiment / trials / 预算参数可配，默认对齐 RoboCurve | cubepick 可跑；mujoco-bowl 待回归达标；isaacsim 仅 Kaggle |
| `mujoco_bowl.py` | ★★自写 MuJoCo bowl 任务 embodiment（当前 **v3：Franka Panda 关节臂版**，分阶段 scripted_solve + 严格 `_go`）。物理回归入口：`.venv/bin/python mujoco_bowl.py [--deterministic]` | 接口已通（G1）；持续提升 0/10，机制已定位（见下） |
| `test_bowl_contract.py` | G1 契约测试（11 用例）：build_toolset / eef_state / move_to 单维·插值·越界·终止 / 夹爪测量语义 | 全绿 |
| `test_bowl_model.py` | G2 组装断言（4 用例）：neq==1 / nexclude==1 / 壁角 20° / equality 对象 | 全绿 |
| `diagnose_bowl.py` | 分阶段诊断：staged（默认 slow / `--pacing fast`）、`--battery`（空载网格+转腕）、`--series`（CSV 时序）、`--assembly`（Task 2 消融） | 可跑 |
| `mujoco_bowl_v2_freefloat.py.bak` | v2 浮动执行器版（无关节臂）：回归 10/10+10/10 全过的**保底实现** | 可跑 |
| `mujoco_bowl_v3_preTask12.py.bak` | v3 修复前快照（SHA256 `d2db7c91…`，与 9/14 规划核查版本一致） | 只读 |
| `verify_robosuite_backend.py` | 新口径 §5 后端核验：robosuite Lift/Panda + OSC_POSE 独立脚本抓放（夹爪符号/世界系 delta 标定见脚本注释） | ✅ 10/10 |
| `robosuite_bowl_env.py` + `robosuite_bowl_asset.xml` | bowl 化任务 RoboBowl（Lift 子类 + 静态漏斗碗 + 入碗判定）；`__main__` = 脚本入碗回归 | ✅ 10/10 |
| `robosuite_bowl.py` | inspect-robots 适配器 RobosuiteBowlEmbodiment（move_to→OSC 世界系 delta、gripper ±1、三相机、eef_state+joint_eff、真值过滤、每 seed 重建） | 契约 8/8 |
| `test_robosuite_bowl_contract.py` | 适配器契约测试（toolset/参考态/单维/插值/越界/done/真值不外泄/播收敛） | 全绿 |
| `verify_robosuite_harness.py` | harness `eval()` 主链路零费冒烟（悬停策略×robosuite-bowl） | ✅ |
| `.venv-robosuite/` | 隔离后端环境：robosuite 1.5.2 + robosuite-models 1.0.0 + mujoco 3.3.0 + **inspect-robots 0.58.0 + agent 0.26.0**（numpy 1.26.4 双栈兼容；勿混入主 venv——robosuite 子依赖 numpy<2 与主 venv 2.4.6 冲突；mujoco 3.13 与 robosuite binding 不兼容、3.2 低于其下限） | bowl 主战场 |
| `mujoco_menagerie/` | sparse clone，仅 `franka_emika_panda/`（官方 CAD 网格+数据表惯量） | 只读勿动 |
| `kaggle_recon_and_setup.py` | 7 个 `# CELL n`：环境侦察 → Vulkan 运行库 → 装 isaacsim+IsaacLab → **BOOT PROOF** → LiftCube 冒烟 → mock 全链路 | 待人工逐格执行 |
| `scene_overview.png` 等 PNG | 环境渲染样张（v2 时期三视角×三时刻拼图） | 展示用 |
| `logs*/` | 各次评测产物（`EvalLog` JSON + transcripts/actions/wire + PNG blob），**只读，勿手改** | 基线留痕 |
| `.env` | `ZHIPUAI_API_KEY`（600 权限，不入库）| — |
| `.venv/` | `inspect-robots==0.58.0` + `inspect-robots-agent==0.26.0` + `[rerun]` extras + `mujoco==3.13.0` | 版本钉死 |

## 关键事实速查（E1 = 源码实证，路径相对本目录）

**harness 双输入模型**：`eval(task, policy, embodiment)`；`Task` = Scene 数据集 + scorer，与 policy/embodiment 解耦。

**agent policy 参数**（`inspect-robots/plugins/inspect-robots-agent/src/inspect_robots_agent/policy.py::AgentPolicyConfig`）：

- RoboCurve 对齐值 = `max_llm_calls=20` + `max_speed_frac=0.25` + `effort="medium"`；**库默认分别是 `max_llm_calls=100`（`policy.py:277`）、`max_speed_frac=0.1`（`policy.py:282`）、`effort=None`（=用 provider 默认，`policy.py:281`）→ 三项全部要显式传**，否则协议身份就变了。`eval_bowl.py` 的 argparse 默认值已对齐（20 / 0.25 / medium / max_steps=200），手写 `LLMAgentPolicy(...)` 时别忘了。
- 本地默认值（E1，`policy.py:271-286`）：`wire="chat"`、`images="always"`、`image_horizon=2`、`depth="render"`。原始 Astra 样本的 wire 实际为 `responses`，其余上述三项与样本一致；本地日志只能证明本地配置，不能代替原始运行证据。
- `base_url` + `api_key_env` 是自定义 OpenAI 兼容端点的一等公民：`_llm.py::resolve_provider` 阶梯第 1 档，key 缺失也允许（本地 vLLM/Ollama）。

**工具面由 embodiment 动作空间语义自动生成**（`_tools.py:115-117`）：

| 动作空间语义 | 生成的移动工具 |
|---|---|
| `eef_delta_pos` / `eef_delta_rot` | `move_by`（mock cubepick 走这条）|
| `eef_*` absolute | `move_to`（RoboCurve 真机走这条）|
| `joint_*` absolute | `move_joints`（Isaac Franka 走这条，8 维）|

Isaac 默认 state 字段 = `joint_pos / joint_vel / eef_pos / eef_quat / gripper`（`inspect-robots-isaacsim/src/inspect_robots_isaacsim/embodiment.py:140-144`），**不含物体位姿** → Isaac 阶段视觉是刚需（也解释了为何无视觉的 `glm-4.6` 在 Isaac 路线上不可用）。

外加 `done` / `give_up` / `take_pic`（`images="on_demand"` 时才有）。每个 move 调用必须带 `note`（人类可读观察-决策说明，进 transcript）。**护栏会把大 delta 插值展开成多步**（`step_frac = min(max_speed_frac/control_hz, 0.05)` × 动作幅度），所以一次调用即可跑完整段位移 —— mock 基线里 8/20 trial 用 **1 次**调用就成功，就是这个机制。

**观测编码**：状态以 `state[key]: [数值]` 纯文本 + 相机帧 PNG data URL 进 user 消息（`policy.py::_observation_content`）→ **`state` 里有没有物体位姿，直接决定该 embodiment 是否强制依赖视觉**。

**mock CubePick**（`inspect_robots/mock/cubepick.py`）：2D、`goal_radius=0.05`、action box `±max_step`、`control_hz=10`、32² top 相机、state 含 `eef_pos`+`cube_pos`（**全可观测 → 视觉并非刚需**）、capabilities 含 `PRIVILEGED_SUCCESS`。

**注册名**（`.venv/bin/inspect-robots list` 现查）：embodiment 只有 `cubepick`；policy `agent|scripted|random|noop`；scorer `success_at_end|episode_length|min_distance_to_goal|reached_goal_state|operator`；task `cubepick-reach`。

**自带的 CLI 比自己写脚本更能干**（`inspect-robots` 会 `init_dotenv`，无需 source .env）：

```bash
.venv/bin/inspect-robots run --task cubepick-reach --policy agent -P model=glm-5.3-flash \
    -P base_url=https://open.bigmodel.cn/api/coding/paas/v4 -P api_key_env=ZHIPUAI_API_KEY \
    -P max_llm_calls=20 -P max_speed_frac=0.25 -P effort=medium --log-dir logs-cli
.venv/bin/inspect-robots inspect <log.json>     # 逐 trial 指标（零成本，已实测）
.venv/bin/inspect-robots summarize <log.json>   # 默认=确定性 markdown digest（零成本）；--model 才走 LLM
.venv/bin/inspect-robots view <log.json>        # 自包含 HTML 报告（20-trial 约 0.5MB）
.venv/bin/inspect-robots video <log.json>       # 存的相机帧 → 每相机一路 MP4（需 ffmpeg）
```

前三条（`inspect`/`summarize`/`view`）已在本目录跑通且**零 API 费**；`summarize`/`view` 建议加 `-o /tmp/xxx` 输出，以免污染日志目录。收周报时先用 `summarize` 出逐 trial 表、再用 `view` 出 HTML 截图。

## 已有基线（Phase 3 之前的管线自检，勿当结论）

| 目录 | 规模 | success_at_end | episode_length | 成本 |
|---|---|---|---|---|
| `logs/` | 5 场景 ×2 epochs | 1.00 | 8.9 | 0（scripted）|
| `logs-mock/` | 3 | 0.67 | 19.0 | 0（56 次本地调用；第 3 trial 撞 20 次预算线 = 预算路径已复现）|
| `logs-glm53flash/` | 3（scene `seed-0..2`）| 1.00 | 14.3 | ~3 分钟 |
| `logs-glm53flash-20/` | **20（scene `seed-0..19`）** | **1.00** | 11.15 | 489s / 90 次调用 / 124k prompt + 11.4k completion token |

## 端点矩阵（E1，`test_glm_vision.py` 实测；换 key 后须重跑复核）

| 端点 | 模型 | 文本 | 图像 | tool_calls | 结论 |
|---|---|---|---|---|---|
| `open.bigmodel.cn/api/paas/v4` | glm-4.6 / 4.7 / 5.3 | ✅ | ❌ (1210 only-text) | — | 429 余额不足，且无视觉 |
| 同上 | `glm-4v-flash` | ✅ | ✅ | ❌ 不发起调用 | 免费但不可用作 policy |
| **`open.bigmodel.cn/api/coding/paas/v4`** | **`glm-5.3-flash`** | ✅ | ✅ | ✅ | **当前唯一可用主配置** |
| 同上 | glm-4.6 | ✅ | ❌ | ✅ | 无视觉，Isaac 阶段不可用 |
| `open.bigmodel.cn/api/anthropic` | glm-4.6 | ✅ | ⚠️ **假通过**（答错颜色）| — | 图像未进模型，禁止用于视觉任务 |

## 硬性规则

1. **保留当前版本基线**：现有安装为 `0.58.0 / 0.26.0`；原运行核心版本与报告概要存在冲突，原 agent 插件版本尚未核实，不能声称完全同版。装新东西先 `pip install --dry-run` 确认不动这两个包；若需版本对照，另建隔离环境并记录差异。
2. **agent 参数只能从 RoboCurve 对齐值出发修改**，且改动必须写进周报表格（`--effort` / `--max-llm-calls` / `--max-speed-frac` 三者构成协议身份）。
3. **协议差异必须显式标注**才能对比：mock 是 eef 位移空间 + 状态全可观测；Isaac 是 `move_joints` + 无物体位姿（必须靠视觉）——两者与真机 eef `move_to` 都不同，成功率只作定性归因。
4. **论断带 E1–E4 等级**（定义见工作区 `AGENTS.md`）；E4 不得用于立项承诺。
5. **日志目录一次一跑**（`--log-dir` 独立命名，如 `logs-<model>-<n>`），不覆盖既有基线；`wire/*/calls.jsonl` 是逐 HTTP 调用的请求/响应，token/成本统计口径以它为准。
6. **凭据**：key 只放 `.env` 或 Kaggle Secrets；禁止写进脚本、日志、周报，禁止粘贴到聊天/公开仓库（`embodied-ai/` 是公开仓）。

## 已知坑

- `eval_bowl.py` 顶部 docstring 里的示例端点（`api/paas/v4` + `glm-4.6v`）**已失效**（无额度且无视觉），真实可用配置见上面端点矩阵。改示例前别照抄。
- 脚本**不自动读 `.env`**（只有 `inspect-robots` CLI 会 `init_dotenv`）→ 忘记 `source .env` 时 `test_glm_vision.py` 直接 KeyError，`eval_bowl.py` 则拿到空 key 得到 401。
- mock 任务**没有判别力**：单步算术 + 一次大 delta 即可通关，任何像样模型都 100%。要做模型间对照，先给 mock 加难度（去掉 `cube_pos` 状态字段强制用视觉 / 障碍 / 多目标 / 收紧 `goal_radius`），或换 `CubePickEmbodiment(goal_radius=...)`。
- `kaggle_recon_and_setup.py` 是 `.py` + `# CELL` 注释，需人工逐格粘进 Kaggle notebook；CELL 4（venv 打包 ~10GB）非必需；CELL 5 BOOT PROOF 反复失败即触发死线 → 停止恋战，回报 CELL 1 输出并切降级方案（A=自写 MuJoCo embodiment，B=cubepick mock 保底）。
- 版本组合钉死 `isaacsim==5.0.0.0` + `IsaacLab v2.3.2`（8 月云 3060 实证，E2，出处 `GR00T-WholeBodyControl/docs/GPU环境搭建指导-IsaacSim.md`）；T4 仅勉强达最低线，渲染崩则回退 `isaacsim==4.5.0.0`（需 `--extra-index-url https://pypi.nvidia.com`）+ IsaacLab v1.x。
- **场景复现口径**：scene id `seed-{i}` 只是 `--seed-base + i` 的标签；实测拿同一个 `init_seed` 直接 `embodiment.reset()` **得不出日志里的 `cube_pos`**（harness 还有自己的 seed 派生，`eval/seed=0`）。要复现某个 trial 必须固定 `--seed-base` + `--seed` 整条调用参数，不要手搓 reset 去“对齐场景”。
- `logs*/wire/**/blobs/*.png` 让单次 20-trial 跑占 ~1.3MB；若要传公开仓库必须整目录排除。
- **v3 过程坑（2026-09-14 实测）**：① move_to chunk 末动作只走 1 个控制步，位置伺服跟踪滞后 ~5cm，LLM 靠逐轮观测补偿，测试断言要用"保持后收敛"而非"零滞后"；② 下探途中闭合会把方块拨走（指头先碰顶/侧），必须沉降到位（~5mm）再闭合——写评测提示词或诊断脚本时都要遵守这个顺序；③ 夹爪实测归一化必须用受控行程 [0.026,0.076]，用关节全行程 0.08 会让"已夹持"读成"半开"；④ 近基座悬停姿态（指下）贴肘折叠边界，静态残差 25–36mm 不可收敛——悬停按转移门（40mm/6°）对待，抓取精度只看下探点；⑤ `_ik_static` 在边界姿态会发散（实测 325mm），重分支恢复必须校验静态解自身残差后再采纳。
- 本目录无 `requirements.txt`/`pyproject.toml`：可复现性目前只靠本文件 + 周报的文字记录。重建环境：`uv venv --python 3.11 && pip install 'inspect-robots[rerun]==0.58.0' inspect-robots-agent==0.26.0` + brew ffmpeg。

## MuJoCo 环境（v2 浮动 → v3 Panda，2026-09-15 交接点）

**目的**：比 mock 更接近真机的 3D 抓放 embodiment，且工具面用 `move_to`（isaacsim 插件是 `move_joints`，协议忠实度反而低）。设计/差距全量记录见周目录 `2026-09-15-MuJoCo碗环境保真度审计.md`。

**v3 设计要点（改代码前必读）**：
1. 动作空间仍是 `eef_abs_pose (x,y,z,yaw,gripper)`——协议层不变；臂侧 DLS IK（xyz+全姿态 6 维误差+关节限位+零空间回中）解关节目标交 Menagerie 位置伺服。与 RoboCurve「模型出末端位姿、机器人侧 IK」同构。
2. **IK 只写 ctrl、不写 qpos**——直接写 qpos = 运动学传送（qvel≈0），接触摩擦无法带动物体（mocap 直驱同坑，踩过两次）。
3. 执行器改造（相对 Menagerie 出厂）：夹爪腱伺服 kp 100→800/kv 10→28（出厂挤压力仅 0.7N 抓不住 64g 方块）；主指垫 8.5mm→22mm 高 + μ=2.0（原指垫提升 1.5cm 即越过方块顶，接触几何性消失）；`impratio=25`。
4. 相机全走自由相机：本版 MuJoCo (3.13.0/macOS 离屏) 的 XML 相机（xyaxes/euler/FIXED 引用）只渲染天空；自由相机 fovy 实测 ~53°≠标称 45°。top=前侧斜视（az=-45/el=-55，避开臂自遮挡），wrist=指间 TCP+0.02 向下看（dist 0.16），side=另一侧斜视。
5. `scripted_solve` 回归脚本：特权信息（方块位姿+yaw）、到位判定+沉降式移动（伺服 ~4cm 跟踪滞后）、闭合前 yaw 对齐、慢速提升 1cm/步。

**v3 当前状态（2026-09-14 Task 4 完成后）：物理候选清单已系统性排除，G3 未通过；唯一正增益 = pad solref 调硬。** agent 工具接口已通（G1）；分阶段诊断基线：闭合 9/10 建立、**持续提升 0/10**（旧流水线 5/10 success 是快节奏瞬时判据，与新 S3 持续持物判据不可互比）。要点：

- **接口与组装（Task 1–2，已修，详见周报 Task12 记录）**：state 新增唯一 (5,) 字段 `eef_state`（move_to 参考态，实测值）；`gripper` 观测=实测指间距、按受控行程 [0.026,0.076]m 归一化（夹住方块报 ~0.34=已闭合）；docs 闭合高度 z≈0.035–0.05；`build_xml(equality=, walls_radian=)` 仅消融用；恢复双指 equality（不对称 0.01–0.04→~1e-4）与 link0/link1 exclude；碗壁 euler 弧度化（65.92°→20°）。
- **失败机制（E2 证据链，详见周报 Task 3 记录）**：夹持力仅 ~3N/侧（"7mm 挤压深度"对刚性方块从未兑现，实际压入 ~0.2mm）；高度保持 1–1.5s 后指间隙振荡越过方块宽度（40.5→41.3mm）产生零压瞬间，方块棘轮下坠、指空闭到 26mm 行程下限。地面静置稳定、空中持重失稳。
- **工作空间边界（E2）**：近基座悬停点（指下姿态）肘关节顶限位，静态残差 25–36mm 不可收敛；下探点（抓取关键位姿）亚微米。悬停门 40mm/6°、下探门 3mm/2° 的分级门控由此而来。
- **执行层已修（E2 证据驱动）**：IK 零空间门控+重分支恢复（坏分支 ctrl 偏差 24mm→下探 0.0007mm）；重力沉降积分补偿（无重力补偿的伺服静态残差 5–15mm→下探亚微米）。
- **Task 4 结论（E2，全表见周报 Task4 记录）**：纯力反馈=蠕吞方块（力伺服无位置锚，~1.2mm/s 合拢）；锚定限力=量纲陷阱（12N@kp800=15mm 虚拟压入越宽挤出）；kp↑=捏肥皂加速甩出（CoM 在夹持区外时增力反向）；kv 100/250、μ 3/4/5、elliptic、noslip 1/2/3 无效；30mm 长垫部分改善（更多 seed 能 lift）；**solref "0.004 1" 唯一正增益**（1004 稳 8s，蠕滑 ~3→2.6mm/s 未归零）。最优组合仍未过 G3。`_GRASP_TCP_OFF=0.018`（+0.001 对齐方块中心已证运动学不可达）。
- **机制新增（Task 4，E2）**：④ 软接触（solref 20ms=10×timestep）是恒定法向力下切向蠕滑的温床，法向力/μ 都不改其量级；⑤ 夹持区不含 CoM 时增力加速甩出——力学路线在当前几何下全部反向；⑥ TCP 下限 ~35–38mm × pad 长度 × 窗口位置 × 桌面构成覆盖三角，单段下探全覆盖不可行，长垫+4cm/步下探过冲 ~12mm 会 pad 磕桌（16.4N 腱力全花在撑桌、gap 卡 67mm）。
- **剩余候选（下一步）**：(a) 两段式下探（安全高度严格门+贴桌宽松门）+ pad 窗口下移 ~7mm 全覆盖几何 + solref 0.004 组合（机制指向明确的最后一个组合实验）；(b) 冻结 v3 先推 Task 5 链路；(c) v2 保底跑协议对照。
- **Task 4 新旋钮**：build_xml 的 grip_kp/grip_kv/pad_mu/pad_len/pad_zoff/pad_solref/pad_solimp/cone/noslip（默认=基线几何）；embodiment 的 grip_mode="force"/grip_force（限力保持闩锁，仅诊断用）。
- 已试过的路线（勿重复）：impratio 10/25、指垫加大、慢提升、yaw 对齐、深探 z+0.018、深闭合目标 0.026/0.030、一步全闭、恢复 equality、修碗壁单位、增量式 vs 平滑提升（同败，非节奏伪影）、fast 开环 playout（无再观测，闭合 0/10）、纯力反馈保持、锚定限力（F=1–20N）、kp 3200、kv 100/250、μ 3/4/5、pad 30mm×3 窗位、深度 +0.001/+0.012/+0.020/+0.025、elliptic、noslip 1/2/3、solref 0.004×{μ3, kv100, force2}。

**机制发现存档（5 条，对 B 路线 MJWarp 有参考价值，详见审计文档）**：① mocap 静态几何接触约束不含几何体自身速度（摩擦无法拖动，需动态体+weld）；② XML 相机离屏只渲染天空（本版/macOS）；③ 自由相机 fovy 与标称不符且无属性可设（像素映射须运行时自标定或避开）；④ 旋转薄壁盒碰撞异常接触（八边形杯壁，换轴对齐漏斗壁绕开）；⑤ 位置伺服夹爪的抓持=挤压深度×摩擦，浅指垫+快速提升=接触几何性消失。

## 进度与下一步

| Phase | 内容 | 状态 |
|---|---|---|
| 0 | 环境就绪 + 版本对齐 | ✅ |
| 1a | mock LLM 全链路（零费） | ✅ |
| 1b | 真实 LLM（glm-5.3-flash @ coding 端点）20 trials 基线 | ✅ |
| 1c | MuJoCo 3D 环境：v2 浮动版（审计+回归全过） | ✅ |
| 1c' | v3 Panda 关节臂版 | 🔶 接口已通（G1/G2 ✅）；Task 3 机制定位 + Task 4 单因素排除完成；G3 未过（solref 0.004 唯一正增益）；新口径下降级为研究路线，保留产物不再作为正式评测前置 |
| 2' | 新口径 §5：核验"现成控制器 + 物理抓放示例"后端 | ✅ robosuite 1.5.2 Panda+Lift+OSC 10/10（见周报《2026-09-14-后端核验-robosuite》） |
| 2'' | 新口径 §4.1–4.2：bowl 化 + inspect-robots 适配器 | ✅ 脚本入碗 10/10 + 契约 8/8 + harness 冒烟（见周报《2026-09-14-bowl化与适配器》） |
| 2a | 独立复核三项阻塞修复（夹爪二值滞回/quat xyzw/相机 group1+翻转/z 下限 0.80）+ 收口项 | ✅ 回归 13 项全绿，**同一 move_to 工具接口完整抓放 10/10**（周报《2026-09-15-三项修复与首次视觉闭环》） |
| 2b | GLM 视觉闭环调试 | 🔶 首跑 0/2（`logs-rsbowl-glm-smoke`：闭环链路工作；sample1 预算耗尽于搬运途中，sample0 闭合时机误判）——模型侧调试中 |
| 2 | Kaggle T4×2 起 Isaac Sim（BOOT PROOF + LiftCube 冒烟） | ⏳ 待人在 Kaggle 逐格执行，回贴 CELL 1/5/6 输出 |
| 3 | `gpt-6-astra` 跑 20 trials 正式对照 | ⛔ 阻塞于 OpenAI key（建议 $50 额度）|
| 加分 | mock 任务加难（强制视觉）后再做多模型对照 | 待办（本机可做）|

## 与周报的分工

- 结论、指标表、对照分析 → `../embodied-ai/2026_0914-0920_GPT6Astra评测复现/`（`readme.md` 周目标 + `2026-09-14-复现启动与agent链路验证.md` 当日全文）。**新结论必须回写那里，本文件只是操作手册。**
- 上游源码锚点 clone（只读参考，勿改）→ `../inspect-robots/`（remote `robocurve/inspect-robots`）；插件在 `plugins/inspect-robots-{agent,isaacsim}/`，其自带 `CLAUDE.md` 在该仓库内为准。

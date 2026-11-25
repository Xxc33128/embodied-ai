# RoboTwin 2.0 适配（RT0–RT7）——对抗性审查报告

> 2026-10-13 补归档：按正文原始日期归入 9/21–9/27 周；来源为本机 GPT-as-Policy-repro 同名工作文档。此次仅整理，不重做文献查询、源码审查或实验。正文结论与数值统一按历史文档记录〔E3〕阅读，原有证据标注表示当时口径，不构成本次复验或立项承诺。
> 归档来源 SHA256：`9cdc17d4b9074aa3fd4400e2fdc6249d552e8f2e1a0a5511d3383b30bae19920`。

- 审查对象：commit 50a4a60（分支 `robotwin2-adapt`，单提交、纯新增 14 文件 +1862 行）
- 审查方式：Mac 侧离线核验——上游 RoboTwin main 分支 raw 文件重新下载对账（不轻信 /tmp 本地副本）、调研报告逐任务程序化比对、边界条件可执行探针、测试实跑；未修改任何被审查代码/测试
- 审查日期：2026-09-23

---

## 判定：PASS−（带保留通过）

**成立的核心结论**：作者声称的五项关键事实**全部核实为真**——50 任务步限逐字一致、14D 布局钉定正确、Easy/Hard 域随机化参数逐项一致、π0 基线 50 项与 π0.5 社区 31 项数字零误差、报告 off-by-one 矛盾的论断成立且保守处理恰当；robotwin 44 项契约测试全过；分支纯新增、零共享模块改动、无复制漂移（transport/runner 与 lp_a2/libero 线逐行同构）。

**保留的理由**：session 层的 `setup_demo` 重置参数协议、`exec_action` 步进候选名、腕相机键名"预期值"三处，均与**审查时即可获得**的上游源码/模型卡证据相矛盾（详见 P2-1/2/3）。三者中只有步进入口被登记为集成点；`reset()` 的错误参数既不在 lock 的"已核验事实"里、也不在四个集成点里，且被一个同义反复的假环境测试锁死。它们在真环境上的失败都是响亮的（不会静默污染数据），且 RT8 本就未声称环境机保护，故不翻判定——但 P2-1 必须在 RT8 动工前修正口径，P2-4（服务器忽略显式 noise 但 meta 谎报 explicit）是 RT9+ 设备对照实验的活陷阱。

---

## 一、事实钉定核验（审查维度 A：作者声称 1–5 逐项复核）

### 事实 1：50 任务步限"逐字抄自" _eval_step_limit.yml —— **成立**

复核命令：

```bash
curl -sL -o /tmp/rt_step_limit_verify.yml \
  https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/env_cfg/task_config/_eval_step_limit.yml
diff /tmp/rt_step_limit.yml /tmp/rt_step_limit_verify.yml   # 无输出（本地副本=上游）
./.venv/bin/python - <<'EOF'   # 全量 50 项逐一比对
import sys; sys.path.insert(0,'src')
from gap_repro.robotwin import tasks as rt
up={}
for l in open('/tmp/rt_step_limit.yml'):
    l=l.strip()
    if l and not l.startswith('#'):
        k,v=l.split(':'); up[k.strip()]=int(v.strip())
panel={t.name:t.max_steps for t in rt.TASKS.values()}
assert set(up)==set(panel) and all(up[n]==panel[n] for n in up)
print(len(up),"tasks, all match")
EOF
```

结果：`50 tasks, all match`——任务名集合与 50 个步限值**全部逐字一致**（含非常规值 put_bottles_dustbin=1700、open_microwave=1500、hanging_mug=900）。本地 /tmp 四个基准文件与重新下载的上游 main 文件 diff 全等（STEP_LIMIT/BASE_TASK/CLEAN/RAND 四项 MATCH），排除"核对基准被篡改"的可能。

### 事实 2：14D 布局钉自 get_obs 的 joint_action.vector —— **成立**

证据（`/tmp/rt_base_task.py` = 上游 `envs/_base_task.py` 逐字）：

- L488–495：`left_jointstate = get_left_arm_jointState()`；`joint_action["left_arm"]=left_jointstate[:-1]`、`["left_gripper"]=left_jointstate[-1]`；`["vector"] = np.array(left_jointstate + right_jointstate)` → **[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]**，与 `actions.py` L7–8 声明一致。
- 交叉印证 1：上游 `scripts/eval_policy_xpolicylab.py` 的 `xpolicylab_action_to_robotwin`（L1091 附近）拼 `flat_action = concat([left_arm, [left_gripper], right_arm, [right_gripper]])`——同一布局。
- 交叉印证 2：SidneyXie/pi05_robotwin 模型卡（HF 页面实查）：absolute joint 14-Dim、chunk 50 整块执行、state (14,) 每臂 waist/shoulder/elbow/forearm_roll/wrist_angle/wrist_rotate/gripper 左右排布——与 `ROBOTWIN_ACTION_DIM=14`、`ACTION_CHUNK=50`、`STATE_DIM=14`（actions.py L29/30/45）全部一致。

复核命令：`sed -n '488,495p' /tmp/rt_base_task.py`；`grep -n "left_arm_dim\|flat_action" /tmp/rt_eval_xpolicylab.py`。

### 事实 3：Easy=demo_clean / Hard=demo_randomized 差异 —— **成立**

lock `eval_settings.hard` 所列 "random_background+cluttered_table+random_light=true, random_table_height=0.03, clean_background_rate=0.02, crazy_random_light_rate=0.02" 与 `/tmp/rt_rand.yml`（= 上游 demo_randomized.yml）L10–17 **逐字段一致**；easy "全关" 与 demo_clean.yml 一致（全部 false、clean_background_rate=1、heights=0）；`eval_instruction: seen/unseen` 两档亦一致。复核命令：`cat /tmp/rt_clean.yml /tmp/rt_rand.yml` 对照 `configs/robotwin.lock.json` L38–41。

### 事实 4：基线两列抄自报告 §3/§4 —— **成立（零误差）**

程序化比对（正则抽取报告 §3 表 31 行 + §4 表 50 行 + §3 缺口清单 18 项，与 `tasks.py` `_TASK_TABLE` 逐任务对数）：

- π0 easy/hard：50/50 全对；`PI0_PANEL_MEAN` 46.4/16.3 与 §4 均值行一致。
- π0.5 社区列：31/31 全对（含加粗行 handover_mic=23、move_stapler_pad=14、scan_object=18）；None 任务集 = §3 缺口 18 项 + place_dual_shoes。
- §3 表内 π0 对照列与 §4 表交叉一致（无内部矛盾）。
- `PI05_COMMUNITY_MEAN=67.91` 按模型卡声称值收录（31 行实算均值 67.77——差异正是缺的第 32 行，见事实 5），口径标注准确。

复核命令：`tests/test_robotwin_tasks.py` 之外可跑审查用脚本（正则剥 `**` 后抽 `| task | n | e / h |` 与 `| task | e | h |` 全量对数）。

### 事实 5：报告 off-by-one 矛盾的论断 —— **成立，保守处理恰当**

自数结果：§3 分任务表**实列 31 行**（家电 5 + 抓放 12 + 双手 5 + 操控 6 + 工具 3）；缺口清单**恰 18 项**；31+18=49≠50；**place_dual_shoes 在两份清单中均未出现**（六类分账逐一核对无遗漏：pick_place 12 行+4 缺口+place_dual_shoes=17 才凑齐）。作者论断逐字复现。

补充佐证作者的"漏抄一行"假说：67.91×32 − 2101（31 行之和）= **72.1**——隐含的第 32 个值是合理整数成功率，且 §4 有 place_dual_shoes π0 基线（15/0，无"不可评"标记），place_dual_shoes 是最可能的被漏行者。`community_gap_tasks()` 返回 19 项、`PI05_TABLE_TASK_COUNT=31` 与 `PI05_CLAIMED_TASK_COUNT=32` 并列登记、模块级"不得静默抹平"备注（tasks.py L87–92）——处理方式符合"不虚构数字"纪律。**恰当。**

### 事实 6：测试声称 —— **部分不实**（见 P2-7）

---

## 二、分级发现

### P1（结论威胁级）：无

（最接近 P1 的是 P2-1/P2-2/P2-3 的 session 接口簇；因失败模式全部响亮、RT8 未声称环境机保护、且不动摇任何已产出结论，按 LP-A2 审查先例归 P2。）

### P2（声明-证据不符 / 未登记协议偏差，RT8 前应修）

**P2-1 `session.reset()` 的 setup_demo 参数协议与上游不符，且未登记为集成点**

- 代码（`src/gap_repro/robotwin/session.py` L84–86）：
  `self.env.setup_demo(use_seed=True, seed=int(seed), eval_times=int(rollout_index), config_path=self.config_name)`
- 上游事实（三处独立证据）：
  1. `envs/adjust_bottle.py` L9–10：`def setup_demo(self, **kwags): super()._init_task_env_(**kwags)`——setup_demo 是透传；
  2. `_base_task.py` L43 起 `_init_task_env_(self, table_xy_bias, table_height_bias, **kwags)` 实际读取的键是 **`seed`、`now_ep_num`、`domain_randomization`（dict）**、task_name、render_freq、data_type、camera、embodiment 等约 20 个展平键——**没有 `use_seed`、没有 `eval_times`、没有 `config_path`**（全仓库 grep 为零命中；`use_seed` 仅作为 yml 中的死键存在，L60 附近只 `np.random.seed(kwags.get("seed", 0))`）；
  3. 官方评测脚本 `scripts/eval_policy_xpolicylab.py` L630：`task_env.setup_demo(now_ep_num=local_episode_id, seed=seed_value, is_test=True, **args)`——`args` 是 `load_task_args` 把 demo_clean.yml/demo_randomized.yml **整个展平**（含 `domain_randomization` dict）后的结果。
- 后果（真环境上）：`config_path` 被静默吞入 `**kwags` → `domain_randomization` 缺失 → `_init_task_env_` L80 `random_setting.get(...)` 对 None 调 `.get` → **AttributeError 崩溃**；即便补上，`eval_times` 也无人消费（episode 计数真实键是 `now_ep_num`）→ rollout 身份丢失；`data_type` 缺失 → get_obs 不含 rgb/joint_action → 下游 ContractError。即 **RT4 的 reset 路径整体不可用**，且 Easy/Hard 区分无法经此签名实现。
- 纪律违反点：session.py L12–13 docstring 称"已钉定（E1，configs/robotwin.lock.json）：setup_demo → _init_task_env_、get_obs 结构、check_success、get_instruction"——委托映射与三个 API 名确实为真，但 lock 的"已核验源码事实（E1）"清单（lock L10）只列 get_obs/步限表/两个 yml，**不含 setup_demo 参数协议**；该参数协议既非已核验事实、又不在四个集成点（①腕相机 ②夹爪 ③state ④步进入口）之列——是一次**未登记的错误假设**（该组参数形态实为 RoboTwin 1.0 时代 API）。
- 测试放大：`tests/test_robotwin_runner.py` L39 `FakeRoboTwinEnv.setup_demo(self, **kwargs)` 吞一切参数；`test_reset_passes_seed_and_config_name`（L176–182）断言的正是 session 自己发出的 kwargs——**同义反复地把错误签名锁进绿色测试**。
- 修复口径：reset 的参数装配（yml 展平 + domain_randomization dict + now_ep_num）登记为第五个集成点（或直接按上游源码钉死），修正 docstring 与 fake。
- 复核命令：`curl -sL https://raw.githubusercontent.com/RoboTwin-Platform/RoboTwin/main/envs/adjust_bottle.py | sed -n '7,11p'`；`sed -n '43,82p' /tmp/rt_base_task.py`；`grep -n "setup_demo" /tmp/rt_eval_xpolicylab.py`。

**P2-2 步进入口候选名 `exec_action` 上游不存在，且机制叙述不实（已登记集成点④，故 P2 非 P1）**

- session.py L45–47 优先探测 `env.exec_action`；L14–18 docstring 称"Base_Task 的专家管线走 CuRobo 规划，学习策略步进由 eval 适配层提供（XPolicyLab 侧）"。
- 上游事实：学习策略步进入口就是 `_base_task.py` **L1552 `def take_action(self, action, action_type: Literal['qpos','ee']='qpos')`**——定义于 Base_Task 本体，不是 eval 适配层的东西；官方 eval 调 `task_env.take_action(flat_action, action_type=robotwin_action_type)`（eval_policy_xpolicylab.py L716/902）。`exec_action` 在上游全仓库零命中。
- 缓解：该点已登记为集成点④且失败响亮（SessionError），RT8 冒烟必触发。但"候选签名按 RoboTwin 评测适配层常见形态排序"的说法没有上游依据——正确候选今天就能从源码钉死（`take_action(a14, action_type="qpos")`），建议 RT8 前直接改，省一次环境机往返。附带发现：上游 `take_action` 自带 `take_action_cnt == step_lim or eval_success` 短路（L1553）与 TOPP 轨迹规划语义（一次"步进"= 一段关节轨迹），与 session"每步一次 check_success 轮询"的 1:1 步语义有粒度差，RT8 集成时需对齐口径。
- 复核命令：`sed -n '1552,1561p' /tmp/rt_base_task.py`；`grep -n "take_action(flat_action" /tmp/rt_eval_xpolicylab.py`。

**P2-3 腕相机键名"预期 left_wrist_camera/right_wrist_camera"与两处现存证据矛盾（已登记集成点①，故 P2）**

- lock L31/L34 与 actions.py L17–18 写"预期 left_wrist_camera/right_wrist_camera……以环境机 get_obs 实测为准"。
- 现存证据（无需环境机）：① 上游 `envs/camera/camera.py` `get_rgba()` 返回键为 **`left_camera`/`right_camera`**（L135–146 建相机名、L323+ get_rgb 按 get_rgba 键派生）；② 官方适配层 `CAMERA_NAME_MAP = {head_camera→cam_head, left_camera→cam_left_wrist, right_camera→cam_right_wrist}`（eval_policy_xpolicylab.py L36–40）；③ **SidneyXie 模型卡明文**："RoboTwin's simulator uses different keys (head_camera, left_camera, right_camera), so a `--rename_map` is required"。
- 实测探针：`rename_camera("left_camera")` 抛 ContractError（未知键）——当前 rename map 对真环境必然全灭（失败响亮，属集成点①既定路径，只改 rename_map 一行）。问题在"预期值"写错：作者已把 _base_task.py 从同一棵树拉下来核验，camera.py 就在旁边，模型卡更是 weight 身份的直接来源——这个"待钉定"本可今天钉对。权重侧键 `cam_high/cam_left_wrist/cam_right_wrist` 已由模型卡证实无误。
- 复核命令：`curl -sL .../envs/camera/camera.py | grep -n "left_camera\|right_camera"`；模型卡页面对照。

**P2-4 服务器忽略请求中的显式 noise，但 meta 谎报 noise_mode="explicit"**

- `scripts/robotwin_policy_server.py` L56–71 `_predict_batch` 从不读取 `d["noise"]`；L91 `meta["noise_mode"] = "explicit" if "noise" in d.files else "internal"`。
- 与声明的契约冲突：policy.py L10–11 与 server docstring L10–11 都声明"显式 noise 只用于设备对照验证机制"（承 LIBERO 线：lp_a2 服务器 L55–56 确实把 `d["noise"]` 透传给 `policy.infer(noise=...)`）。robotwin 版**该验证机制静默失效**：客户端辛苦注入 noise（transport L167–168 会写入），服务器照样内部采样，且审计 meta 标成 "explicit"——若照此跑设备对照，会得出"显式噪声无效果"的**错误实验结论**。
- 修复口径：要么在 `_predict_batch` 把 noise 接进 LeRobot 采样路径并标注集成点，要么删掉 noise 契约声明并让 meta 如实报 "ignored"。
- 复核命令：`grep -n "noise" scripts/robotwin_policy_server.py`（仅 docstring 与 meta 两处，无消费点）；对照 `grep -n "noise" scripts/lp_a2_policy_server.py`（L49–56 有透传）。

**P2-5 `infer_chunk(replan=k)` 无下界守卫：k≤0 被接受，负数从尾部静默截断**

- 实测探针：`replan=0` → 返回 (0,14) 空 chunk；`replan=-3` → 返回 full[:-3]=(47,14)——**静默丢掉最后 3 步**，与 docstring"不静默缩短执行"（policy.py L100–101）直接矛盾。空 chunk 流入 runner 会以 IndexError 落 INFRA_INCOMPLETE（把编程错误误分类为基础设施故障）。
- 对照：LIBERO 线 REPLAN_STEPS=5 为常量、无此参数化风险；此为 robotwin 版新增 replan 参数时引入且未声明的行为差异。修法：`replan` 非正整数抛 ContractError（一行）。
- 复核命令（可直接跑）：构造 FakeTransport 返 (50,14)，`p.infer_chunk(obs,"t","r",replan=0/-3)` 观察被接受。

**P2-6 测试质量：一个零断言空测试 + 引用不存在的服务器测试文件；服务器脚本零覆盖**

- `tests/test_robotwin_policy.py::test_request_payload_keys_policy_side`（L145–157）**AST 级零 assert**——正文只有注释；且 L154–156 注释称"改由 infer 成功 + 服务器契约测试（test_robotwin_policy_server 侧）保证键集"，而 `tests/` 下**不存在任何 server 测试**（`ls tests | grep robotwin` 只有 4 个文件）。线上键集（cam_high/cam_left_wrist/cam_right_wrist/state/prompt）目前靠审查探针兜底证实（实测 wire keys 与 server L76–79 期望一致，协议自洽，但无回归保护）。
- `scripts/robotwin_policy_server.py` 122 行无任何测试：shape 门 (50,14)、必需键检查、.err 路径、meta 构造均裸奔。lp_a2 服务器同样无直接测试（既有先例），但 robotwin 版新增了 shape 硬门与键检查这类值得测的逻辑。
- 复核命令：`./.venv/bin/python -c "import ast; t=ast.parse(open('tests/test_robotwin_policy.py').read()); print([len([n for n in ast.walk(f) if isinstance(n,ast.Assert)]) for f in t.body if isinstance(f,ast.FunctionDef) and f.name=='test_request_payload_keys_policy_side'])"` → `[0]`。

**P2-7 "全量套件 382 passed"的数字在本机不可复现**

- 实跑：`./.venv/bin/python -m pytest tests/ -q` → **364 passed, 1 skipped, 6 xfailed, 1 xpassed, 18 errors**（11 个 test_environment_contract.py + 7 个 test_kinematics.py，全部 BLOCKED: X5A.urdf 资产缺失——两文件均为 main 既有（92a417b/01c05fb），与本次交付无关，且分支未触碰）。
- 364+18=382：声称值大概来自资产齐备的机器（18 个全过时恰为 382 passed）或误把 error 计入 passed。robotwin 44 项（7+13+12+12）单独实跑 **44 passed in 0.27s**——这项声称为真。修法：commit/计划中的数字改为带环境限定（"资产齐备环境 382；Mac 无资产 364+18 blocked"）。
- 复核命令：`./.venv/bin/python -m pytest tests/test_robotwin_{tasks,actions,policy,runner}.py -q`；`./.venv/bin/python -m pytest tests/ -q`。

### P3（记录瑕疵 / 低风险残留）

**P3-1 集成点编号冲突与隐性第 5 集成点；lock 并未真正收录"四个集成点"**
- actions.py L134 `current_pose` 把**末端四元数约定**（wxyz vs xyzw）标为"集成点④"，而 session.py/计划 RT8 的④是**步进入口名**——同号不同物；四元数钉定项实际是计划四集成点清单之外的**隐性第 5 项**。且 lock 正文只明确含①（rename_evidence）③（state 对齐），②（夹爪）只在 actions.py 注释、④只在 session.py/计划——交付说明"lock 文件……4 个待环境机钉定的集成点"与 lock 实际内容不符。修法：lock 增设统一编号的 integration_points 清单（含四元数约定，共 5 项）。
- 复核命令：`grep -rn "集成点" src/gap_repro/robotwin/ configs/robotwin.lock.json docs/superpowers/plans/2026-09-23-robotwin2-adaptation.md`。

**P3-2 `GRIPPER_JOINT_LIMITS` 死常量；"值域越界记录不拒收"的"记录"不存在**
- actions.py L50 定义后全工程零引用；L48–49 声称"值域越界**记录**不拒收（环境侧 step 自带限位钳）"——既无任何记录路径，"环境侧限位钳"亦为未钉定假设（上游 take_action qpos 走 mplib TOPP 规划，超限输入的真实行为是规划异常而非钳位，未见证据）。修法：要么删常量与断言，要么落到集成点②的实测项。
- 复核命令：`grep -rn "GRIPPER_JOINT_LIMITS" src/ tests/ scripts/`（仅定义处一枚）。

**P3-3 文档口径小疵**
- 计划 L33 标题"RT 队列（**RT0–RT4 已交付**，本分支首批提交）"与下方 RT0–RT7 全部 [x] 不一致；tasks.py L8–9"18 任务缺口"与实际收录 19 项 None（L127–131 已正确解释 19=18+1，但 L9 未同步）；runner record 中 `condition` 与 `setting` 同值重复（runner.py L42/44），下游按 LIBERO condition 语义消费时会误读。
- 复核命令：`sed -n '33p' docs/superpowers/plans/2026-09-23-robotwin2-adaptation.md`；`sed -n '8,9p' src/gap_repro/robotwin/tasks.py`。

**P3-4 rollout 间 seed 变化策略未声明**
- session.reset 语义下，同 (task, setting, seed) 的多次 rollout 在真环境将是**逐位相同的场景**（上游 _init_task_env_ 只吃 seed；官方 eval 每 episode 用不同 seed：`st_seed = 100000*(1+seed)` 递增）。CaseSpec 把 seed 与 rollout_index 并列为身份字段，但 campaign 层（RT9）如何随 rollout_index 派生 seed 未在任何文档声明——不补这条，"每任务 100 rollouts"会退化成同一场景重复 100 次。fingerprint_full 会如实暴露（全部相同），可检测，但应前置声明。
- 复核命令：`grep -n "st_seed\|seed" /tmp/rt_eval_xpolicylab.py | head`。

**P3-5 身份哈希与服务器 sha 口径不可互比（继承自 LIBERO 线的既有怪癖）**
- 客户端 identity 的 `actions_sha256` 对 validate 后的 float64 字节哈希（policy.py L107+L114），服务器 meta 的 `actions_sha` 对 float32 字节哈希（server L89–90）——两者永不相等，审计对账时只能各自自证。libero/policy.py 与 lp_a2 服务器存在**完全相同**的口径差（跨线一致、非新引入），登记备查，不在本分支修。

**P3-6 fingerprint_full 的"Easy 下应逐位一致"是未验证预期**
- session.py L128–132 声明 Easy 同 case 跨 reset 指纹逐位一致——依赖 np.random.seed 重放的确定性 + SAPIEN 渲染/物理沉降（check_stable 内 2000+500 步 scene.step）的位级稳定性，后者在真实引擎上未必成立。声明措辞已按 setting 分层（"属预期/应一致"），算自洽，但 RT8 冒烟时应把"Easy 指纹复现性"列为显式检查项而不是假设。

---

## 三、与 LIBERO 线一致性对照（复制漂移专项）

逐行比对结论：**无复制后语义漂移**。

1. `RobotwinFileNpzTransport` vs `FileNpzTransport`（libero/policy.py L191–253）：残留清理（P2-3 修复的镜像面）、tmp 句柄写 + os.rename 原子纪律、err 先查后删、deadline 轮询、响应消费即删——**逐行同构**；差异仅 timeout 300 vs 120（有注释理由）、键集（声明）、payload 构造通用化 vs 显式（行为等价，已实测 wire 键正确）。
2. `run_student_episode` vs libero/runner.py：五态判定、n_chunks 计数、request_id 构造、`t_end` 收尾、infra 不重发不翻判、`entered_policy_stage` 语义、EpisodeResult 字段——**结构对齐**；差异（无 wait10、env_instruction 覆盖 prompt、reset 签名）均在 docstring 声明。VALID_FAILURE 带 native_score=0.0 满足 results.py L47–51 硬约束（实跑验证）。
3. `validate_student_chunk`：与 libero 版逐字同构（仅 dim 14 vs 7），"不裁剪、无 ±1 假设"由 test_no_value_clipping（3.5 过门）实证。差异：短 chunk 从 libero 的 ValueError 改为 ContractError（ContractError 继承 ValueError，收紧而非漂移，但属未声明的微差异）。
4. server vs lp_a2：轮询-原子写-.err-READY marker 同构；npu import 守卫正确（仅 DEVICE=="npu" 时 `import torch_npu`，先于 `.to(DEVICE)`）；差异三处已声明（LeRobot API、(50,14) 硬门、三相机键）+ 一处未声明（P2-4 noise）。
5. 无 180° 旋转预处理：test_no_libero_rotation_preprocess 显式钉住该差异，防将来"好心"补旋转。好。
6. 导入图干净：robotwin 只 import `agent.contract` + `results` + 自身（grep 证实），无 libero/sim 耦合。

---

## 四、越界检查（维度 F）：通过

`git diff main..robotwin2-adapt --stat`：14 文件全部为新增（+1862），merge-base=15b3e89、分支单提交 50a4a60；`results.py`/`agent/contract.py`/`libero/*`/`sim/*` 零改动。工作区两个未跟踪文件（NPU 指南、w11 脚本）不属于本提交。

---

## 五、确认无误清单

1. 步限 50/50 逐字一致（含 4 个非常规值抽查 + 全量脚本比对）。
2. 14D 布局三源一致（get_obs 源码 / eval 适配层 flat_action / 模型卡）。
3. Easy/Hard DR 参数与官方 yml 逐字段一致；eval_instruction seen/unseen 一致。
4. π0 基线 50 项 + π0.5 社区 31 项零误差；§3/§4 内部无矛盾。
5. off-by-one 论断成立（31+18=49，place_dual_shoes 两边皆无；隐含第 32 值≈72 支持漏抄假说）；保守收录 + 显式登记的处理恰当。
6. robotwin 44 项测试实跑全过（0.27s）。
7. chunk=50 整块 / k=50 恰好 / k>50 拒收 / chunk<k 拒收，四个边界实测符合声明（唯 k≤0 失守，P2-5）。
8. rename map 未知键显式报错不静默丢弃（test + 探针）。
9. 成功粘滞位跨 chunk 生效（test_success_sticky_across_chunks）；is_success 非粘滞仅作辅助记录、不翻判（与 LIBERO 线同判）。
10. 五态核算约束（VALID_FAILURE 必带 native_score）满足；预算耗尽=VALID_FAILURE 且 400/50=8 次推理无第九次半块；非整除 chunk（30 vs 400）正确截到 14 次推理、t=400 终止（探针实测）。
11. transport 原子写/残留清理/错误路径/超时与 lp_a2 逐行同构；wire 键集实测与服务器期望一致。
12. 服务器 (50,14) 硬门、必需键检查、npu import 守卫正确。
13. CaseSpec max_steps 防手抄（构造时从面板推导，未知任务早炸）。
14. 分支纯新增，共享模块零触碰；robotwin 导入图无 libero/sim 耦合。
15. 计划文档的完成口径纪律（"模块已实现 ≠ 单测通过 ≠ 环境机集成"）诚实且与实际交付边界相符。

---

## 六、修复优先级建议（供 RT8 前 dev 分支处理）

1. P2-1/P2-2：按上游源码直接钉死 reset 参数装配（yml 展平 + domain_randomization + now_ep_num）与 `take_action(a14, action_type="qpos")`；同步改 fake 与断言，删"已钉定"中超出 lock 证据的表述，两处登记进 lock 集成清单。
2. P2-3：rename map 改 `left_camera/right_camera`（lock 同步），把①从"待钉定"转为"已钉定（camera.py + 模型卡）"。
3. P2-4：noise 透传或删声明；meta 如实。
4. P2-5：replan 下界一行守卫 + 一个测试。
5. P2-6：给空测试补断言（或删）；补最小服务器契约测试（键检查 + shape 门，fake predict）。
6. P2-7：测试数字声明加环境限定。
7. P3-1：lock 增设统一 integration_points 编号清单（5 项，含四元数约定）。

——审查完毕。本报告为唯一新增文件，未修改任何被审查代码/测试。

---

## 后记：P2 修复闭环（2026-09-23 同日，审查者非修复者）

7 项 P2 全部修复，P3 修 1/2/3/4，P3-5 按审查建议登记不修，P3-6 落为集成点⑦显式检查项：

| 项 | 修复方式 | 验证 |
|---|---|---|
| P2-1 | session.reset 重写为官方签名 `setup_demo(now_ep_num, seed, is_test, **展平配置)`；`build_setup_args` 最小装配（注入 task_name/task_config/eval_mode=True）；两份官方 yml vendored 于 configs/robotwin/（PyYAML 入 requirements-dev）；fake env 改锁真实签名，新增 TestSetupArgsProtocol 5 项（含 Easy DR 全关/Hard DR 全开、配置不变异） | test_reset_passes_official_kwargs 等 |
| P2-2 | 步进入口钉死 `take_action(action, action_type="qpos")`；成功读 `eval_success` 粘滞位；环境侧收束（步限/已成功）显式短路防上游静默 no-op 吞步；TOPP 粒度登记为集成点⑤ | test_step_passes_qpos_action_type_and_14d、test_env_step_limit_short_circuit_no_silent_noop |
| P2-3 | rename map 环境侧键改 head_camera/left_camera/right_camera（三源钉定，lock v2 升为 pinned_facts） | test_env_keys_pinned_three_sources |
| P2-4 | 服务器显式 noise 请求改为响亮拒绝（.err，指向集成点④），meta 恒如实 internal；torch 延迟导入 + predict_fn 可注入 | test_explicit_noise_loudly_rejected 等 |
| P2-5 | infer_chunk replan 下界守卫（正整数，拒 0/负/bool/float/None） | test_replan_lower_bound_guard（5 参数化） |
| P2-6 | 空测试替换为真 wire 键断言（responder 捕获请求 npz 对账 REQUEST_KEYS）；新增 tests/test_robotwin_policy_server.py（8 项：键检查/shape 门/noise 拒绝/meta 如实/原子落位/无 torch 导入纪律） | 新文件 |
| P2-7 | 计划 RT7 补测试运行口径（全量需 MUJOCO_GL/GAP_REPRO_DATA，否则 18 项资产阻塞 error 与 robotwin 无关）；本轮全量 401 passed |
| P3-1 | lock v2 增设 pinned_facts + integration_points 统一编号 7 项（含四元数约定③），跨文件引用统一 |
| P3-2 | 删 GRIPPER_JOINT_LIMITS 死常量；"值域越界记录"措辞改为如实声明（不做记录、超限行为归集成点②实测） |
| P3-3 | 计划标题改"RT0–RT7 已交付"；tasks.py 模块 docstring 19 项口径；runner 记录删 condition 重复键（test 守卫） |
| P3-4 | seed 派生纪律三处前置声明（session/CaseSpec docstring + 计划 RT9）：官方 st_seed=100000*(1+base) 递增口径 |

修复后测试：robotwin 组 63 passed（0.27s）；全量 401 passed, 1 skipped, 6 xfailed, 1 xpassed（带 GAP_REPRO_DATA/MUJOCO_GL 环境变量）。上游事实钉定新增 setup_demo 协议/take_action/eval_success/seed 派生四条进 lock v2 pinned_facts（证据：_base_task.py L43-161/L1552-1731、eval_policy_xpolicylab.py L630/L716/L1033/L305）。

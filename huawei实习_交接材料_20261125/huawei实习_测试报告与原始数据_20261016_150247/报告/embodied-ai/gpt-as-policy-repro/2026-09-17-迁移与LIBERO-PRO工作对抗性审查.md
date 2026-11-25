# 迁移与 LIBERO-PRO 工作对抗性审查（第三轮）

> 审查人：独立对抗性审查（只读核查 + 本报告；除本报告外未写/改本工程任何文件）。
> 审查对象：工程根当前状态——主计划线（W0/W1/W4/W5、runtime.md、两个 lock、inputs/acceptance//assets/sim/tasks 源码、脚本、测试）+ LIBERO-PRO 线（LP0/LP1、gap-sim/gap-repro 容器、pi05 服务、pi05_libero 下载）。
> 对照基准：`upstream/GPT-as-Policy@8f3d362b`、`upstream/RoboDojo@ee67a146`、`upstream/XPolicyLab@432f82b`（本地逐行/逐文件比对）；arXiv 2510.03827v1 HTML（论文表）；openpi `main@HEAD examples/libero/main.py`（WebFetch 原文）；<实验服务器> 服务器只读实测（ssh BatchMode）。
> 独立复算规模：panel/scope 规范化哈希 2 项、169 源文件全量重哈希、checkpoint 聚合配方复算、服务器端 17 个 checkpoint 文件全量重哈希（含 7 个大文件约 12.4GB）、355 个闭包文件服务器端 `sha256sum -c` 全量对账、论文 π0.5/π0 两行 24 个数值逐格比对、openpi 客户端 pin 6 项逐项比对、两容器环境实测。
> 结论密度提示：**W1 哈希证据链经全量（非抽查）独立复算全部成立**；主要问题集中在**文档/commit 声明超出实际完成度**（LP1"环境已建成"与 `import libero.libero` 失败相矛盾）、**工作区遗留半成品致测试红**、以及 **LP 线 pi05_libero 权重实质未做内容校验**。

---

## ① 执行摘要

本轮对 W1 身份证据链做了全量独立复算（169 源文件、17 个 checkpoint 文件含 12.4GB 大文件、355 个闭包资产、panel/scope 规范化配方、d15fb8bd 聚合配方），**全部成立，未发现任何哈希或配方错误**；论文 π0.5 锚定表与 openpi 客户端 pin 亦逐项核实为真，并顺带闭合了 lock 中标记为残留的 π0 行复核。真正的问题在声明层与 LP 线：**runtime.md 与 commit f70f224 宣称"LP1 环境已建成 / full LIBERO-PRO stack green"，而实测 `import libero.libero` 因包 `__init__.py` 的交互式 `input()` 在非交互环境直接 EOFError，LIBERO-PRO 核心包当前不可用**（P1）；**工作区有一份未提交的 tower 判据节点化重构（第二轮审查红1/红2 的半成品修复），已把 3 个测试改红且节点求值器尚不存在**（P2）；**pi05_libero 权重 12.4GB 中约 12.4GB（6 个大对象）因 GCS 复合对象无 md5 而只做了尺寸校验**（P2）。README 自相矛盾、`unstable_envs=[False]` 埋雷等上一轮已点名的问题在 HEAD 仍未回收。

---

## ② 确认无误清单（独立复核后为真）

以下各项均由本轮**独立重算/重跑**确认，不是转抄工程文档：

| # | 声明 | 复核方式与结果 |
|---|---|---|
| G1 | panel/scope 规范化配方及嵌入哈希 | 本机对 `upstream/GPT-as-Policy` clone 重算：去掉 `panel_sha256`/`scope_sha256` 后 `json.dumps(sort_keys=True, separators=(",",":"))` 的 sha256 分别等于嵌入值 `9641fa25…`/`56ee460a…`；panel/scope 文件字节哈希 `8218ab8a…`/`05767f60…` 与 `docs/acceptance/w1-input-identity-evidence.json` 一致 |
| G2 | 169/169 原源码内容哈希 | 本机从 `upstream/RoboDojo@ee67a146` 重算全部 169 个文件 sha256，与 panel60 `native_source_files` 零失配、零重复路径、无路径逃逸 |
| G3 | checkpoint 聚合配方 = 上游原配方 | `upstream/GPT-as-Policy/hybrid_rollout/robodojo/pi05_server/checkpoint.py`：`checkpoint_sha256 = sha256(json.dumps(files, sort_keys=True).encode())`，`files` 为 `{relpath: sha256}`（params/** + assets/**）——与 `inputs.aggregate_identity` 逐字一致；用证据 JSON 的 17 文件元数据重算得 `d15fb8bd1d29cb30b69f01b71c66596cb0293c1a8a94111b343c1580dd3e3e5b`；`tests/test_input_identity.py` 金标值 `3fdf0ef3…` 重算吻合 |
| G4 | 共享盘原始 ckpt 内容级 17/17 | **本轮在 <实验服务器> 亲自重跑**：10 个小文件即时 sha256、7 个大文件（约 12.4GB）后台 sha256sum，17/17 与 lock `per_file.lfs_sha256` 全部一致 → 内容级聚合 `d15fb8bd…` 成立（方法可复现，已证） |
| G5 | 50 布局 / 9 轨迹 25 次引用 | 从 panel60 重算：所选 50 case 共 25 条 `native_support_trajectories` 引用、9 个去重文件；sorting 5 case 各对应 0–4.pkl、kong 5 case 共享 0–3.pkl，与证据 JSON records 完全一致；scope 50 行 `layout_sha256` 与 panel 零失配；5 个 seed 字段语义以 panel `seed_semantics` 为准（作者 E2） |
| G6 | 355 闭包文件服务器端内容校验 | 本轮从 `docs/acceptance/w1-asset-closure.json` 生成校验清单，服务器 `/data_nv0/gpt-as-policy-repro/data/hf_cache` 下 `sha256sum -c` 全量通过（EXIT=0，355/355）——最终态声明为真（但见 P2-8：工程自身日志只留有 343/355 的中间记录） |
| G7 | 闭包解析覆盖 | 对 50 个布局 JSON 做原文资产字符串扫描（`Assets/`、`$Robodojo_ASSETS`），5 处显式引用全部被 `extract_layout_refs` 覆盖；顶层键仅 Rigid/Geometry/Room/Table/Ground/Background/Garment，`type` 值仅 `cluttered`；Ground 为程序化几何无资产，与 `assets.py` 规则一致 |
| G8 | W4 子步调度逐行符合上游 | `src/gap_repro/sim/control.py` 的 `interp_count=floor(n*0.8)=8`、`alpha=(i+1)/(interp_count+1)`=1/9…8/9、后 2 子步保持、夹爪插值即 clip 到 scale、mimic 线性组合、支持臂队列每子步至多 1 项且耗尽即断——与 `upstream/RoboDojo/src/eval_client/eval_env.py:500–557` 逐行一致 |
| G9 | π0.5 锚定表 = 论文原表 | arXiv 2510.03827v1 HTML Tables 2–5 的 Pi0.5 Average 行逐格等于 `configs/libero_pro.lock.json` 的 `pi05_table`（goal 0.97/0.97/0.38/0.97/0.00/0.46 等 24 格全对）；五扰动均值复算 = 10.64/20 = **0.532** ✓；Position 0.08–0.38 ✓；Task goal 0.00 其余 0.01 ✓ |
| G10 | π0 行（lock 标记"待正式复核"的 LP0 残留） | 论文同表 Pi0 行：goal 0.92/0.94/0.00/0.93/0.00/0.39、spatial 0.97/0.95/0.00/0.97/0.00/0.60、10: 0.82/0.79/0.00/0.82/0.00/0.27、object 0.98/0.94/0.00/0.90/0.00/0.29 → 五扰动均值 8.79/20 = **0.4395 ≈ 0.44** ✓、Position 全 0.00 ✓。**LP0 该残留项可闭合** |
| G11 | openpi 客户端 pin 6 项 | openpi `main@HEAD examples/libero/main.py`：`num_steps_wait=10`、`replan_steps=5`、180° 旋转（`[::-1, ::-1]` 切片）、`resize_with_pad(224)`、max_steps spatial 220/object 280/goal 300/10:520/90:400、`WebsocketClientPolicy`——与 lock `openpi_client_behavior_pins` 全部一致 |
| G12 | TASK_MAX_STEPS 六条件同值 | PRO fork README `TASK_MAX_STEPS`（L253 起）：goal 300 / spatial 220 / 10: 520 / object 280，TEMP/LAN/OBJECT/SWAP/TASK/ENV 六变体同值——lock 注释为真 |
| G13 | pi05 服务相机键名链 | 原 XPolicyLab 配置 `pi05_base_aloha_full_sim_arx-x5_seed_0`（config.py L635 起）的 repack 即 `images/{cam_high,cam_left_wrist,cam_right_wrist}`；同事服务 `pi05_service.py` 的 `RepackTransform` 把请求键原样映射到同名模型输入；`norm_stats.json` 仅含 `state/actions`，无相机键歧义；冒烟输入键名确实走到了正确模型输入槽位 |
| G14 | LIBERO-PRO 服务器 clone | `repo/upstream/LIBERO-PRO` HEAD = `eafdb809426b13153aa1e4c42d6601844217dfec`，与 lock 一致；PRO 变体 bddl_files（`libero_10_with_blue_stick` 等）与 init_files 在 repo 内就位 |
| G15 | gap-repro 容器 jax pin | 容器内 `import jax` → `0.5.3 cpu`，与 runtime.md 声明一致；服务 `/health → {"ok": true, "device": "npu"}` 在线 |
| G16 | HEAD 测试绿 | 临时 clone 本机复跑 HEAD：**62 passed + 1 xfailed + 15 BLOCKED（缺资产 fail-closed，非 skip）**，0 failed——"容器绿"的声明对提交版成立 |
| G17 | W1 验收门无漏检路径 | `decide_w1_ok` 反例测试齐备；聚合配方对"任何文件集合差异"自检（少文件/多文件/oid 缺失都会导致 ≠d15fb8bd），`params_bytes` 校验兜底大文件字节漂移；runner 对 `panel_sha256_equal_scope_rows` 的补充判定在 L208–210 存在 |
| G18 | 凭据不入库 | 仓库内无密码/token/主机地址；`.gitignore` 覆盖 upstream/data/venv（但见 P3-2：SSH 别名 `<实验服务器>` 以明文出现，与"别名不入库"的自我声明矛盾） |

---

## ③ P1 发现（会导致错误结论 / 验收失效）

### P1-1 LP1 被 runtime.md 与 commit 宣称"已建成/green"，但 LIBERO-PRO 核心包当前不可导入

**证据：**
- `docs/acceptance/runtime.md:92`："实际断点是一连串小坑，**全部修复**。）**LP1 环境已建成**，stage7 冒烟 `robosuite 1.4.0 / bddl ok`"。
- commit `f70f224`："**LP1 environment built on gap-sim: full LIBERO-PRO stack green on aarch64**"。
- 本轮实测（<实验服务器>，gap-sim 容器，只读命令）：
  ```
  python -c "import libero.libero"
  → EOFError: EOF when reading a line
    File "/workspace/repo/upstream/LIBERO-PRO/libero/libero/__init__.py", line 69, in <module>
    answer = input("Do you want to specify a custom path for the dataset folder? (Y/N): ")
  ```
  即 canonical 源 `libero/libero/__init__.py:69` 的**交互式 `input()` 在非交互环境使导入直接崩溃**；`import libero` 仅解析为无 `__file__` 的命名空间包。
- lock 自己记录了修复所在处而未采用：`configs/libero_pro.lock.json:15` maintained_fork 注明"**含 libero 安装导入修复**"；stage7 冒烟（`scripts/setup_libero_env.sh:99–104`）只 import robosuite/bddl，**从不 import libero**。

**影响：** LIBERO-PRO 线的关键验收记录（runtime.md）与提交说明给出了"环境建成/全绿"的结论，而实际上无法加载任何 PRO 任务、无法创建任何 env。若后续按此记录推进 LP2/LP-A0，会在一个不可用的栈上"验收通过"。这正是"会导致错误结论"的一类问题。（对照而言，v0.2 计划文档本身的 LP1 checkbox 未勾选、lock 状态写"进行中"，是诚实的——问题集中在 runtime.md 与 commit message。）

**建议修复：** ①最小修复：预先以非交互方式生成 `libero_config.yaml`（或对 `__init__.py` 打最小补丁集中入 `patches/`，或评估切到 RLinf fork 并记录"原值→我们的值"）；②stage7 冒烟必须包含 `import libero.libero` + 至少一个 `libero.libero.get_task`/env 创建；③更正 runtime.md 与 commit 说明（在新 commit 中更正，不改写历史），把"已建成"降级为"conda 依赖层建成；libero 包导入阻塞，阻塞点 __init__.py:69 input()"。

### P1-2 工作区遗留半成品重构：3 个测试红、节点求值器不存在，且与"第二轮审查已修复"语境无区分标记

**证据：**
- `git status`：`M src/gap_repro/tasks/remaining_tasks.py`（91 行改动，未提交）+ `M scripts/run_w1_asset_closure.py` + 未跟踪 `npunex/`。
- 工作区复跑：`3 failed, 59 passed, 1 xfailed, 15 errors`；HEAD 临时 clone 复跑：`62 passed, 1 xfailed, 15 errors(BLOCKED)`，0 failed——失败精确来自未提交改动。
- 具体断点：`tests/test_remaining_tasks_contracts.py:79` 调 `rt.eval_checks(rt.tower_base_structure_checks(ev_base))` → `TypeError: takes 0 positional arguments`；`:92` 调 `rt.tower_run_reward(ev)` → `AttributeError: no attribute 'tower_run_reward'`（已改名 `tower_run_reward_nodes`）；`:97` `rt.tower_score_stages(ev)` → TypeError。
- 重构产出 `("call", name, kwargs)` 节点（工作区 `remaining_tasks.py:99,105,130,137,139` 等），但**全工程不存在节点求值器**（唯一求值函数仍是 L91 的扁平 `eval_checks`）——即第二轮审查红2（嵌套判据无求值通路）在工作区处于"改了一半"状态。

**影响：** 任何人在当前工作区跑测试/继续开发都会踩红；半成品修复无 commit 边界，丢失或误提交都会破坏"每验收门单独 commit、记录未通过项"的纪律（主计划 §6.2 末段）。虽然 HEAD 本身绿且第二轮审查已如实登记红1/红2 为开放项（无超额声明），但当前工作区状态使"最近一次完整绿"与"正在进行的修复"无法区分。

**建议修复：** 要么尽快完成节点求值器（实现 `check_once` 的 OR/AND 逐层翻转语义）+ 更新 3 个测试后一次提交；要么先 `git stash`/分支保存并回退工作区，让工程根回到可验证的绿态。`npunex/`（vendored 运动学副本，与 `src/gap_repro/reference/` 疑似重复）应明确去留。

---

## ④ P2 发现（显著风险或缺口）

### P2-1 pi05_libero 学生权重实质未做内容校验（LP-A0 的输入地基缺失）

**证据：** 服务器 `data/pi05_libero_manifest.json`（由 `data/dl_pi05_libero.sh` 生成）：16 个对象中 **6 个大对象的 `md5_b64` 为空**（`0eaaecef…`、`155391c1…`、`475fab3e…`、`6c54da5a…`、`896bf93c…`、`efbb4617…`，合计 12,436,062,495 字节 ≈ 12.4GB，即几乎全部权重字节）。脚本逻辑 `good=(not want) or h.hexdigest()==want`——`want=None` 时**任意内容只要尺寸对即判 OK**。下载日志 `DONE ok=16 fail=0` 的"OK"对这 6 个文件只是尺寸+curl 退出码。
**影响：** GCS 复合对象无 md5Hash 但**必有 crc32c**，脚本未用；若下载损坏，LP-A0 数值验收的失败会被错误归因到"转换/设备差异"，正好污染该面板存在的意义（方法论彩排）。lock 的 LP0 清单"checkpoint 身份哈希"实际未闭合（`gs_reachability` 只测了可达性）。
**建议：** 对 6 个大对象补 crc32c 校验（GCS JSON API `mediaLink`/`crc32c` 字段），或与官方任意已发布摘要交叉对账；在此之前 LP-A0 结果只许标记"输入身份未验证"。

### P2-2 README 自相矛盾（上轮黄8点名，至今未改）

**证据：** `README.md:34`"**未完成**：W1 资产引用闭包"vs `README.md:24`/`docs/acceptance/runtime.md:118–119`"355/355 闭合"；`README.md:53`"权重内容下载/校验仍是 W1 的待完成项"vs `README.md:24–26`"已在服务器共享盘逐字节校验（17/17 OK）"。同一文件内两个时间层的陈述并存且无标注。
**影响：** 新读者/评审无法判断哪个是当前态；与"文档版本与源码版本在 campaign 中关联"的纪律相悖。
**建议：** 更新 README"当前状态"为最新事实（W1 全闭合），历史进展移入带日期的小节。

### P2-3 `unstable_envs=[False]` 埋雷仍在 HEAD（上轮黄1未闭合）

**证据：** `src/gap_repro/sim/environment.py:110` `self.unstable_envs = [False]`。原 session 语义 `0 in unstable_envs` 判 invalid；`0 in [False]` ≡ True（Python `False == 0`）。
**影响：** 一旦 session 层接通，全部 episode 会被静默判 `invalid_for_success_rate`、`native_score=None`。
**建议：** 改为 `[0]`/`[False]` 之外的显式类型约定（如 `[False]`→`[0]` 或布尔列表配套改判定），并补一条"unstable 必须是整数索引列表"的契约测试。

### P2-4 openpi 未 clone、客户端 pin 无 commit 锚——LIBERO-PRO 协议保真的 E1 基础不稳

**证据：** `configs/libero_pro.lock.json:44–45` "source": "**openpi main @ main**（E1）"——无 commit SHA；本工程 `upstream/` 仅 GPT-as-Policy/RoboDojo/XPolicyLab 三仓库，**openpi 与 LIBERO-PRO（canonical 与 RLinf fork）均未在工程根 clone**（后者只在服务器容器路径）。lock 的所有 openpi 行为数值本轮靠 WebFetch `main@HEAD` 验证——但 `main` 是移动目标，本次验证值未来可能失效且不可复现。
**影响：** LP2/LP3 要求"客户端行为与官方 `examples/libero/main.py` 逐项对齐"，对齐对象没有身份锁定；另外相机键名在 openpi **服务端** repack 链（`observation/image` → 策略内部键 → 模型槽位）的映射在本地无源码可查，本轮只验证了客户端观测键名为 `observation/image`/`observation/wrist_image`/`observation/state`/`prompt`。
**建议：** clone openpi 并把 commit SHA 写入 lock；LIBERO-PRO 两源也在工程根留浅 clone + SHA 校验；把"camera 键名服务端映射"列入 LP2 冻结清单的显式核对项。

### P2-5 推理用同事副本（robodojo_pi05_pt safetensors 6.8G）无哈希对账记录

**证据：** `upstream.lock.json` `colleague_adapted_version`：服务实际加载 `/data_nv0/gpt-as-policy-repro/data/pi05_hybrid/weights/robodojo_pi05_pt`（"已拷贝到 /data_nv0 副本"）；本轮实测副本与共享盘原件的 `norm_stats.json` md5 一致，但 **model.safetensors（6.8G）没有任何与原件的哈希对账记录**（原件本身也无独立发布的摘要）；`robodojo_pi05_pt_v2` 与 v1 的差异仍是 open question（lock 如实登记）。
**影响：** 冒烟通过（形状/有限值）不可能发现权重拷贝损坏；W6/W7 数值对齐若失败，"副本是否等于同事原件"将是第一个无法回答的问题。
**建议：** 对副本与共享盘原件做一次 `sha256sum` 对账并落盘记录（一次性，分钟级）；v1/v2 差异向同事确认前，在 lock 中标注"推理身份 = 同事交付 v1，与原 JAX checkpoint 的对应关系未验证"。

### P2-6 W1 证据缺"指令来源 + 控制预算"的机读保存（上轮 Y7 缩水项，仍未闭合）

**证据：** scope_v2 case 字段仅 `task/runtime_task/variant/seeds/layout`，**无指令字段**；`docs/acceptance/w1-input-identity-evidence.json` 与 `upstream.lock.json:125`（`per_case_seeds`）均未保存每 case 指令来源与控制步预算（预算只存在于主计划 §5 的散文表）；runtime.md:156 自认"指令来源"未闭合。
**影响：** W5/W9 需要"学生始终收到原任务指令"的可审计来源；当前指令文本与其出处在工程内不可追溯。
**建议：** 在 W5 前把 50 case 的指令文本/来源文件/行号与 `step_lim` 一起冻结进 `configs/campaigns/` 或 w1 证据附录。

### P2-7 runtime.md"外部缺失"清单过时且引用不存在的文件

**证据：** `docs/acceptance/runtime.md:131–138` "外部缺失"仍列"目标机 CPU 架构、卡数/容量、CANN 版本、JAX CPU 跑通原 PRNG"——但同文件 L47–48 已记录 aarch64/192 核/8×910B3 64GB/CANN 8.5.2，L75–79 已记录 jax==0.5.3 装入 gap-repro 并给出 PRNG 锚点（本轮 G15 复核为真）；L134 引用 `configs/runtime.lock.json`——`configs/` 下**不存在**该文件（仅 campaigns/ 空、两份 lock）。
**影响：** W0 的真实剩余缺口（两路线模型接入核验、runtime.lock 冻结）被已满足项淹没；"通过条件对照"（L140–144）的 blocked 范围因此失真。
**建议：** 重写外部缺失清单为当前真实剩余项；要么创建 `configs/runtime.lock.json` 骨架（服务器已测字段先填），要么改引用为 upstream.lock 的 runtime 节。

### P2-8 关键验收动作缺自身留痕（依赖审查者复跑才成立）

**证据：** ①`data/asset_closure.log`（服务器）只有一次 `DOWNLOAD_DONE ok=343/355` 与 12 条 FAILED，**无 355/355 最终运行记录**（本轮 G6 已替其补验）；②共享盘 ckpt `sha256sum -c` 原始输出无落盘（证据只有 lock/evidence 里的结论性 JSON）；③`w1-input-identity-evidence.json` 内部时间层混乱：`checkpoint_metadata.note`（L419）仍写"17 文件内容尚未下载"，而同文件末尾 `checkpoint_content_verification`（L421–429）已完成——无"此 note 已被下节取代"的标注。
**影响：** 当前所有"已完成"结论恰好为真（本轮全量复算背书），但证据链不能自证；未来重放审计需要重做 12GB 级哈希。
**建议：** 服务器端验收命令一律 `tee` 落盘（closure 最终跑一次 `--download` 空转生成全 SKIP 的 355/355 记录即可）；evidence JSON 的陈旧 note 加 superseded 标注。

---

## ⑤ P3 建议

1. **`control.py:10–12` 头注仍写"渐进夹爪作用于 25Hz 目标（MetaControl 层）"**——第二轮黄8 点名的陈旧声明（实现已在 250Hz 子步层，`environment.py:226–231`），HEAD 未改；同批残留还有 `w2-robot-mapping.json` 的四元数顺序记录矛盾。
2. **SSH 别名 `<实验服务器>` 明文出现在 README.md:27,42、runtime.md:53、w1-input-identity-evidence.json:426**——与 runtime.md:44"别名与接入细节按计划不入库"的自我声明矛盾。别名本身低敏，但要么改文案（"已用受控别名，别名值不入公开物"），要么统一脱敏。
3. **容器暴露面**：gap-repro `--privileged --network host`，HTTP 服务绑定在宿主 8642 端口，主机可达者即可调用推理；我们的 pi05 副本以 **rw** 挂载进容器（`docker inspect`：`/data_nv0/.../pi05_hybrid -> /data/pi05_hybrid`，mode 空=rw），只读纪律纯靠自觉。建议副本挂载改 `:ro`、服务加 token 或绑 localhost；同宿主还有第三方容器 `rlinf-study`（Up 7 days），建议在运维记录中登记卡与容器归属台账。
4. **`download_checkpoint.sh:67–78` 内层 `while True` 对网络错误无上限重试**（外层 5 轮只限哈希不符）；虽被"共享盘校验替代"而闲置，若复用应加轮次上限。
5. **`scripts/run_w1_verification.py:36–41` 依赖另一工程目录的绝对路径**（`~/Documents/huawei实习/embodied-ai/weeks/...`）作为 prior——Mac/人换则断；建议把 prior 的最小索引 vendored 进 `docs/acceptance/`（只留 path→sha 映射）。另外 evidence 的 `date` 字段硬编码。
6. **闭包排除了 311 个同目录 sibling 文件**（used object dirs 中的 `metadata.json`/`description.json`）——判定为运行时不读，合理，但排除理由未写进 `w1-asset-closure.json`；建议加一行注记防未来复审重查。
7. **`setup_libero_env.sh` 阶段化输出难判终态**：日志里出现过一次 `STAGE5_REQ_FAIL`（随后重跑成功），append-only 日志无"最终阶段状态汇总"；建议脚本结尾输出各阶段最终状态一行表。
8. **LP0 残留可闭合**：π0 行（PRO 0.44、Position 全 0）本轮已对照论文证实（G10），建议更新 `libero_pro.lock.json` `pi0_row.status`；同时注意论文 landing 页 Comments 标注 v2 "0 tables"（表格转图），lock 应继续锚定 **v1** 并存档取数快照。
9. **测试计数口径**：docs 中 43/43、60/60 为历史快照，当前 HEAD 63 可执行（62 pass + 1 xfail）+15 BLOCKED；建议 runtime.md 加一行"当前全套"计数并注明 BLOCKED=缺资产 fail-closed。
10. **`npunex/` 与 `src/gap_repro/reference/` 内容疑似重复**（vendored kinematics 两份），统一到一处并入库或删除。

---

## ⑥ 复核方法附录（本轮实际执行的只读命令/操作）

**本机（工程根，全部只读）：**
```bash
git log --oneline            # 31 个提交；git status → M scripts/run_w1_asset_closure.py、?? npunex/
git rev-list --count HEAD    # 31
git diff scripts/run_w1_asset_closure.py; git diff src/gap_repro/tasks/remaining_tasks.py
python3（脚本A）: 重算 panel/scope 规范化变体哈希 + 文件字节哈希（upstream/GPT-as-Policy clone）
python3（脚本B）: panel eval_seed/seed_semantics、case 字段、25 轨迹引用映射、scope/panel layout 交叉、task/variant 计数
python3（脚本C）: 169 文件全量 sha256 重算 vs panel；RoboDojo ls-files 206 vs 169 覆盖差集
python3（脚本D）: 聚合配方复算（默认与 compact 分隔符两种）、金标值复算、lock 与 evidence per_file 一致性
python3（脚本E）: 50 布局 JSON 原文资产字符串正则扫描 vs extract_layout_refs 覆盖；顶层键/inst.type 统计
python3（脚本F）: w1-asset-closure.json 对 hf_assets_tree.json 的 object 目录 sibling 文件差集（311 个）
.venv/bin/python -m pytest tests/ -q   # 工作区：3 failed, 59 passed, 1 xfailed, 15 errors
rm -rf /tmp/adv_review_head && git clone <工程> /tmp/adv_review_head && pytest  # HEAD：62 passed, 1 xfailed, 15 BLOCKED
sed -n '495,560p' upstream/RoboDojo/src/eval_client/eval_env.py   # 与 control.py 逐行比对
sed -n '625,665p' upstream/XPolicyLab/policy/Pi_05/openpi/src/openpi/training/config.py  # 原 repack 键名
grep -n previous_load_sha256 upstream/GPT-as-Policy/hybrid_rollout/robodojo/SOURCE.json  # d15fb8bd 出处
python3 -c … 生成 /tmp/closure_checksums.txt（355 行，工程外临时文件）
```

**Web（公开源）：**
```text
WebFetch https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/examples/libero/main.py
         → wait10/replan5/[::-1,::-1]/resize224/max_steps{220,280,300,520,400}/observation/image 等键/WebsocketClientPolicy
WebFetch https://arxiv.org/html/2510.03827v1 → Tables 2–5 的 Pi0/Pi0.5 Average 行逐格数值
```

**<实验服务器> 服务器（ssh -o BatchMode=yes，全部只读；未执行任何修改性命令）：**
```bash
docker ps --format …                                   # gap-sim / gap-repro / rlinf-study
CK=…/59999; stat -c %s + sha256sum（10 小文件）        # 全部匹配
nohup sha256sum（7 个大文件 ~12.4GB）> /tmp/adv_review_bigfiles_sha256.txt  # 17/17 匹配，DONE
cd /data_nv0/gpt-as-policy-repro/data/hf_cache && sha256sum -c - --quiet < /tmp/closure_checksums.txt  # EXIT=0
git -C repo/upstream/LIBERO-PRO rev-parse HEAD          # eafdb809…
cat data/pi05_libero_manifest.json; tail data/pi05_libero_download.log; cat data/dl_pi05_libero.sh
grep -n "cam_high|base_0_rgb|…" data/pi05_hybrid/scripts/pi05_service.py; sed -n 30,90p …
python3 读 norm_stats.json 键（state/actions）；md5sum 副本 vs 共享盘 norm_stats.json（一致）
grep -E "STAGE|build (start|end)" data/libero_env_build.log; grep -c FAILED data/asset_closure.log; grep DOWNLOAD_DONE …
docker inspect gap-repro / gap-sim（Mounts/Privileged/NetworkMode）
docker exec gap-sim … python -c "import libero.libero"   # 只读导入探测 → EOFError __init__.py:69
docker exec gap-repro … python -c "import jax; …"        # 0.5.3 cpu
sed -n 245,275p repo/upstream/LIBERO-PRO/README.md       # TASK_MAX_STEPS
curl -s http://localhost:8642/health                     # {"ok": true, "device": "npu"}
```

**报告文件：** 本文件（工程内唯一写入）。

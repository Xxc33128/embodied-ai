# L0/L1 基线与 case 池证据（LIBERO 单线 v2.0）

> 2026-09-19 · 依据 docs/superpowers/plans/2026-09-19-repro-next-execution.md（L0、L1）。
> 本轮只读探查 + 资产部署 + 清单生成；未启动正式采集，未修改评测代码语义。

## 1. L0 运行基线（全部实测）

| 项 | 值 | 与 lock 对账 |
|---|---|---|
| gap-sim 栈 | python 3.8.13 / mujoco 3.2.3 / robosuite 1.4.0 / torch 1.11.0 | = env_pins ✓ |
| PRO 源 commit（服务器 clone） | `eafdb809426b13153aa1e4c42d6601844217dfec` | = canonical_source.locked_commit ✓ |
| LIBERO 权重 pi05_libero_pt/model.safetensors md5 | `032ad466…` | = lp2_conversion.output ✓ |
| 基线 benchmark 字典 | 4 基线套件 × 10 任务可实例化 | ✓ |
| π₀.₅ robodojo 服务（8642） | health ok —— **不是** LIBERO 权重服务 | LIBERO 权重入口 = `scripts/lp_a2_policy_server.py`（torch cpu/npu 加载 pi05_libero_pt，LP-A2 已验证） |

## 2. 编排拓扑决策（L0）

- **gap-sim**：环境 + 闭环客户端（LiberoSession，osmesa）。
- **gap-repro**：学生策略服务（pi05_libero_pt，torch cpu / npu 卡 1–7）。
- **Mac**：GPT agent 编排（L4 工具循环、agent 契约层已在 `src/gap_repro/agent/`）。
- 交换：共享卷 npz（LP-A2 架构沿用）。部署身份用 `scripts/check_runtime.py --deploy-manifest` 对账。
- 服务器 repo 当前缺 Mac 侧新增文件（agent/ 等）属预期——agent 在 Mac 运行；L3 runner 落地时按角色同步清单。

## 3. 条件 → PRO 资产映射（源码级裁决，钉 eafdb809）

| 论文条件 | benchmark 套件 | 载体 | 扰动器（perturbation.py） |
|---|---|---|---|
| Ori | `{suite}` | 基线 BDDL/init | — |
| Obj | `{suite}_object` | BDDL+init（HF） | ObjectReplacePerturbator |
| **Pos** | `{suite}_swap` | BDDL 的 `(On obj region)` 互换（HF） | SwapPerturbator（纯位置互换，指令/目标不变） |
| Sem | `{suite}_lan` | BDDL `:language` 改写（HF） | LanguagePerturbator |
| Task | `{suite}_task` | BDDL goal 谓词+对象集（HF） | TaskPerturbator |
| Env | `{suite}_env` | **需 EnvironmentReplacePerturbator 生成**（HF 无静态文件） | EnvironmentReplacePerturbator |

- `_temp` benchmark = init 生成的临时目录（EvalEnvCreator `_temp` 输出），**非论文条件**。
- `evaluation_config.yaml` 的 perturbation_mapping（env/swap/object/lan/task）为官方开关名。

## 4. 协议裁决（E1，影响 L2/L3 实现的正确性）

1. **评测指令 = `benchmark.get_task(i).language`**（evaluate.py:186，官方管线逐字）。benchmark 属性来自
   `grab_language_from_filename`（文件名转写）——不是 BDDL `:language`。
   实证自洽：论文 Sem 0.97≈Ori（模型收到原句）、Task 0.01（模型按原句做原任务，新谓词不满足）。
2. BDDL `:language` 段仅作诊断记录（Sem 40/40 确实改写；Ori 基线 BDDL 内嵌句与文件名转写句有措辞差），
   **不进策略输入**——若改用 BDDL 句即偏离论文协议，须显式登记为协议变体。
3. **初态 = `torch.load(init_states/{suite_variant}/{task}.pruned_init)`**（benchmark/__init__.py:267-274）；
   HF `.pruned_init` 即官方格式。实测 6 个变体可加载，形状 (50, n)——**每任务 50 个初态**，
   与论文"每任务 50 集"对应；每集 i 用 init_states[i]（LP-A1 配对口径沿用）。
4. 条件生效载体 = BDDL（环境/谓词/位置）+ 对应 init 文件；指令不随条件变（Pos/Task/Env/Obj 皆然）。

## 5. case 池覆盖（L1 manifest：docs/acceptance/l1-case-inventory.json）

- **240 case**（24 benchmark × 10 任务）；**200 完整**（Ori/Obj/Pos/Sem/Task × 4 套件）；
  **80 缺口全部为 Env**（40 bddl + 40 init 需生成管线）。
- Sem 条件：40/40 的 BDDL `:language` 相对 benchmark 属性确有改写（如 "open the cabinet's
  middle drawer"）。Obj/Pos/Task 的 task0 BDDL 与基线字节级 differs（条件生效实证）。
- 资产身份：HF `zhouxueyang/LIBERO-Pro` SHA256SUMS 仅覆盖 353/676（作者清单不完整，如实记录）；
  可对账 73 项全过、0 失配；其余 320 项以本次下载快照哈希为部署锚（`data/libero_pro_assets/`）。

## 6. 部署记录（可重现）

1. 下载：`data/libero_pro_assets/{bddl_files,init_files,metadata,SHA256SUMS.txt}`（394 文件）。
2. 接入：`scripts/deploy_libero_pro_assets.sh`（容器内执行）——把 20 个变体目录 symlink 进
   钉定 clone 的 `bddl_files/`、`init_files/`（不覆盖既有文件；容器可写层，重建后重跑）。
3. 清单：`scripts/l1_case_inventory.py` → `/workspace/data/l1_case_inventory.json`
   （vendored：`docs/acceptance/l1-case-inventory.json`）。

## 6b. 论文交叉验证裁决（arXiv 2510.03827v1 全文，2026-09-19）

1. 论文 Table 2–5 的位置列名是 **Spa**（spatial，初始位置扰动，§4.1 k=I）——即本档案
   与 lock 中的 "Pos"，`_swap`（位置互换）映射经论文 §4.1 定义复核成立。
2. **论文实验只有 VLA 模型直评**（OpenVLA/π₀/π₀.₅，§5.1）；GPT Direct/Hybrid 是主计划
   方法移植（面板计划 §0/§4），LIBERO 侧契约锚 = GPT-as-Policy 论文 + openpi 客户端，
   论文 2510.03827 内无 GPT 实验——L2/L4 设计引用时不得误标为 LIBERO-PRO 论文内容。
3. **Sem 列 = Ori 等价重跑（论文管线缺陷实锤）**：evaluate.py 指令取 benchmark 属性
   （= 文件名转写原句），_lan 的 BDDL 仅改 :language 段且文件名与基线相同 → 指令、
   环境、谓词与基线全同；init 例外——审查 P1-2 实测 40/40 sha 不同
   （HF 对每变体独立采样初态），故 Sem 是"同指令同环境、不同初态的重跑"
   而非逐位重跑；正式面板按 init[i] 做 Sem-Ori 配对时配的是不同初始状态，
   配对协变量口径须在 L5 冻结如实写明。论文 Sem 0.97/0.97/0.93/0.96 ≈ Ori
   0.97/0.98/0.93/0.98 与此自洽。复现口径（冻结候选）：按论文管线执行（Sem 数据=Ori
   管线重跑），报告单列此发现；若改为把 :language 改写句送入模型则偏离论文协议、
   数字不可比——**该二选一为 L5 冻结前的用户决策点**。
4. Env 条件：论文 §4.3 称"随机替换五类环境"；钉定 eafdb809 实现硬编码
   `new_env="living_room_table"`（随机 choice 被注释）。钉定语义优先（确定性替换），
   偏差已记录；生成管线据此实现。
5. 每任务 50 集（§5.1 "Consistent with the original LIBERO protocol"）与 init (50,n) 吻合。

## 6c. L1 收口结果（2026-09-20，流水线证据）

1. **Env 生成**：`ENV_GEN_DONE files=80 missing=0`（40 BDDL + 40 init，
   PRO@eafdb809 官方函数，seed 42）；manifest =
   `docs/acceptance/l1-env-gen-manifest.json`（内容为 v2 语义含
   vacuous_cases=5——libero_10 基线已在 living_room_table 的 5 任务；
   文件内 schema 字符串仍为 v1，系 v1 文件被 v2 脚本原地升级未改串，
   审查 P3-1 登记）。注意 vacuous 仅指 BDDL 等价，init 为新采样
   （审查 P2-1）。inventory 重估 **240/240 complete, gaps=0**
   （`docs/acceptance/l1-case-inventory.json` 已更新）。
   另："三层指纹"中 fixture digest 对 libero_object 全部任务与
   libero_10 的 5 个 LIVING_ROOM 任务退化为空摘要（无 fixtures_dict，
   90/240 实为两层；session docstring 已载，审查 P3-3）。
2. **逐 case 冒烟 240/240 OK / 0 FAIL**（`l1_smoke_results.json`）：
   reset+三层指纹+原生判据+init 形状全部通过；init_count=50 全对齐论文
   协议。渲染响应检查本轮未生效（审查 P1-1：脚本读错键，240 case 记
   NaN）——脚本已修正（读 mean_abs），抽样补测随本 commit 入档；后续
   全量重跑自动生效。
3. **指令双源分布（新证据，精化 §6b-3 裁决）**：
   - benchmark 属性（=评测馈送）与 BDDL :language 在 **Ori 条件就有
     30/40 case 不同**（如 goal#0：馈送 "open the middle drawer of the
     cabinet" vs BDDL "Open the middle layer of the drawer"）；
     两源一致的是 **libero_10**（10/10）；libero_object 0/10（"Pick the X"
     vs 转写 "pick up the X"）；libero_goal 另有 5 条仅大小写差异
     （differ 标志大小写敏感）。
   - 该分布对 6 条件恒定（Sem/Task 40/40 differ 只说明 :language 被
     扰动器改写，与评测馈送无关）→ **强化 Sem=Ori 裁决**：馈送指令
     在 Ori 与 Sem 完全相同（文件名转写），论文 Sem≈Ori 数字机制成立。
   - 运行口径冻结候选：指令一律取 benchmark 属性（libero/lifelong/evaluate.py:186 同源）；
     BDDL :language 仅作档案对照。
4. 运维修复记录：torch.load 版本兼容（容器 torch 无 weights_only）、
   env 变量须先于 libero_session 导入（T0 残余盲区，libero 包会交互
   提示）、zsh 无词切分（部署脚本 SSH 函数化）、远端管道 pipefail。

## 7. 开放项（L1 收尾清单）

1. **Env 条件生成管线**：离线跑 EnvironmentReplacePerturbator（ood_environment.yaml）生成
   4 套件的 `{suite}_env` BDDL+init，快照哈希入 manifest——之后 240/240。
2. `_object` 等变体的 BDDL 与基线逐任务 diff 细表（本轮只验 task0；manifest 已存全部 sha256 可离线补）。
3. 每条 case 的 reset/观测/dummy-step 冒烟（L1 最后一项）→ L2 学生闭环。

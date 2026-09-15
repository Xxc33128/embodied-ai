# qwen3.8-flash 标准批次执行状态（第 4 个模型批次）

用户指示（2026-09-15 15:45）：按 bowl 标准测试流程跑 20 次；每 trial 一个全新子 agent 连续完成整个 trial；结果只放本目录（`logs-qwen3.8flash-subagent20-20260915-1545/`），不触碰既有 mock / GPT / GLM / DeepSeek 批次。

- 决策模型：qwen3.8-flash（pi 子 agent `bowl-policy`，`qwen-token-plan-cn/qwen3.8-flash`，provider 快照 id 不暴露 → 记 unknown 级）
- 接入方式：persistent-subagent（每 trial 一个无历史子 agent，保留本 trial 记忆；wire 只带最近 2 组图像，但已看过的图无法遗忘 — 与严格 API 等价的差异已披露，与 GPT/GLM/DeepSeek 批次同组口径）
- 子 agent 工具面：`read`（看请求引用的 PNG）+ `bash`（只跑 `subagent_mailbox.py`）；definition 明确禁止读项目文件/源码/私有日志/其他 trial，禁止生成子 agent
- 协议：`robosuite-bowl-paired20-v1`，seed 2000–2019，20 调用 / 900 控制步 / max_speed_frac 0.25 / effort medium；提示词与物理未改，runner 每次加载校验 SHA256
- runner：`run_subagent_trials_qwen.py`（`edff5d10…`），与 `run_subagent_trials.py`（`26a27e92…`）**只差决策模型身份标签**；开跑前 `--smoke` 传输冒烟通过
- 场景 bank：复制自 `logs-persistent-subagent20-20260915-113640/initializations`，`diff -r` 逐文件一致，聚合 SHA256 `7ac039f8…`（算法见 experiment.json）；每 trial 在模型首次调用前做完整快照比对，不一致直接中止
- 公开请求（模型可见）：`/tmp/bowl-qwen38flash-20260915-1545/trial-XX`；私有记录：本目录 `trial-XX`
- 记录：每控制步 20fps 三视图 `video.mp4`（front|side|wrist，含初始帧与终帧）；`initial-*.png` / `final-*.png` 三路；公开请求内嵌即模型所见原图，收口时用 `audit_standard_trials.py` 无损导出 `frames/call-NN-<camera>.png` + `index.json`（SHA256 可追溯到 wire 请求）；`physics.jsonl` 私有轨迹；`outgoing/incoming-NN.json` 逐调用请求与回复；harness 原始 log
- 并发：**动态滚动池**（用户 2026-09-15 15:55 要求，取代原「每波 5 个」）：5 个槽位，某 trial 一结束同一槽位立即起下一个 seed；起库顺序仍是 seed 2000→2019，完成顺序取决于运行时长
  - 每 trial 两个 agent 调用：① 基础设施 launcher（`.pi/agents/bowl-launcher.md`，只给 `bash`，只跑 `launch_trial.sh <index>`，不做任何机器人决策）→ ② 持久策略 worker（`.pi/agents/bowl-policy.md`）做完整个 trial
  - agent 重试关闭（0）：策略 worker 挂掉则让 runner 自行 30 分钟超时写 error.json，该 trial 记基础设施失败，之后从同一冻结初始状态整 trial 重跑（新 attempt，保留原 attempt）
  - orchestrator：pi workflow run `bowl-qwen-dynamic20-mu2dyo2m-k79cko`
- 并发瓶颈实测（2026-09-15 16:25，用户质疑「CPU 是否是瓶颈」后测量；修正此前「DeepSeek 批次让出 CPU 会更快」的口头推断——归因错误）
  - 墙钟去向：GLM 批次 20 个 trial 合计等模型 43289 s / 墙钟 43570 s = **99.4%**；DeepSeek 8 个有结果 trial = **99.3%**。runner 的 `call-NN-ended.json.wait_seconds` 即等决策时间
  - 物理开销：单控制步 20–24 ms → 900 步仅 18–22 s / trial（≈0.7%）
  - 机器：`CPU usage 10.6% user / 6.6% sys / 82.8% idle`，load 2.9（10 核）；本批 5 个 runner 各 0.0% CPU；DeepSeek 12 个空转 runner 合计 0.0% CPU / 0.5 GB（挂在 0.1 s 轮询上）→ 不抢 CPU
  - 单决策延迟（本批 5 个 trial 当前样本）：avg 80–156 s，median 74–122 s，max 375 s；runner 每决策硬超时 1800 s（≈5× 余量）
  - 结论：周期 = 单决策往返延迟 × 决策数  槽位数；能提远的杆是**加槽位**，而槽位天花板是内存（`vm.swapusage used 20.4G/21.5G`、compressor 7.5 GB、每 sim 0.08–0.14 GB），不是 CPU。用户 16:26 决定：**保持 5 槽不加并行**（避开内存卡顶风险），本行仅作为事实记录与后续批次依据
- 阶段评分：批次跑完后 `prepare_blind_review.py` 出匿名复核包，人工 stage 0–4 复核写 `review.json`，不改原始日志

- 事故与处置（16:03，操作者失误，已记 `dispatch.jsonl` 5 条 `attempt_aborted`）：在已起动态池（run `mu2dyo2m-k79cko`，索引 0..19）之后，因编辑旧参数又误启了一个重叠动态池（run `mu2e0rlz-pm6hqm`，索引 5..19），30 秒内发现并停止；但它的 5 个 runner 已为 trial-06..10 建了 attempt 目录（**0 次模型决策**、无 worker 接入）。已 kill 这些孤儿 runner，并把目录/渲染提示/runner stdout+stderr 保留到 `attempts-aborted-duplicate-launch-1603/`，让权威运行轮到 seed 2005..2009 时能干净重建（否则会撞 FileExistsError → launcher 报 RUNNER DIED EARLY → 静默丢 5 个 seed）
- 现有权威运行：`bowl-qwen-dynamic20-mu2dyo2m-k79cko`（单池 5 槽覆盖全部 20 个 seed；trial-01..05 的 runner 由它自己重起，初始化比对再次通过才接入 worker）

## 配额事件：429 insufficient_quota（16:37–16:50）

- 现象：provider `qwen-token-plan-cn`（阿里云百炼 token plan）返回 `429 insufficient_quota: Allocated quota exceeded (token-limit)`，workflow 自动暂停（暂停前累计 38.6M tokens，含缓存读）。
- 不是 pi 并发上限，也不是 CPU/仿真瓶颈（当时 82.8% idle，5 个 runner 各 0.0% CPU）；规范请求 mean 375 KB（内嵌 6 张 224² PNG），5 个持久 worker 各自每轮重发不断增长的上下文（一个机器人决策≈3-4 次模型往返）。
- 用户决定：**不中止批次**，先把在跑的 trial 跑完。
- auto-resume 后的实际后果（已追加 `dispatch.jsonl` 两条事件：事故 + 更正）：
  - 原 worker 被暂停掉封的 trial-01/02/04/06/07 由**新的同模型 worker 接入续跑**（新 worker 只能看到当前规范请求：全文本历史 + 最近 2 组图像，丢了它自己早期的推理记忆）→ 这 5 行归为 `worker-restart-after-quota-pause` 接入子组，与 `persistent-worker` 行分开列，不允入同一张对比表（SOP §3.4）。
  - trial-06/07：resume 重发的 `launch` 在 runner 侧确实报 `FileExistsError`（目录已存在），但 `launch_trial.sh` 轮询的是**共用的 public status.json**，那里是还活着的原 runner 在写，于是脚本误报 READY 并接上了新 worker → **未丢 seed**，但两者进入上面的子组。
  - 新发现的坑（下次批次前必修）：`launch_trial.sh` 无法区分「自己的 runner 就绪」与「别人持有同一个 public 邮箱」；resume / 重复启动会静默给同一个 trial 接第二个 worker。后续应把 public 路径改成按 attempt 唯一（带 pid/attempt token），并在接 worker 前校验 runner 所有权。本次未改脚本，避开影响正在跑的批次。
  - `dispatch.jsonl` 里 trial-06/07 多出几条 `event=started` 对应已死的 FileExistsError runner（无对应进程、无日志），按 `runner_pid` 可识别。
- 干净行（暂停前后完整跑完且 worker 未重启）：trial-03（19 调用, give_up）、trial-05（20 调用, give_up）、trial-08、trial-09 及其后所有 seed。
- 待收口时再定：这 5 行是否用同模型重跑补齐 20 行 persistent-worker 样本（重跑需先把现 attempt 目录移入 `attempts-aborted-*` 保留）。

## R2：降到 2 槽硬扛额度（17:11 第二次 429 后，用户 17:28 决定）

- 口径不变：仍是**每 trial 一个持久无历史子 agent 连续做完整个 trial**；提示词 / 预算 / 物理 / 冻结场景库 / runner 全部不改，只改并发窗口与邮箱路径卫生
- run A（5 槽）`bowl-qwen-dynamic20-mu2dyo2m-k79cko` 17:31 stop（累计 70.76M tokens，两次 429）；run B（2 槽）`bowl-qwen-r2-pool2-mu2h4xrf-2qhfe0` 17:33 起，跑 13 个 seed：trial-06 与 trial-09..20（2005、2008..2019；其中 2005 与 2008/2009/2010 是被暂停污染后重跑）
- 新 launcher `launch_trial_r2.sh`（`3097121a…`）只动基础设施：① attempt 目录已存在则 `ATTEMPT_DIR_EXISTS` 直接拒起（消除 FileExistsError）；② 公开邮箱换根 `/tmp/bowl-qwen38flash-r2-20260915-1730/<trial>`（避开上一窗口残留 status.json 让 launcher 误报 READY → 给同一 trial 接第二个 worker，17:11 就踩过）。17:33 跑前预检：`zsh -n` 、渲染路径相等、`--smoke` 全通过
- 清理：17:11 暂停后残留的 6 个无主 runner（trial-06/09/10/11 及其后自动补位的 12/13）全部 kill，连同 17:32 那次抢跑失败的 trial-09（0 次决策）一并保留在 `attempts-aborted-quota429-r2-1720/`；`dispatch.jsonl` 已追 `r2_transition`
- 当前有效 7 行（seed 2000/2001/2002/2003/2004/2006/2007）：单 worker 干净行 = trial-03、trial-05、trial-08；**worker-restart 子组** = trial-01、trial-02、trial-04、trial-07（暂停后换了新 worker 接续，丢了自身早期推理记忆，主表分开列）
- 亮点（待盲评）：**trial-04 `success_at_end = 1.0`、harness termination=success** — 方块抬到 z=1.003 m、移到碗口上方、落定 0.815 m；但它的关键段是接续 worker 跑的，归 worker-restart 子组

## 模式 A 收口快照（19:26，12 行有效 attempt，seed 2000–2011）

暂停时刻（本地）：#1 16:37、#2 17:11、#3 18:35–18:36；判定“是否换过 worker”的根据 = 该 trial 的 `call-NN-started.json` 时间跳越是否跨过暂停时刻（+ workflow journal 的 agent 状态）。

| trial | seed | 调用 | env_success | 终止 | 接入子组 |
|---|---|---|---|---|---|
| 03 | 2002 | 19 | 0 | give_up | 单 worker 干净 |
| 05 | 2004 | 20 | 0 | give_up | 单 worker 干净 |
| 06 | 2005 | 20 | 0 | give_up | 单 worker 干净（R2 重跑） |
| 08 | 2007 | 18 | 0 | give_up | 单 worker 干净 |
| 09 | 2008 | 19 | 0 | max_steps | 单 worker 干净（R2 重跑） |
| 01 | 2000 | 20 | 0 | give_up | worker-restart @call17 |
| 02 | 2001 | 20 | 0 | give_up | worker-restart @call15 |
| 04 | 2003 | 19 | **1.0** | success | worker-restart @call17（入碗那几步是接续 worker 跑的） |
| 07 | 2006 | 20 | 0 | give_up | worker-restart @call1 |
| 10 | 2009 | 19 | 0 | give_up | worker-restart @call18 |
| 11 | 2010 | 20 | 0 | give_up | worker-restart @call16 |
| 12 | 2011 | 18 | 0 | give_up | worker-restart @call1 |

小结：单 worker 干净 5 行、worker-restart 7 行；`env_success` 1/12（trial-04）。两个时间窗口各自消耗：R1（5 槽）73.6M tokens（cache 69.0M）、R2（2 槽）36.2M（cache 33.8M）→ **持久模式单 trial 实际 7–8.5M tokens**。

## 仓库外部变更导致 runner 失效（19:0x）

另一个会话把 `embodied-ai/sonic-npu/weeks/2026_0914-0920_GPT6Astra评测复现/` 整体移到 `embodied-ai/2026_0914-0920_GPT6Astra评测复现/`（commit 447b362），runner 里写死的 `bowl-eval-v1` 路径失效 → seed 2012/2013/2014 的 runner 一启动就 `FileNotFoundError`（**0 次模型决策、没生成 attempt 目录**，已核实无残留）。
- 修复：`run_subagent_trials_qwen.py` 改成“env `BOWL_EVAL_STANDARD` → 批次本地 `standard/` → 仓库新旧两处”依次探测（`247b8498…`，原 `edff5d10…`）；协议内容仍由 runner 加载时校 SHA256，两份 standard 副本 `diff -r` 一致 → 字节等价。
- 后续 8 个 seed 换模式 B 跑（见 `../logs-qwen3.8flash-perdecision8-20260915-1917/`）。

## R3：回到唯一口径，用并行度当阀门（19:29 用户指令）

用户指令：**不改任何既定测试设置**；能跑几组跑几组，额度没了就等刷新；要控的是并行数量。据此：
- 模式 B（per-decision）于 19:29 停止。它 0 个 trial 跑到终态，4 个 attempt 目录（3/6/11/4 次决策）原样保留在 `../logs-qwen3.8flash-perdecision8-20260915-1917/`，目录内贴了 `OFF_SPEC_NOTE.md`，**不计入本模型 20 行正式成绩**（共花 2.94M tokens）。
- R3 = 原口径不变（每 trial 一个持久无历史子 agent 连续做完 + 冻结提示词/预算/物理/场景库），run `bowl-qwen-r3-pool2-mu2lda0h-szegou`，**2 槽**，跑 seed 2012–2019。
- 选 2 槽的依据：5 槽 ≈35 分钟被掉、2 槽 ≈62 分钟被掉；若仍很快被限，下一轮降到 1 槽。被掉时不做任何补救（不重跑受污染行，只标记）。
- 仅存的基础设施改动（零协议字节变更）：`launch_trial_r2.sh` 导出 `BOWL_EVAL_STANDARD` 指向批次本地 standard 副本（`9dd76d73…`）；`run_subagent_trials_qwen.py` 只改了目录解析回退（`247b8498…`）。runner 加载时照旧校 prompt/system/env-source SHA256。

## 现状

进度：
- [x] 固定波次 wave1 trial-01..05 已起（5/5 初始化比对 matched），但 15:52 因切换到动态滚动池被中断：call≈2、无模型成绩，原 attempt 与公开请求目录、渲染提示、runner stdout/stderr 全部保留在 `attempts-aborted-wave1-1552/`，`dispatch.jsonl` 记 5 条 `attempt_aborted`（valid_attempt=false，SOP §8 基础设施中止），随后从同一冻结初始状态重跑
- [ ] 动态滚动池 20 个 trial（trial-01..20，seed 2000..2019）
- [ ] audit_standard_trials.py → summary.csv / recording-audit.json / index.html / frames
- [ ] prepare_blind_review.py → 匿名复核包（评审不接触模型身份与 notes）
- [ ] 匿名阶段复核（stage_max 0–4）并归档 review.json
- [ ] report.md + 周目录结论回写

收口口径：成功率按复核 stage=4；`env_success` 与 harness `status` 另存；token/成本取不到填 null；基础设施失败保留原 attempt 并从同一初始状态整 trial 重跑。

## 收口（2026-09-15 22:25）

- 20/20 有效 attempt，记录检查 20/20 PASS，`error.json` 0 个。
- 盲评两轮完成（第一轮 5 组×4 包，边界第二轮 3 组×2 包，复核模型 `qwen3.8-max`，看不到模型身份与 notes）：**复核 stage=4 = 4/20**，平均最高阶段 1.45，阶段分布 0/1/2/3/4 = 6/8/1/1/4。`env_success` 与盲评同行无分歧；trial-19 为边界样本（第二复核只看画面给 3，包内轨迹显示已静止在碗内 → 记 4，保守读法 3/20 已在 `review.json` 标注）。
- 分组：单 worker 全程 12 行 = 3/12（25%）；429 换 worker 8 行 = 1/8。结论只建立在同口径的 12 行子组上。
- 产物：`report.md`、`summary.csv`、`paired-by-seed.csv`（与 glm/deepseek 同 seed 对照）、`recording-audit.json`、`index.html`、逐行 `trial-NN/review.json`。
- 结果镜像入公开仓：`embodied-ai/gpt6astra-repro/results/qwen3.8-flash/`（52 MB）；大件（发给模型的原图请求 142 MB、导出帧 52 MB、wire blob 51 MB、盲评包 184 MB、中止 attempt 66 MB）留本机，聚合 SHA256 见 `LOCAL_ONLY_MANIFEST.json`。
- **重要留痕：批次期间 runner 被并行会话改过。** 19:42:53 五个 runner（含 base/glm/deepseek）被统一打上 standard 目录修复 → `run_subagent_trials_qwen.py` 在我这批里先后有三个哈希：`edff5d10…`（15:47–19:31）、`247b8498…`（19:31–19:43）、`5fa57b52…`（19:43 之后，trial-14..20 用它起）。差异只在 standard 目录候选列表，**协议字节没变**：每次加载都校 prompt/system/assembled-system + 环境源码 SHA256，且每 trial 还逐字段比对冻结初始化库 —— 20 行同协议有实证（E2）。

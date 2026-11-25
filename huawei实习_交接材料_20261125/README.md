# 华为实习交接材料 · 索引

汇集日期：2026-11-25。共 6 项：1 个整体归档、4 组实验材料、1 组补充录像。文件名与目录结构均保持原样，未改名、未重新打包。

## 总表

| # | 目录 | 内容 | 来源 | 校验 |
|---|---|---|---|---|
| 1 | `huawei实习_测试报告与原始数据_20261016_150247/` | 2026-10-16 测试报告与原始数据归档，整体拷贝（7,137 个文件，3.03 GB） | 外接盘同名目录 | 自带 `校验/SHA256SUMS`，7,136 / 7,136 通过 |
| 2 | `DreamLab-repo/` | DreamLab 两个实验报告包（2 个 zip，9.1 MB） | 本机工作区 `DreamLab-repo/` | 与源文件 SHA256 一致 |
| 3 | `homebody-repo/videos/` | HomeBody Qwen3.8 换脑实验精选视频 12 段（45 MB） | 本机工作区 `homebody-repo/videos/` | 与源文件 SHA256 一致 |
| 4 | `HALO_qual02_ep10000_生成回放_20261119/` | HALO 首条合格轨迹 qual02 / ep10000 的生成与渲染回放（33 个文件，39 MB） | HALO 生成节点 `/data/halo` | 与节点原件 SHA256 一致 |
| 5 | `HALO_R15_物理跟踪回放_20261123/` | HALO R15 四组配对物理跟踪回放记录（28 个文件，393 MB） | HALO Tracker 节点 `/data/halo` | 与节点原件 SHA256 一致 |
| 6 | `RPent_deep-timing独立8集_补充录像_节点原件/` | RPent 8 集深测全部 10 个 `episode.mp4`（6.0 MB）；第 1 项归档包内只有其中 2 个 | RPent 实验节点 `/data/rpent2` | 与节点原件 SHA256 一致 |

第 1 项在盘上占用约 18G，是 exFAT 大簇的分配开销；实际内容 3,028,101,115 字节，与其 `校验/完成记录.json` 的 `file_content_bytes_approx` 相同。

## 校验方法

在本目录下执行：

```sh
shasum -a 256 -c SHA256SUMS-补充项                                           # 第 2–6 项，85 条
cd huawei实习_测试报告与原始数据_20261016_150247 && shasum -a 256 -c 校验/SHA256SUMS   # 第 1 项，7,136 条
```

第 4–6 项的校验值取自节点原件。第 6 项中两个 `t0_s0` 录像另与第 1 项归档包 `.manifest.json` 登记的 SHA256 核对一致。

## 1. 归档（2026-10-16）

入口：该目录下 `README.md`，指向实验数据总 README v1.3（`报告/embodied-ai/weeks/2026_1012-1018_RPent多模型性能分析/2026-10-16-具身智能测试项目与数据汇总.md`）。速查表 `报告与数据对照.md`。8 个数据包的字节数与 SHA256 见 `校验/数据包清单.tsv`。

`原始数据/本机/` 4 个包、`原始数据/npunew/` 4 个包。每个包旁有 `.manifest.json`（包内逐文件路径、大小、SHA256）、`.source-provenance.json`、`.ready.json`。

归档口径：保留统计记录与必要原始样例，省去其余图像载荷。JSON / JSONL 中的图像以载荷长度和处理摘要标记替代，逐帧 PNG 不保存；报告图表与 HTML 成品完整保留。该归档含会话原始记录与脱敏前分析，属私人离线归档。

### 1.1 `原始数据/本机/GPT-as-Policy_实验批次与代码.tar.gz`

解包根目录为 `GPT-as-Policy-repro/`。

| 实验 | 包内位置（`GPT-as-Policy-repro/data/` 下） |
|---|---|
| 两罐入篮 / Ori，三路线逐轮对照（Hybrid 51 轮全部放行 π） | `decision_v2_20260922/episodes/libero_10-{direct,hybrid,student}/` |
| 打开底层抽屉 / Task，三路线逐轮对照（Hybrid 60 轮，其中 13 轮 eef 接管） | `decision_extra_20260922/episodes/libero_goal_task-task0-{direct,hybrid,student}/` |
| 杯入微波炉并关门 / Ori，三路线逐轮对照 | `decision_extra_20260922/episodes/libero_10-task9-*`、`decision_extra_retry1_20260922/episodes/libero_10-task9-hybrid/` |
| 18 格条件对照（原生结果与人工展示口径） | `init0_20260921_141155/` |
| Hybrid 单格独立复跑 | `goal8_hybrid_repeat_20260921_160205/` |

逐轮对照的每个回合目录内：`decisions.jsonl`（逐轮决策）、`events.jsonl`、`trace.json`、`actions.jsonl`、`metrics.json`、`result.json`、`manifest.json`、`task.bddl`、`playback.html`、`timeline.html`、`video/actions.mp4`。`pi/`（逐轮 π 候选）在 Hybrid 与 Student 回合下都有，Direct 回合没有：两罐入篮 Hybrid 51 份、Student 52 份，底层抽屉两者各 60 份。`frames/` 每回合只留 3 张图。

要看逐轮图像，用归档里未压缩的单文件报告 `报告/本机成品/decision_v2_20260922/index.html`（约 155 MB，自包含）。

对应文档：`报告/embodied-ai/weeks/2026_0921-0927_同机仿真与云边验证/2026-09-22-LIBERO10-task0-Ori-init0-逐轮决策实录.md`（v1.8）。

### 1.2 `原始数据/npunew/deep-timing-20261008_远程原始记录.tar.gz`

解包根目录为 `rpent2/deep-timing-20261008/`。

8 集 = 2 个模型（`gpt-6-astra`、`gpt-6-sol`）× 2 个场景（`t0` 双罐入篮、`t9` 杯入微波炉并关门）× 2 个 seed（`s0`、`s1`）。`t9_s1` 两个模型各有 `attempt-001`、`attempt-002` 两次。

| 关注点 | 包内位置 |
|---|---|
| 双罐入篮、seed 1 逐轮记录（Astra） | `formal-results/gpt-6-astra/libero_10_swap/t0_s1/attempt-001/` |
| 双罐入篮、seed 1 逐轮记录（Sol） | `formal-results/gpt-6-sol/libero_10_swap/t0_s1/attempt-001/` |
| 对应计时数据 | `measurements/gpt-6-{astra,sol}/libero_10_swap/t0_s1/attempt-001/` 下的 `spans-*.jsonl`、`starts-*.jsonl` |

每集 `formal-results` 目录内有 `transcript_10_swap_t0_s1.json`、`codex_10_swap_t0_s1.txt` 及其 `.stream.jsonl`、`run.log`、`states.json`、`10_swap_t0_s1.json`、`agentview_metadata.json/`、`wrist_metadata.json/`。

录像：包内只有 `t0_s0` 的两个 `episode.mp4`（Astra、Sol 各一）。**seed 1 及其余各集的录像不在包内，见第 6 项。**

## 2. DreamLab 两个实验报告包

| 文件 | 字节 | 修改时间 |
|---|---:|---|
| `bottle_experiment_report.zip` | 5,386,401 | 2026-10-26 16:08 |
| `hybrid_experiment_report.zip` | 3,760,644 | 2026-10-27 09:19 |

## 3. HomeBody 换脑实验精选视频（12 段）

来源 `homebody-repo/videos/`。该目录被 gitignore，未随仓库上传。12 个文件修改时间均为 2026-11-12，第三人称视角。对应报告 `embodied-ai/weeks/2026_1109-1115_Qwen换脑与HALO调研/2026-11-12-HomeBody-Qwen3.8换脑报告.md` 第五节「视频回放」，报告中的「现场文件名」即下表文件名。

| 文件 | 字节 |
|---|---:|
| `after_breakfast__win.mp4` | 2,558,043 |
| `bags_to_rear_table__win.mp4` | 4,408,893 |
| `cartons_to_corner_counter__fail.mp4` | 3,979,939 |
| `conditional_juice__fail.mp4` | 1,856,298 |
| `default__win.mp4` | 7,918,017 |
| `farther_bag__fail.mp4` | 1,854,935 |
| `gait_fall__sample.mp4` | 2,457,119 |
| `juice_to_table_by_bin__win.mp4` | 1,929,672 |
| `moved_juice_to_bin__fail.mp4` | 3,177,684 |
| `swapped_reworded__win.mp4` | 7,901,215 |
| `the_other_carton_to_bin__win.mp4` | 2,417,445 |
| `there_and_back__win.mp4` | 4,529,802 |

文件名后缀 `__win` / `__fail` 为结果标记。

## 4. HALO qual02 / ep10000 生成回放（2026-11-19）

目录结构对应节点的绝对路径（本目录代表节点的 `/`）。

```
data/halo/data/motion/qual02/episode_10000/     生成轨迹本体：episode_spec.json、g1_qpos.npy、meta.json、plan.npy
data/halo/data/renders/qual02_mj/               MuJoCo 后端渲染输出
  episodes/episode_10000/                       fpv.mp4、oak.mp4、tpv.mp4、traj.npz、meta.json
  dataset/                                      LeRobot 风格数据集：data/、episode_specs/、meta/、videos/、frames/
  _mjscene/                                     渲染用场景资产
data/halo/diag/ep10000_{oak_f100,tpv_f190,tpv_f30}.png     渲染目视验收抽帧
data/halo/logs/qual02.log                       生成批次日志
data/halo/logs/render_qual02_mj.log             渲染日志
```

`dataset/frames/observation.images.fpv_cam/episode_010000.pt`（29.8 MB）是帧缓存张量，看视频不需要它。

据《HALO复现执行计划-20261117-v2》附录 A（2026-11-24 17:48 版）11-19 条目：ep10000 accept=True，198 帧；walk 1 轮 / turn 3 试 / sit 1 试；拼接完整性 PASS（meta == data == fpv == tpv）。

## 5. HALO R15 物理跟踪回放（2026-11-23）

目录结构对应节点的绝对路径（本目录代表节点的 `/`）。

```
data/halo/diag/play_ab/r15_exp_{f0,p1,p3,p2}.json（及 .log）    修复臂：四组 Terrain Play 回放的逐步记录
data/halo/diag/play_ab/r14_ctrl_{f0,p1,p2,p3}.json（及 .log）   对照臂：同起点、同 seed 的配对回放
data/halo/diag/resample_check/                                  report.json、report.log、r15_exp_stats.json（重采样剂量统计）
data/halo/logs/r15arm.log                                       训练与四次回放的包装日志（用时、checkpoint SHA256）
data/halo/logs/r15_exp.log                                      训练日志
data/halo/ops/r15arm.sh                                         训练与回放入口脚本
data/halo/halo-repo/tracker/logs/rsl_rl/motion_tracking_scene_prerl/2026-11-23_15-06-49_r15_exp_f0mix_fixed/
  model_399.pt                                                  试验 checkpoint
  params/                                                       env.yaml、agent.yaml 等
  events.out.tfevents.1795417628.gpu.3062773.0                  训练曲线事件
```

四组配对（参考起帧 / seed）：0 / 5、86 / 1、175 / 3、252 / 2。**回放未录制视频**，逐步物理量在 JSON 内。

`model_399.pt` SHA256：`0cee07abee2aa8b822a55e3c68459e4f94b75ade8f310738301b2735a4b4907a`。

结果表见《HALO四周执行计划与里程碑交接-20261120》「R15 原登记与四组回放（11 月 23 日）」一节（2026-11-24 15:35 版）。

## 6. RPent 8 集深测补充录像（节点原件）

第 1.2 项的归档按备份口径只保留了代表录像。本项从实验节点取回全部 10 个 `episode.mp4`，路径对应节点的 `/data/rpent2/…`，即归档包内路径前加 `data/`。

```
data/rpent2/deep-timing-20261008/formal-results/
  gpt-6-astra/libero_10_swap/{t0_s0,t0_s1,t9_s0,t9_s1}/attempt-001/episode.mp4
  gpt-6-astra/libero_10_swap/t9_s1/attempt-002/episode.mp4
  gpt-6-sol/libero_10_swap/{t0_s0,t0_s1,t9_s0,t9_s1}/attempt-001/episode.mp4
  gpt-6-sol/libero_10_swap/t9_s1/attempt-002/episode.mp4
```

| 场景 / seed / attempt | Astra（字节） | Sol（字节） |
|---|---:|---:|
| 双罐入篮 `t0_s0` / 001（归档包内已有） | 355,028 | 358,410 |
| **双罐入篮 `t0_s1` / 001** | **343,021** | **955,300** |
| 杯入微波炉并关门 `t9_s0` / 001 | 486,461 | 960,822 |
| 杯入微波炉并关门 `t9_s1` / 001 | 484,745 | 966,169 |
| 杯入微波炉并关门 `t9_s1` / 002 | 304,502 | 821,390 |

双罐入篮、seed 1 的 Astra 与 Sol 对照，取两个模型目录下的 `libero_10_swap/t0_s1/attempt-001/episode.mp4`；逐轮记录与计时数据在第 1.2 项归档包内。

# gpt6astra-repro — GPT-6 Astra 机器人评测复现

复现 OpenAI × RoboCurve「[GPT-6 Astra on robot arms](https://openai.robocurve.org/gpt-6-astra/)」评测的**技术协议**：`inspect-robots` harness + LLM agent policy，在 robosuite Panda 漏斗碗 pick-place 任务上，按统一规程 `bowl-eval-v1`（`robosuite-bowl-paired20-v1`）对多个模型各跑 20 trials。

> ⚠️ 复现的是**协议**（E1/E2），成功率数字只作模型间定性对照。Astra 的 19/20=95% 来自 I2RT YAM 真机，与本仓 robosuite 仿真**不可直接比较**。

## 目录结构

```
gpt6astra-repro/
├── README.md            ← 本文件：方法、控制变量、复跑方法、结果汇总
├── code/                ← 全部脚本（评测 CLI、环境与适配器、批次 runner、测试）
│   └── AGENTS.md        ← 开发操作手册（协议对齐值、已知坑、机制发现）
└── results/
    ├── README.md        ← 各模型批次索引与状态
    ├── glm-5.3-flash/   ← 完成 + 已盲评（20 trials）
    ├── deepseek-v4.1-flash/ ← 完成、盲评未跑（22 attempts）
    └── <model>/
        ├── README.md      ← 本批次结论摘要
        ├── summary.csv    ← 逐 trial：seed / stage / 成败 / 步数 / 调用数 / 终止原因
        ├── report.md      ← 批次报告（GLM 批次有）
        ├── experiment.json / initializations/  ← 共享冻结初态库与指纹
        ├── dispatch-prompt.txt                 ← 本批统一提示词
        ├── blind-review*/                      ← 盲评材料（evidence.json + 映射）
        └── trials/trial-NN/
            ├── video.mp4          ← 三相机渲染视频（最直观证据）
            ├── result.json        ← env_success / 步数 / LLM 调用数 / 终止原因
            ├── review.json        ← 盲评 stage_max 与依据（GLM 批次）
            ├── manifest.json      ← 模型标签 + scene_seed（控制变量指纹）
            ├── initialization*.json ← 本 trial 初态快照核对记录
            ├── incoming-*.json    ← 模型逐轮决策（观察→move_to 指令序列）
            ├── initial/final-*.png ← 首末帧三视角
            └── harness/           ← transcripts / actions / wire calls.jsonl（完整轨迹）
```

## 每次测试怎么控制变量

所有模型批次走**同一条协议**，唯一自由变量是决策模型：

| 控制项 | 做法 |
|---|---|
| 初始状态 | 20 个 scene（seeds 2000–2019）预渲染成**共享冻结初态库**（`experiment.json` + `initializations/`，聚合 SHA256 记录在案）；所有模型批次复制同一 bank |
| 初态核对 | 每 trial 在模型首次调用前对初态做完整快照比对，不一致直接中止（结果见各 trial `initialization-check.json`） |
| 任务协议 | 20 次 LLM 调用 / 900 控制步 / `max_speed_frac=0.25` / `effort=medium` / `image_horizon=2`，物理与提示词逐字节冻结，runner 每次加载校验 SHA256 |
| runner | 各模型 runner 之间**只差决策模型身份标签**（SHA256 差异可核，见 `code/run_subagent_trials*.py`） |
| 提示词 | 每 trial 同一份标准提示词（`dispatch-prompt.txt`；每 trial 仅 mailbox 路径不同） |
| 接入方式 | persistent-subagent：每 trial 一个**无历史**子 agent 连续完成整个 trial（保留本 trial 记忆；wire 只带最近 2 组图像——与严格无状态 API 的差异已在各批次披露，所有模型同组口径） |
| 评分 | 自动 `env_success` 与人工**盲评** stage 0–4 分离；盲评材料匿名化（`prepare_blind_review.py`），评完后写 `review.json`，不改原始日志 |

## 怎么跑测试

```bash
# 环境：Python 3.11，robosuite 主战场用隔离环境（勿混装）
python3 -m venv .venv-robosuite && .venv-robosuite/bin/pip install \
  robosuite==1.5.2 robosuite-models==1.0.0 mujoco==3.3.0 \
  'inspect-robots[rerun]==0.58.0' inspect-robots-agent==0.26.0

# 单模型标准 20-trial 批次（每个 trial 起一个全新子 agent）
.venv-robosuite/bin/python run_subagent_trials.py --log-dir logs-<model>-subagent20-<时间戳>
# 直接 harness 闭环（不同过子 agent 桥接）
.venv-robosuite/bin/python eval_bowl.py --model <model> --base-url <端点> \
    --key-env <KEY环境变量> --embodiment robosuite-bowl --trials 3 --log-dir logs-xxx
```

凭据一律走环境变量（脚本不读 `.env`），不入库。契约与回归测试：`code/test_*.py`。

## 结果汇总（截至 2026-09-15）

| 模型 | trials | env_success | 盲评 stage=4 | stage 分布 | 状态 |
|---|---|---|---|---|---|
| glm-5.3-flash | 20 | 0/20 | 0/20 | 0×11，1×6，2×2，3×1 | 完成，已盲评 |
| deepseek-v4.1-flash | 22 attempts | 1/22 | —（盲评未跑） | — | 完成 |
| qwen3.8-flash | — | — | — | — | **批次进行中，完成后另次同步** |
| gpt-6-astra | — | — | — | — | 未完成（仅 trial-01 有效；中断现场本机保留） |

所有模型盲评成功率均远低于 Astra 真机的 19/20；结合 stage 分布（多数 trial 卡在抓取前段），差距主要来自抓持物理与视觉闭环，属预期内的仿真—真机差距。结论与逐日记录见周报目录 [`../2026_0914-0920_GPT6Astra评测复现/`](../2026_0914-0920_GPT6Astra评测复现/)。

## 未上传内容（本机全量保留）

- `outgoing-*.json`：发给模型的请求体（内嵌 base64 观测图像；文本部分与 `harness/` 记录一致）
- `frames/`、wire 图像 blobs、`physics.jsonl`：中间帧与逐步物理记录，由冻结初态 + `harness/actions` 记录的动作序列确定性可复跑
- 盲评片段视频（`blind-review` 的 mp4）：与 `trials/trial-NN/video.mp4` 同源，仅保留评分依据 `evidence.json`
- 冒烟/诊断/历史 mock 批次、未完成批次（GPT、qwen 进行中）

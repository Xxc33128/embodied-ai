# qwen3.8-flash bowl-eval-v1 结果目录

模式：每个 trial 一个无历史持久子 agent 连续做完整个 trial（agent 定义 bowl-policy，模型 qwen-token-plan-cn/qwen3.8-flash，工具只 read + bash，thinking=medium）。协议 robosuite-bowl-paired20-v1 v1.0，seed 2000–2019 与其他模型批次共用同一份冻结初始化库。

- 结论与口径：`report.md` —— 复核 stage=4 共 **4/20**，平均最高阶段 1.45；12 行单 worker 干净组 3/12，8 行因 provider 429 换 worker 组 1/8。
- 逐行数据：`summary.csv`；`paired-by-seed.csv`（与 glm-5.3-flash、deepseek-v4.1-flash 同 seed 逐行对照）；`recording-audit.json`（20/20 记录检查 PASS）。
- 录像与逐行盲评依据：`trial-NN/video.mp4`（每控制步一帧、20fps 仿真时钟、front|side|wrist 三视图）、`trial-NN/review.json`。
- 可播放汇总页：`index.html`。
- 过程与事故（4 次 provider 429 限流、一次重复起池、一次仓库目录被移动导致 runner 路径失效、一次 per-decision 模式尝试作废）：`RUN_STATUS.md`、`dispatch.jsonl`。
- 协议冻结件：`standard/`（与 bowl-eval-v1 逐字节一致）；20 场景库：`initializations/`；运行时代码快照：`source/`。
- 未入库的大件（发给模型的原图请求、导出帧、盲评包、中止 attempt）只留在实验机，聚合 SHA256 见 `LOCAL_ONLY_MANIFEST.json`。

不含凭据、不含真机数据；数字是仿真 embodiment（robosuite Panda + OSC_POSE），不可与 OpenAI×RoboCurve 报告的 19/20=95% 直接比较。

# 通用大模型对具身智能的影响：比较综述（完整报告 v2.1）

成稿及媒体补齐：2026-11-25。项目实证截止：2026-11-23。未重跑机器人实验。

优先打开 **report.html**。正文10章、9张图、11段原始录像、8集交互计时均内嵌，可以离线阅读。覆盖 Manipulation Direct／Hybrid／Plan／RSI，以及 HomeBody Direct 与 HALO Plan。RSI为方法综述和待验证设计，没有本地正式留出评测成绩。

本版从交接材料新增8段录像：Hybrid两罐全放行与抽屉接管、RPent独立深测t0 seed1双模型对照、HomeBody两条失败案例、HALO ep10000生成参考及DreamLab四瓶入桶成功单集。原有三条Panda录像保留。所有录像完整保留，不剪切、不转码，来源路径及SHA256见 `data/media-manifest.json`。

| 媒体 | 播放定位依据 |
|---|---|
| Panda三条 | 各自13／19／10轮的控制步÷20；Qwen第19轮为give_up终止记录 |
| Hybrid两条 | frame_index.json原始时间索引；51／60轮；抽屉13轮eef接管均有按钮 |
| DreamLab | CSV调用编号和控制步÷25；131次推进物理的调用；调用1未推进，不造零长度片段 |
| HALO ep10000 | 原始meta的WALK／TURN／SIT帧区间÷30；仅为生成参考 |
| RPent两条、HomeBody两条 | 按完整视频时长四等分导航，明确不表示模型轮次；没有可靠逐轮帧映射 |

每条录像有完整播放、半速和区间终点暂停。HomeBody原片为1 fps采样回放，逐秒换帧是源文件特征，不能以此反推控制周期。HALO R15交接材料明确未录制视频：JSON／日志已取得，报告将其标为“未录制”，不再请求不存在的既存MP4。八份数值回放的首末记录与指纹在 `data/halo-tracker-record-summary.json`。

- `full-report.md`：可阅读完整正文，配合assets目录。
- `report.md`：构建输入，含媒体与交互标记。
- `assets/`：6张架构图、3张结果／耗时图、11段录像及原有图像。
- `data/`：逐集CSV、复算结果、媒体时间线、来源指纹和验证记录。
- `evidence/`：24份本地引用来源副本，按S编号命名；保留历史原文，不改写其当时的缺项描述。
- `prepare_handoff_media.py`：从交接原件选择性提取8段原片及最少审计数据，不复制完整请求、会话日志或训练权重。
- `media_utils.py`：按MP4视频轨道解析编码、帧数、帧率与时长，避免音轨覆盖视频元数据。
- `build_report.py`、`verify_report.py`：构建与静态核验；需要Python 3，核验另需Node。
- `video-segments.js`、`test_video_segments.cjs`：离线播放器与事件模拟测试。

在原embodied-ai目录结构中运行：

```sh
# 交接材料变更时重新提取；普通重建不必重复解包。
python3 docs/embodied-llm-survey-20261125/prepare_handoff_media.py
python3 docs/embodied-llm-survey-20261125/build_report.py
python3 docs/embodied-llm-survey-20261125/verify_report.py
node docs/embodied-llm-survey-20261125/test_video_segments.cjs
```

13项构建检查与28项扩展检查覆盖统计复算、引用与链接、来源指纹、内嵌媒体字节和JS语法。事件模拟覆盖303个时间区间、344个按钮、混合视频时钟、独立播放器、重复入口高亮、手动拖动、完整播放、半速和加载前连续选择。事件模拟不替代真实浏览器验证；Chrome实测范围见 `data/browser-verification.json`，记录绑定本版HTML的SHA256。重新构建HTML后，旧浏览器记录若指纹不同会标为过期。

新增媒体不改变实验统计分母：RPent补充目录10个录像包含重试，仍为8个有效episode；DreamLab单集成功不并入pilot样本；HALO生成参考不等于物理跟踪成功。Motus最新完整结果仍为10000步30/50，9000步38/50是固定开发观察集历史最高。

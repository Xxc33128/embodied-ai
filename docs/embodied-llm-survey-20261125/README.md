# 通用大模型对具身智能的影响：比较综述（完整报告 v2.0）

成稿：2026-11-25。项目实证截止：2026-11-23。文献核对：2026-11-25。

优先打开 **report.html**。正文、9张图、3段已有实验视频与8集交互计时均内嵌，可以离线阅读。正文共10章，覆盖 Manipulation Direct／Hybrid／Plan／RSI，以及 HomeBody Direct 与 HALO Plan。RSI为方法综述和待验证设计，没有伪造本地实验成绩。

- `full-report.md`：可编辑完整正文，配合assets目录阅读。
- `report.md`：构建输入，包含媒体与交互插入标记。
- `assets/`：6张架构图、3张结果／耗时图、3段视频与已有图像。
- `data/`：逐集CSV、复算结果、媒体索引、来源指纹和验证记录。
- `evidence/`：24份本次引用的本地来源副本，按S编号命名。内文中的相对路径仍沿用原项目，原始来源路径及SHA256在data/source-manifest.json。
- `build_report.py`、`verify_report.py`：构建与验证脚本。在原embodied-ai目录结构中使用Python 3运行，验证脚本另需Node。

重建：`python3 docs/embodied-llm-survey-20261125/build_report.py`

核验：`python3 docs/embodied-llm-survey-20261125/verify_report.py`

已完成13项构建检查与19项扩展检查：120集成功／耗时复算、8集计时闭合、引用与链接、来源副本指纹、内嵌视频字节及JavaScript语法。2026-11-25已在macOS Chrome本地报告页抽查三条视频的轮次播放、段末暂停和表格入口；V03还实测半速与完整播放重启。详见data/browser-verification.json（绑定HTML SHA256），不是全部按钮的浏览器穷举。全部42轮的时间映射、57个按钮、独立播放器和重复入口高亮已通过Node事件模拟验证。

未重跑机器人实验。HomeBody与HALO部分结论按归档转引。缺失的Hybrid／RPent／HomeBody／HALO代表视频保留证据入口，未用生成动画或上游视频替代。Motus以10000步30/50为最新完整结果，9000步38/50为固定开发观察集历史最高，未当作独立泛化成绩。

此版本保留原有报告，并新增完整RSI章节、原论文方法比较、因果归因与有效性威胁、Motus训练补充、后续实验优先级及术语表。

视频交互修订（2026-11-25）：V01关键轮次按钮显示原视频时间范围，点击后播放该动作段并在终点暂停；增加完整播放、半速、当前轮次提示与选中高亮。原视频不剪切，时间映射仍为控制步÷20。`video-segments.js`在构建时内嵌，离线HTML无需另载脚本。可运行 `node docs/embodied-llm-survey-20261125/test_video_segments.cjs` 验证时间映射、段末暂停、完整播放、手动拖动、倍速和加载前连续选段；这是事件模拟验证，不替代真实浏览器播放验证。

统一视频控制（2026-11-25）：V01／V02／V03分别有13／19／10轮；每条展示5个关键入口，展开表格可播放全部轮次。V02第19轮为give_up终止记录。各视频使用独立时间轴；来源与提取口径见data/video-timeline-provenance.json。缺失Hybrid录像的取件信息见hybrid-video-request.md。

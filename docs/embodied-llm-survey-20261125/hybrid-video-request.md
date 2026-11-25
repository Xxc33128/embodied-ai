# Hybrid录像取件清单

“包B”是归档清单编号，对应压缩包：`GPT-as-Policy_实验批次与代码.tar.gz`。

归档记录中的原位置：

```text
/Volumes/Extreme SSD/huawei实习_测试报告与原始数据_20261016_150247/原始数据/本机/GPT-as-Policy_实验批次与代码.tar.gz
```

请提供以下两条视频，最好连同各自完整回合目录（配置、结果、逐轮日志）一起提供，以便添加分段播放按钮。

| 视频 | 核对身份与结果 |
|---|---|
| 两罐入篮 Hybrid 全放行 | LIBERO-10 task 0、Ori、init 0；episode_id `libero_10-task0-init0-hybrid`；51轮／253步／成功；51轮全部放行；历史核验254帧 |
| 底层抽屉 Hybrid 13次接管 | LIBERO Goal Task task 0、Task、init 0；episode_id `libero_goal_task-task0-init0-hybrid`；60轮／300步／失败；47轮student、13轮eef；历史核验301帧 |

包内视频路径：

```text
GPT-as-Policy-repro/data/decision_v2_20260922/episodes/libero_10-hybrid/video/actions.mp4
GPT-as-Policy-repro/data/decision_extra_20260922/episodes/libero_goal_task-task0-hybrid/video/actions.mp4
```

抽屉任务的实际 instruction 是 `open the bottom drawer of the cabinet`；原始 task_name 仍为 `open_the_middle_drawer_of_the_cabinet`，应结合 episode_id、instruction 和回合计数核对。

当前本地weeks目录仅保存报告、汇总及历史视频核验记录，没有这两条MP4，也没有对应周HTML的内嵌视频。本清单依据归档索引、回合汇总及生成代码确定，尚未读取离线压缩包本体；取件后应再按包旁manifest核验路径与SHA256。

若同事有成品展示页，也可提供归档中的 `报告/本机成品/decision_v2_20260922/index.html`，历史记录称其内嵌视频；仍优先提供原回合目录。

依据：`evidence/S03.md`（归档索引）、项目内 `weeks/2026_0921-0927_同机仿真与云边验证/evidence/2026-09-22-extra-tasks-summary.json`、`2026-09-22-decision-v2-summary.json`、`2026-09-29-decision-v18-verification.json`，以及 `gpt-as-policy-repro/scripts/build_decision_report.py` 中的 `video/actions.mp4` 读取规则。

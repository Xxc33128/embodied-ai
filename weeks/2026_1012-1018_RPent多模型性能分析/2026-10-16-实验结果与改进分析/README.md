# 具身智能实验结果与改进方向报告

版本 v4.1 · 2026年10月16日

[完整报告 PDF](具身智能实验结果与改进方向报告.pdf) · [可编辑 Word](具身智能实验结果与改进方向报告.docx)

本版按“实验条件与结果 → 过程分析 → 结论 → 下一项实验”组织正文。模型层从 bowl 对照和抓空恢复分析能力；Harness 层依次分析 18 格、九回合、RPent 普通模式、Flash 与重试；性能层重点复算 RPent 120 集、独立八集深测，以及 DeepSeek 档位、逐请求窗口、缓存与重放实验。后续训练、JEV 选择器、异步调度和融合架构分别对应前面的实验发现。〔E3〕

## 本次修订

v4.1 将原表 02、03、04 合并为第二章的一张分组表，完整保留 RoboDojo 榜单和两批 bowl 的 12 行数据、指标及条件；后续表号顺延，PPT 拆分同步引用合并表的 A/B/C 组。

## v4 的主要变化

- PDF 正文连续排版。附录 A 单独给出 18 个 PPT 证据单元，每单元可拆成 1–3 张正文或备份内容；按数据和问题拆分，不与 PDF 页数对应。
- 增加逐轮行为和失败过程：Astra 抓空后的修正、Hybrid 全放行与抽屉接管、Flash 名称绑定失败、RPent 重复抓取及恢复耗时。
- 新增配对复算：Spatial Swap 的 SDK 时间差中，调用数项占 69.72%；Long Swap 总时间差的 95.91% 来自非共同成功的 12 对任务。公式、分母和检查结果见 [新增复算](derived-analysis.json)。〔E3〕
- 分开原生任务成功判据与展示接受结果；RPent 普通／Flash 的全样本均值和成功回合均值分别列明。〔E3〕
- DeepSeek 以具体请求说明 TTFT、thinking 与正文／工具输出的不同瓶颈；精简文字并统一标题、表头的中英文字体和字重。

## 正文与 PPT 拆分

| 正文 | 内容 | 后续 PPT 主要数据 |
|---|---|---|
| 第 1 章 | RoboDojo 与 LIBERO | 任务维度、官方榜日期、Score 与 SR |
| 第 2 章 | 模型能力 | 两批 bowl 结果、同 seed 对照、试举与重抓轨迹 |
| 第 3 章 | Harness | 18 格和九回合结果、普通／Flash 分母、失败重试、阶段反馈与 JEV |
| 第 4 章 | RPent 性能 | 120 集分布、共同成功配对、五类时间、深测渲染与恢复时间轴 |
| 第 5 章 | DeepSeek 性能 | 两档配对、TTFT／TPOT 分位数、v5 全部 15 请求、输入缓存与固定请求重放 |
| 第 6 章 | 后续改进 | 训练消融、Harness 对照、动作覆盖预算、low think、模块融合与 Motus 工程结果 |

完整拆分见报告附录 A 与 [ppt-plan.json](ppt-plan.json)。主文包含 26 张编号表、7 幅数据图，另有一张实验范围表。

## 数据附件

沿用已核对的 15 份脱敏 CSV，保留各批次原有身份。新增指标由这些数据复算，未增加实验样本。〔E3〕

| 文件 | 记录数 | 内容 |
|---|---:|---|
| [deepseek16_episodes.csv](data/deepseek16_episodes.csv) | 16 | 非流式与流式两批逐集结果 |
| [deepseek_pairs.csv](data/deepseek_pairs.csv) | 4 | 两批共同成功身份的档位配对 |
| [deepseek_phase_groups.csv](data/deepseek_phase_groups.csv) | 23 | 逐请求阶段分组 |
| [deepseek_protocol.csv](data/deepseek_protocol.csv) | 8 | 流式运行配置摘要 |
| [ds_flash_v5_report_table.csv](data/ds_flash_v5_report_table.csv) | 15 | v5 原报告逐轮展示表 |
| [files_api_report_table.csv](data/files_api_report_table.csv) | 11 | Files API 独立实验逐轮表 |
| [phase_rows_sanitized.csv](data/phase_rows_sanitized.csv) | 458 | 有效事件窗口与 Token 分配估算 |
| [replay42_requests.csv](data/replay42_requests.csv) | 42 | 九个请求体重放 |
| [representative_requests.csv](data/representative_requests.csv) | 3 | 代表长尾请求 |
| [rpent120_episodes.csv](data/rpent120_episodes.csv) | 120 | 正式对照逐集账本 |
| [rpent120_groups.csv](data/rpent120_groups.csv) | 6 | 套件与模型分组 |
| [rpent120_paired.csv](data/rpent120_paired.csv) | 60 | 相同任务与 seed 配对 |
| [rpent8_episodes.csv](data/rpent8_episodes.csv) | 8 | 独立深测逐集账本 |
| [rpent_representative_local_actions.csv](data/rpent_representative_local_actions.csv) | 55 | 代表回合本地动作 |
| [rpent_representative_timeline.csv](data/rpent_representative_timeline.csv) | 275 | 客户端及本地时间区间 |

`table-data.json` 保存表头和展示数值；`figure-data.json` 保存图号、章节、图注，以及请求体图的编号映射；`sources.json` 保存 48 项来源；`content-source.json` 保存可继续编辑的正文结构。`manifest.json` 记录交付文件校验值。

E1 为源码机制，E2 为配置协议，E3 为报告转引、复算、论文结果与明确标注的实验设计，E4 为外部转述。导师最新 low think 的 4.6s→1.4s 按 E4 记录，报告另列完整请求与闭环的复测设计。

上一版保留于 [v3 报告目录](../2026-10-16-三层研究报告/README.md)。

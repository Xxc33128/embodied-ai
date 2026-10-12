# DeepSeek 两场景性能审计 v1.4

日期：2026-10-12。分析既有数据，未执行新的模型/仿真实验。

入口：[HTML报告](RPent-DeepSeek-两场景性能审计.html)。离线可读，无外部依赖。

## 口径

- 原始非流式8集：306条响应记录；原指标另记录一次transport重试，因此不宣称无网络重试。
- 流式补测8集：178条响应attempt、177个逐集唯一逻辑请求；177条有效TTFT/TPOT；medium82条可见thinking窗口。
- 任务仅libero_10_swap t0/t9，seed0/1，medium/none。日志finish success不等于本次独立重验环境终态。
- TTFT为首非空reasoning/content/tool delta；TPOT为客户端均值代理；prefill回归和上传速率估算不是精确服务端阶段。
- 原始JSON的server_span_ms是headers到最后chunk的客户端区间，不是服务端计算时间。
- local_execution的sum_ms/union_ms保留嵌套口径，不可跨族相加；π使用stages同步，会扰动性能。
- 时间轴从首个选中span开始，覆盖范围不等于episode日志起止；记录类名与源文件SHA256，8集各只有1个时钟域。

## 文件

- `audited-data.json`：从两批metrics白名单提取，包含源文件hash；8+8集tokens与逐请求usage已逐字段对账。
- `requests.csv`：两批全部484条响应记录。不同批次列不同，缺失保留空值。
- `io-evidence.json`：远程只导出字段名、类型、形状、长度、hash、usage关联键；无原始prompt/图片/凭据。
- `timeline-evidence.json`：LLM、动作外层、π/SAM client和render的相对区间。轨道有嵌套，不作条长相加。
- `measurement-source.md`、`scene-goals.json`：E1定义与目标证据。
- `extract_*_remote.py`：远程只读白名单提取脚本，需要在原实验节点运行；不启动评测。
- `build_report.py`：读取本目录脱敏JSON并生成HTML；运行 `python3 build_report.py` 即可。

## 交叉审查与验证

三Agent分工：计时/分母与数字审计；场景/接口/协议审计；结构与结论审查。修正失败attempt公式、未验证“最优”措辞、图像历史裁剪与阶段代理指标。

已验证：16集token逐字段对账；8集同clock domain；浏览器8个episode切换、7节内容、无JavaScript错误；390px移动端无页面横向溢出；桌面与移动端截图人工检查。源码与证据均经敏感字段检查，不发布原始日志。旧周目录的未提交草稿未改写或纳入本次提交。

## v1.1 修订

公式改为四集成功样本的逐层实数代入，耗时占比与优化预算前置；新增report_intro.py构建模块。round-context.json和extract_round_context_remote.py保存逐请求消息多重集差分、新带入的历史工具返回的白名单摘要。正文不导出原始会话或思考内容。

消息差分识别新增/改写与删除/改写；工具结果归属为本请求带入的历史工具结果。官方token差额不等于新增文本token。优化预算为条件敏感性估算，不是已测加速。

## v1.2 补核代理日志与批次

新增25份代理指标文件，共484行，与前述8+8集的484条响应属于不同数据源。450行通过请求数、input/output累计值与时间窗匹配25个完整运行区块；32行仅候选关联，2行未归属。repro2/3/4属于Qwen，未混入DeepSeek结论。参考报告11轮实验的transcript可确认，但当前根指标文件没有对应数据；现存files_api的13轮对应另一个运行。

- `proxy-inventory.json`：原文件hash、逐行白名单计时/usage/结构字段，无原始内容。
- `proxy-run-mapping.json`：行范围、实际运行、任务、token对账、时间窗及transcript/run.log指纹。
- `proxy-source-audit.md`：六份当前采集源码的hash与行号；当前源码不能证明所有历史运行版本。
- `proxy-requests.csv`：新增484行记录与派生TTFT筛选、thinking窗口、均值TPOT；原TPOT p50/p90字段为delta字段间隔。
- `extract_proxy_inventory_remote.py`：只读白名单提取器。
- `report_proxy.py`：单独渲染新增审计和15请求模型，未修改原8+8集统计。

355行符合保守流式TTFT筛选（DeepSeek287、Qwen68），329行有thinking可见代理窗口；非流式的完成耗时不冒充TTFT。新模型逐层代入110.534秒请求累计，明确阶段影响因素、分母和20%敏感性预算。请求累计减episode仅作账面核算；整集优化收益要求同轨迹/同成功结果并落在关键路径。prefill拟合与均匀速率思考估算不等于服务端实测。

三项交叉复核已完成：运行归属、源码边界、独立数值与性能模型。页面保留原8集时间轴及逐轮上下文选择器；新增全批归属、计时覆盖和phase字段表。

## v1.3 结论重写

开头只列数据、结论和优化收益；删除修订解释，将日志归属、计时审计移入折叠附录。新增report_conclusions.py，公式直接使用15请求实测分段，原4集模型独立折叠保留。新的文字审查Agent检查结论、删去过程叙述。数值对账与桌面/移动端页面验证通过。

## v1.4 仅保留流式统计

正文移除非流式8集及非流式代理运行。流式8集178条，加15个精确归属DeepSeek运行的280条有效流式请求，共23组选项、458条。原始记录保留备查。新增report_streaming.py和streaming-phases.csv。分段表区分客户端实测窗口、TPOT派生值、按token比例估算的thinking/decode；纯prefill缺测。thinking窗口终点采用首正文或工具delta，与旧content-only口径不同。

思考开关：成功率均2/4；共同成功两集2326.7到1848.2秒，省478.5秒（20.57%），63到57轮。删除混入非流式的43.1%变慢结论。独立Agent核验配对数值，23组切换、458行CSV、估算闭合及浏览器布局检查通过。

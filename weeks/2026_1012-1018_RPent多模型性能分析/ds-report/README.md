# DeepSeek 两场景性能审计 v1.0

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
- `build_report.py`：读取本目录3个JSON并生成HTML；运行 `python3 build_report.py` 即可。

## 交叉审查与验证

三Agent分工：计时/分母与数字审计；场景/接口/协议审计；结构与结论审查。修正失败attempt公式、未验证“最优”措辞、图像历史裁剪与阶段代理指标。

已验证：16集token逐字段对账；8集同clock domain；浏览器8个episode切换、7节内容、无JavaScript错误；390px移动端无页面横向溢出；桌面与移动端截图人工检查。源码与证据均经敏感字段检查，不发布原始日志。旧周目录的未提交草稿未改写或纳入本次提交。

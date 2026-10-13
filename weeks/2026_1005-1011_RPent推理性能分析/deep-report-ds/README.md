# DeepSeek 首批流式补测报告（10/9–10/10）

入口：[历史深度计时报告](DS-DeepSeek-V4.1-Flash-深度计时报告.html)。本目录与原始非流式批次、Codex SDK 的 8 集深测分开保存。〔E2：已归档数据〕

本报告保留当时的计时模型与派生值。有效请求去重、客户端窗口边界及 prefill 代理命名，以 [10/12 后续审计](../../2026_1012-1018_RPent多模型性能分析/ds-report/README.md) 为准；不将历史拟合当作服务端精确计时。〔E3：口径说明〕

`report-data-ds.json` 为本页唯一数据副本；原 `../evidence-ds-20261009/` 中逐字节相同且无引用的副本已移入工作区待清理区。离线重新渲染：

```bash
python3 build_ds_deep_report.py --data report-data-ds.json --out /tmp/deepseek-historical-report.html
```

`assemble_ds_report_data.py` 是原节点提取工具，需要原始批次与私有请求体；本机公开镜像不含这些输入，不在整理时执行。读取现成 JSON 的渲染器不访问实验服务，也不启动仿真。

# DeepSeek 10/9 批次证据

保留原始非流式批次指标、独立流式补测指标、标定与验证 JSON。不同批次单独统计，不混合整集分母。历史数字按已存指标〔E2〕阅读；机制与估算按〔E3〕阅读。

2026-10-13 整理：`metrics.json`、`metrics-shim.json`、`asset-manifest.json` 的公开副本仅将机器根目录替换为 `${RPENT_DATA_ROOT}`，并移除凭据文件位置；数值、实验身份和资产哈希保持原样。敏感原件与同名路径归档在本机私有记录仓，不上传。

文件中的来源 SHA256 指向原始证据／资产，不能拿脱敏后的文件字节重新计算并替代。完整原始请求、图片出站体、凭据和运行日志不在公开镜像内。

报告使用的数据契约只保留在 [../deep-report-ds/report-data-ds.json](../deep-report-ds/report-data-ds.json)。计时有效性与阶段命名的后续结论见 [10/12 审计说明](../../2026_1012-1018_RPent多模型性能分析/ds-report/README.md)。

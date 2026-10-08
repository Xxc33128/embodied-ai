# 深测证据索引

正式报告以 `report-data.json` 的 integration.sources 和逐集 source_artifacts 为准；8集为6个attempt-001与2个attempt-002。

- 最终数值：extract-eight-final-001、trajectory-scopes-eight-final-001、cloud-eight-final-001、map-eight-final-001、**cloud-contract-eight-final-002**、final-artifacts-eight-final-001、batch-summary-eight-final-001。
- cloud-contract-eight-final-001 是未经transcript交叉对账的中间版本，正式使用002。
- 名含 first-pair、t0、six、partial、eight-completed 的文件是阶段审查快照；尤其 eight-completed-20261009-01 包含后来因SAM OOM排除的两次原attempt，不是最终8集分母。保留用于追溯，不替代final来源。
- excluded-attempt-cost-ledger 与 excluded-attempt-model-init-evidence 记录排除原因、原件hash和附加成本；不混入正式成功率或性能。
- io-contract、render-path-audit、logical-sampling-count-convention、vla-calibration 提供源码/方法与校准证据；SAM正式证据区分seed0调用及末对attempt002，健康探针不混入正式数据。
- final-evidence-review、final-protocol-review、final-ui-review 为独立交叉审查；其中hash对应各次审查的明确快照。

这些是脱敏摘要，原始会话、认证、原始敏感日志不在本目录。未知阶段不补零；嵌套计时不能与父项相加。

# bowl-eval-v1 标准包

版本1.0，2026-09-15；状态：规范已制定，正式实验未运行。

- [完整标准流程](../2026-09-15-bowl标准测试流程.md)
- `protocol.json`：模型/环境/观测/记录规范及哈希；不是已实现的CLI参数文件。
- `seeds.csv`：20个正式场景及harness派生seed；每个模型用同一张表。
- `prompts/system.txt`、`prompts/embodiment.txt`、`prompts/task.txt`：标准提示文本。
- `results-template.csv`：汇总表字段模板，没有填入任何实验成绩。

实际请求的system内容应严格等于：system.txt去掉首尾空白 + 两个换行 + `Robot documentation:` + 一个换行 + embodiment.txt去掉首尾空白。任务句来自task.txt。所有供应商都使用这份规范输入；实际wire请求另外完整保存。

环境源码哈希是制定标准时的参考快照，不代表现有runner已经采用新提示词或900步上限。正式批次开始前完成SOP末尾的必要接线检查，生成初始化场景库并冻结批次manifest。

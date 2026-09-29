# 逐轮 HTML 的知识补充维护

版本：v1.0 · 2026-09-29

〔E1〕知识正文来自 embodied-ai 周目录的 `2026-09-23-实验报告对话疑问与解答.md`。先修改这份 MD，再构建 HTML；不要在 HTML 中维护另一份知识正文。标题编号用于正文说明链接，修改章节编号时应同步模板入口并运行验证。

```sh
python3 -m pip install -r requirements-report.txt
python3 scripts/build_decision_report.py data/decision_v2_20260922
python3 scripts/verify_decision_report.py data/decision_v2_20260922
```

〔E1〕构建器自动寻找同级 embodied-ai 或发表镜像上级的权威周目录，也可通过 `--knowledge-md /absolute/path/to/companion.md` 指定。Markdown 依赖只用于生成 HTML，阅读成品不需要 Python 或网络。MD 引用的图片及本地附件直接内嵌；引用需位于权威公开仓库内。外部网站仍作为联网参考链接。

〔E1〕`knowledge_manifest.json` 记录本次采用的 MD 版本及各内嵌文件 SHA256。验证覆盖锚点、重复 ID、配图、下载附件哈希和原有逐轮数据。Node 检查覆盖点击说明展开目标章节；不等同于真实浏览器排版验收。打印时自动展开知识主题，打印后恢复折叠状态。

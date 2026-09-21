# Mac 会话接入：服务器端桥

stdio_socket.py 从服务器已有 Unix socket 转发 JSON-RPC 数据。本脚本不包含凭据，也不负责创建 Mac 模型会话或 SSH 转发。

截至 2026-09-21，只恢复了该服务器端文件；Mac 监听桥源代码仍缺。不要把此目录描述为完整 Mac 接入服务。使用步骤和配置示例见 docs/wiki/Experiments.md。

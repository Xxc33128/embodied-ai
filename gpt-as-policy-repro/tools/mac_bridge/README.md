# Mac 与服务器的连接

- `listen.py`：在 Mac 监听 Unix socket，每个连接启动独立 Codex app-server；本次新写。
- `stdio_socket.py`：服务器端标准输入／输出与 Unix socket 之间转发；从原实验归档。

[启动方法与 SSH 转发](../../docs/wiki/Mac-bridge.md)。两个脚本均不包含账号凭据。连接检查通过不等于模型实验通过。

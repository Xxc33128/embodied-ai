# Mac 监听程序

`tools/mac_bridge/listen.py` 是 2026-09-21 按现有 JSON-RPC 接口补写的版本，不是 9 月 20 日原程序的恢复副本。只用 Python 标准库，每个连接启动独立 Codex app-server；连接结束后回收进程，Ctrl+C 关闭监听。

## 1. 在 Mac 启动

Mac 需要 Python 3.8+、支持 `app-server --listen stdio://` 的 Codex，以及使用者自己的模型权限。先用 `codex --version` 和 `codex app-server --help` 核对本机命令。登录与模型配置留在 Mac，不复制账号凭据。

在仓库的 `gpt-as-policy-repro` 目录执行，保持终端运行：

```bash
python3 tools/mac_bridge/listen.py --socket "/tmp/gap-mac-$(id -u)/codex.sock"
```

看到 `READY` 表示监听已就绪，不代表模型权限已验证。如果 `codex` 不在 PATH，可显式指定可执行文件：

```bash
python3 tools/mac_bridge/listen.py --socket "/tmp/gap-mac-$(id -u)/codex.sock" -- /absolute/path/to/codex app-server --listen stdio://
```

监听目录必须归当前用户所有、权限为 700；socket 权限为 600。已有同名文件时程序拒绝覆盖。异常退出留下 socket 时，先确认旧进程已结束，再只删除该 socket。程序不监听公网端口。

## 2. 从 Mac 建立 SSH 转发

另开 Mac 终端。`NPU_HOST` 是本机 SSH 配置中的服务器别名；`REMOTE_DATA` 是服务器宿主机挂载到容器 `/workspace/data` 的目录，须替换成实际路径。路径不要含空格。

```bash
NPU_HOST=your-npu
REMOTE_DATA=/absolute/host/data
ssh "$NPU_HOST" "mkdir -p '$REMOTE_DATA/mac_bridge' && chmod 700 '$REMOTE_DATA/mac_bridge'"
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3   -R "$REMOTE_DATA/mac_bridge/codex.sock:/tmp/gap-mac-$(id -u)/codex.sock" "$NPU_HOST"
```

保持两个 Mac 终端运行。服务器 SSH 服务须允许 Unix socket 转发；仿真容器须挂载该 data 目录，并有权访问 socket。重连若提示地址占用，确认旧 SSH 转发已结束后，只移除服务器上遗留的 `mac_bridge/codex.sock`，不要删除数据目录。

## 3. 在仿真容器检查连接

按 [实验运行](https://github.com/Xxc33128/embodied-ai/wiki/Experiments) 保存 `gpt.local.json`，在 `gap-sim` 中、`/workspace/repo` 下执行：

```bash
export PYTHONPATH=/workspace/repo/src
python - <<'CHECK'
import json
from gap_repro.agent.client import AppServerTransport
with open('/workspace/data/mac_bridge/gpt.local.json') as f:
    transport = AppServerTransport(json.load(f))
try:
    transport.connect()
    transport.request('initialize', {
        'clientInfo': {'name': 'gap-connection-check', 'version': '1'},
        'capabilities': {'experimentalApi': True}
    }, timeout=15)
    transport.notify('initialized', {})
    print('initialize OK')
finally:
    transport.close()
CHECK
```

这一步不发起模型推理，只验证通信。然后按 Experiments 页运行一个 Direct 回合，才能验证账号、模型和工具调用。

## 已验证范围

5 项自动测试覆盖大数据与半关闭后的末尾响应、连接隔离、断开与退出时进程回收、拒绝覆盖已有文件，以及服务器端 `stdio_socket.py` 的往返传输。另已通过本机及真实 SSH 隧道访问 Mac Codex 的 `initialize` 检查，本机版本为 `0.155.0-alpha.9.2`。未用本次新程序运行完整机器人回合。

"""LP3/L4：模型接入传输层（路线 A app-server / 路线 B API 网关）。

接入信息未提供前，两条路线都可实例化但 connect() 抛 ConfigMissingError，
错误信息即"需要用户提供什么"的清单——契约层与其余代码不依赖模型在线。

路线 A 实现逐字移植 upstream skill/transport.py::StdioAppServer
（commit 8f3d362b；JSON-RPC over stdio、双线程读、RPC 日志 jsonl、
进程组关闭）；适配仅限构造（argv 由 app_server_cmd shlex 切分、
日志目录由 workspace 指定）。路线 B 为 HTTP POST（凭据只取环境变量名），
事件面与路线 A 统一：request 产出 completed 事件入队，next_message 排空。
真实端点行为在联调（用户提供凭据）时核对；本实现保证契约层可全路径测试。
"""
import json
import os
import queue
import shlex
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


class ConfigMissingError(RuntimeError):
    def __init__(self, route, missing):
        self.route, self.missing = route, list(missing)
        super().__init__(
            f"route {route!r} missing config: {', '.join(self.missing)} "
            f"（待用户提供）")


class Transport:
    """agent 传输接口：request（工具往返）+ next_message（事件流）。

    语义对齐 upstream skill/run.py::CodexPolicy 所需的最小面：
    request(method, params, timeout) → dict；next_message(timeout) → event dict
    （method ∈ turn/completed | item/tool/call | thread/tokenUsage/updated）。
    """

    route = "abstract"

    def __init__(self, config=None):
        self.config = dict(config or {})

    def missing_config(self):
        raise NotImplementedError

    def connect(self):
        missing = self.missing_config()
        if missing:
            raise ConfigMissingError(self.route, missing)

    def request(self, method, params, timeout):
        raise NotImplementedError

    def next_message(self, timeout):
        raise NotImplementedError

    def close(self):
        pass


class AppServerTransport(Transport):
    """路线 A：持久 Codex app-server（stdio JSON-RPC）。

    需要用户提供：app_server_cmd（启动命令）、model、provider、effort(xhigh)、
    workspace（RPC 日志目录）。实现锚：upstream transport.py::StdioAppServer。
    """

    route = "app_server"
    REQUIRED = ("app_server_cmd", "model", "provider", "effort")

    def missing_config(self):
        return [k for k in self.REQUIRED if not self.config.get(k)]

    def connect(self):
        super().connect()
        argv = (shlex.split(self.config["app_server_cmd"])
                if isinstance(self.config["app_server_cmd"], str)
                else list(self.config["app_server_cmd"]))
        # workspace（RPC 日志目录）是实现细节而非用户决策：默认临时目录
        self._server = _StdioJsonRpc(
            argv, Path(self.config.get("workspace")
                       or Path("/tmp/gap_repro_appserver")),
            model=self.config.get("model"),
            provider=self.config.get("provider"),
            effort=self.config.get("effort"))

    def request(self, method, params, timeout):
        return self._server.request(method, params, timeout)

    def notify(self, method, params):
        return self._server.notify(method, params)

    def reply(self, identifier, result):
        return self._server.reply(identifier, result)

    def next_message(self, timeout):
        return self._server.next_message(timeout)

    def close(self):
        if hasattr(self, "_server"):
            self._server.close()


class ApiGatewayTransport(Transport):
    """路线 B：API/网关（HTTP），无执行工具能力时由 Mac runner 执行工具。

    需要用户提供：endpoint、api_key_env（环境变量名，不收裸密钥）、model、
    effort、限流/重试策略（rate_limit_s 可选，默认 0）。
    """

    route = "api_gateway"
    REQUIRED = ("endpoint", "api_key_env", "model", "effort")

    def missing_config(self):
        missing = [k for k in self.REQUIRED if not self.config.get(k)]
        env = self.config.get("api_key_env")
        if env and not os.environ.get(env):
            missing.append(f"environment variable {env}")
        return missing

    def connect(self):
        super().connect()
        self._events = queue.Queue()
        self._rate_limit_s = float(self.config.get("rate_limit_s", 0) or 0)
        self._last_request_t = 0.0
        self.last_result = None

    def request(self, method, params, timeout):
        """POST <endpoint>：body={method,params,model,effort}，Bearer 凭据
        取自 api_key_env 环境变量（密钥不落盘不记日志）。响应 dict 回传。

        P1-5 修正：不再把每个响应自动包装成 turn/completed 事件（此前泵
        首个 next_message 即误取 completed 而终止）。route B 的事件流由
        ingest_response(result) 按网关返回的工具调用形状转换——真实包络
        字段在联调（用户提供凭据）时核对并钉定。"""
        key = os.environ[self.config["api_key_env"]]
        payload = json.dumps({
            "method": method, "params": params,
            "model": self.config["model"], "effort": self.config["effort"],
        }).encode()
        wait = self._rate_limit_s - (time.time() - self._last_request_t)
        if wait > 0:
            time.sleep(wait)
        self._last_request_t = time.time()
        req = urllib.request.Request(
            self.config["endpoint"], data=payload, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode())
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"api gateway request {method} failed: HTTP {error.code}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"api gateway request {method} failed: {error.reason}") from error
        self.last_result = result
        return result

    def ingest_response(self, result, *, tool_call_field="tool_calls"):
        """网关响应 → 泵事件。联调钉定前的约定形状：
        {"tool_calls": [{"id","name"/"tool","arguments"}], "done": bool}；
        done=True 或无工具调用 → turn/completed。"""
        calls = result.get(tool_call_field) if isinstance(result, dict) else None
        if not calls:
            self._events.put({"method": "turn/completed", "result": result})
            return 0
        for i, call in enumerate(calls):
            self._events.put({"method": "item/tool/call", "id": call.get("id", i),
                              "params": {"tool": call.get("tool", call.get("name")),
                                         "arguments": call.get("arguments")}})
        if result.get("done"):
            self._events.put({"method": "turn/completed", "result": result})
        return len(calls)

    def next_message(self, timeout):
        try:
            message = self._events.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError(
                "Timed out waiting for a gateway event") from error
        return message

    def close(self):
        pass


# ---- 路线 A 的 JSON-RPC stdio 机制（移植 upstream StdioAppServer，逐字语义） ----

class _StdioJsonRpc:
    """Small JSON-RPC transport for a single local Codex app-server process."""

    def __init__(self, argv, workspace, *, popen=subprocess.Popen,
                 model=None, provider=None, effort=None):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._incoming = queue.Queue()
        self._responses = {}
        self._response_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._next_id = 1
        self._closed = False
        self._stdout_closed = threading.Event()
        self._in_log = (self.workspace / "rpc_in.jsonl").open("a", buffering=1)
        self._out_log = (self.workspace / "rpc_out.jsonl").open("a", buffering=1)
        self._stderr_log = (self.workspace / "stderr.log").open("a", buffering=1)
        agent_cwd = self.workspace / 'agent'
        self.process = popen(argv, cwd=agent_cwd if agent_cwd.is_dir() else self.workspace,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)
        if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
            raise RuntimeError("Codex app-server stdio pipes are unavailable")
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stdout(self):
        try:
            for line in self.process.stdout:
                self._out_log.write(line)
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as error:
                    self._incoming.put(RuntimeError(f"Non-JSON Codex app-server output: {error}"))
                    continue
                identifier = message.get("id")
                if identifier is not None and "method" not in message:
                    with self._response_lock:
                        destination = self._responses.get(identifier)
                    if destination is not None:
                        destination.put(message)
                        continue
                self._incoming.put(message)
        finally:
            self._stdout_closed.set()
            # Startup/config errors close stdout before replying to initialize.
            # Wake outstanding RPCs immediately instead of waiting their full timeout.
            with self._response_lock:
                for destination in self._responses.values():
                    try:
                        destination.put_nowait(dict(error=dict(message='Codex app-server stdout closed; inspect stderr.log')))
                    except queue.Full:
                        pass
            self._incoming.put(EOFError("Codex app-server stdout closed"))

    def _read_stderr(self):
        for line in self.process.stderr:
            self._stderr_log.write(line)

    def _send(self, message):
        if self._closed or self._stdout_closed.is_set() or self.process.poll() is not None:
            raise RuntimeError("Codex app-server is not running")
        line = json.dumps(message, separators=(",", ":")) + "\n"
        with self._write_lock:
            self._in_log.write(line)
            self.process.stdin.write(line)
            self.process.stdin.flush()

    def request(self, method, params, timeout):
        with self._response_lock:
            identifier = self._next_id
            self._next_id += 1
            destination = queue.Queue(maxsize=1)
            self._responses[identifier] = destination
        try:
            self._send({"jsonrpc": "2.0", "id": identifier, "method": method, "params": params})
            try:
                response = destination.get(timeout=timeout)
            except queue.Empty as error:
                raise TimeoutError(f"Timed out waiting for Codex RPC {method}") from error
            if "error" in response:
                raise RuntimeError(f"Codex RPC {method} failed: {response['error']}")
            return response.get("result")
        finally:
            with self._response_lock:
                self._responses.pop(identifier, None)

    def notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def reply(self, identifier, result):
        self._send({"jsonrpc": "2.0", "id": identifier, "result": result})

    def next_message(self, timeout):
        try:
            message = self._incoming.get(timeout=timeout)
        except queue.Empty as error:
            raise TimeoutError("Timed out waiting for a Codex turn event") from error
        if isinstance(message, BaseException):
            raise message
        return message

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        self._stdout_thread.join(timeout=2)
        self._stderr_thread.join(timeout=2)
        for stream in (self._in_log, self._out_log, self._stderr_log):
            stream.close()


def dispatch_tool_call(name, arguments, handlers, audit_sink=None, call_index=0):
    """工具调用分发（语义对齐 upstream run.py 循环体：JSON 参数、未知工具拒绝、审计落盘）。"""
    if audit_sink is not None:
        audit_sink(dict(call_index=call_index, tool=name, arguments=arguments))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValueError(f"Tool arguments must be a JSON object: {error}") from error
    if name not in handlers:
        raise ValueError("Unknown host service; native Codex tools are executed by app-server")
    return handlers[name](**arguments)

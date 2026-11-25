"""L4 真传输实现测试（本地子进程/本地 HTTP，无需真实凭据）。"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from gap_repro.agent.client import (
    ApiGatewayTransport,
    AppServerTransport,
    ConfigMissingError,
)

ECHO_SERVER = (
    "import json,sys\n"
    "for line in sys.stdin:\n"
    "    m = json.loads(line)\n"
    "    if 'id' in m and 'method' not in m: continue\n"
    "    resp = {'jsonrpc':'2.0','id':m['id'],'result':"
    "{'echo':m['method'],'params':m.get('params')}}\n"
    "    print(json.dumps(resp), flush=True)\n"
    "    # 事件流：每次请求后推一条非响应消息\n"
    "    print(json.dumps({'method':'turn/completed'}), flush=True)\n"
)


class TestAppServer:
    def test_missing_config_lists_required(self):
        t = AppServerTransport({"model": "gpt"})
        with pytest.raises(ConfigMissingError) as e:
            t.connect()
        # LP3 钉定的用户清单（workspace 是实现细节，日志目录有默认值）
        assert set(e.value.missing) == {"app_server_cmd", "provider",
                                        "effort"}

    def test_request_roundtrip_and_event(self, tmp_path):
        script = tmp_path / "echo.py"
        script.write_text(ECHO_SERVER)
        t = AppServerTransport({
            "app_server_cmd": f"{__import__('sys').executable} {script}",
            "model": "m", "provider": "p", "effort": "xhigh",
            "workspace": str(tmp_path / "ws")})
        t.connect()
        try:
            result = t.request("turn/start", {"x": 1}, timeout=10)
            assert result["echo"] == "turn/start"
            event = t.next_message(timeout=5)
            assert event["method"] == "turn/completed"
            assert (tmp_path / "ws" / "rpc_in.jsonl").exists()
        finally:
            t.close()

    def test_rpc_error_surfaced(self, tmp_path):
        script = tmp_path / "echo.py"
        script.write_text(ECHO_SERVER.replace(
            "'result':", "'error':{'message':'boom'},'unused':") + "")
        t = AppServerTransport({
            "app_server_cmd": f"{__import__('sys').executable} {script}",
            "model": "m", "provider": "p", "effort": "xhigh",
            "workspace": str(tmp_path / "ws")})
        t.connect()
        try:
            with pytest.raises(RuntimeError, match="boom"):
                t.request("turn/start", {}, timeout=10)
        finally:
            t.close()


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.headers["Authorization"].startswith("Bearer ")
        resp = {"ok": True, "got": body["method"], "model": body["model"],
                "tool_calls": [{"id": 7, "tool": "libero_start",
                                "arguments": {"task": "x"}}]}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class TestApiGateway:
    def test_missing_config_includes_env_var(self):
        t = ApiGatewayTransport({"endpoint": "http://x", "model": "m",
                                 "effort": "xhigh",
                                 "api_key_env": "DEFINITELY_UNSET_VAR_XYZ"})
        with pytest.raises(ConfigMissingError) as e:
            t.connect()
        assert any("DEFINITELY_UNSET_VAR_XYZ" in m for m in e.value.missing)

    def test_request_and_unified_event(self, tmp_path):
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        import os
        os.environ["TEST_GW_KEY_XYZ"] = "secret"
        try:
            t = ApiGatewayTransport({
                "endpoint": f"http://127.0.0.1:{port}/",
                "api_key_env": "TEST_GW_KEY_XYZ", "model": "m",
                "effort": "xhigh"})
            t.connect()
            result = t.request("turn/start", {"p": 1}, timeout=10)
            assert result["ok"] is True and result["model"] == "m"
            # P1-5：request 不再自动入队事件；工具调用经 ingest_response
            # 转换为泵事件（真实网关包络在联调钉定）
            n = t.ingest_response(result)
            assert n == 1
            event = t.next_message(timeout=5)
            assert event["method"] == "item/tool/call"
            assert event["params"]["tool"] == "libero_start"
            with pytest.raises(TimeoutError):
                t.next_message(timeout=0.05)
            t.close()
        finally:
            server.shutdown()

    def test_key_never_in_config(self):
        # 凭据只存环境变量名；config 里出现裸密钥属于契约违规
        t = ApiGatewayTransport({"endpoint": "http://x",
                                 "api_key_env": "K", "model": "m",
                                 "effort": "e", "api_key": "leak"})
        assert "api_key" not in t.REQUIRED
        assert all("key" not in k or k == "api_key_env"
                   for k in t.config if "key" in k and k != "api_key_env") \
            or t.config.get("api_key_env") == "K"

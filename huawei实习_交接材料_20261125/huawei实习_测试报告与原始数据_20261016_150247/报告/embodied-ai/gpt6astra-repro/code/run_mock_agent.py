"""Phase 1a: 零成本验证完整 agent loop。

内置一个 OpenAI 兼容的 mock LLM 服务器（stdlib 实现，无外部依赖），
它解析观测文本里的 state[eef_pos]/state[cube_pos]，通过 move_by 工具
调用闭环走向方块——完整走通 chat wire、工具调用、插值限速、日志链路，
不需要任何 API key。

用法: .venv/bin/python run_mock_agent.py
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from inspect_robots import eval
from inspect_robots.mock import CubePickEmbodiment
from inspect_robots.scene import Scene
from inspect_robots.scorer import episode_length, success_at_end
from inspect_robots.task import Epochs, Task
from inspect_robots_agent import LLMAgentPolicy

PORT = 8877

_STATE_RE = re.compile(r"state\[([^\]]+)\]: \[([^\]]*)\]")


def _floats(raw: str) -> list[float]:
    return [float(v) for v in raw.split(",") if v.strip()]


class MockAgent:
    """从观测文本提取状态，输出 move_by 工具调用。"""

    def __init__(self) -> None:
        self.calls = 0

    def respond(self, body: dict) -> dict:
        self.calls += 1
        # 1) 从 tools schema 找 move 工具名、参数 key、维度名
        move_name, values_key, dims = "move_by", "deltas", ["x", "y"]
        for tool in body.get("tools") or []:
            fn = tool.get("function", {})
            if fn.get("name", "").startswith("move"):
                move_name = fn["name"]
                props = fn.get("parameters", {}).get("properties", {})
                for key in ("targets", "deltas"):
                    if key in props:
                        values_key = key
                        m = re.search(r"Valid names: ([\w., ]+)", props[key].get("description", ""))
                        if m:
                            dims = [d.strip() for d in m.group(1).split(",")]
                        break
                break
        # 2) 从最近的含 state 的消息解析状态
        state: dict[str, list[float]] = {}
        for msg in reversed(body.get("messages") or []):
            content = msg.get("content")
            texts: list[str] = []
            if isinstance(content, str):
                texts = [content]
            elif isinstance(content, list):
                texts = [p.get("text", "") for p in content if isinstance(p, dict)]
            joined = "\n".join(texts)
            found = dict(_STATE_RE.findall(joined))
            if found:
                state = {k: _floats(v) for k, v in found.items()}
                break
        eef = state.get("eef_pos")
        cube = state.get("cube_pos")
        if eef is None or cube is None or len(eef) != len(cube) or len(eef) != len(dims):
            call = {
                "name": "give_up",
                "arguments": json.dumps(
                    {"reason": f"mock: state unavailable ({sorted(state)})", "hindsight": "none"}
                ),
            }
        else:
            deltas = {}
            for i, dim in enumerate(dims):
                d = cube[i] - eef[i]
                deltas[dim] = round(max(-0.03, min(0.03, d)), 4)
            call = {
                "name": move_name,
                "arguments": json.dumps(
                    {
                        values_key: deltas,
                        "note": f"mock agent: eef={eef} cube={cube}, stepping {deltas}",
                    }
                ),
            }
        return {
            "id": f"chatcmpl-mock-{self.calls}",
            "object": "chat.completion",
            "model": body.get("model", "mock"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None, "tool_calls": [
                        {"id": f"call-{self.calls}", "type": "function", "function": call}
                    ]},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }


class Handler(BaseHTTPRequestHandler):
    agent = MockAgent()

    def do_POST(self):  # noqa: N802
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        payload = self.agent.respond(body)
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # 静音默认访问日志
        pass


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[mock] LLM server on http://127.0.0.1:{PORT}/v1")

    task = Task(
        name="cubepick-mock-agent-loop",
        scenes=[Scene(id=f"layout-{i}", instruction="reach the cube", init_seed=i) for i in range(3)],
        scorer=[success_at_end(), episode_length()],
        max_steps=80,
        epochs=Epochs(count=1, reducer="mean"),
    )
    # 参数对齐 RoboCurve agent policy：20 次调用预算 + 25% 速度上限 + medium effort
    policy = LLMAgentPolicy(
        model="mock-solver",
        base_url=f"http://127.0.0.1:{PORT}/v1",
        api_key_env="MOCK_LLM_KEY",
        max_llm_calls=20,
        max_speed_frac=0.25,
        effort="medium",
    )
    (log,) = eval(task, policy, CubePickEmbodiment(), log_dir="logs-mock")
    server.shutdown()

    print(f"\nstatus:  {log.status}")
    print(f"trials:  {log.results.total_trials}   mock LLM calls: {Handler.agent.calls}")
    for name, value in sorted(log.results.metrics.items()):
        print(f"  {name}: {value:.4g}")


if __name__ == "__main__":
    main()

"""L4 模型驱动桥测试（fake 传输脚本化事件，本地全路径）。"""

from __future__ import annotations

import base64
import json
import queue

import numpy as np
import pytest

from gap_repro.libero.gpt_bridge import (
    LiberoRollout,
    run_rollout,
)
from gap_repro.libero.policy import LiberoStudentPolicy
from gap_repro.results import VALID_FAILURE, VALID_SUCCESS
from tests.test_libero_agent_loop import _takeover_response
from tests.test_libero_runner import FakeSession, _FakeTransport, _case


class ScriptedTransport:
    """fake 传输：预置工具调用脚本（模型行为），实现 request/next_message/reply。"""

    def __init__(self, tool_calls):
        self.tool_calls = list(tool_calls)  # [{"name","arguments"}]
        self.events = queue.Queue()
        self.replies = []
        self.config = {"model": "m", "provider": "p", "effort": "xhigh"}

    def request(self, method, params, timeout):
        if method == "initialize":
            return {}
        if method == "thread/start":
            return {"thread": {"id": "t1"}}
        if method == "turn/start":
            for i, call in enumerate(self.tool_calls):
                self.events.put({"method": "item/tool/call", "id": 100 + i,
                                 "params": {"threadId": "t1",
                                            "tool": call["name"],
                                            "arguments": call["arguments"]}})
            return {"turn": {"id": "u1"}}
        raise AssertionError(f"unexpected request {method}")

    def notify(self, method, params):
        pass

    def next_message(self, timeout):
        return self.events.get(timeout=timeout)

    def reply(self, identifier, result):
        self.replies.append((identifier, result))
        if identifier == 100 + len(self.replies) - 1 \
                and not self.events.queue:
            self.events.put({"method": "turn/completed"})

    def close(self):
        pass


def _obs_packet_ok(reply):
    assert reply["success"] is True
    items = reply["contentItems"]
    assert items[0]["type"] == "inputText"  # upstream content_items 形状
    assert any(i["type"] == "inputImage" for i in items)
    uri = next(i["imageUrl"] for i in items if i["type"] == "inputImage")
    assert uri.startswith("data:image/png;base64,")
    base64.b64decode(uri.split(",", 1)[1])
    return json.loads(items[0]["text"])


class TestDirectRollout:
    def test_full_direct_episode(self):
        # 模型脚本：start → act(向目标) → act 后到达预算 → 终局失败
        case = _case(max_steps=5)
        rollout = LiberoRollout(case, session=FakeSession())
        transport = ScriptedTransport([
            {"name": "libero_start",
             "arguments": {"task": case.instruction}},
            {"name": "libero_act",
             "arguments": {"response": dict(_takeover_response(),
                                            steps=3)}},
        ])
        # max_steps=5：start 后一次 act 3 步（5 步预算内）→ 返回观测包；
        # 脚本耗尽 turn/completed → 模型弃驶 → infra（不伪造终局）
        with pytest.raises(RuntimeError, match="abandoned"):
            run_rollout(rollout, transport, method="gpt_only")

    def test_direct_budget_failure(self):
        case = _case(max_steps=5)
        rollout = LiberoRollout(case, session=FakeSession())
        transport = ScriptedTransport([
            {"name": "libero_start", "arguments": {"task": case.instruction}},
            {"name": "libero_act",
             "arguments": {"response": dict(_takeover_response(), steps=5)}},
        ])
        status, record = run_rollout(rollout, transport, method="gpt_only")
        assert status == VALID_FAILURE
        assert record["success"] is False
        assert record["done_step"] == 5  # 预算耗尽

    def test_input_error_reply_not_fatal(self):
        case = _case(max_steps=3)
        rollout = LiberoRollout(case, session=FakeSession())
        transport = ScriptedTransport([
            {"name": "libero_start", "arguments": {"task": "wrong task"}},
            {"name": "libero_act",
             "arguments": {"response": dict(_takeover_response(), steps=3)}},
        ])
        # start 被拒 → act 因未 start 亦被拒（协议守卫）→ 模型弃驶 → infra
        with pytest.raises(RuntimeError, match="abandoned"):
            run_rollout(rollout, transport, method="gpt_only")
        assert rollout.rejected_calls == 2
        # 被拒调用的 reply success=False 且携带可修正反馈
        first_reply = transport.replies[0][1]
        assert first_reply["success"] is False
        assert "input_error" in first_reply["contentItems"][0]["text"]


class TestHybridRollout:
    def test_full_hybrid_cycle(self):
        case = _case(max_steps=12)
        rollout = LiberoRollout(
            case, policy=LiberoStudentPolicy(_FakeTransport()),
            session=FakeSession())
        def exec_resp(step0):
            return {"response": {
                "mode": "student",
                "reason": "first decision" if step0 else "continuing",
                "assessment": {"task_progress": {
                    "verified_completed": [], "currently_attempting": "x",
                    "remaining": ["y"]},
                    "current_subgoal": "x",
                    "execution_status": "not_started" if step0 else "progressing",
                    "execution_evidence": "episode started" if step0 else "moving",
                    "expected_next_intent": "grasp",
                    "predicted_next_intent": "grasp",
                    "intent_status": "aligned",
                    "intent_evidence": "proposal toward handle"}}}
        calls = [{"name": "libero_start",
                  "arguments": {"task": case.instruction}}]
        for i in range(4):  # 3 轮 × 5 步 = 15 > 12 预算 → 预算终局
            calls.append({"name": "pi05_infer", "arguments": {}})
            calls.append({"name": "libero_execute",
                          "arguments": exec_resp(i == 0)})
        transport = ScriptedTransport(calls)
        status, record = run_rollout(rollout, transport, method="pi05_plus_gpt")
        assert status == VALID_FAILURE  # 12 步预算耗尽（未 done）
        chunks = record["chunks"]
        assert chunks[0]["origin"] == "student"
        assert rollout.rejected_calls == 0

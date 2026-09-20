"""L2 LIBERO GPT 响应 schema 测试：结构对齐 upstream + F08 单臂防泄漏。"""

from __future__ import annotations

import pytest

from gap_repro.agent.contract import dumps as cdumps
from gap_repro.libero.agent_contract import (
    libero_direct_response_schema,
    libero_response_schema,
    libero_tool_specs,
    validate_mode_steps,
)


class TestSingleArm:
    def test_no_dual_arm_wrapper(self):
        # F08：edit/target 必须是裸对象，不得出现 left/right 包装
        s = libero_response_schema()
        for field in ("edit", "target"):
            props = s["properties"][field]["properties"]
            assert "left" not in props and "right" not in props
        t = libero_direct_response_schema()
        assert "left" not in t["properties"]["target"]["properties"]

    def test_target_fields_match_paper(self):
        t = libero_response_schema()["properties"]["target"]
        assert set(t["required"]) == {"position", "quaternion_wxyz",
                                      "gripper_closed"}
        assert set(t["properties"]) == {"position", "quaternion_wxyz",
                                        "gripper_closed"}

    def test_assessment_structure_verbatim(self):
        # assessment 与 upstream 域无关字段逐字段一致（键集合断言）
        s = libero_response_schema()["properties"]["assessment"]
        assert set(s["required"]) == {
            "task_progress", "current_subgoal", "execution_status",
            "execution_evidence", "expected_next_intent",
            "predicted_next_intent", "intent_status", "intent_evidence"}
        assert set(s["properties"]["execution_status"]["enum"]) == {
            "not_started", "progressing", "failed", "uncertain", "recovered"}
        assert set(s["properties"]["intent_status"]["enum"]) == {
            "aligned", "misaligned", "uncertain"}


class TestStepBounds:
    def test_student_max_5(self):
        s = libero_response_schema()["properties"]["steps"]
        assert s["maximum"] == 5  # LIBERO replan=5（RoboDojo 15 不移植）

    def test_validate_mode_steps(self):
        validate_mode_steps("student", 5)
        validate_mode_steps("eef", 5)
        validate_mode_steps("edit", 3)
        with pytest.raises(ValueError, match="at most 5"):
            validate_mode_steps("student", 6)
        with pytest.raises(ValueError, match="unknown mode"):
            validate_mode_steps("hybrid", 1)
        with pytest.raises(ValueError, match="positive"):
            validate_mode_steps("edit", 0)
        with pytest.raises(ValueError, match="positive"):
            validate_mode_steps("edit", True)


class TestDirectSchema:
    def test_eef_only(self):
        d = libero_direct_response_schema()
        assert d["properties"]["mode"]["enum"] == ["eef"]
        assert d["properties"]["steps"]["maximum"] == 5
        assert set(d["required"]) == {"request_id", "mode", "steps", "reason",
                                      "target"}


class TestToolSpecs:
    def test_hybrid_tools(self):
        tools = libero_tool_specs("pi05_plus_gpt")
        names = [t["name"] for t in tools]
        assert names == ["libero_start", "pi05_infer", "libero_execute"]
        ex = tools[2]
        assert ex["inputSchema"]["properties"]["response"] == \
            libero_response_schema()

    def test_direct_tools(self):
        tools = libero_tool_specs("gpt_only")
        names = [t["name"] for t in tools]
        assert names == ["libero_start", "libero_act"]
        act = tools[1]
        desc = act["description"]
        assert "dual-arm" not in desc  # F08 文案防泄漏
        assert act["inputSchema"]["properties"]["response"] == \
            libero_direct_response_schema()

    def test_every_tool_closes_schema(self):
        for m in ("pi05_plus_gpt", "gpt_only"):
            for t in libero_tool_specs(m):
                assert t["inputSchema"]["additionalProperties"] is False
                assert t["type"] == "function"


def test_canonical_dump_matches_contract_helper():
    # 与 RoboDojo 侧同一规范序列化（审计对账同源）
    s = libero_response_schema()
    assert cdumps(s) == cdumps(libero_response_schema())

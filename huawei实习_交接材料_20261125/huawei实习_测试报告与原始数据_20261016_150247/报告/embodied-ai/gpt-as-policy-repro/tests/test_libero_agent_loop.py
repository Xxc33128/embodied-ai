"""L4 假传输闭环测试：Direct/Hybrid 模式、接管 gate、交还、白名单。"""

from __future__ import annotations

import numpy as np
import pytest

from gap_repro.agent.gate import OBSERVATION_KEYS, build_observation_packet
from gap_repro.libero.agent_loop import (
    FakeGptClient,
    _ok_assessment,
    run_hybrid_episode,
)
from gap_repro.libero.policy import CaseSpec, LiberoStudentPolicy
from gap_repro.libero.runner import run_student_episode  # noqa: F401 (对账用)
from gap_repro.results import VALID_FAILURE, VALID_SUCCESS
from tests.test_libero_runner import FakeSession, _FakeTransport, _case


def _policy():
    return LiberoStudentPolicy(_FakeTransport())


def _hyb_case(**kw):
    return _case(**kw)


def _assess(**over):
    a = _ok_assessment(False)  # 决策点均在 step>0
    a.update(over)
    return a


def _student(step0):
    return {"mode": "student", "reason": "student continues",
            "assessment": _ok_assessment(step0)}


def _edit_response():
    return {"mode": "edit", "steps": 2, "reason": "object dropped, adjust",
            "edit": {"delta_position": [0.02, 0.0, 0.0],
                     "delta_rotation_vector": [0.0, 0.1, 0.0],
                     "gripper": "closed"},
            "assessment": _assess(execution_status="failed",
                                  execution_evidence="object dropped")}


def _takeover_response():
    # FakeSession 静态 EEF [0.4,0.4,0.4]、姿态 (x,y,z,w)=(0,0,1,0)：
    # 目标须在决策界内（0.055 m / 0.425 rad，P0-1 实测尺度）
    return {"mode": "eef", "steps": 5, "reason": "correct the trajectory",
            "target": {"position": [0.42, 0.38, 0.36],
                       "quaternion_wxyz": [0.0, 0.0, 1.0, 0.0],
                       "gripper_closed": True},
            "assessment": _assess(intent_status="misaligned",
                                  intent_evidence="moving away from drawer")}


class TestApprovePath:
    def test_default_student_cycles(self):
        result, rec = run_hybrid_episode(
            _hyb_case(max_steps=15), policy=_policy(),
            gpt_client=FakeGptClient(), session=FakeSession())
        assert result.status == VALID_FAILURE  # 未 done → 预算失败
        assert all(c["gpt_mode"] == "student" for c in rec["cycles"])
        assert all(c["origin"] == "student" for c in rec["chunks"])
        assert rec["stop_requested"] is False

    def test_packet_whitelist_exact(self):
        gpt = FakeGptClient()
        run_hybrid_episode(_hyb_case(max_steps=5), policy=_policy(),
                           gpt_client=gpt, session=FakeSession())
        rid, packet, feedback = gpt.packets[0]
        assert set(packet) == set(OBSERVATION_KEYS)
        assert packet["instruction"] == "open the drawer"
        assert packet["executed_gripper_closed"] is None  # 尚未执行
        rid, pkt, feedback = gpt.packets[0]
        assert feedback is None  # 首次评估无反馈


class TestGate:
    def test_illegal_takeover_gets_feedback_then_corrects(self):
        # aligned+progressing 的 edit → gate 拒绝 → 同会话反馈修正（L4）；
        # 修正响应是合法 student → episode 继续
        bad = {"mode": "edit", "steps": 2,
               "edit": {"delta_position": [0.01, 0, 0],
                        "delta_rotation_vector": [0, 0, 0],
                        "gripper": "keep"},
               "assessment": _assess()}  # aligned/progressing
        gpt = FakeGptClient([bad, _student(True)])
        result, rec = run_hybrid_episode(_hyb_case(max_steps=15),
                                         policy=_policy(), gpt_client=gpt,
                                         session=FakeSession())
        # envelope 检查（request_id/reason）先于 gate——与 upstream
        # validate_response 的检查顺序一致
        assert rec["assess_retries"][0]["error"].startswith(
            "A visible observation/decision explanation is required")
        assert gpt.packets[1][2] is not None  # 第二次 assess 携带反馈
        assert all(c["gpt_mode"] in ("student",) for c in rec["cycles"])

    def test_model_never_corrects_is_infra(self):
        bad = {"mode": "edit", "steps": 2,
               "edit": {"delta_position": [0.01, 0, 0],
                        "delta_rotation_vector": [0, 0, 0],
                        "gripper": "keep"},
               "assessment": _assess()}
        result, rec = run_hybrid_episode(_hyb_case(max_steps=15),
                                         policy=_policy(),
                                         gpt_client=FakeGptClient([bad, bad]),
                                         session=FakeSession())
        assert result.status == "infrastructure_incomplete"
        assert "failed to correct" in rec["infra_error"]
        assert len(rec["assess_retries"]) == 2


class TestEditPath:
    def test_edit_executed_and_recorded(self):
        # edit 不能发生在 step0（gate：无执行证据即无接管）→ 先 student 一轮
        result, rec = run_hybrid_episode(
            _hyb_case(max_steps=15), policy=_policy(),
            gpt_client=FakeGptClient([_student(True), _edit_response()]),
            session=FakeSession())
        first = rec["chunks"][1]  # chunks[0] 是首个 student cycle
        assert first["origin"] == "edit(execution_failure)"
        assert rec["cycles"][1]["gpt_mode"] == "edit"
        assert rec["cycles"][1]["edit_info"]["translation_m"] == pytest.approx(0.02)
        # 交还：后续 cycle 回到 student
        assert rec["cycles"][2]["origin"] == "student"


class TestTakeoverPath:
    def test_takeover_then_handback(self):
        result, rec = run_hybrid_episode(
            _hyb_case(max_steps=25), policy=_policy(),
            gpt_client=FakeGptClient([_student(True), _takeover_response()]),
            session=FakeSession())
        first = rec["chunks"][1]
        assert first["origin"] == "takeover(wrong_intent)"
        assert rec["cycles"][1]["takeover_info"]["steps"] == 5
        assert rec["cycles"][2]["origin"] == "student"  # 交还

    def test_done_during_takeover_is_success(self):
        sess = FakeSession(succeed_at=16)  # student chunk(10-14) 后接管段 done
        result, rec = run_hybrid_episode(
            _hyb_case(max_steps=25), policy=_policy(),
            gpt_client=FakeGptClient([_student(True), _takeover_response()]),
            session=sess)
        assert result.status == VALID_SUCCESS
        assert result.native_score == 1.0


class TestStopPath:
    def test_stop_ends_early_native_verdict(self):
        stop = {"mode": "stop", "reason": "task state is stable",
                "assessment": _assess(execution_status="recovered",
                                      execution_evidence="task state stable")}
        result, rec = run_hybrid_episode(
            _hyb_case(max_steps=50), policy=_policy(),
            gpt_client=FakeGptClient([_student(True), stop]),
            session=FakeSession(succeed_at=None))
        # FakeSession.succeed_at=None → is_success False → stop 终局失败
        assert result.status == VALID_FAILURE
        assert rec["stop_requested"] is True
        assert rec["success"] is False

    def test_stop_with_true_predicate_is_success(self):
        stop = {"mode": "stop", "reason": "task state is stable",
                "assessment": _assess(execution_status="recovered",
                                      execution_evidence="task state stable")}
        sess = FakeSession(succeed_at=None)
        sess.is_success = lambda: True  # 终态原生判据成立
        result, rec = run_hybrid_episode(
            _hyb_case(max_steps=50), policy=_policy(),
            gpt_client=FakeGptClient([_student(True), stop]), session=sess)
        assert result.status == VALID_SUCCESS
        assert rec["stop_requested"] is True


def test_observation_packet_rejects_truth_keys():
    with pytest.raises(Exception, match="non-whitelisted"):
        build_observation_packet(step_id=0, bddl_answer="the drawer")


class TestDirectPath:
    """Direct（gpt_only）：mode 枚举仅 eef（upstream direct schema）；
    集终局 = done 或预算；非 eef 响应走同会话修正。"""

    def test_direct_runs_and_records(self):
        from gap_repro.libero.agent_loop import run_direct_episode
        gpt = FakeGptClient([_takeover_response()])  # 合法 eef 响应（5 步）
        result, rec = run_direct_episode(_hyb_case(max_steps=20),
                                         gpt_client=gpt,
                                         session=FakeSession())
        assert rec["method"] == "direct"
        assert all(c["origin"] == "direct" for c in rec["chunks"])
        assert rec["cycles"][0]["takeover_info"]["steps"] == 5
        # 无 stop 概念
        assert rec["stop_requested"] is False

    def test_direct_rejects_student_mode_with_correction(self):
        from gap_repro.libero.agent_loop import run_direct_episode
        gpt = FakeGptClient([_student(True), _takeover_response()])
        result, rec = run_direct_episode(_hyb_case(max_steps=20),
                                         gpt_client=gpt,
                                         session=FakeSession())
        assert rec["assess_retries"][0]["error"].startswith(
            "direct route requires mode 'eef'")
        assert rec["cycles"][0]["gpt_mode"] == "eef"  # 修正后执行

    def test_direct_budget_failure(self):
        from gap_repro.libero.agent_loop import run_direct_episode
        result, rec = run_direct_episode(_hyb_case(max_steps=10),
                                         gpt_client=FakeGptClient(default="eef"),
                                         session=FakeSession())
        assert result.status == VALID_FAILURE
        assert result.native_score == 0.0

    def test_direct_success(self):
        from gap_repro.libero.agent_loop import run_direct_episode
        sess = FakeSession(succeed_at=2)  # 首个决策段内 done（控制步号）
        result, rec = run_direct_episode(_hyb_case(max_steps=20),
                                         gpt_client=FakeGptClient(
                                             [_takeover_response()]),
                                         session=sess)
        assert result.status == VALID_SUCCESS

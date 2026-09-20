"""L3 runner 学生路径测试（fake session/policy，本地无环境）。"""

from __future__ import annotations

import numpy as np
import pytest

from gap_repro.libero.policy import CaseSpec, LiberoStudentPolicy
from gap_repro.libero.runner import run_student_episode
from gap_repro.results import (
    INFRA_INCOMPLETE,
    VALID_FAILURE,
    VALID_SUCCESS,
)


def _case(**kw):
    base = dict(campaign_id="c", episode_id="e0", benchmark="libero_goal",
                condition="Ori", task_index=0, task_name="t",
                instruction="open the drawer", init_index=0, max_steps=20,
                method="student_only")
    base.update(kw)
    return CaseSpec(**base)


class FakeSession:
    """openpi 语义仿真：wait 阶段沉降；每 chunk 第 3 步可选 done。"""

    def __init__(self, *, succeed_at=None, fail_reset=False):
        self.t = 0
        self.replans = 0
        self.succeed_at = succeed_at      # 全局步号（含 wait）触发 done
        self.fail_reset = fail_reset
        self.n_init_states_n = 50
        self.steps_taken = 0

    @property
    def init_states(self):
        return [object()] * self.n_init_states_n

    def reset_to(self, index, wait_steps=10, fixture_poses=None):
        if self.fail_reset:
            raise RuntimeError("sim exploded on reset")
        self.t = 0
        self.steps_taken = 0
        return {"agentview_image": np.zeros((4, 4, 3), np.uint8),
                "robot0_eye_in_hand_image": np.zeros((4, 4, 3), np.uint8),
                "robot0_eef_pos": np.array([0.4, 0.4, 0.4]),  # 与 step 一致
                "robot0_eef_quat": np.array([0.0, 1.0, 0.0, 0.0]),
                "robot0_gripper_qpos": np.array([0.02, -0.02])}

    def step(self, action):
        self.t += 1
        self.steps_taken += 1
        done = (self.succeed_at is not None and self.t >= self.succeed_at)
        return ({"agentview_image": np.zeros((4, 4, 3), np.uint8),
                 "robot0_eye_in_hand_image": np.zeros((4, 4, 3), np.uint8),
                 "robot0_eef_pos": np.array([0.4, 0.4, 0.4]),  # 近静止仿真
                 "robot0_eef_quat": np.array([0.0, 1.0, 0.0, 0.0]),
                 "robot0_gripper_qpos": np.array([0.02, -0.02])},
                0.0, done, {})

    def is_success(self):
        return self.succeed_at is not None

    def fingerprint_state(self, obs):
        return f"fp{self.t}"

    def fingerprint_full(self, obs):
        return f"fpf{self.t}"


def _policy():
    return LiberoStudentPolicy(_FakeTransport())


class _FakeTransport:
    def __init__(self):
        self.n_calls = 0

    def infer(self, element, request_id):
        self.n_calls += 1
        actions = np.full((10, 7), 0.01, dtype=np.float32)
        actions[:, 6] = -1.0
        return actions, {"device": "fake", "noise_mode": "internal"}


def _run(session, case=None, **kw):
    case = case or _case()
    return run_student_episode(case, policy=_policy(), session=session, **kw)


class TestHappyPath:
    def test_success_before_budget(self):
        # P1-1 修正：wait 只在 reset_to 内（fake 不计步）→ 控制步号
        result, rec = _run(FakeSession(succeed_at=3))
        assert result.status == VALID_SUCCESS
        assert result.native_score == 1.0
        assert rec["success"] is True
        assert rec["done_step"] == 3

    def test_replan_every_5_steps(self):
        # max20 控制步；重规划发生在 0,5,10,15（4 次）
        result, rec = _run(FakeSession())
        assert rec["done_step"] == 20  # 未 done → 记满预算
        assert len(rec["chunks"]) == 4

    def test_entered_policy_stage(self):
        result, rec = _run(FakeSession())
        assert result.entered_policy_stage is True
        assert result.init_fingerprint == "fp0"

    def test_chunk_records_identity_and_span(self):
        _, rec = _run(FakeSession(succeed_at=3))
        c = rec["chunks"][0]
        assert c["request_id"] == "c|e0|r0"
        assert c["t_start"] == 0   # 控制步 0 即首决策（P1-1）
        assert c["t_end"] == 3
        assert c["done_in_chunk"] is True
        assert c["identity"]["noise"] is None
        assert c["identity"]["meta"]["noise_mode"] == "internal"

    def test_final_is_success_check_is_auxiliary(self):
        # 超时未 done：即使 _check_success 为 True 也不翻转判定（done-only）
        result, rec = _run(FakeSession(succeed_at=None))
        assert result.status == VALID_FAILURE
        assert result.native_score == 0.0
        assert rec["final_is_success_check"] is False


class TestFailurePaths:
    def test_reset_failure_is_infra(self):
        result, rec = _run(FakeSession(fail_reset=True))
        assert result.status == INFRA_INCOMPLETE
        assert result.native_score is None
        assert result.entered_policy_stage is False
        assert "sim exploded" in result.reason

    def test_policy_failure_is_infra_no_retry(self):
        sess = FakeSession()
        t = _FakeTransport()
        orig = t.infer

        def boom(element, request_id):
            raise TimeoutError("policy server down")

        t.infer = boom
        result, rec = run_student_episode(
            _case(), policy=LiberoStudentPolicy(t), session=sess)
        assert result.status == INFRA_INCOMPLETE
        assert t.n_calls == 0  # 异常发生在 infer 内部，未产生已执行 chunk
        assert "policy server down" in rec["infra_error"]

    def test_audit_sink_called(self):
        seen = []
        _run(FakeSession(), audit_sink=seen.append)
        assert len(seen) == 1 and seen[0]["schema"] == "gap_repro.lp_runner_episode.v1"


class TestBudget:
    def test_max_steps_from_case_respected(self):
        sess = FakeSession()
        _, rec = _run(sess, case=_case(max_steps=7))
        assert sess.steps_taken == 7  # 控制步（wait 在 reset_to 内，fake 不计）

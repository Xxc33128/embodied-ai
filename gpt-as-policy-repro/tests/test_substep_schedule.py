"""W4 子步调度验收：25Hz ACK → 10×250Hz，插值/夹爪 clip+mimic/支持臂队列/渐进夹爪。

对照原实现（E1）：eval_env.py 插值与支持臂消费、control_manager.py 渐进夹爪。
"""

from __future__ import annotations

import numpy as np
import pytest

from gap_repro.sim import control, robots

ARM_KEY = "left_arm_joint_state"
GRIP_KEY = "left_ee_joint_state"


def make_sched() -> control.SubstepScheduler:
    return control.SubstepScheduler(interpolation_nums=10,
                                    gripper_scale=robots.GRIPPER_SCALE,
                                    mimic=robots.MIMIC[1:])


class TestArmInterpolation:
    def test_alpha_sequence_then_hold(self):
        sched = make_sched()
        seq = sched.make_sequence(
            {ARM_KEY: {"position": [0.9] * 6}},
            current_arm={ARM_KEY: [0.0] * 6}, current_gripper={})
        for i in range(8):  # 前 8 子步 alpha=(i+1)/9
            expect = 0.9 * (i + 1) / 9
            assert seq[i][ARM_KEY]["position"] == pytest.approx([expect] * 6)
        assert seq[8][ARM_KEY]["position"] == pytest.approx([0.9] * 6)
        assert seq[9][ARM_KEY]["position"] == pytest.approx([0.9] * 6)

    def test_interp_count_is_floor_80pct(self):
        assert make_sched().interp_count == 8

    def test_interpolates_from_measured_current(self):
        sched = make_sched()
        seq = sched.make_sequence(
            {ARM_KEY: {"position": [1.0] * 6}},
            current_arm={ARM_KEY: [0.5] * 6}, current_gripper={})
        assert seq[0][ARM_KEY]["position"] == pytest.approx([0.5 + 0.5 / 9] * 6)


class TestGripperInterpolation:
    def test_clip_mimic_and_hold(self):
        sched = make_sched()
        seq = sched.make_sequence(
            {GRIP_KEY: {"position": [1.0]}},  # 超出 scale，目标应被 clip 到 0.044
            current_arm={}, current_gripper={GRIP_KEY: -0.01})
        lo, hi = robots.GRIPPER_SCALE
        for i in range(8):
            alpha = (i + 1) / 9
            v = np.clip(-0.01 + alpha * (1.0 - -0.01), lo, hi)  # 早期值不被 clip，后期被 clip
            assert seq[i][GRIP_KEY]["position"][0] == pytest.approx(v)
            assert seq[i][GRIP_KEY]["position"][1] == pytest.approx(v)  # mimic 1.0*v+0
        assert seq[8][GRIP_KEY]["position"] == pytest.approx([hi, hi])
        assert seq[9][GRIP_KEY]["position"] == pytest.approx([hi, hi])


class TestSupportQueue:
    KEYS = {"franka_arm_joint_state": {"position": [0.1] * 6},
            "franka_ee_joint_state": {"position": [0.02]}}

    def _entry(self, marker):
        import copy
        d = copy.deepcopy(self.KEYS)
        d["marker"] = marker
        return d

    def test_empty_queue_leaves_sequence_untouched(self):
        sched = make_sched()
        seq = sched.make_sequence({}, current_arm={}, current_gripper={})
        assert all("marker" not in s for s in seq)

    def test_queue_of_one_consumes_first_substep_only(self):
        sched = make_sched()
        sched.push_support_actions([self._entry("e0")])
        seq = sched.make_sequence({}, current_arm={}, current_gripper={})
        assert seq[0]["marker"] == "e0"
        assert all("marker" not in s for s in seq[1:])

    def test_queue_of_nine_consumes_first_nine(self):
        sched = make_sched()
        sched.push_support_actions([self._entry(f"e{i}") for i in range(9)])
        seq = sched.make_sequence({}, current_arm={}, current_gripper={})
        assert [s.get("marker") for s in seq[:9]] == [f"e{i}" for i in range(9)]
        assert "marker" not in seq[9]

    def test_queue_of_ten_exactly_one_step(self):
        sched = make_sched()
        sched.push_support_actions([self._entry(f"e{i}") for i in range(10)])
        seq = sched.make_sequence({}, current_arm={}, current_gripper={})
        assert [s.get("marker") for s in seq] == [f"e{i}" for i in range(10)]
        assert len(sched.support_queue) == 0  # 原队列耗尽后保持空，不产生缺省动作

    def test_queue_of_eleven_spills_to_next_step(self):
        sched = make_sched()
        sched.push_support_actions([self._entry(f"e{i}") for i in range(11)])
        seq1 = sched.make_sequence({}, current_arm={}, current_gripper={})
        assert [s.get("marker") for s in seq1] == [f"e{i}" for i in range(10)]
        seq2 = sched.make_sequence({}, current_arm={}, current_gripper={})
        assert seq2[0].get("marker") == "e10"
        assert "marker" not in seq2[1]


class TestProgressiveGripper:
    SCALE = robots.GRIPPER_SCALE

    def test_within_eps_jumps_to_target(self):
        cur = -0.01
        target = cur + 0.001  # 差 0.001 << 行程 0.054*0.2=0.0108
        assert control.apply_progressive_gripper(target, cur, self.SCALE) == pytest.approx(target)

    def test_beyond_eps_moves_20pct_of_range(self):
        cur, hi = self.SCALE
        got = control.apply_progressive_gripper(hi, cur, self.SCALE)
        assert got == pytest.approx(cur + (hi - cur) * 0.2)

    def test_closing_direction(self):
        lo, _ = self.SCALE
        got = control.apply_progressive_gripper(lo, 0.044, self.SCALE)
        assert got == pytest.approx(0.044 - (0.044 - lo) * 0.2)


def test_opening_to_joint_matches_scale():
    assert robots.opening_to_joint(0.0) == pytest.approx(robots.GRIPPER_SCALE[0])
    assert robots.opening_to_joint(1.0) == pytest.approx(robots.GRIPPER_SCALE[1])


def test_velocity_clamp():
    v = control.clamp_substep_target(0.0, 0.5)  # 0.5 >> 5*0.004=0.02
    assert v == pytest.approx(0.02)
    assert control.clamp_substep_target(0.0, -0.5) == pytest.approx(-0.02)
    assert control.clamp_substep_target(0.0, 0.01) == pytest.approx(0.01)  # 界内不动
    assert control.clamp_substep_target(1.0, 1.0) == pytest.approx(1.0)

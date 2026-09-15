"""robosuite-bowl 适配器的 G1 契约测试（新口径 §4.2，零付费 API）。

先红后绿的把关项与 v3 同构：build_toolset 构建、eef_state 参考态、
move_to 单维/插值/越界/终止、夹爪测量语义。运行：
  .venv-robosuite/bin/python -m unittest test_robosuite_bowl_contract -v
"""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import numpy as np

from inspect_robots_agent._tools import build_toolset

from robosuite_bowl import RobosuiteBowlEmbodiment

_NOISE0 = {"execution_noise": (0.0, 0.0, 0.0), "cameras": False}


def _scene(i="c"):
    return type("S", (), {"instruction": "pick the red cube into the bowl", "id": i})()


class TestToolsetBuilds(unittest.TestCase):
    def test_bowl_toolset_builds(self):
        emb = RobosuiteBowlEmbodiment(**_NOISE0)
        try:
            obs = emb.reset(_scene(), seed=0)
            toolset = build_toolset(
                emb.info.action_space, emb.info.observation_space,
                control_hz=emb.info.control_hz,
                max_speed_frac=0.25, images="always")
            self.assertEqual(toolset._state_key, "eef_state")
            self.assertEqual(toolset._move_tool, "move_to")
            self.assertEqual(toolset._labels, ("x", "y", "z", "yaw", "gripper"))
            self.assertIsNotNone(obs)
        finally:
            emb.close()


class TestStateContract(unittest.TestCase):
    def test_fields_and_units(self):
        emb = RobosuiteBowlEmbodiment(**_NOISE0)
        try:
            fields = {f.key: f for f in emb.info.observation_space.state.fields}
            self.assertIn("eef_state", fields)
            self.assertEqual(fields["eef_state"].shape, (5,))
            self.assertNotEqual(fields["eef_state"].unit, "m")
            # 原评测观测含 joint_eff（E2），适配器对齐暴露
            self.assertIn("joint_eff", fields)
            self.assertIn("joint_pos", fields)
        finally:
            emb.close()

    def test_ranges_and_truth_not_exposed(self):
        emb = RobosuiteBowlEmbodiment(**_NOISE0)
        try:
            obs = emb.reset(_scene("r"), seed=3)
            s = np.asarray(obs.state["eef_state"], dtype=np.float64)
            lo, hi = emb.info.action_space.low, emb.info.action_space.high
            self.assertTrue(np.all(np.isfinite(s)))
            self.assertTrue(np.all(s >= lo - 1e-6) and np.all(s <= hi + 1e-6))
            # 口径 §4.2：物体真值不得进入模型观测
            for key in obs.state:
                self.assertNotIn("cube", key.lower())
                self.assertNotIn("object", key.lower())
        finally:
            emb.close()


class TestMoveToBehavior(unittest.TestCase):
    def _make(self):
        emb = RobosuiteBowlEmbodiment(**_NOISE0)
        obs = emb.reset(_scene("m"), seed=5)
        toolset = build_toolset(
            emb.info.action_space, emb.info.observation_space,
            control_hz=emb.info.control_hz, max_speed_frac=0.25, images="always")
        return emb, obs, toolset

    def test_single_axis_preserves_others(self):
        emb, obs, toolset = self._make()
        try:
            cur = np.asarray(obs.state["eef_state"], dtype=np.float64)
            call = SimpleNamespace(id="t", name="move_to", arguments=json.dumps(
                {"targets": {"x": float(cur[0] + 0.03)}, "note": "contract"}))
            res = toolset.execute(call, obs)
            self.assertIsNone(res.error)
            last = np.asarray(res.chunk.actions[-1].data, dtype=np.float64)
            self.assertAlmostEqual(last[0], cur[0] + 0.03, places=6)
            np.testing.assert_allclose(last[1:5], cur[1:5], atol=1e-9)
        finally:
            emb.close()

    def test_interpolation_within_limits(self):
        emb, obs, toolset = self._make()
        try:
            cur = np.asarray(obs.state["eef_state"], dtype=np.float64)
            call = SimpleNamespace(id="t", name="move_to", arguments=json.dumps(
                {"targets": {"x": float(cur[0] + 0.15)}, "note": "contract"}))
            res = toolset.execute(call, obs)
            acts = res.chunk.actions
            self.assertGreater(len(acts), 1)
            prev = cur
            for a in acts:
                d = np.abs(np.asarray(a.data, dtype=np.float64) - prev)
                self.assertTrue(np.all(d <= toolset._step_limits + 1e-9))
                prev = np.asarray(a.data, dtype=np.float64)
        finally:
            emb.close()

    def test_out_of_bounds_and_note(self):
        emb, obs, toolset = self._make()
        try:
            call = SimpleNamespace(id="t", name="move_to", arguments=json.dumps(
                {"targets": {"x": 10.0}, "note": "contract"}))
            res = toolset.execute(call, obs)
            self.assertIsNone(res.chunk)
            self.assertIn("outside", res.error)
            call2 = SimpleNamespace(id="t", name="move_to", arguments=json.dumps(
                {"targets": {"x": 0.0}}))
            self.assertIn("note is required", toolset.execute(call2, obs).error)
        finally:
            emb.close()

    def test_done_requests_stop(self):
        emb, obs, toolset = self._make()
        try:
            call = SimpleNamespace(id="t", name="done", arguments=json.dumps(
                {"summary": "ok", "hindsight": "none"}))
            res = toolset.execute(call, obs)
            self.assertTrue(res.chunk.actions[0].meta["request_stop"])
        finally:
            emb.close()

    def test_playout_reaches_target(self):
        emb, obs, toolset = self._make()
        try:
            cur = np.asarray(obs.state["eef_state"], dtype=np.float64)
            call = SimpleNamespace(id="t", name="move_to", arguments=json.dumps(
                {"targets": {"x": float(cur[0] + 0.06), "z": 0.95}, "note": "contract"}))
            res = toolset.execute(call, obs)
            final = np.asarray(res.chunk.actions[-1].data, dtype=np.float64)
            for a in res.chunk.actions:
                emb.step(a)
            for _ in range(15):
                emb.step(a)
            now = np.asarray(emb._observe(emb._env._get_observations()).state["eef_state"])
            self.assertLess(np.linalg.norm(now[:3] - final[:3]), 0.02,
                            f"保持后残差应 <2cm，实测 {now[:3]} vs {final[:3]}")
        finally:
            emb.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)

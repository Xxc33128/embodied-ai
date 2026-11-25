"""Task 1（规划 2026-09-14 §4）：bowl 本体 × agent 工具面契约测试。

先红后绿：当前 v3 的 state 无 (5,) 字段，build_toolset 必须报 ToolsetError；
补上 eef_state 后全部转绿。零付费 API——只用真实 Toolset 与本地仿真。

运行：.venv/bin/python -m unittest test_bowl_contract -v
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import numpy as np

from inspect_robots_agent._tools import build_toolset

from mujoco_bowl import MuJoCoBowlEmbodiment

_NOISE0 = {"execution_noise": (0.0, 0.0, 0.0)}


def _blank_render(emb) -> None:
    """关掉三路离屏渲染（契约测试只关心 state/动作，不关心像素）。"""
    emb._render = lambda cam: np.zeros((emb._img_h, emb._img_w, 3), dtype=np.uint8)


def _move(toolset, obs, **targets):
    args = {"targets": targets, "note": "contract test motion"}
    call = SimpleNamespace(id="t0", name="move_to", arguments=json.dumps(args))
    return toolset.execute(call, obs)


class TestToolsetBuilds(unittest.TestCase):
    """G1 第一道门：工具面必须能从 bowl 的 space 声明构建出来。"""

    def test_bowl_toolset_builds(self):
        emb = MuJoCoBowlEmbodiment(**_NOISE0)
        try:
            _blank_render(emb)
            obs = emb.reset(type("S", (), {"instruction": "t", "id": "c0"})(), seed=0)
            toolset = build_toolset(
                emb.info.action_space, emb.info.observation_space,
                control_hz=emb.info.control_hz,
                max_speed_frac=0.25, images="always")
            # absolute 模式必须锁定到唯一 (5,) 参考字段，且就是我们的 eef_state
            self.assertEqual(toolset._state_key, "eef_state")
            self.assertEqual(toolset._move_tool, "move_to")
            self.assertEqual(toolset._labels, ("x", "y", "z", "yaw", "gripper"))
            self.assertIsNotNone(obs)
        finally:
            emb.close()


class TestStateContract(unittest.TestCase):
    """eef_state：shape/顺序/量纲/取值范围，测量值不冒充命令值。"""

    def test_eef_state_field_declared(self):
        emb = MuJoCoBowlEmbodiment(**_NOISE0)
        try:
            fields = {f.key: f for f in emb.info.observation_space.state.fields}
            self.assertIn("eef_state", fields)
            self.assertEqual(fields["eef_state"].shape, (5,))
            # 混合量纲不得标成全是米
            self.assertNotEqual(fields["eef_state"].unit, "m")
        finally:
            emb.close()

    def test_eef_state_matches_action_order_and_ranges(self):
        emb = MuJoCoBowlEmbodiment(**_NOISE0)
        try:
            _blank_render(emb)
            emb.reset(type("S", (), {"instruction": "t", "id": "c1"})(), seed=3)
            s = np.asarray(emb._observe().state["eef_state"], dtype=np.float64)
            self.assertEqual(s.shape, (5,))
            self.assertTrue(np.all(np.isfinite(s)))
            lo, hi = emb.info.action_space.low, emb.info.action_space.high
            self.assertTrue(np.all(s >= lo - 1e-6), f"below action low: {s} vs {lo}")
            self.assertTrue(np.all(s <= hi + 1e-6), f"above action high: {s} vs {hi}")
        finally:
            emb.close()

    def test_gripper_reports_measurement_not_command(self):
        """到位（张开）后命令闭合，手指被 4cm 方块挡住：实测开度须停在
        ~0.3-0.45（gap≈0.043 在 [0.026,0.076] 行程内），不能回 0（命令值）
        也不能报 0.5+（关节全行程归一化，语义错位）。
        注：必须先下探到位再闭合——下探途中闭合会把方块拨走（闭合挤出）。"""
        emb = MuJoCoBowlEmbodiment(**_NOISE0)
        try:
            _blank_render(emb)
            emb.reset(type("S", (), {"instruction": "t", "id": "c2"})(), seed=1000)
            cube = emb._cube_pos()
            tgt = lambda g: type("A", (), {"data": np.array(
                [cube[0], cube[1], cube[2] + 0.018, 0.0, g])})()
            for _ in range(4):
                emb.step(tgt(1.0))   # 张开下探
            for _ in range(40):      # 沉降：伺服收敛到位（<5mm）才闭合
                res = emb.step(tgt(1.0))
                if np.linalg.norm(emb._site_pos() - np.array([cube[0], cube[1], cube[2] + 0.018])) < 0.005:
                    break
            cube_before = emb._cube_pos().copy()
            for _ in range(8):
                emb.step(tgt(0.0))   # 到位后才闭合
            np.testing.assert_allclose(emb._cube_pos()[:2], cube_before[:2], atol=0.006,
                                       err_msg="闭合前面 served 方块已被拨走，场景失效")
            obs = emb._observe()
            measured = float(obs.state["gripper"][0])
            self.assertGreater(measured, 0.1,
                               "方块挡指时实测开度应>0.1；返回<=0 说明报的是命令值")
            self.assertLess(measured, 0.45,
                            "夹住方块应报'已闭合夹持'；>0.45 说明归一化口径是全行程")
            self.assertAlmostEqual(float(obs.state["eef_state"][4]), measured, places=6)
        finally:
            emb.close()


class TestMoveToBehavior(unittest.TestCase):
    """经真实 Toolset.execute 调 move_to：单维、插值、越界、终止。"""

    def _make(self):
        emb = MuJoCoBowlEmbodiment(**_NOISE0)
        _blank_render(emb)
        obs = emb.reset(type("S", (), {"instruction": "t", "id": "m"})(), seed=5)
        toolset = build_toolset(
            emb.info.action_space, emb.info.observation_space,
            control_hz=emb.info.control_hz, max_speed_frac=0.25, images="always")
        return emb, obs, toolset

    def test_single_axis_preserves_other_dims(self):
        emb, obs, toolset = self._make()
        try:
            cur = np.asarray(obs.state["eef_state"], dtype=np.float64)
            res = _move(toolset, obs, x=float(cur[0] + 0.03))
            self.assertIsNone(res.error)
            acts = res.chunk.actions
            self.assertGreaterEqual(len(acts), 1)
            last = np.asarray(acts[-1].data, dtype=np.float64)
            self.assertAlmostEqual(last[0], cur[0] + 0.03, places=6)
            np.testing.assert_allclose(last[1:5], cur[1:5], atol=1e-9)
        finally:
            emb.close()

    def test_yaw_only_and_gripper_only(self):
        emb, obs, toolset = self._make()
        try:
            cur = np.asarray(obs.state["eef_state"], dtype=np.float64)
            res = _move(toolset, obs, yaw=0.2)
            self.assertIsNone(res.error)
            last = np.asarray(res.chunk.actions[-1].data, dtype=np.float64)
            self.assertAlmostEqual(last[3], 0.2, places=6)
            np.testing.assert_allclose(last[[0, 1, 2, 4]], cur[[0, 1, 2, 4]], atol=1e-9)

            res2 = _move(toolset, emb._observe(), gripper=0.0)
            self.assertIsNone(res2.error)
            last2 = np.asarray(res2.chunk.actions[-1].data, dtype=np.float64)
            self.assertAlmostEqual(last2[4], 0.0, places=6)
        finally:
            emb.close()

    def test_large_displacement_interpolates_within_step_limits(self):
        emb, obs, toolset = self._make()
        try:
            cur = np.asarray(obs.state["eef_state"], dtype=np.float64)
            res = _move(toolset, obs, x=float(cur[0] + 0.30))
            self.assertIsNone(res.error)
            acts = res.chunk.actions
            self.assertGreater(len(acts), 1, "0.30m 位移必须被护栏插值成多步")
            limits = toolset._step_limits
            prev = cur
            for a in acts:
                d = np.abs(np.asarray(a.data, dtype=np.float64) - prev)
                self.assertTrue(np.all(d <= limits + 1e-9),
                                f"单步位移超限: {d} > {limits}")
                prev = np.asarray(a.data, dtype=np.float64)
            # 末动作必须精确落在目标上
            self.assertAlmostEqual(prev[0], cur[0] + 0.30, places=6)
        finally:
            emb.close()

    def test_out_of_bounds_rejected_without_actions(self):
        emb, obs, toolset = self._make()
        try:
            res = _move(toolset, obs, x=10.0)
            self.assertIsNone(res.chunk)
            self.assertIn("outside", res.error)
        finally:
            emb.close()

    def test_unknown_dim_and_missing_note_rejected(self):
        emb, obs, toolset = self._make()
        try:
            res = _move(toolset, obs, zz=0.1)
            self.assertIn("unknown dimension", res.error)
            call = SimpleNamespace(id="t1", name="move_to",
                                   arguments=json.dumps({"targets": {"x": 0.0}}))
            res2 = toolset.execute(call, obs)
            self.assertIn("note is required", res2.error)
        finally:
            emb.close()

    def test_done_requests_stop(self):
        emb, obs, toolset = self._make()
        try:
            call = SimpleNamespace(id="t2", name="done",
                                   arguments=json.dumps({"summary": "ok", "hindsight": "none"}))
            res = toolset.execute(call, obs)
            self.assertIsNone(res.error)
            self.assertTrue(res.chunk.actions[0].meta["request_stop"])
        finally:
            emb.close()

    def test_playout_reaches_target_in_sim(self):
        """动作 chunk 实际执行后，实测 eef_state 须逼近目标（伺服滞后内）。

        位置伺服有物理跟踪滞后（规划 §3 已记录），chunk 末动作只执行 1 个
        控制步时残差≈单步位移量级；补保持步（LLM 经观测自然补偿）后须收敛。"""
        emb, obs, toolset = self._make()
        try:
            cur = np.asarray(obs.state["eef_state"], dtype=np.float64)
            res = _move(toolset, obs, x=float(cur[0] + 0.10), z=0.15)
            self.assertIsNone(res.error)
            final = np.asarray(res.chunk.actions[-1].data, dtype=np.float64)
            for a in res.chunk.actions:
                emb.step(a)
            for _ in range(12):  # 保持：伺服沉降
                emb.step(a)
            now = np.asarray(emb._observe().state["eef_state"], dtype=np.float64)
            self.assertLess(np.linalg.norm(now[:3] - final[:3]), 0.02,
                            f"保持 12 步后 TCP 残差应 <2cm，实测 {now[:3]} vs {final[:3]}")
        finally:
            emb.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)

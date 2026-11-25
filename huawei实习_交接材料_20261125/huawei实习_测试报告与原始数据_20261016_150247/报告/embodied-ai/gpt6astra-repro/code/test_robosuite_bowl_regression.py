"""robosuite-bowl 回归测试（针对 2026-09-15 独立复核的三项阻塞 + 收口项）。

原契约测试的盲区：没测开合方向、转腕实际转动、画面内容、噪声轨迹重放、
以及"经真实 move_to 工具链的完整抓放"。本文件逐项补齐。

运行：.venv-robosuite/bin/python -m unittest test_robosuite_bowl_regression -v
"""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import numpy as np

from inspect_robots_agent._tools import build_toolset

from robosuite_bowl import RobosuiteBowlEmbodiment

_NOISE0 = {"execution_noise": (0.0, 0.0, 0.0), "cameras": False}


def _scene(i="r"):
    return type("S", (), {"instruction": "pick the red cube into the bowl", "id": i})()


def _move(emb, toolset, targets):
    call = SimpleNamespace(id="t", name="move_to",
                           arguments=json.dumps({"targets": targets, "note": "regression"}))
    res = toolset.execute(call, emb._observe(emb._env._get_observations()))
    assert res.error is None, res.error
    return res


def _play(emb, res, settle=0):
    for a in res.chunk.actions:
        emb.step(a)
    for _ in range(settle):
        emb.step(a)
    return emb._observe(emb._env._get_observations())


class TestGripperDirection(unittest.TestCase):
    """① 协议 0=闭/1=开 → 实测开度必须同向（复核前反向）。"""

    def test_close_open_close_sequence(self):
        emb = RobosuiteBowlEmbodiment(**_NOISE0)
        try:
            obs = emb.reset(_scene("g"), seed=1000)
            toolset = build_toolset(emb.info.action_space, emb.info.observation_space,
                                    control_hz=emb.info.control_hz,
                                    max_speed_frac=0.25, images="always")
            eef = np.asarray(obs.state["eef_state"][:3])

            o = _play(emb, _move(emb, toolset, {"gripper": 0.0}), settle=40)
            self.assertLess(float(o.state["gripper"][0]), 0.15,
                            "指令 0（闭合）后实测开度应 <0.15")

            o = _play(emb, _move(emb, toolset, {"gripper": 1.0}), settle=40)
            self.assertGreater(float(o.state["gripper"][0]), 0.85,
                               "指令 1（张开）后实测开度应 >0.85")

            o = _play(emb, _move(emb, toolset, {"gripper": 0.0}), settle=40)
            self.assertLess(float(o.state["gripper"][0]), 0.15)
            _ = eef
        finally:
            emb.close()


class TestYawTracking(unittest.TestCase):
    """② xyzw 解析修复后：转腕目标须实际收敛，平移不得带偏姿态。"""

    def _make(self):
        emb = RobosuiteBowlEmbodiment(**_NOISE0)
        obs = emb.reset(_scene("y"), seed=5)
        toolset = build_toolset(emb.info.action_space, emb.info.observation_space,
                                control_hz=emb.info.control_hz,
                                max_speed_frac=0.25, images="always")
        return emb, obs, toolset

    def test_yaw_positive_negative_and_hold(self):
        emb, obs, toolset = self._make()
        try:
            cur = np.asarray(obs.state["eef_state"], dtype=np.float64)
            o = _play(emb, _move(emb, toolset, {"yaw": 0.4}), settle=25)
            self.assertLess(abs(float(o.state["eef_yaw"][0]) - 0.4), 0.1,
                            f"yaw 目标 0.4 未收敛：{o.state['eef_yaw']}")
            o = _play(emb, _move(emb, toolset, {"yaw": -0.4}), settle=25)
            self.assertLess(abs(float(o.state["eef_yaw"][0]) + 0.4), 0.1)
            # 仅平移：姿态保持
            o = _play(emb, _move(emb, toolset, {"x": float(cur[0] + 0.05)}), settle=15)
            self.assertLess(abs(float(o.state["eef_yaw"][0]) + 0.4), 0.1,
                            "平移后 yaw 漂移超限")
        finally:
            emb.close()


class TestCameraSeesBowl(unittest.TestCase):
    """③ 三路相机必须能看到碗（复核前 group=0 全部不可见），且方向正确。"""

    def test_bowl_visible_in_all_cameras(self):
        emb = RobosuiteBowlEmbodiment(cameras=True, execution_noise=(0, 0, 0))
        try:
            obs = emb.reset(_scene("cam"), seed=1000)
            self.assertEqual(set(obs.images), {"front", "side", "wrist"})
            for name, img in obs.images.items():
                arr = np.asarray(img).astype(int)
                blue = (arr[:, :, 2] - np.maximum(arr[:, :, 0], arr[:, :, 1])) > 25
                self.assertGreater(int(blue.sum()), 80,
                                   f"{name} 相机里看不到碗（蓝色像素 {int(blue.sum())}）")
            # 方向：front 视角下碗在画面下半（桌面上）
            arr = np.asarray(obs.images["front"]).astype(int)
            blue = (arr[:, :, 2] - np.maximum(arr[:, :, 0], arr[:, :, 1])) > 25
            rows = np.where(blue.any(axis=1))[0]
            self.assertGreater(float(rows.mean()), 0.45 * arr.shape[0],
                               "翻转后 front 视图中碗仍在上半幅，图像方向可疑")
        finally:
            emb.close()


class TestSeedReplay(unittest.TestCase):
    """收口：同 seed 同动作序列（含噪声）→ EEF 轨迹逐步一致。"""

    def test_same_seed_same_trajectory(self):
        def run():
            emb = RobosuiteBowlEmbodiment()  # 默认噪声
            try:
                emb.reset(_scene("rep"), seed=1000)
                traj = []
                rs = np.random.RandomState(7)  # 动作序列固定
                for _ in range(20):
                    a = np.array([0.0, 0.0, 0.90, 0.0, 1.0]) + rs.uniform(-0.01, 0.01, 5)
                    r = emb.step(type("A", (), {"data": a})())
                    traj.append(tuple(np.round(r.observation.state["eef_state"], 6)))
                return traj
            finally:
                emb.close()

        t1, t2 = run(), run()
        self.assertEqual(t1, t2, "同 seed 两次运行的 EEF 轨迹不一致（噪声源未重置？）")


class TestToolInterfaceFullPickPlace(unittest.TestCase):
    """经真实 move_to 工具链的完整抓放（模型同一接口）。闭合后只移动、
    不重指定开度。10 seeds ≥9 成功。"""

    def test_full_pick_place_10_seeds(self):
        n_ok = 0
        details = []
        for i in range(10):
            seed = 1000 + i
            emb = RobosuiteBowlEmbodiment(execution_noise=(0, 0, 0), cameras=False)
            try:
                obs = emb.reset(_scene(f"fp{seed}"), seed=seed)
                toolset = build_toolset(emb.info.action_space, emb.info.observation_space,
                                        control_hz=emb.info.control_hz,
                                        max_speed_frac=0.25, images="always")
                cube = np.asarray(emb._cube_pos())
                bowl = np.asarray(emb._env.bowl_xy)
                ok = False
                # 悬停（开爪）→ 下探 → 闭合 → 提升 → 搬运 → 下降 → 张开
                seq = [
                    {"x": float(cube[0]), "y": float(cube[1]),
                     "z": float(cube[2] + 0.12), "gripper": 1.0},
                    {"x": float(cube[0]), "y": float(cube[1]),
                     "z": float(cube[2] + 0.004)},
                    {"gripper": 0.0},
                    {"z": 1.02},                                   # 不重指定开度
                    {"x": float(bowl[0]), "y": float(bowl[1]), "z": 1.02},
                    {"z": 0.92},
                    {"gripper": 1.0},
                ]
                for targets in seq:
                    res = _move(emb, toolset, targets)
                    settle = 35 if set(targets) == {"gripper"} else 20
                    obs = _play(emb, res, settle=settle)
                # 释放后方块需下落入碗：持续观察 60 步采样成功
                for _ in range(60):
                    r = emb.step(type("A", (), {"data": np.zeros(5)})())
                    if r.info["success"]:
                        ok = True
                        break
                n_ok += int(ok)
                details.append((seed, ok))
            finally:
                emb.close()
        print("\n  tool-interface pick-place:", details)
        self.assertGreaterEqual(n_ok, 9, f"工具接口完整抓放仅 {n_ok}/10 成功")


if __name__ == "__main__":
    unittest.main(verbosity=2)

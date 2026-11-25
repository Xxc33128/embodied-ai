"""Task 2（规划 2026-09-14 §5）：bowl 模型组装正确性断言。

先红后绿：
  - 双指 equality（Menagerie 原 panda.xml 的 joint 耦合）在 v3 组装时丢失
  - 碗壁 euler 数值是角度制（20），被 XML 顶部 angle="radian" compiler 当
    弧度解释，实际主角 ≈65.92°
  - contact exclude（link0-link1）同样丢失（父子对默认已被过滤，恢复是
    为模型语义忠实，非宣称碰撞故障——规划 §5 第 4 条）

运行：.venv/bin/python -m unittest test_bowl_model -v
"""

from __future__ import annotations

import unittest

import numpy as np

from mujoco_bowl import MuJoCoBowlEmbodiment

_WALL_TILT_RAD = np.radians(20.0)


def _principal_angle(q: np.ndarray) -> float:
    """四元数 (w,x,y,z) 的旋转主角（规划 §5 指定算法）。"""
    return float(2 * np.arccos(np.clip(abs(q[0]), 0.0, 1.0)))


class TestModelAssembly(unittest.TestCase):
    def setUp(self):
        self.emb = MuJoCoBowlEmbodiment(execution_noise=(0.0, 0.0, 0.0))
        self.model = self.emb._model

    def tearDown(self):
        self.emb.close()

    def test_finger_equality_restored(self):
        """v3 组装丢了 <equality> 双指耦合 → neq 必须回到 1。"""
        self.assertEqual(self.model.neq, 1,
                         "双指 joint equality 丢失：左/右指不再同步，闭合易不对称挤出")

    def test_link0_link1_exclude_preserved(self):
        """v3 组装丢了 <contact><exclude link0 link1/>。父子对默认已过滤，
        恢复只为与原模型语义一致（neq/nexclude 作为组装完整性哨兵）。"""
        self.assertEqual(self.model.nexclude, 1)

    def test_wall_tilt_is_20_deg(self):
        """碗壁主角必须是设计的 20°，而不是 20 个弧度（≈65.92°）。"""
        for i in range(4):
            q = self.model.body(f"bowl_wall_{i}").quat
            angle = _principal_angle(np.asarray(q, dtype=np.float64))
            self.assertAlmostEqual(angle, _WALL_TILT_RAD, delta=1e-4,
                                   msg=f"bowl_wall_{i} 主角 {np.degrees(angle):.2f}° ≠ 20°")

    def test_equation_joints_are_fingers(self):
        """equality 必须耦合的是左右指关节（防接错体）。mjEQ_JOINT=2。"""
        eq_type = int(self.model.eq_type[0])
        self.assertEqual(eq_type, 2, "eq_type 2 = joint 耦合（与原模型一致）")
        j1 = self.emb._model.joint("finger_joint1")
        j2 = self.emb._model.joint("finger_joint2")
        obj1 = int(self.model.eq_obj1id[0])
        obj2 = int(self.model.eq_obj2id[0])
        self.assertEqual(obj1, j1.id)
        self.assertEqual(obj2, j2.id)


if __name__ == "__main__":
    unittest.main(verbosity=2)

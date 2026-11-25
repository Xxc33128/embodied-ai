"""L2 LIBERO 单臂 7D 动作契约测试（纯本地，无 MuJoCo 依赖）。

常数钉定依据见 src/gap_repro/libero/actions.py 模块 docstring：
- 夹爪符号 −1 开 / +1 合（l2_action_probe 实测）
- 缩放 0.05 m / 0.5 rad（活环境控制器实读）
- 学生值可略超 ±1（实测 −1.003），工具值严格 ±1
"""

from __future__ import annotations

import numpy as np
import pytest

from gap_repro.agent.contract import ContractError, rotation_vector_to_quat
from gap_repro.libero.actions import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    LIBERO_ACTION_DIM,
    LIBERO_CORRECTION_MAX_STEPS,
    LIBERO_DECISION_MAX_ROTATION,
    LIBERO_DECISION_MAX_TRANSLATION,
    POS_ACTION_SCALE,
    ROT_ACTION_SCALE,
    direct_target_to_chunk,
    edit_to_chunk,
    gripper_closed_to_action,
    pose_to_policy_state,
    quat_to_rotation_vector,
    quat_xyzw_to_wxyz,
    validate_student_chunk,
    validate_tool_action,
)


def _pose(pos, q_wxyz):
    return {"position": np.asarray(pos, dtype=np.float64),
            "quaternion_wxyz": np.asarray(q_wxyz, dtype=np.float64)}


IDENT = np.array([1.0, 0.0, 0.0, 0.0])


class TestGripperSign:
    def test_probe_pinned_sign(self):
        # 实测证据（l2_action_probe.json）：−1 开口度 0.0794 > +1 的 0.0012
        assert GRIPPER_OPEN == -1.0 and GRIPPER_CLOSE == +1.0
        assert gripper_closed_to_action(True) == +1.0
        assert gripper_closed_to_action(False) == -1.0


class TestStudentChunk:
    def test_accepts_real_observed_values(self):
        # 学生实测值（LP-A2）：pos 分量 0.96、gripper −1.003 必须原样通过
        chunk = np.full((5, LIBERO_ACTION_DIM), 0.1)
        chunk[0, 0] = 0.96
        chunk[0, 6] = -1.003
        out = validate_student_chunk(chunk)
        assert out[0, 6] == -1.003  # 不裁剪（客户端一致性）

    def test_rejects_non_finite(self):
        chunk = np.zeros((1, 7))
        chunk[0, 2] = np.nan
        with pytest.raises(ContractError, match="non-finite"):
            validate_student_chunk(chunk)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ContractError, match=r"n, 7"):
            validate_student_chunk(np.zeros((1, 14)))  # 14D 双臂拒收（F08）

    def test_full_horizon_chunk_accepted(self):
        # 服务器响应为全程 (10,7)；replan 预算由切片执行段保证（policy 层）
        assert validate_student_chunk(np.zeros((10, 7))).shape == (10, 7)


class TestToolAction:
    def test_ok(self):
        a = validate_tool_action([0.5, -0.5, 0.1, 0.2, 0.0, 0.0, 1.0])
        assert a.shape == (7,)

    def test_rejects_over_bound(self):
        with pytest.raises(ContractError, match="±1"):
            validate_tool_action([1.01, 0, 0, 0, 0, 0, 1.0])

    def test_rejects_non_finite(self):
        with pytest.raises(ContractError):
            validate_tool_action([0.1, 0, 0, np.inf, 0, 0, 1.0])


class TestQuatRotvec:
    def test_identity(self):
        assert np.allclose(quat_to_rotation_vector(IDENT), np.zeros(3))

    def test_ninety_about_z(self):
        q = rotation_vector_to_quat([0, 0, np.pi / 2])
        assert np.allclose(quat_to_rotation_vector(q), [0, 0, np.pi / 2],
                           atol=1e-12)

    def test_roundtrip_large_angle(self):
        w = np.array([0.3, -0.4, 0.2])
        w = w / np.linalg.norm(w) * 2.5  # >π/2 的大角（w>0 支内与 log-map 一致）
        assert np.allclose(quat_to_rotation_vector(rotation_vector_to_quat(w)),
                           w, atol=1e-9)

    def test_openpi_hemisphere_semantics(self):
        # P2-7：state 编码 = openpi _quat2axisangle 逐字（robosuite 版）：
        # (x,y,z,w) 输入、w<0 时角度沿原 axis 超过 π、无归一化
        from gap_repro.libero.actions import axisangle_from_xyzw
        # w=cos(0.05), axis ẑ, (x,y,z,w)
        q = [0.0, 0.0, np.sin(0.05), np.cos(0.05)]
        rv = axisangle_from_xyzw(q)
        assert rv[2] == pytest.approx(0.1, abs=1e-9)
        q_neg = [0.0, 0.0, -np.sin(0.05), -np.cos(0.05)]  # w<0 同轴
        rv_neg = axisangle_from_xyzw(q_neg)
        assert rv_neg[2] == pytest.approx(-(2 * np.pi - 0.1), abs=1e-9)
        # 近单位 w → 零向量分支
        assert np.allclose(axisangle_from_xyzw([0, 0, 0, 1.0]), np.zeros(3))

    def test_non_unit_rejected(self):
        with pytest.raises(ContractError, match="unit"):
            quat_to_rotation_vector([2.0, 0, 0, 0])

    def test_xyzw_to_wxyz(self):
        # robosuite 观测 (x,y,z,w)=(0,0,0.7071,0.7071) → wxyz
        out = quat_xyzw_to_wxyz([0, 0, 0.7071, 0.7071])
        assert np.allclose(out, [0.7071, 0, 0, 0.7071])


class TestPolicyState:
    def test_matches_openpi_concat_order(self):
        # LP-A2 契约：8 维 = pos(3) + axisangle(3) + 双指 qpos(2)；
        # 四元数为 robosuite (x,y,z,w) 原样（P2-7 逐字端口）
        pos = np.array([0.1, 0.2, 0.3])
        q_xyzw = [0.0, 0.0, np.sin(0.25), np.cos(0.25)]  # 0.5 rad 绕 z
        g = np.array([0.0208, -0.0208])
        s = pose_to_policy_state(pos, q_xyzw, g)
        assert s.shape == (8,)
        assert np.allclose(s[:3], pos)
        assert np.allclose(s[3:6], [0, 0, 0.5], atol=1e-9)
        assert np.allclose(s[6:], g)

    def test_rejects_bad_pos(self):
        with pytest.raises(ContractError):
            pose_to_policy_state([0.1, 0.2], IDENT, [0.0, 0.0])

    def test_rejects_3_finger_qpos(self):
        with pytest.raises(ContractError, match="dual-finger"):
            pose_to_policy_state([0.1, 0.2, 0.3], IDENT, [0.0, 0.0, 0.0])


class TestDirectTarget:
    def test_reaches_target_exactly(self):
        cur = _pose([0.4, 0.0, 0.1], IDENT)
        tgt_q = rotation_vector_to_quat([0, 0, 0.3])  # 0.3 rad < 0.425 界
        tgt = {"position": [0.44, 0.02, 0.11],  # |Δ|=0.0458 < 0.055 界
               "quaternion_wxyz": tgt_q, "gripper_closed": True}
        chunk, info = direct_target_to_chunk(tgt, cur, steps=5)
        assert chunk.shape == (5, 7)
        # 每步物理增量 = 总量/步数；反解累计位移应精确回到目标
        per_pos = chunk[:, :3] * POS_ACTION_SCALE
        per_rot = chunk[:, 3:6] * ROT_ACTION_SCALE
        assert np.allclose(per_pos.sum(axis=0), tgt["position"]
                           - cur["position"], atol=1e-12)
        assert np.isclose(np.linalg.norm(per_rot.sum(axis=0)), 0.3, atol=1e-9)
        # 沿同一轴等分（geodesic）
        axis = per_rot[0] / np.linalg.norm(per_rot[0])
        for r in per_rot:
            assert np.allclose(r / np.linalg.norm(r), axis, atol=1e-9)
        assert chunk[0, 6] == GRIPPER_CLOSE
        assert info["gripper_action"] == GRIPPER_CLOSE

    def test_unreachable_in_steps_rejected(self):
        # 0.3 m > 5 步 × 0.011 m 决策上界 → 拒绝（不静默截断）
        cur = _pose([0.0, 0.0, 0.0], IDENT)
        tgt = {"position": [0.3, 0, 0], "quaternion_wxyz": IDENT,
               "gripper_closed": False}
        with pytest.raises(ContractError, match="0.300000 m >|translation"):
            direct_target_to_chunk(tgt, cur, steps=5)

    def test_decision_rotation_bound(self):
        cur = _pose([0.0, 0.0, 0.0], IDENT)
        q = rotation_vector_to_quat([1.2, 0, 0])  # > 1.0 rad 预注册界
        tgt = {"position": [0.0, 0, 0], "quaternion_wxyz": q,
               "gripper_closed": False}
        with pytest.raises(ContractError, match="rotation"):
            direct_target_to_chunk(tgt, cur, steps=5)

    def test_bad_target_structure(self):
        cur = _pose([0.0, 0.0, 0.0], IDENT)
        tgt = {"position": [0.1, 0, 0], "quaternion_wxyz": [1, 0, 0],
               "gripper_closed": True}  # 非 4 元素
        with pytest.raises(ContractError, match="4 elements"):
            direct_target_to_chunk(tgt, cur, steps=3)

    def test_steps_bound(self):
        cur = _pose([0.0, 0.0, 0.0], IDENT)
        tgt = {"position": [0.01, 0, 0], "quaternion_wxyz": IDENT,
               "gripper_closed": False}
        with pytest.raises(ContractError, match="at most"):
            direct_target_to_chunk(tgt, cur, steps=6)
        assert LIBERO_CORRECTION_MAX_STEPS == 5


class TestEditChunk:
    def test_edit_basic(self):
        cur = _pose([0.5, 0.1, 0.2], IDENT)
        edit = {"delta_position": [0.02, 0.0, 0.0],
                "delta_rotation_vector": [0.0, 0.15, 0.0],
                "gripper": "open"}
        chunk, info = edit_to_chunk(edit, cur, steps=2)
        assert chunk.shape == (2, 7)
        assert np.allclose(chunk[:, :3] * POS_ACTION_SCALE,
                           np.tile([0.01, 0, 0], (2, 1)))
        assert chunk[0, 6] == GRIPPER_OPEN
        assert info["rotation_rad"] == pytest.approx(0.15)

    def test_keep_uses_hold_value(self):
        # P1-4 修正：keep = 保持学生夹爪值（gripper_hold 由调用方传入）
        cur = _pose([0.5, 0.1, 0.2], IDENT)
        edit = {"delta_position": [0.01, 0, 0],
                "delta_rotation_vector": [0, 0, 0], "gripper": "keep"}
        chunk, info = edit_to_chunk(edit, cur, steps=2, gripper_hold=+1.0)
        assert chunk[0, 6] == +1.0
        with pytest.raises(ContractError, match="gripper_hold"):
            edit_to_chunk(edit, cur, steps=2)  # 未传 hold → 拒绝
        with pytest.raises(ContractError, match="open/closed/keep"):
            edit_to_chunk({"delta_position": [0.01, 0, 0],
                           "delta_rotation_vector": [0, 0, 0],
                           "gripper": "squeeze"}, cur, steps=2)

    def test_edit_over_decision_bound(self):
        cur = _pose([0.5, 0.1, 0.2], IDENT)
        edit = {"delta_position": [LIBERO_DECISION_MAX_TRANSLATION + 0.01, 0, 0],
                "delta_rotation_vector": [0, 0, 0], "gripper": "closed"}
        with pytest.raises(ContractError, match="translation"):
            edit_to_chunk(edit, cur, steps=5)


class TestPreRegisteredBounds:
    def test_bounds_documented_values(self):
        # P0-1 修正：实测满幅传递（probe v2），非 config output_max
        assert POS_ACTION_SCALE == 0.011
        assert ROT_ACTION_SCALE == 0.085
        assert LIBERO_DECISION_MAX_TRANSLATION == 0.055
        assert LIBERO_DECISION_MAX_ROTATION == 0.425

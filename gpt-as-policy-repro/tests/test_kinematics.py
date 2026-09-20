"""W2 FK 验收：原 ArmFK（vendored，解析 URDF）vs MuJoCo 双 X5 模型。

门槛（E1，原 DualKinematics.check）：位置 <2mm、角度 <0.01 rad。
姿态集：零位（原 init_state）、关节限位边缘、固定种子随机可达位。
环境缺失时显式失败标记 BLOCKED（不得静默 skip 充当通过）。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from gap_repro.reference.kinematics_original import ArmFK
from gap_repro.sim import robots

ASSETS = Path(os.environ.get("GAP_REPRO_DATA", ""))
URDF = ASSETS / "hf_cache/Assets/Robots/x5/X5A.urdf"
ARM_JOINTS = robots.ARM_JOINTS


def _require_env():
    import mujoco  # noqa: F401

    if not URDF.is_file():
        pytest.fail(
            f"BLOCKED: X5A.urdf 不存在（{URDF}）。请先在资产齐备的环境运行"
            f"（设 GAP_REPRO_DATA=资产缓存根）", pytrace=False)


@pytest.fixture(scope="module")
def model_meta():
    _require_env()
    model, meta = robots.compile_dual_x5(URDF)
    data = __import__("mujoco").MjData(model)
    return model, data, meta


@pytest.fixture(scope="module")
def arm_fk():
    return ArmFK(str(URDF), ARM_JOINTS, base="base_link", tip="link6")


def _qpos_for(model, side, q6):
    q = np.zeros(model.nq)
    for i, j in enumerate(robots.ARM_JOINTS):
        q[model.joint(f"{side}_{j}").qposadr[0]] = q6[i]
    return q


def _compare(model, data, fk, side, q6):
    import mujoco

    data.qpos[:] = _qpos_for(model, side, q6)
    got = robots.relative_fk(model, data, f"{side}_base_link", f"{side}_link6")
    ref = fk.matrix(q6)
    pos_err = float(np.linalg.norm(ref[:3, 3] - got[:3, 3]))
    rot_err = float(np.linalg.norm(
        __import__("scipy.spatial.transform", fromlist=["Rotation"])
        .Rotation.from_matrix(ref[:3, :3] @ got[:3, :3].T).as_rotvec()))
    return pos_err, rot_err


def test_zero_pose_fk_matches(model_meta, arm_fk):
    model, data, _ = model_meta
    for side in ("left", "right"):
        pos_err, rot_err = _compare(model, data, arm_fk, side, np.zeros(6))
        assert pos_err < 0.002, f"{side} zero pose pos_err={pos_err*1000:.3f}mm"
        assert rot_err < 0.01, f"{side} zero pose rot_err={rot_err:.5f}rad"


def test_random_reachable_poses_fk_matches(model_meta, arm_fk):
    model, data, _ = model_meta
    rng = np.random.default_rng(0)
    worst_pos = worst_rot = 0.0
    for _ in range(25):
        q6 = rng.uniform(-np.pi, np.pi, 6)  # URDF 限位 ±10，取工作密度区
        for side in ("left", "right"):
            pos_err, rot_err = _compare(model, data, arm_fk, side, q6)
            worst_pos, worst_rot = max(worst_pos, pos_err), max(worst_rot, rot_err)
    assert worst_pos < 0.002, f"worst pos_err={worst_pos*1000:.4f}mm"
    assert worst_rot < 0.01, f"worst rot_err={worst_rot:.6f}rad"


def test_near_joint_limit_fk_matches(model_meta, arm_fk):
    model, data, _ = model_meta
    rng = np.random.default_rng(1)
    worst_pos = worst_rot = 0.0
    for _ in range(10):
        edges = rng.choice([-9.0, 9.0], 6) + rng.uniform(-0.2, 0.2, 6)
        for side in ("left", "right"):
            pos_err, rot_err = _compare(model, data, arm_fk, side, edges)
            worst_pos, worst_rot = max(worst_pos, pos_err), max(worst_rot, rot_err)
    assert worst_pos < 0.002, f"worst pos_err={worst_pos*1000:.4f}mm"
    assert worst_rot < 0.01, f"worst rot_err={worst_rot:.6f}rad"


def test_dual_roots_composition(model_meta):
    import mujoco

    model, data, meta = model_meta
    mujoco.mj_forward(model, data)
    for side, root in robots.DUAL_ROOTS.items():
        bid = model.body(f"{side}_base_link").id
        pos = data.xpos[bid]
        assert np.allclose(pos, root["pos"], atol=1e-3), f"{side} root {pos}"
        q = data.xquat[bid]
        assert np.allclose(np.abs(q), np.abs(np.array(root["quat_wxyz"]) / np.linalg.norm(root["quat_wxyz"])),
                           atol=2e-3), f"{side} quat {q}"


def test_gripper_semantics_and_mimic_elements(model_meta):
    """0闭/1开映射与 joint8=joint7 等式元素必须存在且系数正确。"""
    import mujoco

    model, _, _ = model_meta
    assert robots.opening_to_joint(0.0) == pytest.approx(-0.01)
    assert robots.opening_to_joint(1.0) == pytest.approx(0.044)
    assert robots.opening_to_joint(0.5) == pytest.approx(0.017)
    for side in ("left", "right"):
        j8 = model.joint(f"{side}_joint8").id
        j7 = model.joint(f"{side}_joint7").id
        hits = [i for i in range(model.neq)
                if model.eq_type[i] == int(mujoco.mjtEq.mjEQ_JOINT)
                and model.eq_obj1id[i] == j8 and model.eq_obj2id[i] == j7]
        assert len(hits) == 1, f"{side} mimic equality missing"
        assert np.allclose(model.eq_data[hits[0]][:5], [0.0, 1.0, 0.0, 0.0, 0.0])


def test_armature_and_no_gravity(model_meta):
    """T4：原 Isaac `disable_gravity=True` 由逐 body gravcomp=1 实现。

    全局重力必须开启（场景自由物体需要下落）；机器人所有 body 单独抵消
    重力。全局 mjDSBL_GRAVITY 会让物体也不动，已废弃。
    """
    import mujoco

    model, _, _ = model_meta
    for side in ("left", "right"):
        for j in ARM_JOINTS:
            dof = model.joint(f"{side}_{j}").dofadr[0]
            assert model.dof_armature[dof] == pytest.approx(0.01)
    assert not (model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY))
    assert np.any(model.opt.gravity != 0.0)
    for side in ("left", "right"):
        for bid in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if name.startswith(f"{side}_"):
                assert model.body_gravcomp[bid] == pytest.approx(1.0), name


def test_actuator_mapping(model_meta):
    """W4：Isaac ImplicitActuator → position 伺服（kp/kv/forcerange/ctrlrange）。"""
    model, _, _ = model_meta
    n = 0
    for side in ("left", "right"):
        a = model.actuator(f"{side}_gripper_act")
        assert a.gaintype[0] == 0
        assert a.gainprm[0] == pytest.approx(2300.0)
        assert a.biasprm[1] == pytest.approx(-2300.0)
        assert a.biasprm[2] == pytest.approx(-100.0)
        assert np.allclose(a.ctrlrange, list(robots.GRIPPER_SCALE))
        assert np.allclose(a.forcerange, [-100.0, 100.0])
        n += 1
        for j in ARM_JOINTS:
            a = model.actuator(f"{side}_{j}_act")
            assert a.gainprm[0] == pytest.approx(4400.0)
            assert a.biasprm[1] == pytest.approx(-4400.0)
            assert a.biasprm[2] == pytest.approx(-40.0)
            assert np.allclose(a.forcerange, [-100.0, 100.0])
            n += 1
    assert n == model.nu == 14  # 2×(6+1)，mimic 的 joint8 不单独驱动

"""T4d：三相机视觉契约（cam_head / cam_left_wrist / cam_right_wrist）。

覆盖 audit F07（视觉部分）：原 `get_obs(include_vision=True)` 此前直接
BLOCKED；现在经 VisionRenderer 输出 HxWx4 uint8，并验证：
- 相机名/分辨率/内参来源登记（robots.CAMERA_MANIFEST）；
- 图像非空非全黑、同状态重渲染一致、物体移动后像素变化；
- 后端不可用时 fail-closed（BLOCKED 而非空图）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gap_repro.sim import robots
from gap_repro.sim.environment import MuJoCoEnvironment

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LAYOUT_DIR = DATA / "hf_cache/Assets/Eval_Layout/RoboDojo/arx_x5/0"
OBJ_ROOT = DATA / "w11_scene_obj"
URDF = DATA / "hf_cache/Assets/Robots/x5/X5A.urdf"
LAYOUT = "organize_table_0"

HAVE_URDF = URDF.exists()
HAVE_SCENE = (LAYOUT_DIR / f"{LAYOUT}.json").exists() and OBJ_ROOT.exists()

CONTRACT_KEYS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


@pytest.fixture(scope="module")
def scene_env():
    if not (HAVE_URDF and HAVE_SCENE):
        pytest.skip("organize_table_0 资产不在本机")
    env = MuJoCoEnvironment(URDF, "organize_table", layout_name=LAYOUT,
                            obj_root=str(OBJ_ROOT), layout_dir=str(LAYOUT_DIR))
    env.reset()
    return env


def _vision(env):
    try:
        return env.get_obs(include_vision=True)["vision"]
    except RuntimeError as e:
        if "BLOCKED" in str(e):
            pytest.skip(f"渲染后端不可用: {e}")
        raise


def test_camera_contract_names_and_manifest():
    assert set(CONTRACT_KEYS) == set(robots.CAMERA_MANIFEST)
    assert robots.CAMERA_MANIFEST["cam_head"]["source"].endswith("camera_config.yml")


def test_head_camera_orientation_matches_source():
    xy = robots.euler_xyz_deg_to_xyaxes((30.0, 0.0, 0.0))
    x, y = np.array(xy[:3]), np.array(xy[3:])
    assert np.allclose(x, [1, 0, 0], atol=1e-9)
    assert np.allclose(y, [0, np.cos(np.pi / 6), np.sin(np.pi / 6)], atol=1e-9)
    z = np.cross(x, y)  # 视线 = -z
    assert np.allclose(-z, [0, 0.5, -np.sqrt(3) / 2], atol=1e-9)


def test_vision_shapes_and_dtype(scene_env):
    vision = _vision(scene_env)
    assert set(vision) == set(CONTRACT_KEYS)
    for key, v in vision.items():
        color = v["color"]
        assert color.dtype == np.uint8, key
        assert color.shape == (480, 640, 4), (key, color.shape)
        assert np.all(color[..., 3] == 255), key
        assert v["resolution"] == (640, 480)
    # 原名别名契约（w5 inventory）：不出现旧称
    assert "left_wrist_cam" not in vision and "right_wrist_cam" not in vision


def test_images_non_black_and_stable(scene_env):
    v1 = _vision(scene_env)
    v2 = _vision(scene_env)
    for key in CONTRACT_KEYS:
        img = v1[key]["color"][..., :3].astype(float)
        assert img.std() > 1.0, f"{key} 图像近似常量（可能全黑）"
        assert img.mean() > 1.0, f"{key} 图像近似全黑"
        assert np.array_equal(v1[key]["color"], v2[key]["color"]), \
            f"{key} 同状态重渲染不一致"


def test_moving_object_changes_pixels(scene_env):
    before = _vision(scene_env)["cam_head"]["color"].copy()
    e = scene_env.scene_manifest["objects"]["alarm"]
    bid = scene_env.model.body(e["body"]).id
    adr = int(scene_env.model.body_jntadr[bid])
    qadr = int(scene_env.model.jnt_qposadr[adr])
    scene_env.data.qpos[qadr] += 0.15
    scene_env._mujoco.mj_forward(scene_env.model, scene_env.data)
    after = _vision(scene_env)["cam_head"]["color"]
    diff = np.abs(after[..., :3].astype(int) - before[..., :3].astype(int)).sum()
    assert diff > 1000, "物体移动后画面未变化（相机帧与状态不一致？）"


def test_include_vision_false_no_renderer():
    if not HAVE_URDF:
        pytest.skip("URDF 不在本机")
    env = MuJoCoEnvironment(URDF, "organize_table")
    env.reset()
    obs = env.get_obs(include_vision=False)
    assert obs["vision"] == {}
    assert env._vision is None


def test_backend_unavailable_is_blocked_not_empty(monkeypatch):
    import mujoco

    if not HAVE_URDF:
        pytest.skip("URDF 不在本机")
    env = MuJoCoEnvironment(URDF, "organize_table")
    env.reset()

    def _boom(*a, **k):
        raise OSError("no GL backend")

    monkeypatch.setattr(mujoco, "Renderer", _boom)
    with pytest.raises(RuntimeError, match="BLOCKED"):
        env.get_obs(include_vision=True)

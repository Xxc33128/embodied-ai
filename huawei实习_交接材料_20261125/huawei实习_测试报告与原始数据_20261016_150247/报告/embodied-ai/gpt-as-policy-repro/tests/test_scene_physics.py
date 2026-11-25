"""T4：真实场景物理契约（organize_table_0 原 case）。

覆盖 audit F04 修复：
- Rigid 物体为 freejoint 动力学刚体（不再是固定 body）；Geometry 保持静态；
- 质量/摩擦来自 layout physics 映射并带来源，缺值不再一律 0.05kg；
- 桌面尺寸来自 layout 的 Table 元数据；
- 物体在重力下真实运动，评分读取的 provider 状态与 MjData 同步；
- reset 恢复冻结初态，不残留上一集物体状态。

数据缺失时整组 skip（Mac 轻量环境无资产），目标机必须实跑验收。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gap_repro.sim import scene

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LAYOUT_DIR = DATA / "hf_cache/Assets/Eval_Layout/RoboDojo/arx_x5/0"
OBJ_ROOT = DATA / "w11_scene_obj"
URDF = DATA / "hf_cache/Assets/Robots/x5/X5A.urdf"
LAYOUT = "organize_table_0"

pytestmark = pytest.mark.skipif(
    not (LAYOUT_DIR / f"{LAYOUT}.json").exists() or not OBJ_ROOT.exists()
    or not URDF.exists(),
    reason="organize_table_0 资产不在本机（目标机必须实跑）")


@pytest.fixture(scope="module")
def scene_built():
    import mujoco

    model, data, manifest = scene.build_scene_mjcf(
        URDF, LAYOUT, obj_root=str(OBJ_ROOT), layout_dir=str(LAYOUT_DIR))
    return model, data, manifest


def _free_qadr(model, entry):
    import mujoco

    bid = model.body(entry["body"]).id
    adr = int(model.body_jntadr[bid])
    assert int(model.jnt_type[adr]) == int(mujoco.mjtJoint.mjJNT_FREE)
    return int(model.jnt_qposadr[adr])


class TestSceneComposition:
    def test_rigid_objects_are_free_bodies(self, scene_built):
        import mujoco

        model, _, manifest = scene_built
        dynamic = {k: e for k, e in manifest["objects"].items() if e["dynamic"]}
        assert set(dynamic) == {"mouse", "alarm", "keyboard", "garage"}
        for label, e in dynamic.items():
            bid = model.body(e["body"]).id
            assert int(model.body_jntnum[bid]) == 1, label
            adr = int(model.body_jntadr[bid])
            assert int(model.jnt_type[adr]) == int(mujoco.mjtJoint.mjJNT_FREE), label

    def test_geometry_objects_are_static_visual_by_default(self, scene_built):
        model, _, manifest = scene_built
        geometry = {k: e for k, e in manifest["objects"].items()
                    if e["section"] == "Geometry"}
        assert len(geometry) == 6
        for label, e in geometry.items():
            bid = model.body(e["body"]).id
            assert int(model.body_jntnum[bid]) == 0, label
            assert e["dynamic"] is False
            gid = model.geom(e["geom"]).id
            assert int(model.geom_contype[gid]) == 0, label
            assert int(model.geom_conaffinity[gid]) == 0, label

    def test_mass_and_friction_mapped_from_layout(self, scene_built):
        model, _, manifest = scene_built
        e = manifest["objects"]["alarm"]
        assert e["mass"] == pytest.approx(0.35)
        assert e["mass_source"] == "layout"
        assert e["friction"] == pytest.approx(0.45)
        assert e["friction_source"] == "layout_friction"
        gid = model.geom(e["collision_geom"]).id
        assert float(model.geom_friction[gid][0]) == pytest.approx(0.45)
        assert float(model.body_mass[model.body("obj_alarm").id]) == pytest.approx(0.35)

    def test_missing_mass_uses_upstream_default_not_0_05(self, scene_built):
        model, _, manifest = scene_built
        e = manifest["objects"]["keyboard"]
        assert e["mass"] == pytest.approx(scene.RIGID_MASS_DEFAULT)
        assert e["mass_source"] == "upstream_default_0.5"
        e2 = manifest["objects"]["mouse"]
        assert e2["mass"] == pytest.approx(0.09)

    def test_table_geometry_from_layout_metadata(self, scene_built):
        model, _, manifest = scene_built
        layout = scene.load_layout(LAYOUT, str(LAYOUT_DIR))
        t = layout["Table"]
        assert manifest["table"]["half_extents"] == pytest.approx(
            [s / 2 for s in t["scale"]])
        gid = model.geom("table").id
        assert model.geom_size[gid] == pytest.approx([s / 2 for s in t["scale"]])

    def test_unsupported_type_fails_closed(self):
        with pytest.raises(NotImplementedError, match="garment"):
            scene.map_physics({"type": "garment"}, "Garment")


class TestScenePhysics:
    def test_free_object_falls_and_rests_on_support(self, scene_built):
        import mujoco

        model, data, manifest = scene_built
        e = manifest["objects"]["alarm"]
        qadr = _free_qadr(model, e)
        mujoco.mj_resetData(model, data)
        data.qpos[qadr:qadr + 3] = np.array(e["default_pos"]) + [0, 0, 0.2]
        data.qpos[qadr + 3:qadr + 7] = e["default_quat_wxyz"]
        mujoco.mj_forward(model, data)
        z0 = float(data.xpos[model.body("obj_alarm").id][2])
        for _ in range(1500):
            mujoco.mj_step(model, data)
        z1 = float(data.xpos[model.body("obj_alarm").id][2])
        assert z1 < z0 - 0.05  # 真实下落
        table_top = manifest["table"]["pos"][2] + manifest["table"]["half_extents"][2]
        assert z1 > table_top - 0.05  # 停在桌面上而非穿过
        assert np.isfinite(data.qvel).all()

    def test_reset_restores_default_object_poses(self, scene_built):
        from gap_repro.sim.environment import MuJoCoEnvironment

        env = MuJoCoEnvironment(URDF, "organize_table", layout_name=LAYOUT,
                                obj_root=str(OBJ_ROOT),
                                layout_dir=str(LAYOUT_DIR))
        env.reset()
        # 上一集残留：物体移位/带速度/仿真时间推进
        e = env.scene_manifest["objects"]["garage"]
        qadr = _free_qadr(env.model, e)
        vadr = int(env.model.jnt_dofadr[
            int(env.model.body_jntadr[env.model.body(e["body"]).id])])
        env.data.qpos[qadr:qadr + 3] = [0.0, 0.0, 0.2]
        env.data.qvel[vadr:vadr + 6] = 1.0
        env.data.time = 5.0
        env.reset()
        assert env.data.time == 0.0
        for label, ent in env.scene_manifest["objects"].items():
            if not ent["dynamic"]:
                continue
            qadr = _free_qadr(env.model, ent)
            vadr = int(env.model.jnt_dofadr[
                int(env.model.body_jntadr[env.model.body(ent["body"]).id])])
            assert env.data.qpos[qadr:qadr + 3] == pytest.approx(ent["default_pos"])
            assert np.allclose(env.data.qpos[qadr + 3:qadr + 7],
                               ent["default_quat_wxyz"], atol=1e-8), label
            assert np.allclose(env.data.qvel[vadr:vadr + 6], 0.0)

    def test_provider_matches_mjdata_and_scoring_sees_physics(self, scene_built):
        from gap_repro.sim.environment import MuJoCoEnvironment

        model, _, manifest = scene_built
        env = MuJoCoEnvironment(URDF, "organize_table", layout_name=LAYOUT,
                                obj_root=str(OBJ_ROOT),
                                layout_dir=str(LAYOUT_DIR))
        env.reset()
        alarm = env.provider.objects["alarm"]
        bid = env.model.body("obj_alarm").id
        assert np.allclose(alarm.position, env.data.xpos[bid])
        assert env._call("is_moved", {"label": "alarm", "dis_threshold": 0.05}) == 0.0

        # 物理式移动：改 freejoint qpos 后 forward（等价一步真实位移结果）
        e = manifest["objects"]["alarm"]
        qadr = _free_qadr(env.model, e)
        env.data.qpos[qadr] += 0.10
        env._mujoco.mj_forward(env.model, env.data)
        env._sync_provider_objects()
        moved = env.provider.objects["alarm"]
        assert np.allclose(moved.position, env.data.xpos[bid])
        assert env._call("is_moved", {"label": "alarm", "dis_threshold": 0.05}) == 1.0

    def test_scene_env_take_action_updates_object_sync(self, scene_built):
        from gap_repro.sim.environment import MuJoCoEnvironment

        env = MuJoCoEnvironment(URDF, "organize_table", layout_name=LAYOUT,
                                obj_root=str(OBJ_ROOT),
                                layout_dir=str(LAYOUT_DIR))
        env.reset()
        command = {
            "left_arm_joint_state": {"position": [0.0] * 6},
            "right_arm_joint_state": {"position": [0.0] * 6},
            "left_ee_joint_state": {"position": [1.0]},
            "right_ee_joint_state": {"position": [1.0]},
        }
        env.take_action(command)
        for label, e in env.scene_manifest["objects"].items():
            bid = env.model.body(e["body"]).id
            assert np.allclose(env.provider.objects[label].position,
                               env.data.xpos[bid])

"""W11/T4 场景组装：layout JSON → 可操作物理 MJCF（自由刚体 + 固定 fixture）。

E1 来源：
- `Assets/Eval_Layout/RoboDojo/arx_x5/0/<layout>.json`（实例位姿/朝向/尺度/physics）
- `env/scene_manager/objects/rigid.py`（质量/摩擦映射与默认值）
- `env/scene_manager/objects/geometry.py`（geometry 默认不参与碰撞）
- `env/scene_manager/objects/table.py`、`ground.py`（桌面/地面尺寸语义）
- `env/scene_manager/scene_manager.py`（physics.type → 物体类）

T4 关键修复（audit F04）：
- Rigid 物体是有 freejoint 的动力学刚体（此前是固定 body，不可操作）；
- layout 的 default_ori/scale/physics 进入模型（此前只用 pos、统一 mass=0.05）；
- 质量/摩擦按源映射并记录来源，缺值显式标记，不再一律 0.05kg；
- 桌面尺寸/位置从 layout 的 Table 元数据读取（此前硬编码 [0.5,0.35] 半尺寸）；
- manifest 输出 label→body/geom 映射与局部 bbox，供 environment 从 MjData 同步。

已知差异（记录在 manifest.diffs）：
- 碰撞代理：优先同名 collision OBJ；当前转换产物只有 visual 网格时用其凸包，
  与原 PhysX convexDecomposition 不完全一致；
- Geometry 物体按源默认 collision=False 仅为视觉（可用 physics.collision 显式打开）；
- Room 视觉墙体未建模；Ground 以有厚度 box 近似（原为实现为 cube + env_spacing）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gap_repro.sim import robots

OBJ_ROOT = "/workspace/data/w11_scene_obj"
LAYOUT_DIR = "/workspace/data/hf_cache/Assets/Eval_Layout/RoboDojo/arx_x5/0"

# 上游默认值（E1）
RIGID_MASS_DEFAULT = 0.5          # rigid.py:65 `get("mass", 0.5)`
RIGID_MASS_CAP = 0.5              # rigid.py:65 `min(mass, 0.5)`
RIGID_MASS_MIN_FALLBACK = 0.05    # rigid.py:206-207 `mass <= 0 → 0.05`
FRICTION_STATIC_DEFAULT = 0.6     # rigid.py:74
FRICTION_DYNAMIC_DEFAULT = 1.5    # rigid.py:75
TABLE_FRICTION = 0.8              # table.py:103-104
GROUND_FRICTION = 1.0             # 本工程 layout Ground.physics_material
GROUND_HALF_XY = 2.0              # 原 Ground 尺寸由 env_spacing 决定；单环境近似
GROUND_MUJOCO_ROLLING = 0.0001
GROUND_MUJOCO_TORSIONAL = 0.005

SUPPORTED_TYPES = {"rigid", "geometry"}


@dataclass
class PhysicsSpec:
    mass: float = 0.0
    mass_source: str = ""
    sliding_friction: float = 0.0
    friction_source: str = ""
    dynamic: bool = True
    collision: bool = True


def map_physics(physics: dict | None, section: str) -> PhysicsSpec:
    """layout physics dict → MuJoCo 参数（全部带来源标签）。"""
    p = dict(physics or {})
    spec = PhysicsSpec()
    raw_mass = p.get("mass", RIGID_MASS_DEFAULT)
    mass = float(min(raw_mass, RIGID_MASS_CAP))
    if mass <= 0:
        mass = RIGID_MASS_MIN_FALLBACK
        spec.mass_source = "upstream_fallback_mass<=0"
    elif "mass" in p:
        spec.mass_source = "layout" if raw_mass <= RIGID_MASS_CAP else "layout_capped_0.5"
    else:
        spec.mass_source = "upstream_default_0.5"
    spec.mass = mass

    if "friction" in p:
        spec.sliding_friction = float(p["friction"])
        spec.friction_source = "layout_friction"
    elif "static_friction" in p:
        spec.sliding_friction = float(p["static_friction"])
        spec.friction_source = "layout_static_friction"
    elif "dynamic_friction" in p:
        spec.sliding_friction = float(p["dynamic_friction"])
        spec.friction_source = "layout_dynamic_friction"
    else:
        spec.sliding_friction = FRICTION_STATIC_DEFAULT
        spec.friction_source = "upstream_static_default_0.6"

    ptype = p.get("type", "rigid" if section == "Rigid" else "geometry")
    if ptype not in SUPPORTED_TYPES:
        raise NotImplementedError(
            f"scene object type '{ptype}' not supported by MuJoCo scene builder "
            f"(supported: {sorted(SUPPORTED_TYPES)})")
    spec.dynamic = ptype == "rigid"
    spec.collision = bool(p.get("collision", spec.dynamic))
    return spec


def extract_objects(layout: dict) -> dict:
    """layout → {label: {pos, quat_wxyz, scale, physics, type, ...}}。

    `default_ori` 与上游一致按 (w,x,y,z) 直通（Isaac set_local_pose 约定，
    见 rigid.py:94；robot dual_x5.yml 同为 wxyz）。
    """
    out = {}
    for key in ("Rigid", "Geometry"):
        for cat, items in (layout.get(key) or {}).items():
            if not isinstance(items, list):
                continue
            for inst in items:
                if not isinstance(inst, dict):
                    continue
                label = inst.get("label", f"{cat}_{len(out)}")
                dp = inst.get("default_pos", [0, 0, 0.78])
                ori = inst.get("default_ori", [1.0, 0.0, 0.0, 0.0])
                out[label] = {
                    "category": cat,
                    "category_idx": inst.get("category_idx", 0),
                    "section": key,
                    "pos": [float(x) for x in dp[:3]],
                    "quat_wxyz": [float(x) for x in ori[:4]],
                    "scale": [float(x) for x in (inst.get("scale") or [1, 1, 1])[:3]],
                    "physics": inst.get("physics") or {},
                }
    return out


def _find_mesh(obj_root, section, cat, idx, suffix: str):
    section = section or "Rigid"
    nested = Path(obj_root) / section / cat / f"{idx:05d}" / f"object{suffix}.obj"
    flat = Path(obj_root) / f"{section}_{cat}_{idx:05d}{suffix}_object.obj"
    clutter = Path(obj_root) / "Clutter" / cat / f"{idx:05d}" / f"object{suffix}.obj"
    for cand in (nested, flat, clutter):
        if cand.exists():
            return cand
    return None


def find_obj_mesh(obj_root, section, cat, idx):
    """布局物体 → visual OBJ 网格；缺失返回 None（既有契约）。"""
    return _find_mesh(obj_root, section, cat, idx, "")


def find_collision_mesh(obj_root, section, cat, idx):
    """collision OBJ（转换产物含 `_collision` 变体时）；缺失返回 None。"""
    return _find_mesh(obj_root, section, cat, idx, "_collision")


def load_layout(layout_name="organize_table_0", layout_dir=None):
    root = Path(layout_dir or LAYOUT_DIR)
    return json.loads((root / f"{layout_name}.json").read_text())


def _add_table(spec, layout: dict):
    import mujoco

    t = layout.get("Table") or {}
    scale = [float(x) for x in (t.get("scale") or [1.4, 1.1, 0.05])]
    pos = [float(x) for x in (t.get("default_pos") or [0.0, -0.05, 0.74])]
    half = [scale[0] / 2, scale[1] / 2, scale[2] / 2]
    spec.worldbody.add_geom(
        name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=pos,
        rgba=[0.65, 0.5, 0.4, 1],
        friction=[TABLE_FRICTION, GROUND_MUJOCO_TORSIONAL, GROUND_MUJOCO_ROLLING])
    return {"scale": scale, "pos": pos, "half_extents": half}


def _add_ground(spec, layout: dict):
    import mujoco

    g = layout.get("Ground") or {}
    thickness = float(g.get("thickness", 0.1))
    gpos = [float(x) for x in (g.get("default_pos") or [0.0, 0.0, 0.0])]
    center = [gpos[0], gpos[1], gpos[2] - thickness / 2]
    spec.worldbody.add_geom(
        name="ground", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[GROUND_HALF_XY, GROUND_HALF_XY, thickness / 2], pos=center,
        rgba=[0.35, 0.35, 0.35, 1],
        friction=[GROUND_FRICTION, GROUND_MUJOCO_TORSIONAL, GROUND_MUJOCO_ROLLING])
    return {"thickness": thickness, "pos": center}


def mesh_local_bbox(model, geom_id):
    mid = int(model.geom_dataid[geom_id])  # mjGEOM_MESH: dataid = mesh id
    adr = int(model.mesh_vertadr[mid])
    num = int(model.mesh_vertnum[mid])
    verts = model.mesh_vert[adr:adr + num]
    return verts.min(axis=0), verts.max(axis=0)


def build_scene_mjcf(urdf, layout_name="organize_table_0", obj_root=OBJ_ROOT,
                     layout_dir=None):
    """组装双 X5 + 桌面/地面 + 物体 → (model, data, manifest)。

    物体映射：physics.type=rigid → freejoint 动力学刚体；geometry → 固定 body
    （碰撞默认关闭，与原 GeometryObject 一致）。未知类型 fail-closed。
    """
    import mujoco

    layout = load_layout(layout_name, layout_dir)
    objs = extract_objects(layout)
    spec = robots.build_dual_x5_spec(urdf)
    manifest = {"layout": layout_name, "objects": {}, "diffs": []}

    manifest["table"] = _add_table(spec, layout)
    manifest["ground"] = _add_ground(spec, layout)
    spec.worldbody.add_light(pos=[0.0, -1.0, 1.8], dir=[0.0, 0.3, -1.0],
                             diffuse=[1, 1, 1])

    for label, info in objs.items():
        cat, idx = info["category"], info.get("category_idx", 0)
        obj_file = find_obj_mesh(obj_root, info.get("section"), cat, idx)
        if obj_file is None:
            manifest["diffs"].append(f"missing_mesh:{label}")
            raise FileNotFoundError(
                f"layout {layout_name}: {label} 网格缺失（obj_root={obj_root}）")
        col_file = find_collision_mesh(obj_root, info.get("section"), cat, idx)
        pspec = map_physics(info["physics"], info["section"])
        mesh_name = f"mesh_{label}"
        spec.add_mesh(name=mesh_name, file=str(obj_file),
                      scale=list(info["scale"]))
        body = spec.worldbody.add_body(name=f"obj_{label}", pos=info["pos"],
                                       quat=info["quat_wxyz"])
        if pspec.dynamic:
            body.add_freejoint()
        # 视觉 geom：始终存在，不参与碰撞（bbox/观测用）
        body.add_geom(name=f"geom_{label}", type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=mesh_name, mass=0.0, contype=0, conaffinity=0,
                      rgba=[1, 1, 1, 1])
        entry = {
            "label": label, "body": f"obj_{label}", "geom": f"geom_{label}",
            "section": info["section"], "category": cat, "category_idx": idx,
            "dynamic": pspec.dynamic, "collision": pspec.collision,
            "mass": pspec.mass, "mass_source": pspec.mass_source,
            "friction": pspec.sliding_friction,
            "friction_source": pspec.friction_source,
            "obj_path": str(obj_file),
            "default_pos": list(info["pos"]),
            "default_quat_wxyz": list(info["quat_wxyz"]),
            "scale": list(info["scale"]),
        }
        if pspec.collision:
            if col_file is not None:
                col_mesh = f"colmesh_{label}"
                spec.add_mesh(name=col_mesh, file=str(col_file),
                              scale=list(info["scale"]))
                proxy = "collision_mesh"
            else:
                col_mesh = mesh_name  # 凸包代理（与 visual 同网格）
                proxy = "visual_convex_hull"
                manifest["diffs"].append(f"collision_proxy_visual:{label}")
            body.add_geom(name=f"col_{label}", type=mujoco.mjtGeom.mjGEOM_MESH,
                          meshname=col_mesh, mass=pspec.mass, condim=3,
                          friction=[pspec.sliding_friction, GROUND_MUJOCO_TORSIONAL,
                                    GROUND_MUJOCO_ROLLING],
                          rgba=[0, 0, 0, 0])
            entry["collision_geom"] = f"col_{label}"
            entry["collision_proxy"] = proxy
        else:
            entry["collision_proxy"] = "none"
        manifest["objects"][label] = entry

    model = spec.compile()
    data = mujoco.MjData(model)
    for label, entry in manifest["objects"].items():
        bid = model.body(entry["body"]).id
        gid = model.geom(entry["geom"]).id
        lo, hi = mesh_local_bbox(model, gid)
        entry["body_id"] = bid
        entry["geom_id"] = gid
        entry["local_bbox_min"] = [float(x) for x in lo]
        entry["local_bbox_max"] = [float(x) for x in hi]
    return model, data, manifest

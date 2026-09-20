#!/usr/bin/env python3
"""W11 场景渲染验证：双 X5 + 真实 OBJ 对象 + 桌面 → 三相机 RGB。"""
import sys, os
os.environ["MUJOCO_GL"] = "osmesa"
sys.path.insert(0, "/workspace/repo/src")
from pathlib import Path
import numpy as np
import mujoco
from gap_repro.sim import scene
from gap_repro.sim import robots

urdf = Path("/workspace/data/hf_cache/Assets/Robots/x5/X5A.urdf")
model, data, _manifest = scene.build_scene_mjcf(urdf, "organize_table_0",
                                                obj_root="/workspace/data/w11_scene_obj")
for side in ("left", "right"):
    for i, j in enumerate(robots.ARM_JOINTS):
        data.qpos[model.joint(f"{side}_{j}").qposadr[0]] = [0.3,-0.4,0.5,-0.9,0.4,0.2][i]
mujoco.mj_forward(model, data)
r = mujoco.Renderer(model, 480, 640)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
out_dir = Path("/workspace/data/w11_renders")
out_dir.mkdir(exist_ok=True)
for cid in range(model.ncam):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
    cam.fixedcamid = cid
    r.update_scene(data, camera=cam)
    img = r.render()
    from PIL import Image
    Image.fromarray(img).save(str(out_dir / f"{name}.png"))
    print(f"RENDER {name} {img.shape} std={img.std():.1f}")
print(f"TOTAL_BODIES {model.nbody} TOTAL_MESHES {model.nmesh}")
print("W11_SCENE_OK")

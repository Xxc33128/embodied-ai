#!/usr/bin/env python3
"""W3：三相机渲染存证（OSMesa，官方 mujoco venv）。"""
import sys
from pathlib import Path
import numpy as np
import mujoco
sys.path.insert(0, "/workspace/repo/src")
from gap_repro.sim import robots

urdf = Path("/workspace/data/hf_cache/Assets/Robots/x5/X5A.urdf")
spec = robots.build_dual_x5_spec(urdf)
spec.visual.global_.offwidth = 1280
spec.visual.global_.offheight = 720
spec.worldbody.add_light(pos=[0.0, -1.0, 1.8], dir=[0.0, 0.3, -1.0], diffuse=[1, 1, 1])
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.05],
                        pos=[0, 0, 0], rgba=[0.3, 0.32, 0.36, 1])
model = spec.compile()
data = mujoco.MjData(model)
for side in ("left", "right"):
    for i, j in enumerate(robots.ARM_JOINTS):
        data.qpos[model.joint(f"{side}_{j}").qposadr[0]] = [0.3, -0.4, 0.5, -0.9, 0.4, 0.2][i]
mujoco.mj_forward(model, data)
r = mujoco.Renderer(model, 480, 640)
out = {}
for cam_id in range(model.ncam):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = cam_id
    r.update_scene(data, camera=cam)
    img = r.render()
    out[name] = img
    from PIL import Image
    Path("/workspace/data/w3_cameras").mkdir(exist_ok=True)
    Image.fromarray(img).save(f"/workspace/data/w3_cameras/{name}.png")
    print("RENDER", name, img.shape, "std", round(float(img.std()), 1), flush=True)
print("W3_CAMERAS_OK")
